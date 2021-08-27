#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: July 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Class containing one version of a DOSA architectural draft
#  *
#  *

import math
import json

from dimidium.lib.util import OptimizationStrategies
from dimidium.lib.ArchNode import ArchNode
from dimidium.lib.devices.dosa_device import DosaBaseHw
from dimidium.lib.devices.dosa_roofline import RooflineRegions


class ArchDraft(object):

    # _bstr_fmt_ = "{:06}"
    # _bid_max_ = 99999

    def __init__(self, name, version, strategy: OptimizationStrategies, batch_size, sample_size, target_sps=-1, target_latency=-1,
                 target_resources=-1, tvm_node=None):
        self.name = name
        self.version = version
        self.strategy = strategy
        self.batch_size = batch_size
        self.target_sps = target_sps
        self.target_latency = target_latency
        self.target_resources = target_resources
        self.main_tvm_handle = tvm_node
        self.sample_size_B = sample_size
        self.nodes = {}
        self.nid_cnt = 0
        self.input_layer = None
        self.output_layer = None
        self.calc_throughput = -1
        self.calc_latency = -1
        self.calc_resources = -1
        self.target_hw_set = []
        self.fallback_hw_set = []
        self.tmp_notes = {}

    def __repr__(self):
        return "ArchDraft({}, {}, {})".format(self.name, self.version, self.strategy)

    def as_dict(self):
        res = {'name': self.name, 'version': self.version, 'strategy': str(self.strategy),
               'batch_size': self.batch_size, 'target_sps': self.target_sps, 'target_latency': self.target_latency,
               'target_resources': self.target_resources,
               'input': str(self.input_layer), 'output': str(self.output_layer),
               'main_tvm_handle': str(self.main_tvm_handle)[:100],
               'target_hw_set': [], 'fallback_hw_set': [],
               'nodes': {}}
        for thw in self.target_hw_set:
            tn = type(thw).__name__
            res['target_hw_set'].append(tn)
        for fhw in self.fallback_hw_set:
            fn = type(fhw).__name__
            res['fallback_hw_set'].append(fn)
        for ni in self.nodes:
            n = self.nodes[ni]
            res['nodes'][ni] = n.as_dict()
        return res

    def __str__(self):
        ret = self.as_dict()
        return json.dumps(ret, indent=2)

    def add_node(self, node: ArchNode):
        # bstr = self._bstr_fmt_.format(self.bid_cnt)
        n_id = self.nid_cnt
        self.nid_cnt += 1
        # if self.bid_cnt > self._bid_max_:
        #    print("[DOSA:ArchDraft:ERROR] Brick Id overflow occurred!")
        node.set_node_id(n_id)
        self.nodes[n_id] = node

    def insert_node(self, node: ArchNode, new_id):
        if new_id > self.nid_cnt or new_id < 0:
            print("[DOSA:ArchDraft:ERROR] insert node with invalid new_id, skipping.")
            return
        for i in reversed(range(new_id, self.nid_cnt)):
            self.nodes[i+1] = self.nodes[i]
            self.nodes[i+1].set_node_id(i+1)
        node.set_node_id(new_id)
        self.nodes[new_id] = node
        self.nid_cnt += 1

    def set_tvm_handle(self, tvm_node):
        self.main_tvm_handle = tvm_node

    def set_input_layer(self, in_dpl):
        self.input_layer = in_dpl

    def set_output_layer(self, out_dpl):
        self.output_layer = out_dpl

    def brick_iter_gen(self):
        for ni in self.nodes:
            nn = self.nodes[ni]
            for bi in nn.bricks:
                bb = nn.bricks[bi]
                yield bb

    def node_iter_gen(self):
        for ni in self.nodes:
            nn = self.nodes[ni]
            yield nn

    def get_bricks_num(self):
        ret = 0
        for ni in self.nodes:
            ret += len(self.nodes[ni].bricks)
        return ret

    def add_possible_target_hw(self, nth: DosaBaseHw):
        self.target_hw_set.append(nth)

    def add_possible_fallback_hw(self, nfh: DosaBaseHw):
        self.fallback_hw_set.append(nfh)

    def legalize(self):
        # 0. split based on "used_perf"
        # build list of original nodes
        orig_nodes_handles = []
        for nn in self.node_iter_gen():
            orig_nodes_handles.append(nn)
        for nn in orig_nodes_handles:
            nn.update_used_perf()
            all_new_nodes = []
            cur_node = nn
            while cur_node.used_perf_F > nn.max_perf_F:
                cur_used_perf_F = 0
                for i in range(0, cur_node.bid_cnt):
                    cur_used_perf_F += cur_node.bricks[i].req_flops
                    if cur_used_perf_F > nn.max_perf_F:
                        if i == 0:
                            i = 1
                        new_node = cur_node.split_horizontal(i)  # including update_used_perf
                        all_new_nodes.append(new_node)
                        cur_node = new_node
                        break
            if len(all_new_nodes) > 0:
                new_nid = nn.get_node_id() + 1
                for new_node in all_new_nodes:
                    self.insert_node(new_node, new_nid)
                    new_nid += 1
        assert len(self.nodes) == self.nid_cnt
        self.update_required_perf()  # to consider new latencies
        # 1. compute parallelization for engine and stream (i.e. regions 1 and 4)
        #  update: compute parallelization (i.e. blocking of compute ops) is difficult, hence also round robin here
        for nn in self.node_iter_gen():
            # only nodes with one brick should be affected
            need_to_split = False
            split_factor = 0
            for lb in nn.local_brick_iter_gen():
                # engine and stream
                rr = nn.roofline.get_region(lb.oi_stream, lb.req_flops)
                if rr == RooflineRegions.ABOVE_TOP or rr == RooflineRegions.ABOVE_BRAM:
                    need_to_split = True
                    nsf = lb.req_flops / nn.max_perf_F
                    if nsf > split_factor:
                        split_factor = nsf
                rr = nn.roofline.get_region(lb.oi_engine, lb.req_flops)
                if rr == RooflineRegions.ABOVE_TOP or rr == RooflineRegions.ABOVE_BRAM:
                    need_to_split = True
                    nsf = lb.req_flops / nn.max_perf_F
                    if nsf > split_factor:
                        split_factor = nsf
            if need_to_split:
                split_factor_up = math.ceil(split_factor)
                if split_factor_up < 2:
                    split_factor_up = 2
                nn.split_vertical(factor=split_factor_up)  # including update of used perf
        # 2. select engine or stream: check if both in same region, select the region that is "to the right"
        # 3. data parallelization for all above IN_HOUSE
        # 4. merge sequential nodes (no data par, no twins, same target_hw) if possible, based on used_perf,
        # (i.e move bricks after each other)

    def update_required_perf(self):
        if self.strategy == OptimizationStrategies.THROUGHPUT:
            if self.target_sps < 0:
                print("[DOSA:archGen:ERROR] Optimization strategy ({}) does not fit target numbers in constraint target_sps ({}). Stop."
                      .format(self.strategy, self.target_sps))
                return -1
            # optimizing towards throughput
            target_throughput = self.target_sps * self.sample_size_B
            # annotate input & output
            self.input_layer['inp_Bs'] = self.input_layer['inpB'] * self.target_sps
            self.input_layer['out_Bs'] = self.input_layer['outB'] * self.target_sps
            self.output_layer['inp_Bs'] = self.output_layer['inpB'] * self.target_sps
            self.output_layer['out_Bs'] = self.output_layer['outB'] * self.target_sps
            # annotate bricks
            # pipelined design: no communication latency
            for node in self.node_iter_gen():
                # take data parallelism into account
                local_data_par_level = node.data_parallelism_level
                for brick in node.local_brick_iter_gen():
                    brick.input_bw_Bs = brick.input_bytes * (self.target_sps/local_data_par_level)
                    brick.output_bw_Bs = brick.output_bytes * (self.target_sps/local_data_par_level)
                    brick.req_flops = brick.flops * (self.target_sps/local_data_par_level)
        elif self.strategy == OptimizationStrategies.LATENCY:
            # optimizing towards latency
            if self.target_latency < 0:
                print("[DOSA:archGen:ERROR] Optimization strategy ({}) does not fit target numbers in constraint target_latency ({}). Stop."
                      .format(self.strategy, self.target_latency))
                return -1
            # first, try with 1/N distribution
            # consider communication latency
            comm_latency = 0
            for nn in self.node_iter_gen():
                if nn.target_hw is not None:
                    cl1 = nn.target_hw.get_comm_latency_s()
                    # just consider it two times, don't consider successor
                    comm_latency += 2 * cl1
            compute_latency = self.target_latency - comm_latency
            if compute_latency < 0:
                print("[DOSA:archGen:ERROR] Communication latency ({}) is higher than target latency ({}). "
                      .format(comm_latency, self.target_latency) +
                      "Impossible to generate architecture. Stop.")
                return -1
            latency_per_brick = compute_latency / float(self.get_bricks_num())
            for node in self.node_iter_gen():
                # take data parallelism into account
                local_data_par_level = node.data_parallelism_level
                for brick in node.local_brick_iter_gen():
                    brick.req_latency = latency_per_brick
                    # calc_latency is depending on mode
                    # brick.req_perf_engine = (brick.oi_engine * brick.input_bytes) / latency_per_brick
                    # brick.req_perf_stream = (brick.oi_stream * brick.input_bytes) / latency_per_brick
                    brick.req_flops = brick.flops / (latency_per_brick*local_data_par_level)
        else:
            # optimizing towards resource footprint
            if self.target_resources < 0:
                print("[DOSA:archGen:ERROR] Optimization strategy ({}) does not fit target numbers in constraint target_resources ({}). Stop."
                      .format(self.strategy, self.target_resources))
                return -1
            # find max resources in flops
            max_resources = 0
            max_res_dev = "unknown"
            for d in self.target_hw_set:
                dr = d.get_max_flops()
                if dr > max_resources:
                    max_resources = dr
                    max_res_dev = d.name
            allowed_resources = max_resources * self.target_resources
            self.tmp_notes['max_res_dev'] = max_res_dev
            self.tmp_notes['allowed_resources'] = allowed_resources
            # first, try with 1/N distribution
            # ignore latency, data_par etc.
            resource_per_brick = allowed_resources / self.get_bricks_num()
            for brick in self.brick_iter_gen():
                brick.req_flops = resource_per_brick
        return 0
