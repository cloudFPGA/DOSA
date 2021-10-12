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

from dimidium.lib.util import OptimizationStrategies, BrickImplTypes, DosaRv
from dimidium.middleend.archGen.ArchNode import ArchNode
from dimidium.middleend.archGen.ArchBrick import ArchBrick
from dimidium.middleend.archGen.ArchOp import ArchOp
from dimidium.backend.devices.dosa_device import DosaBaseHw, placeholderHw
from dimidium.backend.devices.dosa_roofline import RooflineRegions, get_rightmost_roofline_region
from dimidium.backend.operatorSets.BaseOSG import sort_osg_list


class ArchDraft(object):

    # _bstr_fmt_ = "{:06}"
    # _bid_max_ = 99999

    def __init__(self, name, version, strategy: OptimizationStrategies, batch_size, sample_size, target_sps=-1,
                 target_latency=-1,
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
        self.possible_hw_types = []

    def __repr__(self):
        return "ArchDraft({}, {}, {})".format(self.name, self.version, self.strategy)

    def as_dict(self):
        res = {'name': self.name, 'version': self.version, 'strategy': str(self.strategy),
               'batch_size': self.batch_size, 'target_sps': self.target_sps, 'target_latency': self.target_latency,
               'target_resources': self.target_resources,
               'input': str(self.input_layer), 'output': str(self.output_layer),
               'main_tvm_handle': str(self.main_tvm_handle)[:100],
               'possible_hw_types': [], 'target_hw_set': [], 'fallback_hw_set': [],
               'nodes': {}}
        for thw in self.target_hw_set:
            tn = type(thw).__name__
            res['target_hw_set'].append(tn)
        for fhw in self.fallback_hw_set:
            fn = type(fhw).__name__
            res['fallback_hw_set'].append(fn)
        for ph in self.possible_hw_types:
            phs = repr(ph)
            res['possible_hw_types'].append(phs)
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
            self.nodes[i + 1] = self.nodes[i]
            self.nodes[i + 1].set_node_id(i + 1)
        node.set_node_id(new_id)
        self.nodes[new_id] = node
        self.nid_cnt += 1

    def delete_node(self, id_to_delete):
        if id_to_delete > self.nid_cnt or id_to_delete < 0:
            print("[DOSA:ArchDraft:ERROR] delete node with invalid id, skipping.")
            return
        del self.nodes[id_to_delete]
        self.nid_cnt -= 1
        for i in range(id_to_delete, self.nid_cnt):
            self.nodes[i].set_node_id(i)

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

    def op_iter_gen(self):
        for ni in self.nodes:
            nn = self.nodes[ni]
            for bi in nn.bricks:
                bb = nn.bricks[bi]
                for op in bb.ops:
                    oo = bb.ops[op]
                    yield oo

    def get_bricks_num(self):
        ret = 0
        for ni in self.nodes:
            ret += len(self.nodes[ni].bricks)
        return ret

    def add_possible_target_hw(self, nth: DosaBaseHw):
        self.target_hw_set.append(nth)

    def add_possible_fallback_hw(self, nfh: DosaBaseHw):
        self.fallback_hw_set.append(nfh)

    def get_total_nodes_cnt(self):
        total_nodes = 0
        for nn in self.node_iter_gen():
            total_nodes += nn.data_parallelism_level
        return total_nodes

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
                if split_factor_up < 1:
                    split_factor_up = 1
                nn.split_vertical(factor=split_factor_up)  # including update of used perf
        # 2. select engine or stream: check if both in same region, select the region that is "to the right"
        for nn in self.node_iter_gen():
            if nn.bid_cnt == 1:
                # only one brick -> stream
                nn.bricks[0].set_impl_type(BrickImplTypes.STREAM)
                continue
            for lb in nn.local_brick_iter_gen():
                if lb.flops == 0:
                    # means re-mapping of input -> always stream in FPGAs
                    lb.set_impl_type(BrickImplTypes.STREAM)
                    continue
                r1 = nn.roofline.get_region(lb.oi_engine, lb.req_flops)
                r2 = nn.roofline.get_region(lb.oi_stream, lb.req_flops)
                br = get_rightmost_roofline_region(r1, r2)
                if br == 1:
                    lb.set_impl_type(BrickImplTypes.ENGINE)
                else:
                    lb.set_impl_type(BrickImplTypes.STREAM)
        # ensure all are decided
        for bb in self.brick_iter_gen():
            assert bb.selected_impl_type != BrickImplTypes.UNDECIDED
        # 3. data parallelization for all above IN_HOUSE, based on selected impl type
        for nn in self.node_iter_gen():
            # only nodes with one brick should be affected
            need_to_split = False
            split_factor = 0
            for lb in nn.local_brick_iter_gen():
                oi_selected = lb.get_oi_selected_impl()
                rr = nn.roofline.get_region(oi_selected, lb.req_flops)
                if rr == RooflineRegions.IN_HOUSE:
                    # nothing to do in all cases
                    continue
                ap_engine = nn.roofline.get_max_perf_at_oi(lb.oi_engine, ignore_net=True)
                ap_stream = nn.roofline.get_max_perf_at_oi(lb.oi_stream)
                if lb.selected_impl_type == BrickImplTypes.ENGINE:
                    # oi_stream must be below network even if Enigne is selected
                    # ap_stream is containing only user data to transmit
                    if lb.req_flops > ap_stream:
                        need_to_split = True
                        nsf = lb.req_flops / ap_stream
                        if nsf > split_factor:
                            split_factor = nsf
                    elif lb.req_flops > ap_engine:
                        need_to_split = True
                        nsf = lb.req_flops / ap_engine
                        if nsf > split_factor:
                            split_factor = nsf
                else:
                    if lb.req_flops > ap_stream:
                        need_to_split = True
                        nsf = lb.req_flops / ap_stream
                        if nsf > split_factor:
                            split_factor = nsf
            if need_to_split:
                split_factor_up = math.ceil(split_factor)
                if split_factor_up < 1:
                    split_factor_up = 1
                nn.split_vertical(factor=split_factor_up)  # including update of used perf
        # 4. for each node: turn lone engine impls into streams
        #  (i.e. if the sequence is 1 engine, 2 stream, and 3 & 4 engine --> first engine doesn't make sense)
        #  in other words: ensure that all engine sets are bigger or equal 2
        #  no need to update req. perf or split nodes, is already considered in step 3
        for nn in self.node_iter_gen():
            cur_engine_set = []
            turn_engine_to_stream_list = []
            for bi in range(0, nn.bid_cnt):
                bb = nn.bricks[bi]
                if bb.selected_impl_type == BrickImplTypes.STREAM:
                    if len(cur_engine_set) < 2:
                        turn_engine_to_stream_list.extend(cur_engine_set)
                    cur_engine_set = []
                    continue
                cur_engine_set.append(bi)
            # last time
            if len(cur_engine_set) < 2:
                turn_engine_to_stream_list.extend(cur_engine_set)
            for bi in turn_engine_to_stream_list:
                bb = nn.bricks[bi]
                bb.set_impl_type(BrickImplTypes.STREAM)
        # 5. merge sequential nodes (no data par, no twins, same targeted_hw) if possible, based on used_perf,
        #  (i.e move bricks after each other)
        node_ids_to_delete = []
        for ni in range(0, len(self.nodes)):
            if ni in node_ids_to_delete:
                continue
            n1 = self.nodes[ni]
            if n1.data_parallelism_level > 1:
                continue
            if ni < (len(self.nodes) - 1):
                n2 = self.nodes[ni + 1]
                if n2.data_parallelism_level > 1:
                    continue
                if n1.targeted_hw == n2.targeted_hw:
                    # TODO: move bricks after each other?
                    #  does it make sense? just reduces the resource usage somewhere else?
                    if (n1.used_perf_F + n2.used_perf_F) <= n1.max_perf_F:
                        # merge nodes totally
                        print("[DOSA:archGen:INFO] merging sequential, non-parallel nodes {} and {} totally."
                              .format(n1.node_id, n2.node_id))
                        node_ids_to_delete.append(n1 + 1)
                        for bb in n2.local_brick_iter_gen():
                            n1.add_brick(bb)
        for nd in node_ids_to_delete:
            self.delete_node(nd)
        # 6. update OSGs and possible hw targets
        self.update_possible_osgs()
        self.update_possible_hw_types()
        # 7. decide for hw, if targeted hw is possible, use this one
        #  if not, use other possible hw with largest roof_F
        #  if only smaller roof_F are available, use fallback hw
        #  (optimizing hw --> not part of legalizing)
        for nn in self.node_iter_gen():
            if nn.targeted_hw in nn.possible_hw_types:
                nn.selected_hw_type = nn.targeted_hw
            else:
                targeted_roof_F = nn.targeted_hw.get_roof_F()
                max_roof_F = 0
                selected_hw = None
                for phw in nn.possible_hw_types:
                    # if only smaller roof_F are available, use fallback hw
                    phw_roof_F = phw.get_roof_F()
                    if phw_roof_F >= targeted_roof_F:
                        if phw_roof_F > max_roof_F:
                            max_roof_F = phw_roof_F
                            selected_hw = phw
                if selected_hw is not None:
                    nn.selected_hw_type = selected_hw
                    print("[DOSA:archGen:INFO] Targeted hw {} not possible for node {}, selected {} instead."
                          .format(nn.targeted_hw, nn.node_id, nn.selected_hw_type))
                else:
                    if len(self.fallback_hw_set) > 0:
                        fallback_hw_found = False
                        for fhw in self.fallback_hw_set:
                            if fhw in nn.possible_hw_types:
                                # take first one, order represents priority
                                nn.selected_hw_type = fhw
                                fallback_hw_found = True
                                print("[DOSA:archGen:WARNING] Targeted hw {} not possible for node {}, forced to use \
                                      fallback hw {} instead.".format(nn.targeted_hw, nn.node_id,
                                                                        nn.selected_hw_type))
                                break
                        if not fallback_hw_found:
                            print("[DOSA:archGen:ERROR] Targeted hw {} not possible for node {}, failed to find \
                                  replacement or fallback. Impossible to legalize draft"
                                  .format(nn.targeted_hw, nn.node_id))
                            return DosaRv.ERROR
        # ensure, all HW is decided
        for nn in self.node_iter_gen():
            assert nn.selected_hw_type != placeholderHw
        # 8. decide for OSG
        for nn in self.node_iter_gen():
            decided_hw_class = nn.selected_hw_type.hw_class
            # for now, decide for one OSG, if possible (i.e. avoid switching at all costs)
            # TODO: later, consider individual switching costs
            all_possible_osgs = []
            for lb in nn.local_brick_iter_gen():
                for posg in lb.possible_osgs:
                    if decided_hw_class in posg.device_classes:
                        if posg not in all_possible_osgs:
                            all_possible_osgs.append(posg)
            all_possible_sorted = sort_osg_list(all_possible_osgs)
            found_consens = False
            consens_osg = None
            if len(all_possible_sorted) >= 1:
                not_possible_osgs = []
                for cur_posg in all_possible_sorted:
                    suitable = True
                    if cur_posg in not_possible_osgs:
                        continue
                    for lb in nn.local_brick_iter_gen():
                        if cur_posg not in lb.possible_osgs:
                            not_possible_osgs.append(cur_posg)
                            suitable = False
                    if suitable:
                        # oder still represents priority, so take this one
                        found_consens = True
                        consens_osg = cur_posg
                        break
            if found_consens:
                for lb in nn.local_brick_iter_gen():
                    lb.selected_osg = consens_osg
            else:
                # no common osg found, use whatever is best
                print("[DOSA:archGen:INFO] couldn't find common OSG for node {}, use individual ones \
                (higher switching costs).".format(nn.node_id))
                for lb in nn.local_brick_iter_gen():
                    for posg in lb.possible_osgs:
                        if decided_hw_class in posg.device_classes:
                            # order represents priority, so take first possible one
                            lb.selected_osg = posg
                            continue
        # 9. update kernel uuids & req. perf
        self.update_uuids()
        self.update_required_perf()
        return DosaRv.OK

    def update_required_perf(self):
        if self.strategy == OptimizationStrategies.THROUGHPUT:
            if self.target_sps < 0:
                print(
                    "[DOSA:archGen:ERROR] Optimization strategy ({}) does not fit target numbers in constraint \
                    target_sps ({}). Stop."
                    .format(self.strategy, self.target_sps))
                return DosaRv.ERROR
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
                    brick.input_bw_Bs = brick.input_bytes * (self.target_sps / local_data_par_level)
                    brick.output_bw_Bs = brick.output_bytes * (self.target_sps / local_data_par_level)
                    brick.req_flops = brick.flops * (self.target_sps / local_data_par_level)
        elif self.strategy == OptimizationStrategies.LATENCY:
            # optimizing towards latency
            if self.target_latency < 0:
                print(
                    "[DOSA:archGen:ERROR] Optimization strategy ({}) does not fit target numbers in constraint \
                     target_latency ({}). Stop."
                    .format(self.strategy, self.target_latency))
                return DosaRv.ERROR
            # first, try with 1/N distribution
            # consider communication latency
            comm_latency = 0
            for nn in self.node_iter_gen():
                if nn.targeted_hw is not None:
                    cl1 = nn.targeted_hw.get_comm_latency_s()
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
                    brick.req_flops = brick.flops / (latency_per_brick * local_data_par_level)
        else:
            # optimizing towards resource footprint
            if self.target_resources < 0:
                print(
                    "[DOSA:archGen:ERROR] Optimization strategy ({}) does not fit target numbers in constraint \
                    target_resources ({}). Stop."
                    .format(self.strategy, self.target_resources))
                return DosaRv.ERROR
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
        return DosaRv.OK

    def update_uuids(self):
        next_kuuid = 0
        next_buuid = 0
        for nn in self.node_iter_gen():
            next_kuuid = nn.update_kernel_uuids(next_kuuid)
            next_buuid = nn.update_brick_uuids(next_buuid)

    def update_possible_osgs(self):
        for nn in self.node_iter_gen():
            nn.update_possible_osgs()

    def update_possible_hw_types(self):
        cur_possible_hw_types = []
        for nn in self.node_iter_gen():
            nn.update_possible_hw_types()
            nn_phw = nn.possible_hw_types
            # add all possibilities
            for nn_pht in nn_phw:
                if nn_pht not in cur_possible_hw_types:
                    cur_possible_hw_types.append(nn_pht)
        not_possible_hw_types = []
        for nn in self.node_iter_gen():
            nn_phw = nn.possible_hw_types
            # now, remove all non-common options
            for cpht in cur_possible_hw_types:
                if cpht not in nn_phw:
                    not_possible_hw_types.append(cpht)
        not_possible_hw_types = list(set(not_possible_hw_types))
        for npht in not_possible_hw_types:
            del cur_possible_hw_types[cur_possible_hw_types.index(npht)]
        self.possible_hw_types = cur_possible_hw_types
