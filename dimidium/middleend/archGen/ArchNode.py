#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: July 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Class containing one DOSA node
#  *
#  *

import json

import dimidium.lib.singleton as dosa_singleton
from dimidium.lib.units import gigaU
from dimidium.middleend.archGen.ArchBrick import ArchBrick
from dimidium.backend.devices.dosa_device import DosaBaseHw, placeholderHw
from dimidium.backend.devices.dosa_roofline import DosaRoofline
from dimidium.lib.util import BrickImplTypes
from dimidium.middleend.archGen.ArchBlock import ArchBlock
from dimidium.backend.codeGen.EngineContainer import EngineContainer
from dimidium.middleend.archGen.CommPlan import CommPlan


class ArchNode(object):

    # _bstr_fmt_ = "{:06}"
    # _bid_max_ = 99999

    def __init__(self, node_id=-1, target_hw=None):
        self.node_id = node_id
        self.targeted_hw = target_hw
        self.bricks = {}
        self.bid_cnt = 0
        # self.latency_to_next_node = 0
        self.data_parallelism_level = 1  # round robin data parallelization, 1 = NO parallelism
        # self.twins = []  # compute parallelization
        self.predecessors = []
        self.successors = []
        self.ranks = []
        self.inp_ranks = []
        self.out_ranks = []
        self.req_iter_hz = -1
        self.roofline = None
        self.max_perf_F = -1
        self.max_perf_iter_based = -1
        self.used_perf_F = -1
        self.used_perf_share = 0
        self.used_comp_util_share = 0
        self.used_mem_util_share = 0
        self.max_iter_hz = -1
        self.used_iter_hz = -1
        self.possible_hw_types = []
        self.selected_hw_type = placeholderHw
        self.arch_block_list = []
        self.engine_container_refs = []
        self.build_tool = None
        self.comm_plan = None
        self.used_comm_lib = None
        self.skip_in_roofline = False
        self.total_pipeline_store = 0
        self.over_utilized_node = False

    def __repr__(self):
        return "ArchNode({}, {})".format(self.node_id, self.targeted_hw)

    def as_dict(self):
        res = {'node_id': self.node_id, 'targeted_hw': str(self.targeted_hw),
               'data_paral_level': self.data_parallelism_level,  # 'twin_nodes': [],
               'ranks': self.ranks, 'inp_ranks': self.inp_ranks, 'out_ranks': self.out_ranks,
               'pred_nodes': [], 'succ_nodes': [], 'possible_hw_types': [],
               'selected_hw_type': repr(self.selected_hw_type),
               'estimations': {'comp_util%': self.used_comp_util_share*100, 'mem_util%': self.used_mem_util_share*100,
                               # 'used_perf%': self.used_perf_share*100},
                               'req_iter_hz': self.req_iter_hz, 'used_iter_hz': self.used_iter_hz,
                               'impl_Gflop_eqiv': self.used_perf_F / gigaU,
                               'max_Gflop_based_on_impl_eqiv': self.max_perf_iter_based / gigaU},
               'over_utilized_node': self.over_utilized_node,
               'blocks': [], 'engineContainers': [],
               'bricks': {}}
        # for tn in self.twins:
        #    res['twin_nodes'].append(tn.node_id)
        for pn in self.predecessors:
            res['pred_nodes'].append(pn.node_id)
        for sn in self.successors:
            res['succ_nodes'].append(sn.node_id)
        for ph in self.possible_hw_types:
            phs = repr(ph)
            res['possible_hw_types'].append(phs)
        for bi in self.bricks:
            b = self.bricks[bi]
            res['bricks'][bi] = b.as_dict()
        for ab in self.arch_block_list:
            res['blocks'].append(repr(ab))
        for ec in self.engine_container_refs:
            res['engineContainers'].append(repr(ec))
        return res

    def __str__(self):
        ret = self.as_dict()
        return json.dumps(ret, indent=2)

    def add_brick(self, brick: ArchBrick, new_bid=None):
        if new_bid is None:
            # append at the end
            # bstr = self._bstr_fmt_.format(self.bid_cnt)
            b_id = self.bid_cnt
        else:
            b_id = new_bid
            new_bricks = self.bricks
            # new_bricks[new_bid] = brick
            for i in range(new_bid, self.bid_cnt):
                new_bricks[i+1] = self.bricks[i]
                new_bricks[i+1].set_brick_id(i+1)
            self.bricks = new_bricks
        self.bid_cnt += 1
        # if self.bid_cnt > self._bid_max_:
        #    print("[DOSA:ArchDraft:ERROR] Brick Id overflow occurred!")
        brick.set_brick_id(b_id)
        self.bricks[b_id] = brick

    def del_brick(self, b_id):
        if b_id == 0:
            print("[DOSA:ArchNode:WARNING] trying to delete last Brick of Node, skipping.")
            return -2
        if b_id >= self.bid_cnt or b_id < 0:
            print("[DOSA:ArchNode:ERROR] trying to delete invalid brick_id, skipping.")
            return -1
        for i in range(0, self.bid_cnt-1):
            if i <= (b_id - 1):
                continue
            self.bricks[i] = self.bricks[i+1]  # that's why -1 above
            self.bricks[i].set_brick_id(i)
        del self.bricks[self.bid_cnt-1]
        self.bid_cnt -= 1
        return 0

    def split_horizontal(self, b_id_to_new_node):
        if b_id_to_new_node == 0 or b_id_to_new_node >= self.bid_cnt or b_id_to_new_node < 0:
            print("[DOSA:ArchNode:ERROR] invalid split attempt, skipping.")
            return None
        new_node = ArchNode(target_hw=self.targeted_hw)
        new_node.max_perf_F = self.max_perf_F
        new_node.roofline = self.roofline
        # new_node.latency_to_next_node = self.latency_to_next_node
        self.successors.append(new_node)
        new_node.predecessors.append(self)
        for i in range(b_id_to_new_node, self.bid_cnt):
            new_node.add_brick(self.bricks[i])
        for i in reversed(range(b_id_to_new_node, self.bid_cnt)):
            del self.bricks[i]
        self.bid_cnt = len(self.bricks)
        self.update_used_perf_util_contr(add_switching_costs=True)
        new_node.update_used_perf_util_contr(add_switching_costs=True)
        return new_node  # afterwards draft.add_node/insert_node must be called

    def split_vertical(self, factor=2):
        assert factor > 1
        self.data_parallelism_level *= factor
        for lb in self.local_brick_iter_gen():
            lb.annotate_parallelization(factor)
        self.update_used_perf_util_contr()

    def set_node_id(self, node_id):
        self.node_id = node_id

    def get_node_id(self):
        return self.node_id

    def local_brick_iter_gen(self):
        for bi in self.bricks:
            bb = self.bricks[bi]
            yield bb

    def set_targeted_hw(self, target_hw: DosaBaseHw):
        self.targeted_hw = target_hw
        self.update_roofline()
        self.used_perf_F = -1

    def add_pred_node(self, p):
        assert type(p) is ArchNode
        self.predecessors.append(p)

    def add_succ_node(self, p):
        assert type(p) is ArchNode
        self.successors.append(p)

    # def add_twin_node(self, t):
    #     assert type(t) is ArchNode
    #     self.twins.append(t)

    def update_roofline(self):
        assert self.targeted_hw is not None
        nrl = DosaRoofline()
        nrl.from_perf_dict(self.targeted_hw.get_performance_dict())
        self.roofline = nrl
        self.max_perf_F = nrl.roof_F

    # def update_used_perf_util(self):
    #     total_perf_F = 0
    #     total_comp_per = 0
    #     total_mem_per = 0
    #     for lb in self.local_brick_iter_gen():
    #         total_perf_F += lb.req_flops
    #         if self.targeted_hw is not None:
    #             lb.update_util_estimation(self.targeted_hw)
    #             total_comp_per += lb.req_util_comp
    #             total_mem_per += lb.req_util_mem
    #     self.used_perf_F = total_perf_F
    #     self.used_comp_util_share = total_comp_per
    #     self.used_mem_util_share = total_mem_per
    #     self.used_perf_share = self.used_perf_F/self.max_perf_F

    def update_used_perf_util_contr(self, prefer_engine=False, add_switching_costs=False):
        # min_iter_hz_req = float('inf')
        max_iter_hz_req = 0
        min_iter_hz_impl = float('inf')
        total_comp_per = 0
        total_mem_per = 0
        total_switching_comp_share = 0
        total_switching_mem_share = 0
        total_flops_tmp = 0
        all_flops_valid = True
        cur_osg = None
        total_tensor_store = 0
        for lb in self.local_brick_iter_gen():
            if lb.req_iter_hz > max_iter_hz_req:
                max_iter_hz_req = lb.req_iter_hz
            if self.targeted_hw is not None:
                lb.update_util_estimation_contr(self.targeted_hw, prefer_engine)
                total_comp_per += lb.req_util_comp
                total_mem_per += lb.req_util_mem
                if lb.tmp_osg != cur_osg:
                    total_switching_comp_share += lb.switching_comp_share
                    total_switching_mem_share += lb.switching_mem_share
                    cur_osg = lb.tmp_osg
                if lb.iter_hz < min_iter_hz_impl:
                    min_iter_hz_impl = lb.iter_hz
                if lb.used_flops > 0:
                    total_flops_tmp += lb.used_flops
                    total_tensor_store += lb.local_pipeline_store
                else:
                    all_flops_valid = False
        if add_switching_costs:
            total_comp_per += total_switching_comp_share
            total_mem_per += total_switching_mem_share
        max_util = max(total_comp_per, total_mem_per)
        max_iter = (1.0/max_util) * min_iter_hz_impl
        self.max_iter_hz = max_iter
        self.req_iter_hz = max_iter_hz_req
        self.used_iter_hz = min_iter_hz_impl
        self.used_comp_util_share = total_comp_per
        self.used_mem_util_share = total_mem_per
        self.used_perf_share = self.used_iter_hz/self.max_iter_hz
        if all_flops_valid:
            self.used_perf_F = total_flops_tmp
            self.max_perf_iter_based = total_flops_tmp * (max_iter/min_iter_hz_impl)
            self.total_pipeline_store = total_tensor_store
        if max_util > dosa_singleton.config.utilization.dosa_xi:
            self.over_utilized_node = True
        else:
            self.over_utilized_node = False

    def update_kernel_uuids(self, kuuid_start):
        # TODO: take parallelism into account?
        next_uuid = kuuid_start
        for bb in self.local_brick_iter_gen():
            next_uuid = bb.update_global_ids(next_uuid)
        return next_uuid

    def update_brick_uuids(self, buuid_start):
        # TODO: take parallelism into account?
        next_uuid = buuid_start
        for bb in self.local_brick_iter_gen():
            bb.set_brick_uuid(next_uuid)
            next_uuid += 1
        return next_uuid

    def update_block_ids(self, bluuid_start):
        next_uuid = bluuid_start
        local_id = 0
        for ab in self.arch_block_list:
            ab.set_local_block_id(local_id)
            ab.set_block_uuid(next_uuid)
            local_id += 1
            next_uuid += 1
        return next_uuid

    # def update_possible_osgs(self):
    #     for bb in self.local_brick_iter_gen():
    #         bb.update_possible_osgs()

    def update_possible_hw_types(self):
        cur_possible_hw_types = []
        for bb in self.local_brick_iter_gen():
            bb.update_possible_hw_types()
            bb_phw = bb.possible_hw_types
            # add all possibilities
            for bb_pht in bb_phw:
                if bb_pht not in cur_possible_hw_types:
                    cur_possible_hw_types.append(bb_pht)
        not_possible_hw_types = []
        for bb in self.local_brick_iter_gen():
            # now, remove all non-common options
            bb_phw = bb.possible_hw_types
            for cpht in cur_possible_hw_types:
                if cpht not in bb_phw:
                    not_possible_hw_types.append(cpht)
        not_possible_hw_types = list(set(not_possible_hw_types))
        for npht in not_possible_hw_types:
            del cur_possible_hw_types[cur_possible_hw_types.index(npht)]
        self.possible_hw_types = cur_possible_hw_types

    def update_block_list(self):
        self.arch_block_list = []
        cur_impl_type = None
        cur_osg = None
        cur_block = None
        cur_block_id = 0
        for bb in self.local_brick_iter_gen():
            if bb.selected_osg != cur_osg or bb.selected_impl_type != cur_impl_type \
                or (cur_osg is not None and cur_block is not None
                    # and cur_impl_type != BrickImplTypes.ENGINE
                    #   with additional compiler steps for Engine --> doesn't make sense any more
                    and len(cur_block.brick_list) >= cur_osg.suggested_max_block_length):  # TODO: remove?
                if cur_block is not None:
                    self.arch_block_list.append(cur_block)
                cur_impl_type = bb.selected_impl_type
                cur_osg = bb.selected_osg
                cur_block = ArchBlock(self, cur_block_id, cur_impl_type, cur_osg, [bb])
                cur_block_id += 1
            else:
                cur_block.add_brick(bb)
        if cur_block is not None:
            self.arch_block_list.append(cur_block)

        self.engine_container_refs = []
        for ab in self.arch_block_list:
            if ab.block_impl_type == BrickImplTypes.ENGINE:
                new_container = EngineContainer(ab)
                self.engine_container_refs.append(new_container)

    def build(self):
        if self.build_tool is None:
            self.build_tool = self.selected_hw_type.create_build_tool(self.node_id)
            self.build_tool.create_build_dir(self.node_id)
        assert self.comm_plan is not None and self.used_comm_lib is not None
        self.used_comm_lib.build(self.comm_plan, self.build_tool)
        for ab in self.arch_block_list:
            ab.build(self.build_tool)
        self.build_tool.write_build_scripts()

    # def synth(self):
    #     if self.build_tool is None:
    #         print("[DOSA:ArchNode:INFO] Must build before synthesis, starting build now...")
    #         self.build()
    #     for ab in self.arch_block_list:
    #         ab.synth()

    def generate_communication(self, comm_lib, pipeline_store_until_now):
        self.used_comm_lib = comm_lib
        self.comm_plan = CommPlan(self, pipeline_store_until_now)
