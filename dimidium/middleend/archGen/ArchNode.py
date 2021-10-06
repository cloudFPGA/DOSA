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

from dimidium.middleend.archGen.ArchBrick import ArchBrick
from dimidium.backend.devices import DosaBaseHw
from dimidium.backend.devices.dosa_roofline import DosaRoofline
from dimidium.lib.util import BrickImplTypes


class ArchNode(object):

    # _bstr_fmt_ = "{:06}"
    # _bid_max_ = 99999

    def __init__(self, node_id=-1, target_hw=None):
        self.node_id = node_id
        self.target_hw = target_hw
        self.bricks = {}
        self.bid_cnt = 0
        # self.latency_to_next_node = 0
        self.data_parallelism_level = 1  # round robin data parallelization, 1 = NO parallelism
        # self.twins = []  # compute parallelization
        self.predecessors = []
        self.successors = []
        self.roofline = None
        self.max_perf_F = -1
        self.used_perf_F = -1

    def __repr__(self):
        return "ArchNode({}, {})".format(self.node_id, self.target_hw)

    def as_dict(self):
        res = {'node_id': self.node_id, 'target_hw': str(self.target_hw),
               'data_paral_level': self.data_parallelism_level,  # 'twin_nodes': [],
               'pred_nodes': [], 'succ_nodes': [],
               'bricks': {}}
        # for tn in self.twins:
        #    res['twin_nodes'].append(tn.node_id)
        for pn in self.predecessors:
            res['pred_nodes'].append(pn.node_id)
        for sn in self.successors:
            res['succ_nodes'].append(sn.node_id)
        for bi in self.bricks:
            b = self.bricks[bi]
            res['bricks'][bi] = b.as_dict()
        return res

    def __str__(self):
        ret = self.as_dict()
        return json.dumps(ret, indent=2)

    def add_brick(self, brick: ArchBrick):
        # bstr = self._bstr_fmt_.format(self.bid_cnt)
        b_id = self.bid_cnt
        self.bid_cnt += 1
        # if self.bid_cnt > self._bid_max_:
        #    print("[DOSA:ArchDraft:ERROR] Brick Id overflow occurred!")
        brick.set_brick_id(b_id)
        self.bricks[b_id] = brick

    def del_brick(self, b_id):
        if b_id == 0:
            print("[DOSA:ArchNode:ERROR] trying to delete last Brick of Node, skipping.")
            return
        if b_id >= self.bid_cnt or b_id < 0:
            print("[DOSA:ArchNode:ERROR] trying to delete invalid brick_id, skipping.")
            return
        for i in range(0, self.bid_cnt-1):
            if i <= (b_id - 1):
                continue
            self.bricks[i] = self.bricks[i+1]  # that's why -1 above
            self.bricks[i].set_brick_id(i)
        del self.bricks[self.bid_cnt-1]
        self.bid_cnt -= 1

    def split_horizontal(self, b_id_to_new_node):
        if b_id_to_new_node == 0 or b_id_to_new_node >= self.bid_cnt or b_id_to_new_node < 0:
            print("[DOSA:ArchNode:ERROR] invalid split attempt, skipping.")
            return None
        new_node = ArchNode(target_hw=self.target_hw)
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
        self.update_used_perf()
        new_node.update_used_perf()
        return new_node  # afterwards draft.add_node/insert_node must be called

    def split_vertical(self, factor=2):
        assert factor > 1
        self.data_parallelism_level *= factor
        for lb in self.local_brick_iter_gen():
            lb.req_flops /= factor
            if lb.selected_impl_type == BrickImplTypes.STREAM:
                lb.req_flops_stream /= factor
            elif lb.selected_impl_type == BrickImplTypes.ENGINE:
                lb.req_flops_engine /= factor
        self.update_used_perf()

    def set_node_id(self, node_id):
        self.node_id = node_id

    def get_node_id(self):
        return self.node_id

    def local_brick_iter_gen(self):
        for bi in self.bricks:
            bb = self.bricks[bi]
            yield bb

    def set_target_hw(self, target_hw: DosaBaseHw):
        self.target_hw = target_hw
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
        assert self.target_hw is not None
        nrl = DosaRoofline()
        nrl.from_perf_dict(self.target_hw.get_performance_dict())
        self.roofline = nrl
        self.max_perf_F = nrl.roof_F

    def update_used_perf(self):
        total_perf_F = 0
        for lb in self.local_brick_iter_gen():
            total_perf_F += lb.req_flops
        self.used_perf_F = total_perf_F

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


