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

from dimidium.lib.ArchBrick import ArchBrick
from dimidium.lib.devices.dosa_device import DosaBaseHw
from dimidium.lib.devices.dosa_roofline import DosaRoofline


class ArchNode(object):

    # _bstr_fmt_ = "{:06}"
    # _bid_max_ = 99999

    def __init__(self, node_id=-1, target_hw=None):
        self.node_id = node_id
        self.target_hw = target_hw
        self.bricks = {}
        self.bid_cnt = 0
        self.latency_to_next_node = 0
        self.number_of_round_robin = 0  # data parallelization
        self.twins = []  # compute parallelization
        self.predecessors = []
        self.successors = []
        self.roofline = None
        self.max_perf_F = -1
        self.used_perf_F = -1

    def __repr__(self):
        return "ArchNode({}, {})".format(self.node_id, self.target_hw)

    def as_dict(self):
        res = {'node_id': self.node_id, 'target_hw': str(self.target_hw),
               'data_paral_level': self.number_of_round_robin, 'twin_nodes': [],
               'pred_nodes': [], 'succ_nodes': [],
               'bricks': {}}
        for tn in self.twins:
            res['twin_nodes'].append(tn.node_id)
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

    # def add_brick_dict(self, brick_dict):
    #    nb = ArchBrick(brick_id=None, dpl_dict=brick_dict)
    #    self.add_brick(nb)

    def set_node_id(self, node_id):
        self.node_id = node_id

    def get_node_id(self):
        return self.node_id

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

    def add_twin_node(self, t):
        assert type(t) is ArchNode
        self.twins.append(t)

    def update_roofline(self):
        assert self.target_hw is not None
        nrl = DosaRoofline()
        nrl.from_perf_dict(self.target_hw.get_performance_dict)
        self.roofline = nrl
        self.max_perf_F = nrl.roof_F
