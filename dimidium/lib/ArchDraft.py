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

from dimidium.lib.util import OptimizationStrategies
from dimidium.lib.ArchBrick import ArchBrick


class ArchDraft(object):

    # _bstr_fmt_ = "{:06}"
    # _bid_max_ = 99999

    def __init__(self, name, version, strategy: OptimizationStrategies, batch_size, target_sps=-1, target_latency=-1,
                 target_resources=-1, tvm_node=None):
        self.name = name
        self.version = version
        self.strategy = strategy
        self.batch_size = batch_size
        self.target_sps = target_sps
        self.target_latency = target_latency
        self.target_resources = target_resources
        self.main_tvm_node = tvm_node
        self.bricks = {}
        self.input_layer = None
        self.output_layer = None
        self.bid_cnt = 0

    def add_brick(self, brick: ArchBrick):
        # bstr = self._bstr_fmt_.format(self.bid_cnt)
        b_id = self.bid_cnt
        self.bid_cnt += 1
        # if self.bid_cnt > self._bid_max_:
        #    print("[DOSA:ArchDraft:ERROR] Brick Id overflow occurred!")
        brick.set_brick_id(b_id)
        self.bricks[b_id] = brick

    # def add_brick_dict(self, brick_dict):
    #    nb = ArchBrick(brick_id=None, dpl_dict=brick_dict)
    #    self.add_brick(nb)

    def set_tvm_node(self, tvm_node):
        self.main_tvm_node = tvm_node

    def set_input_layer(self, in_dpl):
        self.input_layer = in_dpl

    def set_output_layer(self, out_dpl):
        self.output_layer = out_dpl

