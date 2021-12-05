#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Dec 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Filter for Optimization passes
#  *
#  *

import abc

from dimidium.middleend.archGen.ArchBrick import ArchBrick
from dimidium.middleend.archGen.ArchOp import ArchOp


class ArchFilter(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def match_brick(self, brick: ArchBrick):
        print("[DOSA:Filter:ERROR] NOT YET IMPLEMENTED.")
        return False

    @abc.abstractmethod
    def match_op(self, op: ArchOp):
        print("[DOSA:Filter:ERROR] NOT YET IMPLEMENTED.")
        return False


class OpCallFilter(ArchFilter):

    def __init__(self, op_call_list):
        self.op_call_list = op_call_list

    def match_brick(self, brick: ArchBrick):
        print("[DOSA:Filter:ERROR] Can't filter Bricks with op calls")
        return False

    def match_op(self, op: ArchOp):
        return op.op_call in self.op_call_list


class OiThresholdFilter(ArchFilter):

    def __init__(self, upper_oi_threshold):
        self.upper_oi_thres = upper_oi_threshold

    def match_brick(self, brick: ArchBrick):
        return brick.oi_engine <= self.upper_oi_thres and brick.oi_stream <= self.upper_oi_thres

    def match_op(self, op: ArchOp):
        return op.oi_engine <= self.upper_oi_thres and op.oi_stream <= self.upper_oi_thres

