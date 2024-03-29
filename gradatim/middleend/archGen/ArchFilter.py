#  /*******************************************************************************
#   * Copyright 2019 -- 2024 IBM Corporation
#   *
#   * Licensed under the Apache License, Version 2.0 (the "License");
#   * you may not use this file except in compliance with the License.
#   * You may obtain a copy of the License at
#   *
#   *     http://www.apache.org/licenses/LICENSE-2.0
#   *
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS,
#   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   * See the License for the specific language governing permissions and
#   * limitations under the License.
#  *******************************************************************************/
#

#  *
#  *                       cloudFPGA
#  *    =============================================
#  *     Created: Dec 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Filter for Optimization passes
#  *
#  *

import abc
import numpy as np

from gradatim.middleend.archGen.ArchBrick import ArchBrick
from gradatim.middleend.archGen.ArchOp import ArchOp


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


class OpCallSameDimFilter(ArchFilter):
    """Matches operations of a list where input and output dimensions are same"""

    def __init__(self, op_call_list, reduce_dims=False):
        self.op_call_list = op_call_list
        self.reduce_dims = reduce_dims

    def match_brick(self, brick: ArchBrick):
        print("[DOSA:Filter:ERROR] Can't filter Bricks with op calls")
        return False

    def match_op(self, op: ArchOp):
        if op.op_call in self.op_call_list:
            if not self.reduce_dims:
                if op.dims.inp == op.dims.out:
                    return True
            else:
                if np.prod(op.dims.inp) == np.prod(op.dims.out):
                    return True
        return False


class MergeBrickContrFilter(ArchFilter):
    """Matches operations of a list where input and output dimensions are same"""

    def __init__(self, consider_pseudo_contracts=False):
        self.consider_pseudo_contracts = consider_pseudo_contracts

    def match_brick(self, brick: ArchBrick):
        if brick.selected_contract is not None:
            if brick.selected_contract.is_contract_to_be_merged:
                return True
            if self.consider_pseudo_contracts and brick.selected_contract.is_pseudo_contract:
                return True
        return False

    def match_op(self, op: ArchOp):
        print("[DOSA:Filter:ERROR] Can't filter Bricks with op calls")
        return False

