#  /*******************************************************************************
#   * Copyright 2019 -- 2023 IBM Corporation
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
#  *        Optimization passes for archGen
#  *
#  *

import copy
import numpy as np
import tvm.relay as relay

import gradatim.lib.singleton as dosa_singleton
# from gradatim.middleend.archGen.ArchDraft import ArchDraft
from gradatim.middleend.archGen.ArchFilter import ArchFilter
from gradatim.lib.dosa_dtype import get_bitwidth_of_DosaDtype


def append_bricks_pass(input_draft, arch_filter: ArchFilter, work_on_copy=False):
    """append operations to previous bricks"""
    if work_on_copy:
        # without is faster
        arch_draft = copy.deepcopy(input_draft)
    else:
        arch_draft = input_draft
    # TODO make dynamic, handle cross node op merging
    assert len(arch_draft.nodes) == 1
    nn = arch_draft.nodes['0']
    bis_to_del = []
    prev_bb = None
    for bi in range(0, len(nn.bricks)):
        bb = nn.bricks[bi]
        original_oo_handles = {}
        index_sub_correct = 0
        for oi in range(0, len(bb.ops)):
            original_oo_handles[oi] = bb.ops[oi]
        for oi in range(0, len(bb.ops)):
            # oo = bb.ops[oi]
            oo = original_oo_handles[oi]
            corrected_pos = oi - index_sub_correct
            # we can only merge to previous brick, if it is the first operation,
            #  and it is not the first brick
            if arch_filter.match_op(oo) and corrected_pos == 0 and prev_bb is not None:
                oo.original_brick_tvm_handle = bb.tvm_node
                prev_bb.add_arch_op(oo)
                bb.del_arch_op(oo)
                index_sub_correct += 1
        if len(bb.ops) == 0:
            bis_to_del.append(bi)
        else:
            prev_bb = bb
    bis_to_del.reverse()
    for dbi in bis_to_del:
        nn.del_brick(dbi)
    return arch_draft


def merge_bricks_pass(input_draft, arch_filter: ArchFilter, work_on_copy=False, verbose=False):
    """merge brick to previous bricks"""
    if work_on_copy:
        # without is faster
        arch_draft = copy.deepcopy(input_draft)
    else:
        arch_draft = input_draft
    # TODO make dynamic, handle cross node op merging
    assert len(arch_draft.nodes) == 1
    nn = arch_draft.nodes[0]
    bis_to_del = []
    original_bb_handles = {}
    for bi in range(0, len(nn.bricks)):
        original_bb_handles[bi] = nn.bricks[bi]
    prev_bb = None
    for bi in range(0, len(nn.bricks)):
        # bb = nn.bricks[bi]
        bb = original_bb_handles[bi]
        if arch_filter.match_brick(bb) and prev_bb is not None:
            if verbose:
                print("[DOSA:archOpt:INFO] Merging brick {} into previous brick {}.".format(repr(bb), repr(prev_bb)))
            for op in bb.local_op_iter_gen():
                op.original_brick_tvm_handle = bb.tvm_node
                prev_bb.add_arch_op(op, not_update_counters=False)
            bis_to_del.append(bi)
            # is done in arch_op
            # if prev_bb.selected_contract is not None:
            #     # updating also contracts
            #     sosg = prev_bb.selected_contract.osg
            #     prev_bb.available_contracts = []
            #     sosg.annotate_brick(prev_bb, prev_bb.selected_contract.device)
            #     prev_bb.selected_contract = prev_bb.available_contracts[0]
        else:
            prev_bb = bb
    bis_to_del.reverse()
    for dbi in bis_to_del:
        nn.del_brick(dbi)
    return arch_draft


def delete_ops_pass(input_draft, arch_filter: ArchFilter, work_on_copy=False):
    """delete ops of bricks if filter matches"""
    if work_on_copy:
        # without is faster
        arch_draft = copy.deepcopy(input_draft)
    else:
        arch_draft = input_draft
    # TODO make dynamic, handle cross node op merging
    assert len(arch_draft.nodes) == 1
    nn = arch_draft.nodes[0]
    bis_to_del = []
    original_bb_handles = {}
    for bi in range(0, len(nn.bricks)):
        original_bb_handles[bi] = nn.bricks[bi]
    for bi in range(0, len(nn.bricks)):
        bb = original_bb_handles[bi]
        lops = bb.ops
        ops_to_del = []
        for lop_i in range(0, len(lops)):
            op = lops[lop_i]
            if arch_filter.match_op(op):
                ops_to_del.append(lop_i)
        if len(ops_to_del) >= len(lops):
            bis_to_del.append(bi)
        else:
            ops_to_del.reverse()
            for doi in ops_to_del:
                bb.del_arch_op(doi)
    bis_to_del.reverse()
    for dbi in bis_to_del:
        nn.del_brick(dbi)
    return arch_draft


class MergeOpClass:

    def __init__(self):
        self._method_cache = {}

    def merge(self, op_merge_receive, op_to_be_merged, verbose=False):
        if self._method_cache is None:
            self._method_cache = {}
        method_name = op_merge_receive.op_call.split('.')[-1]
        mergeF = self._method_cache.get(method_name, None)
        if mergeF is None:
            method = 'merge_into_' + method_name
            # get method or default
            mergeF = getattr(self, method, self.fallback_merge)
            self._method_cache[method_name] = mergeF
        return mergeF(op_merge_receive, op_to_be_merged, verbose)

    def fallback_merge(self, op_merge_receive, op_to_be_merged, verbose=False):
        print(f'[DOSA:MergeOps:ERROR] Attempt to merge {repr(op_to_be_merged)} into {repr(op_merge_receive)}, but '
              f'this is not possible. STOP.')
        exit(1)

    def merge_into_multi_threshold(self, op_merge_receive, op_to_be_merged, verbose=False):
        if op_to_be_merged.op_call not in ['nn.multi_threshold']:
            print(f"[DOSA:MergeOps:ERROR] Can't merge {repr(op_to_be_merged)} into multi_threshold "
                  f"{repr(op_merge_receive)}. STOP.")
            exit(1)
        assert op_merge_receive.dims.out == op_to_be_merged.dims.inp
        assert op_to_be_merged.dims.inp == op_to_be_merged.dims.out
        assert op_merge_receive.dims.param == op_to_be_merged.dims.param
        assert op_merge_receive.used_dtype == op_to_be_merged.used_dtype

        # actually, we only need to replace the constants?
        nbit_in = get_bitwidth_of_DosaDtype(op_merge_receive.used_dtype)
        upper_bound = np.power(2, nbit_in - 1) - 1
        lower_bound = -np.power(2, nbit_in - 1)
        out_values = np.arange(lower_bound, upper_bound)
        # upper_bound = np.power(2, nbit_in) - 1
        # out_values = np.arange(0, upper_bound)

        # in_out_1 = np.vstack([op_merge_receive.tvm_args['by_position'][1]['ref'].data.numpy()[0], out_values])
        in_values_1 = op_merge_receive.tvm_args['by_position'][1]['ref'].data.numpy()
        in_values_2 = op_to_be_merged.tvm_args['by_position'][1]['ref'].data.numpy()
        assert len(in_values_1) == len(in_values_2)
        orig_array_dtype = in_values_1.dtype

        out_array = []
        # here, duplicate values are allowed!
        for channel_id in range(in_values_1.shape[0]):
            out_to_in_1 = {}
            vector_1 = in_values_1[channel_id]
            vector_2 = in_values_2[channel_id]
            assert len(out_values) == len(vector_1)
            assert len(vector_1) == len(vector_2)
            for i in range(len(vector_1)):
                out_to_in_1[int(out_values[i])] = vector_1[i]
            out_vector = []
            for i in range(len(vector_1)):
                new_entry = int(out_to_in_1[int(vector_2[i])])
                out_vector.append(new_entry)
            out_array.append(out_vector)
        # # next try...
        # out_array = []
        # # here, duplicate values are allowed!
        # for channel_id in range(in_values_1.shape[0]):
        #     vector_1 = in_values_1[channel_id]
        #     vector_2 = in_values_2[channel_id]
        #     assert len(out_values) == len(vector_1)
        #     assert len(vector_1) == len(vector_2)
        #     out_vector = []
        #     for i in range(len(vector_1)):
        #         new_entry = vector_1[int(vector_2[i])]
        #         out_vector.append(new_entry)
        #     out_array.append(out_vector)
        new_array = np.array(out_array)
        assert new_array.shape == in_values_1.shape
        new_c = relay.const(new_array, dtype=orig_array_dtype)

        # replace old constant
        op_merge_receive.tvm_args['by_position'][1]['ref'] = new_c
        op_merge_receive.tvm_args['vars'][0]['ref'] = new_c
        op_merge_receive.merged_ops = [op_to_be_merged]
        print(f"[DOSA:MergeOps:INFO] Merged multi_threshold {repr(op_to_be_merged)} into {repr(op_merge_receive)} "
              f"successfully (parent: {op_merge_receive.parent_fn}).")
        assert op_merge_receive.tvm_args['by_position'][1]['ref'] == new_c
        return op_merge_receive


def merge_ops_within_brick_pass(input_draft, arch_filter: ArchFilter, work_on_copy=False):
    """delete ops of bricks if filter matches"""
    if work_on_copy:
        # without is faster
        arch_draft = copy.deepcopy(input_draft)
    else:
        arch_draft = input_draft
    # assert len(arch_draft.nodes) == 1
    merger = MergeOpClass()
    for bb in arch_draft.brick_iter_gen():
        lops = bb.ops
        ops_to_del = []
        new_ops = []
        prev_op = None
        prev_op_id = -1
        for lop_i in range(0, len(lops)):
            op = lops[lop_i]
            if lop_i in ops_to_del:
                continue
            if arch_filter.match_op(op) and prev_op is not None and arch_filter.match_op(prev_op):
                ops_to_del.append(prev_op_id)
                ops_to_del.append(lop_i)
                new_op = merger.merge(prev_op, op)
                new_ops.append(new_op)
            prev_op = op
            prev_op_id = lop_i
        ops_to_del.reverse()
        for doi in ops_to_del:
            bb.del_arch_op(doi)
        for nop in new_ops:
            bb.add_arch_op(nop)
    return arch_draft

