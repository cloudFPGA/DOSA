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

import dimidium.lib.singleton as dosa_singleton
# from dimidium.middleend.archGen.ArchDraft import ArchDraft
from dimidium.middleend.archGen.ArchFilter import ArchFilter


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
                prev_bb.add_arch_op(op, update_counters=True)
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


