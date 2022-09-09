#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
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


def merge_bricks_pass(input_draft, arch_filter: ArchFilter, work_on_copy=False):
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
            for op in bb.local_op_iter_gen():
                op.original_brick_tvm_handle = bb.tvm_node
                prev_bb.add_arch_op(op, update_counters=True)
            bis_to_del.append(bi)
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


