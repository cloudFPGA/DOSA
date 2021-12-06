#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Jun 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        ArchGen flow
#  *
#  *

import copy
import time
import json
import tvm
# import tvm.relay as relay

import dimidium.lib.singleton as dosa_singleton
from dimidium.frontend.TvmPrintMeta import PrintMeta
from dimidium.middleend.astProc.oiVisitor import OiPipeline
from dimidium.middleend.astProc.oiCalculation import OiCalculator
from dimidium.lib.util import OptimizationStrategies, BrickImplTypes, DosaRv
from dimidium.middleend.archGen.ArchBrick import ArchBrick
from dimidium.middleend.archGen.ArchDraft import ArchDraft
from dimidium.middleend.archGen.ArchOp import ArchOp
from dimidium.middleend.archGen.ArchNode import ArchNode
from dimidium.middleend.astProc.oiVisitor import oiV_fn_main_str, oiV_input_str, oiV_output_str, oiV_func_str
from dimidium.backend.operatorSets.BaseOSG import BaseOSG
from dimidium.backend.devices.dosa_roofline import RooflineRegions
from dimidium.middleend.archGen.ArchFilter import OiThresholdFilter
from dimidium.middleend.archGen.archOpt import merge_bricks_pass


def arch_gen(mod, params, name, strategy: OptimizationStrategies, available_osgs: [BaseOSG], available_devices,
             batch_size=1, sample_size=1, target_sps=-1, target_latency=-1, target_resources=-1,
             arch_target_devices=None, arch_fallback_devices=None, debug=False, profiling=False, verbose=False):
    arch_gen_start = time.time()
    oi_calc = OiCalculator(default_oi=1.0)
    oi_pass = OiPipeline(fallback_size_t=32, oiCalc=oi_calc)
    assert oi_pass.info.name == "OiPipeline"

    # "read only" passes
    seq2ro_calls = [
        oi_pass,
    ]

    pass_instruments = []
    if debug:
        pass_instruments.append(PrintMeta())

    tvm_pass_start = time.time()
    seq2_ro = tvm.transform.Sequential(seq2ro_calls)

    with tvm.transform.PassContext(opt_level=3, instruments=pass_instruments):
        ignore = seq2_ro(mod)

    if verbose:
        print(mod.astext(show_meta_data=False))
    tvm_pass_end = time.time()

    oi_pass.reorder_fn_calls()
    oi_results = oi_pass.get_oi_results()
    bw_results = oi_pass.get_bw_results()
    data_per_layer = oi_pass.get_data_per_layer()
    oi_fused_wise = oi_pass.get_oi_fused_wise()
    oi_main_view = oi_pass.get_oi_main_view()
    fn_call_stats = oi_pass.get_fn_call_cnts()
    tvm_nodes = oi_pass.get_tvm_nodes()
    tvm_call_args = oi_pass.get_call_args()

    if debug:
        # print(oi_results)
        # print(bw_results)
        # print("\n[DEBUG] data_per_layer")
        # print(json.dumps(data_per_layer, indent=2, sort_keys=False))
        # print("\n[DEBUG] oi_fused_wise")
        # print(json.dumps(oi_fused_wise, indent=2, sort_keys=False))
        print("\n[DEBUG] oi_main_view")
        print(json.dumps(oi_main_view, indent=2, sort_keys=False))
        print("\n[DEBUG] fn_call_stats")
        print(json.dumps(fn_call_stats, indent=2, sort_keys=False))

    creating_draft_start = time.time()
    inital_draft = create_arch_draft(name, strategy, available_osgs, batch_size, sample_size, target_sps, target_latency, target_resources,
                                     data_per_layer, tvm_nodes, tvm_call_args, mod, params)

    for do in arch_target_devices:
        inital_draft.add_possible_target_hw(do)

    if arch_fallback_devices is not None:
        for d in arch_fallback_devices:
            do = available_devices.types_dict[d]
            inital_draft.add_possible_fallback_hw(do)
    creating_draft_end = time.time()

    # batch_flatten_ops = ['nn.batch_flatten']
    zero_oi_filter = OiThresholdFilter(0.0)
    opt_draft_start = time.time()
    merge_bricks_pass(inital_draft, zero_oi_filter, work_on_copy=False)
    opt_draft_end = time.time()

    annotating_draft_start = time.time()
    # annotated_draft = annotate_required_performance(inital_draft)
    annotated_draft = inital_draft
    annotate_required_performance(annotated_draft)
    annotating_draft_end = time.time()

    # if debug:
    #     print("\n[DEBUG] initial annotated draft:")
    #     # print(inital_draft)
    #     print(annotated_draft)

    check_annot_start_1 = time.time()
    still_valid = check_annotations(annotated_draft)
    check_annot_end_1 = time.time()

    find_best_start = time.time()
    best_draft = find_best_draft(annotated_draft, verbose=verbose)
    find_best_end = time.time()

    check_annot_start_2 = time.time()
    still_valid = check_annotations(best_draft)
    check_annot_end_2 = time.time()
    print("\t...done.")

    if debug or verbose:
        print("\n[DEBUG] best draft found:")
        print(best_draft)

    print("\nDOSA: Found best and valid draft, generating architecture and software in {}...\n"
          .format(dosa_singleton.config.global_build_dir))
    build_start = time.time()
    best_draft.build()
    build_stop = time.time()

    # synth_start = time.time()
    # best_draft.synth()
    # synth_stop = time.time()

    other_opts = []
    if debug:
        other_opts.append(annotated_draft)
        for opt_s in OptimizationStrategies:
            if opt_s == strategy:
                continue
            inital_draft.strategy = opt_s
            new_draft = copy.deepcopy(inital_draft)
            # new_draft = annotate_required_performance(inital_draft)
            annotate_required_performance(new_draft)
            # opt_s_n = str(opt_s).split('.')[-1]
            # other_opts[opt_s_n] = new_draft
            if opt_s == OptimizationStrategies.LATENCY:
                check_annotations(new_draft)
            other_opts.append(new_draft)
        inital_draft.strategy = strategy

    ret = {'mod': mod, 'base_dpl': data_per_layer, 'fused_view': oi_main_view, 'draft': best_draft,
           'debug_obj': None}

    arch_gen_end = time.time()
    if profiling:
        prof_dict = {'archGen_time_total_s': arch_gen_end - arch_gen_start,
                     'tvm_pass_time_s': tvm_pass_end - tvm_pass_start,
                     'creating_draft_time_s': creating_draft_end - creating_draft_start,
                     'optimizing_draft_time_s': opt_draft_end - opt_draft_start,
                     'creating_annotations_time_s': annotating_draft_end - annotating_draft_start,
                     'check_annotations_time_1_s': check_annot_end_1 - check_annot_start_1,
                     'find_best_draft_time_s': find_best_end - find_best_start,
                     'check_annotations_time_2_s': check_annot_end_2 - check_annot_start_2,
                     'build_total_time_s': build_stop - build_start}
                     #, 'synth_total_time_s': synth_stop - synth_start}
        ret['profiling'] = prof_dict
        if debug or verbose:
            print("\n[DEBUG] profiling information: ")
            print(json.dumps(prof_dict, indent=2))

    if debug:
        ret['debug_obj'] = {}
        ret['debug_obj']['other_opts'] = other_opts
    return ret


def create_arch_draft(name, strategy: OptimizationStrategies,  available_osgs: [BaseOSG], batch_size, sample_size, target_sps, target_latency,
                      target_resources, data_per_layer, tvm_nodes, tvm_call_args, tvm_mod, tvm_params):
    # construct main function calls
    main_fn_exec = []
    for lid in data_per_layer:
        layer = data_per_layer[lid]
        if layer['fn'] == oiV_fn_main_str:
            main_fn_exec.append(layer)

    draft = ArchDraft(name, 'initial', strategy, batch_size, sample_size, target_sps, target_latency, target_resources,
                      tvm_mod=tvm_mod, tvm_params=tvm_params)
    node_0 = ArchNode(0)

    # for fid in fn_view:
    #     fn = fn_view[fid]
    #     assert fn['fn'] == fn_main_str
    for fn in main_fn_exec:
        t_node = tvm_nodes[fn['tid']]
        t_args = None
        if fn['tid'] in tvm_call_args.keys():  # for input layer
            t_args = tvm_call_args[fn['tid']]
        brick = ArchBrick(dpl_dict=fn, tvm_node=t_node, tvm_args=t_args)
        fn_str = fn['name']
        if fn_str == oiV_input_str:
            # brick.set_tvm_mod(None)
            draft.set_input_layer(fn)
            draft.set_tvm_mod(t_node)
        elif fn_str == oiV_output_str:
            # brick.set_tvm_mod(None)
            draft.set_output_layer(fn)
        else:
            for lid in data_per_layer:
                layer = data_per_layer[lid]
                if layer['fn'] == fn_str and layer['op'] != oiV_func_str:
                    op_t = tvm_nodes[layer['tid']]
                    op_a = tvm_call_args[layer['tid']]
                    aop = ArchOp(dpl_dict=layer, tvm_node=op_t, tvm_args=op_a)
                    brick.add_arch_op(aop)
            node_0.add_brick(brick)
    draft.add_node(node_0)

    # annotate OSGs
    for bb in draft.brick_iter_gen():
        for osg in available_osgs:
            osg.annotate_brick(bb)
    draft.update_possible_osgs()
    draft.update_possible_hw_types()

    return draft


def annotate_required_performance(arch_draft: ArchDraft):
    # arch_draft = copy.deepcopy(input_draft)
    rc = arch_draft.update_required_perf()
    if rc != DosaRv.OK:
        exit(1)
    arch_draft.version += '_annotated'
    # return arch_draft


def check_perf_annotations(draft: ArchDraft, fallback_impl_type=BrickImplTypes.ENGINE):
    if draft.strategy == OptimizationStrategies.THROUGHPUT:
        target_throughput = draft.target_sps * draft.sample_size_B
        for node in draft.node_iter_gen():
            # take data parallelism into account
            local_data_par_level = node.data_parallelism_level
            for bb in node.local_brick_iter_gen():
                cur_perf = bb.calc_flops
                if cur_perf < 0:
                    cur_perf = bb.req_flops
                selected_impl = bb.selected_impl_type
                if selected_impl == BrickImplTypes.UNDECIDED:
                    selected_impl = fallback_impl_type
                if selected_impl == BrickImplTypes.ENGINE:
                    cur_oi = bb.oi_engine
                    cur_inp = bb.input_bytes + bb.parameter_bytes
                else:
                    cur_oi = bb.oi_stream
                    cur_inp = bb.input_bytes
                if cur_perf == 0 or cur_oi == 0:
                    continue
                total_cur_perf = cur_perf * local_data_par_level
                cur_local_tp = total_cur_perf / cur_oi
                req_local_tp = cur_inp * draft.target_sps
                # local_time = cur_inp / local_tp
                if cur_local_tp < req_local_tp:
                    print("[DOSA:archVerify:ERROR] Brick {} does not fulfill local throughput requirement (req: {} current: {} B/s)."
                          .format(repr(bb), req_local_tp, cur_local_tp))
                    return False
        print("[DOSA:archVerify:INFO] Draft {} fulfills throughput requirement.".format(repr(draft)))
        return True
    elif draft.strategy == OptimizationStrategies.LATENCY:
        total_time = 0
        for node in draft.node_iter_gen():
            # take data parallelism into account
            local_data_par_level = node.data_parallelism_level
            for bb in node.local_brick_iter_gen():
                cur_perf = bb.calc_flops
                if cur_perf < 0:
                    cur_perf = bb.req_flops
                # selected_impl = bb.selected_impl_type
                # if selected_impl == BrickImplTypes.UNDECIDED:
                #     selected_impl = fallback_impl_type
                # if selected_impl == BrickImplTypes.ENGINE:
                #     cur_oi = bb.oi_engine
                #     cur_inp = bb.input_bytes + bb.parameter_bytes
                # else:
                #     cur_oi = bb.oi_stream
                #     cur_inp = bb.input_bytes
                if cur_perf == 0 or bb.flops == 0:
                    continue
                total_cur_perf = cur_perf * local_data_par_level
                local_time = bb.flops / total_cur_perf
                total_time += local_time
        for ni in draft.nodes:
            nn = draft.nodes[ni]
            if nn.targeted_hw is not None:
                cl1 = nn.targeted_hw.get_comm_latency_s()
                total_time += 2 * cl1
        if total_time > float(draft.target_latency):
            print("[DOSA:archVerify:ERROR] Draft {} does not fulfill latency requirement (req: {} current: {} s)."
                  .format(repr(draft), float(draft.target_latency), total_time))
            return False
        print("[DOSA:archVerify:INFO] Draft {} fulfills latency requirement.".format(repr(draft)))
        return True
    else:
        # checking resource footprint
        print("not yet implemented")
    return False


def check_annotations(draft: ArchDraft, fallback_impl_type=BrickImplTypes.ENGINE):
    perf_result = check_perf_annotations(draft, fallback_impl_type)
    if not perf_result:
        return False
    # check if resources are possible
    draft.update_possible_hw_types()
    one_possilbe_hw = False
    for phw in draft.possible_hw_types:
        if phw in draft.target_hw_set:
            one_possilbe_hw = True
    if not one_possilbe_hw:
        print("[DOSA:archVerify:INFO] Draft {} does not fulfill resource type requirement (target: {} possible: {})."
              .format(repr(draft), draft.target_hw_set, draft.possible_hw_types))
        for phw in draft.possible_hw_types:
            if phw in draft.fallback_hw_set:
                one_possilbe_hw = True
        if not one_possilbe_hw:
            print("[DOSA:archVerify:ERROR] Draft {} does not fulfill resource or fallback type requirement \
                   (target: {}, fallback: {} possible: {})."
                  .format(repr(draft), draft.target_hw_set, draft.fallback_hw_set, draft.possible_hw_types))
            return False
    print("[DOSA:archVerify:INFO] Draft {} fulfills resource type requirements.".format(repr(draft)))
    # if roofline present, check if all bricks in all nodes are "IN_HOUSE"
    not_in_house = []
    for nn in draft.node_iter_gen():
        if nn.roofline is None:
            continue
        for lb in nn.local_brick_iter_gen():
            oi_selected = lb.get_oi_selected_impl()
            rr = nn.roofline.get_region(oi_selected, lb.req_flops)
            if rr != RooflineRegions.IN_HOUSE:
                msg_str = "({}, {})".format(nn.node_id, lb.brick_uuid)
                not_in_house.append(msg_str)
    if len(not_in_house) > 0:
        print("[DOSA:archVerify:ERROR] Draft {} does not fulfill roofline requirement. The following bircks \
        (node_id, brick_uuid) are above roofs: {}".format(repr(draft), not_in_house))
        return False
    print("[DOSA:archVerify:INFO] Draft {} fulfills roofline requirement.".format(repr(draft)))
    return True


# update draft so that roofline and types are possible
def find_best_draft(draft: ArchDraft, verbose=False) -> ArchDraft:
    # draft = copy.deepcopy(input_draft)
    assert len(draft.target_hw_set) >= 1
    assert len(draft.fallback_hw_set) >= 1
    draft_list = []
    node_count_list = []   # index equals index in target_hw_set and draft_list
    for thw in draft.target_hw_set:
        # A) for each target hw, create legal draft and count number of nodes
        tmp_draft = copy.deepcopy(draft)
        # populate first target hw
        for nn in tmp_draft.node_iter_gen():
            nn.set_targeted_hw(thw)  # this includes the generation of the roofline
        # legalize this version
        rv = tmp_draft.legalize(verbose=verbose)
        if rv != DosaRv.OK:
            continue
        # save state
        node_count_list.append(tmp_draft.nid_cnt)
        draft_list.append(tmp_draft)
    if len(draft_list) == 0:
        print("[DOSA:archGen:ERROR] unable to find legal architecture draft. Stop.")
        exit(1)
    #  then, select the type of hw with the lowest number of nodes
    best_version_i = node_count_list.index(min(node_count_list))
    best_draft = draft_list[best_version_i]
    # TODO: add additional cost factor to devices? E.g. if 3 small nodes are cheaper then 1 big one?
    draft = best_draft
    # TODO B) then, for not full nodes, see if other hw could be used
    # C) then, check if all operations can be implemented?
    #  split accordingly and merge again?
    #  --> done in legalizing
    # D) annotate network latencies?
    #   --> not necessary, done implicitly via targeted_hw
    draft.version = 'selected_best'
    return draft


def optimize_draft(draft: ArchDraft) -> [ArchDraft]:
    return False


