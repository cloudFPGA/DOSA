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
import json
import tvm
import tvm.relay as relay
from dimidium.lib.oiVisitor import OiPipeline
from dimidium.lib.oiCalculation import OiCalculator
from dimidium.lib.util import OptimizationStrategies, BrickImplTypes
from dimidium.lib.ArchBrick import ArchBrick
from dimidium.lib.ArchDraft import ArchDraft
from dimidium.lib.ArchOp import ArchOp
from dimidium.lib.ArchNode import ArchNode
from dimidium.lib.oiVisitor import oiV_fn_main_str, oiV_input_str, oiV_output_str, oiV_func_str, oiV_func_call_str
from dimidium.lib.devices import types_dict as device_types_dict


@tvm.instrument.pass_instrument
class PrintMeta:
    """Print the name of the pass, the IR, only before passes execute."""

    def run_before_pass(self, mod, info):
        print("Running pass: {}".format(info))
        # print(mod)


def arch_gen(mod, params, name, strategy: OptimizationStrategies, batch_size, sample_size, target_sps=-1, target_latency=-1,
             target_resources=-1, arch_target_devices=None, arch_fallback_devices=None, debug=False):
    oi_calc = OiCalculator(default_oi=1.0)
    oi_pass = OiPipeline(fallback_size_t=32, oiCalc=oi_calc)
    assert oi_pass.info.name == "OiPipeline"

    # first, TVM optimization pass
    seq1_calls = [
        relay.transform.FoldConstant(),
        relay.transform.FastMath(),
        relay.transform.CanonicalizeCast(),
        relay.transform.DeadCodeElimination(),
        relay.transform.FuseOps(),
        relay.transform.RemoveUnusedFunctions(),
        relay.transform.EliminateCommonSubexpr(),
        # tvm.transform.PrintIR(),
        relay.transform.SimplifyInference(),
        relay.transform.FoldExplicitPadding(),
        relay.transform.ForwardFoldScaleAxis(),
        relay.transform.InferType(),
        # relay.transform.AnnotateSpans(),  # not working with input.1 name...
    ]

    # "read only" passes
    seq2ro_calls = [
        oi_pass,
    ]

    pass_instruments = []
    if debug:
        pass_instruments.append(PrintMeta())
        seq1_calls.append(tvm.transform.PrintIR())

    seq1 = tvm.transform.Sequential(seq1_calls)
    seq2_ro = tvm.transform.Sequential(seq2ro_calls)

    with tvm.transform.PassContext(opt_level=3, instruments=pass_instruments):
        mod2 = seq1(mod)
        ignore = seq2_ro(mod2)

    oi_pass.reorder_fn_calls()
    oi_results = oi_pass.get_oi_results()
    bw_results = oi_pass.get_bw_results()
    data_per_layer = oi_pass.get_data_per_layer()
    oi_fused_wise = oi_pass.get_oi_fused_wise()
    oi_main_view = oi_pass.get_oi_main_view()
    fn_call_stats = oi_pass.get_fn_call_cnts()
    tvm_nodes = oi_pass.get_tvm_nodes()

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

    inital_draft = create_arch_draft(name, strategy, batch_size, sample_size, target_sps, target_latency, target_resources,
                                     data_per_layer, tvm_nodes)

    for do in arch_target_devices:
        inital_draft.add_possible_target_hw(do)

    if arch_fallback_devices is not None:
        for d in arch_fallback_devices:
            do = device_types_dict[d]
            inital_draft.add_possible_fallback_hw(do)

    annotated_draft = annotate_required_performance(inital_draft)

    if debug:
        print("\n[DEBUG] initial annotated draft:")
        # print(inital_draft)
        print(annotated_draft)

    still_valid = check_annotations(annotated_draft)
    best_draft = find_best_draft(annotated_draft)
    still_valid = check_annotations(best_draft)

    if debug:
        other_opts = []
        for opt_s in OptimizationStrategies:
            if opt_s == strategy:
                continue
            inital_draft.strategy = opt_s
            new_draft = annotate_required_performance(inital_draft)
            # opt_s_n = str(opt_s).split('.')[-1]
            # other_opts[opt_s_n] = new_draft
            if opt_s == OptimizationStrategies.LATENCY:
                check_annotations(new_draft)
            other_opts.append(new_draft)
        inital_draft.strategy = strategy

    ret = {'mod': mod2, 'base_dpl': data_per_layer, 'fused_view': oi_main_view, 'draft': annotated_draft,
           'debug_obj': None}
    if debug:
        ret['debug_obj'] = {}
        ret['debug_obj']['other_opts'] = other_opts
    return ret


def create_arch_draft(name, strategy: OptimizationStrategies, batch_size, sample_size, target_sps, target_latency,
                      target_resources, data_per_layer, tvm_nodes):
    # construct main function calls
    main_fn_exec = []
    for lid in data_per_layer:
        layer = data_per_layer[lid]
        if layer['fn'] == oiV_fn_main_str:
            main_fn_exec.append(layer)

    draft = ArchDraft(name, 'initial', strategy, batch_size, sample_size, target_sps, target_latency, target_resources)
    node_0 = ArchNode(0)

    # for fid in fn_view:
    #     fn = fn_view[fid]
    #     assert fn['fn'] == fn_main_str
    for fn in main_fn_exec:
        t_node = tvm_nodes[fn['tid']]
        brick = ArchBrick(dpl_dict=fn, tvm_node=t_node)
        fn_str = fn['name']
        if fn_str == oiV_input_str:
            # brick.set_tvm_handle(None)
            draft.set_input_layer(fn)
            draft.set_tvm_handle(t_node)
        elif fn_str == oiV_output_str:
            # brick.set_tvm_handle(None)
            draft.set_output_layer(fn)
        else:
            for lid in data_per_layer:
                layer = data_per_layer[lid]
                if layer['fn'] == fn_str and layer['op'] != oiV_func_str:
                    op_t = tvm_nodes[layer['tid']]
                    aop = ArchOp(dpl_dict=layer, tvm_node=op_t)
                    brick.add_arch_op(aop)
            node_0.add_brick(brick)
    draft.add_node(node_0)

    return draft


def annotate_required_performance(input_draft: ArchDraft):
    arch_draft = copy.deepcopy(input_draft)
    if arch_draft.strategy == OptimizationStrategies.THROUGHPUT:
        if arch_draft.target_sps < 0:
            print("[DOSA:archGen:ERROR] Optimization strategy ({}) does not fit target numbers in constraint target_sps ({}). Stop."
                  .format(arch_draft.strategy, arch_draft.target_sps))
            exit(1)
        # optimizing towards throughput
        target_throughput = arch_draft.target_sps * arch_draft.sample_size_B
        # annotate input & output
        arch_draft.input_layer['inp_Bs'] = arch_draft.input_layer['inpB'] * arch_draft.target_sps
        arch_draft.input_layer['out_Bs'] = arch_draft.input_layer['outB'] * arch_draft.target_sps
        arch_draft.output_layer['inp_Bs'] = arch_draft.output_layer['inpB'] * arch_draft.target_sps
        arch_draft.output_layer['out_Bs'] = arch_draft.output_layer['outB'] * arch_draft.target_sps
        # annotate bricks
        # for ni in arch_draft.nodes:
        #     nn = arch_draft.nodes[ni]
        #     for bi in nn.bricks:
        #        brick = nn.bricks[bi]
        for brick in arch_draft.brick_iter_gen():
            brick.input_bw_Bs = brick.input_bytes * arch_draft.target_sps
            brick.output_bw_Bs = brick.output_bytes * arch_draft.target_sps
            brick.req_flops = brick.flops * arch_draft.target_sps
            # calc_latency is depending on mode
    elif arch_draft.strategy == OptimizationStrategies.LATENCY:
        # optimizing towards latency
        if arch_draft.target_latency < 0:
            print("[DOSA:archGen:ERROR] Optimization strategy ({}) does not fit target numbers in constraint target_latency ({}). Stop."
                  .format(arch_draft.strategy, arch_draft.target_latency))
            exit(1)
        # first, try with 1/N distribution
        latency_per_brick = arch_draft.target_latency / float(arch_draft.get_bricks_num())
        for brick in arch_draft.brick_iter_gen():
            brick.req_latency = latency_per_brick
            # brick.req_perf_engine = (brick.oi_engine * brick.input_bytes) / latency_per_brick
            # brick.req_perf_stream = (brick.oi_stream * brick.input_bytes) / latency_per_brick
            brick.req_flops = brick.flops / latency_per_brick
    else:
        # optimizing towards resource footprint
        if arch_draft.target_resources < 0:
            print("[DOSA:archGen:ERROR] Optimization strategy ({}) does not fit target numbers in constraint target_resources ({}). Stop."
                  .format(arch_draft.strategy, arch_draft.target_resources))
            exit(1)
        # find max resources in flops
        max_resources = 0
        max_res_dev = "unknown"
        for d in arch_draft.target_hw_set:
            dr = d.get_max_flops()
            if dr > max_resources:
                max_resources = dr
                max_res_dev = d.name
        allowed_resources = max_resources * arch_draft.target_resources
        arch_draft.tmp_notes['max_res_dev'] = max_res_dev
        arch_draft.tmp_notes['allowed_resources'] = allowed_resources
        # first, try with 1/N distribution
        resource_per_brick = allowed_resources / arch_draft.get_bricks_num()
        for brick in arch_draft.brick_iter_gen():
            brick.req_flops = resource_per_brick
    arch_draft.version += '_annotated'
    return arch_draft


def check_annotations(draft: ArchDraft, fallback_impl_type=BrickImplTypes.ENGINE):
    if draft.strategy == OptimizationStrategies.THROUGHPUT:
        target_throughput = draft.target_sps * draft.sample_size_B
        # TODO: consider vertical parallelized nodes
        #  maybe with "flop_paralleization_factor" per brick?
        for bb in draft.brick_iter_gen():
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
            cur_local_tp = cur_perf / cur_oi
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
        # TODO: consider vertical parallelized nodes
        #  maybe with "flop_parallelization_factor" per brick?
        for bb in draft.brick_iter_gen():
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
            local_time = bb.flops / cur_perf
            total_time += local_time
        for ni in draft.nodes:
            nn = draft.nodes[ni]
            total_time += nn.latency_to_next_node
        if total_time > draft.target_latency:
            print("[DOSA:archVerify:ERROR] Draft {} does not fulfill latency requirement (req: {} current: {} s)."
                  .format(repr(draft), draft.target_latency, total_time))
            return False
        print("[DOSA:archVerify:INFO] Draft {} fulfills latency requirement.".format(repr(draft)))
        return True
    else:
        # checking resource footprint
        print("not yet implemented")
    # TODO: if roofline present, check if all bricks in all nodes are "IN_HOUSE"
    return False


# update draft so that roofline and types are possible
def find_best_draft(input_draft: ArchDraft) -> ArchDraft:
    draft = copy.deepcopy(input_draft)
    assert len(draft.target_hw_set) >= 1
    assert len(draft.fallback_hw_set) >= 1
    draft_list = []
    node_count_list = []   # index equals index in target_hw_set and draft_list
    for thw in draft.target_hw_set:
        # A) for each target hw, create legal draft and count number of nodes
        tmp_draft = copy.deepcopy(draft)
        # populate first target hw
        for nn in tmp_draft.node_iter_gen():
            nn.set_target_hw(thw)  # this includes the generation of the roofline
        # legalize this version
        tmp_draft.legalize()
        # save state
        node_count_list.append(tmp_draft.nid_cnt)
        draft_list.append(tmp_draft)
    #  then, select the type of hw with the lowest number of nodes
    best_version_i = node_count_list.index(min(node_count_list))
    best_draft = draft_list[best_version_i]
    draft = best_draft
    # TODO: B) then, for not full nodes, see if other hw could be used
    # TODO C) then, check if all operations can be implemented?
    #  split accordingly and merge again?
    # TODO D) annotate network latencies
    draft.version = 'selected_best'
    return draft


def optimize_draft(draft: ArchDraft) -> [ArchDraft]:
    return False


