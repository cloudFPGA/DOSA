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

import json
import tvm
import tvm.relay as relay
from dimidium.lib.oiVisitor import OiPipeline
from dimidium.lib.oiCalculation import OiCalculator
from dimidium.lib.util import OptimizationStrategies
from dimidium.lib.ArchBrick import ArchBrick
from dimidium.lib.ArchDraft import ArchDraft
from dimidium.lib.ArchOp import ArchOp
from dimidium.lib.oiVisitor import oiV_fn_main_str, oiV_input_str, oiV_output_str, oiV_func_str, oiV_func_call_str


@tvm.instrument.pass_instrument
class PrintMeta:
    """Print the name of the pass, the IR, only before passes execute."""

    def run_before_pass(self, mod, info):
        print("Running pass: {}".format(info))
        # print(mod)


def arch_gen(mod, params, name, strategy: OptimizationStrategies, batch_size, target_sps=-1, target_latency=-1,
             target_resources=-1, debug=False):
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

    inital_draft = create_arch_draft(name, strategy, batch_size, target_sps, target_latency, target_resources,
                                     data_per_layer, tvm_nodes)

    ret = {'mod': mod2, 'oi_results': oi_results, 'bw_results': bw_results, 'dpl': data_per_layer,
           'fused_view': oi_main_view}

    return ret


def calculate_required_performance(detail_list, target_sps, used_batch_size=1, unit=1, debug_print=False):
    """

    :param detail_list: detailed layer list from model summary
    :param target_sps: target samplerate in samples per second
    :param used_batch_size: batch size that is configured in the neuronal network
    :return:
    """
    # assert target_batch_size == 1
    # calculate latency
    e2e_latency = float(1) / float(target_sps)
    n_layers = len(detail_list) - 2  # subtracting input & output
    assert n_layers >= 1
    latency_per_layer = e2e_latency / float(n_layers)
    if debug_print:
        print(
            "calculating FLOPs for target e2e latency {}s ({}s for each layer if equal distribution is assumed).".format(
                e2e_latency, latency_per_layer))
    annotated_list = []
    cmpl_list = []
    uinp_list = []
    for e in detail_list:
        # calculate input and output bandwidth
        i_bw = e['inpB'] * target_sps
        o_bw = e['outB'] * target_sps
        # calculate FLOPs
        req_flop = e['flop'] * target_sps
        req_flop_u = req_flop / unit
        e['inpBs'] = i_bw
        e['outBs'] = o_bw
        e['rFLOP'] = req_flop
        e['eqLat'] = latency_per_layer
        annotated_list.append(e)
        cmpl = e['cmpl']
        if cmpl == 1:
            continue
        uinp = e['uinp']
        name = e['name']
        layer = e['layer']
        cn = {'name': name + "_" + layer + "_engine", 'oi': cmpl, 'perf': req_flop_u}
        un = {'name': name + "_" + layer + "_stream", 'oi': uinp, 'perf': req_flop_u}
        cmpl_list.append(cn)
        uinp_list.append(un)
    if debug_print:
        print(json.dumps(annotated_list, indent=2, sort_keys=False))
    return annotated_list, cmpl_list, uinp_list


def create_arch_draft(name, strategy: OptimizationStrategies, batch_size, target_sps, target_latency,
                      target_resources, data_per_layer, tvm_nodes):
    # construct main function calls
    main_fn_exec = []
    for lid in data_per_layer:
        layer = data_per_layer[lid]
        if layer['fn'] == oiV_fn_main_str:
            main_fn_exec.append(layer)

    draft = ArchDraft(name, 'initial', strategy, batch_size, target_sps, target_latency, target_resources)

    # for fid in fn_view:
    #     fn = fn_view[fid]
    #     assert fn['fn'] == fn_main_str
    for fn in main_fn_exec:
        t_node = tvm_nodes[fn['tid']]
        brick = ArchBrick(dpl_dict=fn, tvm_node=t_node)
        fn_str = fn['name']
        if fn_str == oiV_input_str:
            # brick.set_tvm_node(None)
            draft.set_input_layer(fn)
            draft.set_tvm_node(t_node)
        elif fn_str == oiV_output_str:
            # brick.set_tvm_node(None)
            draft.set_output_layer(fn)
        else:
            for lid in data_per_layer:
                layer = data_per_layer[lid]
                if layer['fn'] == fn_str and layer['op'] != oiV_func_str:
                    op_t = tvm_nodes[layer['tid']]
                    aop = ArchOp(dpl_dict=layer, tvm_node=op_t)
                    brick.add_arch_op(aop)
            draft.add_brick(brick)

    return draft
