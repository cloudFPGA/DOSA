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


@tvm.instrument.pass_instrument
class PrintMeta:
    """Print the name of the pass, the IR, only before passes execute."""

    def run_before_pass(self, mod, info):
        print("Running pass: {}".format(info))
        print(mod)


def arch_gen(mod, params, debug=False):

    oi_calc = OiCalculator(default_oi=1.0)
    oi_pass = OiPipeline(size_t=32, oiCalc=oi_calc)
    assert oi_pass.info.name == "OiPipeline"

    # first, TVM optimization pass
    seq1 = tvm.transform.Sequential(
        [
            relay.transform.FoldConstant(),
            relay.transform.DeadCodeElimination(),
            relay.transform.FuseOps(),
            relay.transform.EliminateCommonSubexpr(),
            tvm.transform.PrintIR(),
        ]
    )

    # "read only" passes
    seq2_ro = tvm.transform.Sequential(
        [
            oi_pass,
        ]
    )

    with tvm.transform.PassContext(opt_level=3, instruments=[PrintMeta()]):
        mod2 = seq1(mod)
        ignore = seq2_ro(mod2)

    oi_results = oi_pass.get_oi_results()
    bw_results = oi_pass.get_bw_results()
    data_per_layer = oi_pass.get_data_per_layer()
    oi_fused_wise = oi_pass.get_oi_fused_wise()
    oi_main_view = oi_pass.get_oi_main_view()
    fn_call_stats = oi_pass.get_fn_call_cnts()

    if debug:
        # print(oi_results)
        # print(bw_results)
        print("\n[DEBUG] data_per_layer")
        print(json.dumps(data_per_layer, indent=2, sort_keys=False))
        print("\n[DEBUG] oi_fused_wise")
        print(json.dumps(oi_fused_wise, indent=2, sort_keys=False))
        print("\n[DEBUG] oi_main_view")
        print(json.dumps(oi_main_view, indent=2, sort_keys=False))
        print("\n[DEBUG] fn_call_stats")
        print(json.dumps(fn_call_stats, indent=2, sort_keys=False))

    ret = {'mod': mod2, 'oi_results': oi_results, 'bw_results': bw_results, 'dpl': data_per_layer,
           'fused_view': oi_main_view}

    return ret


def calculate_required_performance(detail_list, target_fps, used_batch_size=1, unit=1, debug_print=True):
    """

    :param detail_list: detailed layer list from model summary
    :param target_fps: target framerate in Bytes per second
    :param used_batch_size: batch size that is configured in the neuronal network
    :return:
    """
    # assert target_batch_size == 1
    # calculate latency
    e2e_latency = float(1)/float(target_fps)
    n_layers = len(detail_list) - 2  # subtracting input & output
    assert n_layers >= 1
    latency_per_layer = e2e_latency/float(n_layers)
    if debug_print:
        print("calculating FLOPs for target e2e latency {}s ({}s for each layer if equal distribution is assumed).".format(e2e_latency, latency_per_layer))
    annotated_list = []
    cmpl_list = []
    uinp_list = []
    for e in detail_list:
        # calculate input and output bandwidth
        i_bw = e['inpB']*target_fps
        o_bw = e['outB']*target_fps
        # calculate FLOPs
        req_flop = e['flop']*target_fps
        req_flop_u = req_flop/unit
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

