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

    oi_calc = OiCalculator(default_oi=1)
    oi_pass = OiPipeline(size_t=32, oiCalc=oi_calc)
    assert oi_pass.info.name == "OiPipeline"

    # first, TVM optimization pass
    seq1 = tvm.transform.Sequential(
        [
            relay.transform.FoldConstant(),
            tvm.transform.PrintIR(),
            # relay.transform.FuseOps(),
        ]
    )

    # "read only" passes
    seq2_ro = tvm.transform.Sequential(
        [
            oi_pass,
        ]
    )

    with tvm.transform.PassContext(opt_level=2, instruments=[PrintMeta()]):
        mod2 = seq1(mod)
        ignore = seq2_ro(mod2)

    oi_results = oi_pass.get_oi_results()
    bw_results = oi_pass.get_bw_results()

    if debug:
        print(oi_results)
        print(bw_results)

    ret = {'mod': mod2, 'oi_results': oi_results, 'bw_results': bw_results}

    return ret

