#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Oct 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *       Tvm Print Meta Pass
#  *
#  *

import tvm


@tvm.instrument.pass_instrument
class PrintMeta:
    """Print the name of the pass, the IR, only before passes execute."""

    def run_before_pass(self, mod, info):
        print("Running pass: {}".format(info))
        # print(mod)

