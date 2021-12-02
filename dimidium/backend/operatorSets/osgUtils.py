#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Dec 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Utility library for OSGs for DOSA arch gen
#  *
#  *

from tvm.tir.expr import IntImm


def convert_IntImm_array(intImm_arr):
    ret_arr = []
    for e in intImm_arr:
        if isinstance(e, IntImm):
            ret_arr.append(e.value)
        else:
            inner_arr = convert_IntImm_array(e)
            ret_arr.append(inner_arr)
    return ret_arr

