#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Oct 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Enum for different data types used in DOSA
#  *
#  *

from enum import Enum
import numpy as np
from fxpmath import Fxp


class DosaDtype(Enum):
    UNKNOWN = 0
    float32 = 1
    float16 = 2
    int32   = 3
    int16   = 4
    int8    = 5
    uint8   = 6
    double  = 7
    int4    = 8
    int3    = 9
    int2    = 10

    def __repr__(self):
        return DosaDtype_to_string(self)


complete_dtype_list = [DosaDtype.float32, DosaDtype.float16, DosaDtype.int32, DosaDtype.int16, DosaDtype.int8,
                       DosaDtype.uint8, DosaDtype.double, DosaDtype.int4, DosaDtype.int3, DosaDtype.int2]


def convert_tvmDtype_to_DosaDtype(dtype):
    if dtype == 'float32':
        return DosaDtype.float32
    elif dtype == 'float16':
        return DosaDtype.float16
    elif dtype == 'int32':
        return DosaDtype.int32
    elif dtype == 'int16':
        return DosaDtype.int16
    elif dtype == 'int8':
        return DosaDtype.int8
    elif dtype == 'uint8':
        # TODO: catch also uint below 8 here?
        return DosaDtype.uint8
    elif dtype == 'int4':
        return DosaDtype.int4
    return DosaDtype.UNKNOWN
    # return DosaDtype.float32  # default?


def get_bitwidth_of_DosaDtype(dtype: DosaDtype) -> int:
    if dtype == DosaDtype.float32:
        return 32
    if dtype == DosaDtype.float16:
        return 16
    if dtype == DosaDtype.int32:
        return 32
    if dtype == DosaDtype.int16:
        return 16
    if dtype == DosaDtype.int8:
        return 8
    if dtype == DosaDtype.uint8:
        return 8
    if dtype == DosaDtype.double:
        return 64
    if dtype == DosaDtype.int4:
        return 4
    # unknown, take default
    return 32


def DosaDtype_to_string(dtype: DosaDtype) -> str:
    if dtype == DosaDtype.float32:
        return 'float32'
    if dtype == DosaDtype.float16:
        return 'float16'
    if dtype == DosaDtype.int32:
        return 'int32'
    if dtype == DosaDtype.int16:
        return 'int16'
    if dtype == DosaDtype.int8:
        return 'int8'
    if dtype == DosaDtype.uint8:
        return 'uint8'
    if dtype == DosaDtype.double:
        return 'double'
    if dtype == DosaDtype.int4:
        return 'int4'
    return 'unknown'


def DosaDtype_is_signed(dtype: DosaDtype) -> bool:
    if dtype == DosaDtype.uint8:
        return False
    return True


def bitw_to_scaleFactor(nbits):
    return np.power(2, (nbits - 1)) - 1


def data_array_convert_to_DosaDtype(orig_data: np.ndarray, target_dtype: DosaDtype,
                                    data_already_scaled=True, numpy_array_type=None) -> np.ndarray:
    signed = DosaDtype_is_signed(target_dtype)
    precision = get_bitwidth_of_DosaDtype(target_dtype)
    # TODO: support custom fixed point
    # scale_factor = bitw_to_scaleFactor(precision)
    if data_already_scaled:
        # rescaled_data = orig_data / scale_factor
        # so we need to treat it as raw data / representation
        quant_data = Fxp([0], signed=signed, n_word=precision, n_frac=precision-1)
        quant_data.set_val(orig_data, raw=True)
    else:
        # rescaled_data = orig_data
        quant_data = Fxp(orig_data, signed=signed, n_word=precision, n_frac=(precision - 1))
    ret = quant_data.val
    if numpy_array_type is not None:
        ret = ret.astype(numpy_array_type)
    return ret


# def data_array_convert_to_DosaDtype(orig_data: np.ndarray, target_dtype: DosaDtype) -> np.ndarray:
#     # TODO: check if supported by numpy
#     ret_val = orig_data.astype(DosaDtype_to_string(target_dtype))
#     return ret_val
#
