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


class DosaDtype(Enum):
    UNKNOWN = 0
    float32 = 1
    float16 = 2
    int32   = 3
    int16   = 4
    uint8   = 5
    double  = 6


def convert_tvmDtype_to_DosaDtype(dtype):
    if dtype == 'float32':
        return DosaDtype.float32
    elif dtype == 'float16':
        return DosaDtype.float16
    elif dtype == 'int32':
        return DosaDtype.int32
    elif dtype == 'int16':
        return DosaDtype.int16
    elif dtype == 'uint8':
        # TODO: catch also uint below 8 here?
        return DosaDtype.uint8
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
    if dtype == DosaDtype.uint8:
        return 8
    if dtype == DosaDtype.double:
        return 64
    # unkown, take default
    return 32

