#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Feb 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Python module to export dosa_infer C function in Python
#  *
#  *
import os
import ctypes
import numpy as np
import time


__filedir__ = os.path.dirname(os.path.abspath(__file__))
__so_lib_name__ = 'dosa_infer_pass.so'
# __so_lib_name__ = 'lib_test.so'


class DosaRoot:

    def __init__(self, used_bitwidth, signed=True):
        libname = os.path.abspath(__filedir__ + '/' + __so_lib_name__)
        self.c_lib = ctypes.CDLL(libname)
        self.c_lib.infer.restype = ctypes.c_int
        self.nbits = used_bitwidth
        self.n_bytes = (used_bitwidth + 7) / 8
        if used_bitwidth == 8:
            if signed:
                self.ndtype = np.int8
            else:
                self.ndtype = np.uint8
        elif used_bitwidth == 16:
            if signed:
                self.ndtype = np.int16
            else:
                self.ndtype = np.uint16
        else:
            if signed:
                self.ndtype = np.int32
            else:
                self.ndtype = np.uint32
        # MPI init
        # TODO
        # self.c_lib.init()

    # def _prepare_data(self, tmp_array):
    #     bin_str = ''
    #     msg_len = 0
    #     with np.nditer(tmp_array) as it:
    #         for x in it:
    #             br = np.binary_repr(x, width=self.nbits)
    #             bin_str += br
    #             msg_len += 1
    #     ret = bytes(br, 'ascii')
    #     return ret, msg_len

    def infer(self, x: np.ndarray, output_shape, debug=False):
        # input_data, input_length = self._prepare_data(x)
        input_data = x.astype(self.ndtype)
        input_length = self.n_bytes * input_data.size
        output_length = self.n_bytes
        for d in output_shape:
            output_length *= d
        out_length_32bit = np.ceil(output_length * (self.nbits/32))
        output_placeholder = bytearray(out_length_32bit)
        output_array = ctypes.c_int * len(output_placeholder)

        infer_start = time.time()
        # int infer(int *input, int input_length, int *output, int output_length);
        rc = self.c_lib.infer(input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), ctypes.c_int(input_length),
                              output_array.from_buffer(output_placeholder), ctypes.c_int(output_length))
        infer_stop = time.time()
        infer_time = infer_stop - infer_start
        if debug:
            print(f'DOSA inference run returned {rc} after {infer_time}s.')

        output_deserialized = np.frombuffer(output_placeholder, dtype=self.ndtype)
        output = np.reshape(output_deserialized, newshape=output_shape)
        return output


