#  /*******************************************************************************
#   * Copyright 2019 -- 2024 IBM Corporation
#   *
#   * Licensed under the Apache License, Version 2.0 (the "License");
#   * you may not use this file except in compliance with the License.
#   * You may obtain a copy of the License at
#   *
#   *     http://www.apache.org/licenses/LICENSE-2.0
#   *
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS,
#   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   * See the License for the specific language governing permissions and
#   * limitations under the License.
#  *******************************************************************************/
#

#  *
#  *                       cloudFPGA
#  *    =============================================
#  *     Created: Jun 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *       Library for calculating the OI for selected operations
#  *
#  *

import numpy as np
import math
import tvm
import tvm.relay as relay


class OiCalculator(object):

    def __init__(self, default_oi):
        self.default_oi = default_oi
        self._method_cache = None

    def calc(self, op_name, data_dim, param_dim, out_dim, attrs, size_b):
        """
        Calculate OI for an operation
        """

        if self._method_cache is None:
            self._method_cache = {}

        # method_name = op.__class__.__name__
        method_name = op_name.split('.')[-1]
        visitor = self._method_cache.get(method_name, None)
        if visitor is None:
            method = 'calc_' + method_name
            # get method or default
            visitor = getattr(self, method, self.generic_calc)
            self._method_cache[method_name] = visitor

        return visitor(op_name, data_dim, param_dim, out_dim, attrs, size_b)

    def generic_calc(self, op_name, data_dim, param_dim, out_dim, attrs, size_b):
        """ Called if no explicit calculator function exists for an operation
        """
        print("[DOSA::OiCalculator] generic OI calc called for {}".format(op_name))
        return self.default_oi, self.default_oi, self.default_oi

    def calc_conv2d(self, op_name, data_dim, param_dim, out_dim, attrs: relay.op.op_attrs.Conv2DAttrs, size_b):
        # conv2d = cross correlation = "sliding dot product"
        layout_in = data_dim[0]
        layout_out = out_dim[0]
        sk = 1
        for e in attrs.kernel_size:
            sk *= int(e)
        flop_per_cc = 2 * sk - 1   # sk multiplications, sk-1 additions
        cc_per_in_channel = layout_out[2] * layout_out[3]  # each elem in the output plane is result of a conv
        flop_per_in_c = flop_per_cc * cc_per_in_channel
        batch_n = layout_in[0]
        in_channel = layout_in[1]
        out_channel = layout_out[1]
        flop_per_out_c = flop_per_in_c * in_channel + 1  # +1 is bias
        flop_total = flop_per_out_c * out_channel * batch_n
        # calculate bw requirements
        if len(param_dim) > 0:
            param_B = size_b
            if type(param_dim[0]) is list:
                for pd in param_dim[0]:
                    param_B *= pd
            else:
                for pd in param_dim:
                    param_B *= pd
        else:
            param_B = 0
        input_B = size_b
        for e in layout_in:
            input_B *= e
        # calculate oi complete (input + params)
        data_cmpl = param_B + input_B
        oi_cmpl = float(flop_total) / float(data_cmpl)
        # calculate oi only "user" input
        oi_uinp = float(flop_total) / float(input_B)
        return oi_cmpl, oi_uinp, flop_total

    def calc_bias_add(self, op_name, data_dim, param_dim, out_dim, attrs: relay.op.op_attrs.BiasAddAttrs, size_b):
        # one addition per element
        layout_in = data_dim[0]
        layout_out = out_dim[0]
        batch_n = layout_in[0]
        in_channel = layout_in[1]
        out_channel = layout_out[1]
        cc_per_batch = 1
        for d in layout_in[1:]:
            cc_per_batch *= d
        flop_per_cc = 1  # exactly one addition per element
        flop_total = cc_per_batch * flop_per_cc * batch_n
        # calculate bw requirements
        if len(param_dim) > 0:
            param_B = size_b
            if type(param_dim[0]) is list:
                for pd in param_dim[0]:
                    param_B *= pd
            else:
                for pd in param_dim:
                    param_B *= pd
        else:
            param_B = 0
        input_B = size_b
        for e in layout_in:
            input_B *= e
        # calculate oi complete (input + params)
        data_cmpl = param_B + input_B
        oi_cmpl = float(flop_total) / float(data_cmpl)
        # calculate oi only "user" input
        oi_uinp = float(flop_total) / float(input_B)
        return oi_cmpl, oi_uinp, flop_total

    def calc_tanh(self, op_name, data_dim, param_dim, out_dim, attrs, size_b):
        layout_in = data_dim[0]
        layout_out = out_dim[0]
        batch_n = layout_in[0]
        in_channel = layout_in[1]
        out_channel = layout_out[1]
        cc_per_batch = 4  # 4 FLOPs per element?
        for d in layout_in[1:]:
            cc_per_batch *= d
        flop_per_cc = 1
        flop_total = cc_per_batch * flop_per_cc * batch_n
        # calculate bw requirements
        param_B = 0
        input_B = size_b
        for e in layout_in:
            input_B *= e
        # calculate oi complete (input + params)
        data_cmpl = param_B + input_B
        oi_cmpl = float(flop_total) / float(data_cmpl)
        # calculate oi only "user" input
        oi_uinp = float(flop_total) / float(input_B)
        return oi_cmpl, oi_uinp, flop_total

    def calc_relu(self, op_name, data_dim, param_dim, out_dim, attrs, size_b):
        # one compare per element
        layout_in = data_dim[0]
        layout_out = out_dim[0]
        batch_n = layout_in[0]
        in_channel = layout_in[1]
        out_channel = layout_out[1]
        cc_per_batch = 1
        for d in layout_in[1:]:
            cc_per_batch *= d
        flop_per_cc = 1  # exactly one addition per element
        flop_total = cc_per_batch * flop_per_cc * batch_n
        # calculate bw requirements
        param_B = 0
        input_B = size_b
        for e in layout_in:
            input_B *= e
        # calculate oi complete (input + params)
        data_cmpl = param_B + input_B
        oi_cmpl = float(flop_total) / float(data_cmpl)
        # calculate oi only "user" input
        oi_uinp = float(flop_total) / float(input_B)
        return oi_cmpl, oi_uinp, flop_total

    def calc_max_pool2d(self, op_name, data_dim, param_dim, out_dim, attrs: relay.op.op_attrs.MaxPool2DAttrs, size_b):
        layout_in = data_dim[0]
        layout_out = out_dim[0]
        sk = 1
        for e in attrs.pool_size:
            sk *= int(e)
        flop_per_cc = sk - 1   # sk-1 compares
        cc_per_in_channel = layout_out[2] * layout_out[3]  # each elem in the output plane is result of a conv
        flop_per_in_c = flop_per_cc * cc_per_in_channel
        batch_n = layout_in[0]
        in_channel = layout_in[1]
        out_channel = layout_out[1]
        flop_per_out_c = flop_per_in_c * in_channel
        flop_total = flop_per_out_c * out_channel * batch_n
        # calculate bw requirements
        param_B = 0
        input_B = size_b
        for e in layout_in:
            input_B *= e
        # calculate oi complete (input + params)
        data_cmpl = param_B + input_B
        oi_cmpl = float(flop_total) / float(data_cmpl)
        # calculate oi only "user" input
        oi_uinp = float(flop_total) / float(input_B)
        return oi_cmpl, oi_uinp, flop_total

    def calc_batch_flatten(self, op_name, data_dim, param_dim, out_dim, attrs, size_b):
        # Flattens the input into a 2-D array.
        # does actually nothing from a flop perspective?
        return 0.0, 0.0, 0.0

    def calc_dense(self, op_name, data_dim, param_dim, out_dim, attrs: relay.op.op_attrs.DenseAttrs, size_b):
        # Applies a linear transformation: :math:`Y = XW^T`.
        # - **data**: `(x1, x2, ..., xn, input_dim)`
        # - **weight**: `(units, input_dim)`
        # - **out**: `(x1, x2, ..., xn, units)`.
        # assuming matrix is already transformed
        layout_in = data_dim[0]
        layout_out = out_dim[0]
        a_n = layout_in[1]
        flop_per_out_elem = 2 * a_n - 1  # multiplication + sum
        out_dim = layout_out[1]
        out_channel = layout_out[0]
        flop_per_out_channel = out_dim * flop_per_out_elem
        flop_total = out_channel * flop_per_out_channel
        if len(param_dim) > 0:
            param_B = size_b
            if type(param_dim[0]) is list:
                for pd in param_dim[0]:
                    param_B *= pd
            else:
                for pd in param_dim:
                    param_B *= pd
        else:
            param_B = 0
        input_B = size_b
        for e in layout_in:
            input_B *= e
        # calculate oi complete (input + params)
        data_cmpl = param_B + input_B
        oi_cmpl = float(flop_total) / float(data_cmpl)
        # calculate oi only "user" input
        oi_uinp = float(flop_total) / float(input_B)
        return oi_cmpl, oi_uinp, flop_total

    def calc_add(self, op_name, data_dim, param_dim, out_dim, attrs, size_b):
        # one addition per element
        layout_in = data_dim[0]
        layout_out = out_dim[0]
        batch_n = layout_in[0]
        cc_per_batch = 1
        for d in layout_in[1:]:
            cc_per_batch *= d
        flop_per_cc = 1  # exactly one addition per element
        flop_total = cc_per_batch * flop_per_cc * batch_n
        # calculate bw requirements
        if len(param_dim) > 0:
            param_B = size_b
            if type(param_dim[0]) is list:
                for pd in param_dim[0]:
                    param_B *= pd
            else:
                for pd in param_dim:
                    param_B *= pd
        else:
            param_B = 0
        input_B = size_b
        for e in layout_in:
            input_B *= e
        # calculate oi complete (input + params)
        data_cmpl = param_B + input_B
        oi_cmpl = float(flop_total) / float(data_cmpl)
        # calculate oi only "user" input
        oi_uinp = float(flop_total) / float(input_B)
        return oi_cmpl, oi_uinp, flop_total

    def calc_global_avg_pool2d(self, op_name, data_dim, param_dim, out_dim, attrs, size_b):
        layout_in = data_dim[0]
        layout_out = out_dim[0]
        batch_n = layout_in[0]
        in_channel = layout_in[1]
        out_channel = layout_out[1]
        sum_dim_x = layout_in[2] / layout_out[2]
        sum_dim_y = layout_in[3] / layout_out[3]
        flop_per_cc = sum_dim_x * sum_dim_y - 1 + 3  # kernel_size*2-1 additions, +3 for averaging
        cc_per_in_channel = layout_out[2] * layout_out[3]  # each elem in the output plane is result of a conv
        flop_per_out_c = flop_per_cc * cc_per_in_channel * in_channel  #in channel = out channel
        flop_total = flop_per_out_c * out_channel * batch_n
        # calculate bw requirements
        param_B = 0
        input_B = size_b
        for e in layout_in:
            input_B *= e
        # calculate oi complete (input + params)
        data_cmpl = param_B + input_B
        oi_cmpl = float(flop_total) / float(data_cmpl)
        # calculate oi only "user" input
        oi_uinp = float(flop_total) / float(input_B)
        return oi_cmpl, oi_uinp, flop_total

    def calc_multi_threshold(self, op_name, data_dim, param_dim, out_dim, attrs, size_b):
        # for the FPGA, it should be done within 1 cycle  --> ignore for now
        return 0.0, 0.0, 0.0
