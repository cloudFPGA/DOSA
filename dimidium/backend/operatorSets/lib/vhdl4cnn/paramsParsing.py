#  /*******************************************************************************
#   * Copyright 2019 -- 2023 IBM Corporation
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
#  *     Created: Nov 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Library for Haddoc2 code generation.
#  *        Please see the folder Readme.md for more information.
#  *
#  *

import numpy as np
import math
import time


def write_in_size(layer_name, value, target):
    target.write("constant ")
    target.write(layer_name)
    target.write("_IN_SIZE      :  integer := ")
    target.write(str(value) + " ;\n")


def write_out_size(layer_name, value, target):
    target.write("constant ")
    target.write(layer_name)
    target.write("_OUT_SIZE     :  integer := ")
    target.write(str(value) + " ;\n")


def write_kernel_size(layer_name, value, target):
    target.write("constant ")
    target.write(layer_name)
    target.write("_KERNEL_SIZE  :  integer := ")
    target.write(str(value) + " ;\n")


def write_image_width(layer_name, value, target):
    target.write("constant ")
    target.write(layer_name)
    target.write("_IMAGE_WIDTH  :  integer := ")
    target.write(str(value) + " ;\n")


def to_shiftNorm(kernel):
    ### DEPRECATED ###
    norm = np.abs(np.sum(kernel[...]))
    real_shift = math.log(norm, 2)
    int_shift = int(real_shift)
    return int_shift + 1


def to_fixedPoint(data, scale_factor):
    scaled_data = scale_factor * data
    return np.array(np.round(scaled_data), dtype=int)


def to_scaleFactor(nbits):
    return math.pow(2, (nbits - 1)) - 1


def write_bias_value(bias_data, name, nbits, target):
    layer_name = name

    # scale_factor = to_scaleFactor(nbits)
    out_size = bias_data.shape[0]

    target.write("constant ")
    target.write(layer_name)
    target.write("_BIAS_VALUE   :  pixel_array ")
    target.write("   (" + layer_name + "_OUT_SIZE - 1 downto 0) := \n")
    # NOT scaling AGAIN
    # bias_fp = to_fixedPoint(bias_data, scale_factor)

    target.write(" (")
    # to deal with 'type does not match with a string literal'
    # if out_size == 1:
    #    target.write(" others => ")
    for n in range(out_size):
        # bias_bin = np.binary_repr(bias_fp[n], width=nbits)
        target.write(f" {n} => ")
        bias_bin = np.binary_repr(bias_data[n], width=nbits)
        target.write("\"" + bias_bin + "\"")
        if n == (out_size - 1):
            target.write(");\n")
        else:
            target.write(",")


def write_kernel_value(kernel_data, layer_name, nbits, target):
    #kernel_data  = data

    scale_factor = to_scaleFactor(nbits)
    out_size = kernel_data.shape[0]
    in_size = kernel_data.shape[1]
    kernel_size = kernel_data.shape[2]

    # constant CONV1_KERNEL_VALUE : pixel_matrix
    target.write("constant ")
    target.write(layer_name)
    target.write("_KERNEL_VALUE :  pixel_matrix ")
    # (0 to CONV1_LAYER_SIZE * CONV1_LAYER_SIZE - 1, 0 to CONV1_KERNEL_SIZE*CONV1_KERNEL_SIZE - 1) :=
    target.write(" (" + layer_name + "_OUT_SIZE - 1 downto 0,")
    target.write("  " + layer_name + "_IN_SIZE * " + layer_name +
                 "_KERNEL_SIZE * " + layer_name + "_KERNEL_SIZE - 1 downto 0)")
    target.write(" :=\n")

    # NOT scaling AGAIN
    # kernel_fp = to_fixedPoint(kernel_data, scale_factor)
    kernel_fp = kernel_data
    target.write(" (")
    # to deal with 'type does not match with a string literal'
    # if out_size == 1:
    #     target.write(" others => ")
    upper_bound = scale_factor
    lower_bound = -scale_factor - 1

    # In some Networks, such AlexNet, neurons from layer l are not totally connected to layer l+1
    # But only a group is connected. We manage this as follows:
    #target.write(" (")
    for n in range(out_size):
        target.write(f" {n} => (")
        for m in range(in_size):
            for i in range(0, kernel_size):
                for j in range(0, kernel_size):
            # for i in range(kernel_size - 1, -1, -1):
            #     for j in range(kernel_size - 1, -1, -1):
                    if kernel_fp[n][m][i][j] > upper_bound:
                        print(f"[:HADDOC:INFO] weight at {n}, {m}, {i}, {j} above upper bound ({kernel_fp[n][m][i][j]}), setting to {upper_bound}.")
                        kernel_fp[n][m][i][j] = upper_bound
                    if kernel_fp[n][m][i][j] < lower_bound:
                        print(f"[:HADDOC:INFO] weight at {n}, {m}, {i}, {j} below lower bound ({kernel_fp[n][m][i][j]}), setting to {lower_bound}.")
                        kernel_fp[n][m][i][j] = lower_bound
                    kernel_bin = np.binary_repr(
                        kernel_fp[n][m][i][j], width=nbits)
                    target.write("\"" + kernel_bin + "\"")
                    # if (m != in_size - 1 or i != 0 or j != 0):
                    if (m != in_size - 1) or (i != kernel_size - 1) or (j != kernel_size - 1):
                            target.write(",")
            # if (m == in_size - 1):
            #     if (n != out_size - 1):
            #         target.write("),\n (")
            #     else:
            #         target.write(")")
        target.write(")")
        if n != (out_size - 1):
            target.write(",\n  ")
    target.write("\n);\n")


# def parse_convLayer(target, cnn, layer_name, previous_layer_name, nbits):
#     kernel_data = cnn.params[layer_name][0].data
#     in_size = cnn.params[layer_name][0].data.shape[1]
#     out_size = cnn.blobs[layer_name].data.shape[1]
#     previous_layer_size = cnn.blobs[previous_layer_name].data.shape[1]
#     kernel_size = cnn.params[layer_name][0].data.shape[2]
#     image_width = cnn.blobs[previous_layer_name].data.shape[2]
#     bias_data = np.zeros(out_size, dtype=float)
#     try:
#         bias_data = cnn.params[layer_name][1].data
#     except (IndexError):
#         bias_data = np.zeros(out_size, dtype=float)
#     except (NameError):
#         bias_data = np.zeros(out_size, dtype=float)
#     ## Write layer params ##
#     target.write("--" + layer_name + "\n")
#     write_image_width(layer_name, image_width, target)
#     write_in_size(layer_name, previous_layer_size, target)
#     write_out_size(layer_name, out_size, target)
#     write_kernel_size(layer_name, kernel_size, target)
#     write_bias_value(bias_data, layer_name, nbits, target)
#     write_kernel_value(kernel_data, layer_name, nbits, target)
#     target.write("----------------------------------------------------------")
#     target.write("--------------------------------------------------------\n")
#
#
# def parse_poolLayer(target, cnn, layer_name, previous_layer_name):
#     kernel_size = 2  # For now only a subsampling factor of 4 is supported
#     out_size = cnn.blobs[layer_name].data.shape[1]
#     image_width = cnn.blobs[previous_layer_name].data.shape[2]
#     target.write("--" + layer_name + "\n")
#     write_image_width(layer_name, image_width, target)
#     write_out_size(layer_name, out_size, target)
#     write_kernel_size(layer_name, kernel_size, target)
#     target.write("----------------------------------------------------------")
#     target.write("--------------------------------------------------------\n")


def write_fileHead(target, block_id):
    target.write("--------------------------------------------------------\n")
    target.write("-- This file is generated with Haddoc2 utility and DOSA\n")
    target.write("-- Generated on : " + time.ctime() + "\n")
    target.write(
        "--------------------------------------------------------\n\n")
    target.write("library ieee;\n")
    target.write("    use    ieee.std_logic_1164.all;\n")
    target.write("library work;\n")
    target.write("    use    work.cnn_types.all;\n")
    target.write("package params_b{} is\n".format(block_id))


def write_pixelWidth(target, pixelWidth):
    target.write("constant GENERAL_BITWIDTH    : integer :=" +
                 str(pixelWidth) + ";\n")


def write_fileEnd(target):
    target.write("end package;")
