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


def WriteComponents(target):
    target.write("--Components\n")
    # WriteInputLayerComponent(target)
    WriteDynInputLayerComponent(target)
    # WriteDisplayLayerComponent(target)
    WriteDynOutputLayerComponent(target)
    WriteConvlayerComponent(target)
    WritePoolLayerComponent(target)
    target.write("\n")


def WriteInputLayerComponent(target):
    target.write("component InputLayer\n")
    target.write("generic (\n")
    target.write("  BITWIDTH      : integer;\n")
    target.write("  INPUT_BIT_WIDTH : integer;\n")
    target.write("  NB_OUT_FLOWS    : integer\n")
    target.write(");\n")
    target.write("port (\n")
    target.write("  clk      : in  std_logic;\n")
    target.write("  reset_n  : in  std_logic;\n")
    target.write("  enable   : in  std_logic;\n")
    target.write("  in_data  : in  std_logic_vector(INPUT_BIT_WIDTH-1 downto 0);\n")
    target.write("  in_dv    : in  std_logic;\n")
    target.write("  in_fv    : in  std_logic;\n")
    target.write("  out_data : out pixel_array(NB_OUT_FLOWS-1 downto 0);\n")
    target.write("  out_dv   : out std_logic;\n")
    target.write("  out_fv   : out std_logic\n")
    target.write(");\n")
    target.write("end component InputLayer;\n\n")


def WriteDynInputLayerComponent(target):
    target.write("component DynInputLayer\n")
    target.write("generic (\n")
    target.write("  BITWIDTH      : integer;\n")
    target.write("  INPUT_BIT_WIDTH : integer;\n")
    target.write("  NB_OUT_FLOWS    : integer\n")
    target.write(");\n")
    target.write("port (\n")
    target.write("  clk      : in  std_logic;\n")
    target.write("  reset_n  : in  std_logic;\n")
    target.write("  enable   : in  std_logic;\n")
    target.write("  in_data  : in  std_logic_vector(INPUT_BIT_WIDTH-1 downto 0);\n")
    target.write("  in_dv    : in  std_logic;\n")
    target.write("  in_fv    : in  std_logic;\n")
    target.write("  out_data : out pixel_array(NB_OUT_FLOWS-1 downto 0);\n")
    target.write("  out_dv   : out std_logic;\n")
    target.write("  out_fv   : out std_logic\n")
    target.write(");\n")
    target.write("end component DynInputLayer;\n\n")


def WriteConvlayerComponent(target):
    target.write("component ConvLayer\n")
    target.write("generic (\n")
    target.write("  BITWIDTH   : integer;\n")
    target.write("  IMAGE_WIDTH  : integer;\n")
    target.write("  PROD_WIDTH    : integer;\n")
    target.write("  KERNEL_SIZE  : integer;\n")
    target.write("  NB_IN_FLOWS  : integer;\n")
    target.write("  NB_OUT_FLOWS : integer;\n")
    target.write("  USE_RELU_ACTIVATION : boolean;\n")
    target.write("  USE_TANH_ACTIVATION : boolean;\n")
    target.write("  KERNEL_VALUE : pixel_matrix;\n")
    target.write("  BIAS_VALUE   : pixel_array\n")
    target.write(");\n")
    target.write("port (\n")
    target.write("  clk      : in  std_logic;\n")
    target.write("  reset_n  : in  std_logic;\n")
    target.write("  enable   : in  std_logic;\n")
    target.write("  in_data  : in  pixel_array(NB_IN_FLOWS - 1 downto 0);\n")
    target.write("  in_dv    : in  std_logic;\n")
    target.write("  in_fv    : in  std_logic;\n")
    target.write("  out_data : out pixel_array(NB_OUT_FLOWS - 1 downto 0);\n")
    target.write("  out_dv   : out std_logic;\n")
    target.write("  out_fv   : out std_logic\n")
    target.write(");\n")
    target.write("end component ConvLayer;\n\n")


def WriteDisplayLayerComponent(target):
    target.write("component DisplayLayer is\n")
    target.write("generic(\n")
    target.write("  BITWIDTH : integer;\n")
    target.write("  NB_IN_FLOWS: integer\n")
    target.write(");\n")
    target.write("port(\n")
    target.write("  in_data  : in  pixel_array(NB_IN_FLOWS-1 downto 0);\n")
    target.write("  in_dv    : in  std_logic;\n")
    target.write("  in_fv    : in  std_logic;\n")
    target.write("  sel      : in  std_logic_vector(31 downto 0);\n")
    target.write("  out_data : out std_logic_vector(BITWIDTH-1 downto 0);\n")
    target.write("  out_dv   : out std_logic;\n")
    target.write("  out_fv   : out std_logic\n")
    target.write(");\n")
    target.write("end component;\n\n")


def WriteDynOutputLayerComponent(target):
    target.write("component DynOutputLayer is\n")
    target.write("generic(\n")
    target.write("  BITWIDTH : integer;\n")
    target.write("  NB_IN_FLOWS: integer\n")
    target.write(");\n")
    target.write("port(\n")
    target.write("  in_data  : in  pixel_array(NB_IN_FLOWS-1 downto 0);\n")
    target.write("  in_dv    : in  std_logic;\n")
    target.write("  in_fv    : in  std_logic;\n")
    target.write("  out_data : out std_logic_vector(NB_IN_FLOWS*BITWIDTH-1 downto 0);\n")
    target.write("  out_dv   : out std_logic;\n")
    target.write("  out_fv   : out std_logic\n")
    target.write(");\n")
    target.write("end component;\n\n")


def WritePoolLayerComponent(target):
    target.write("component PoolLayer\n")
    target.write("generic \n(")
    target.write("  BITWIDTH   : integer;\n")
    target.write("  IMAGE_WIDTH  : integer;\n")
    target.write("  KERNEL_SIZE  : integer;\n")
    target.write("  NB_OUT_FLOWS : integer\n")
    target.write(");\n")
    target.write("port (")
    target.write("  clk      : in  std_logic;\n")
    target.write("  reset_n  : in  std_logic;\n")
    target.write("  enable   : in  std_logic;\n")
    target.write("  in_data  : in  pixel_array(NB_OUT_FLOWS - 1 downto 0);\n")
    target.write("  in_dv    : in  std_logic;\n")
    target.write("  in_fv    : in  std_logic;\n")
    target.write("  out_data : out pixel_array(NB_OUT_FLOWS - 1 downto 0);\n")
    target.write("  out_dv   : out std_logic;\n")
    target.write("  out_fv   : out std_logic\n")
    target.write(");\n")
    target.write("end component PoolLayer;\n")


def WriteLayerSignal(target, layer_name):
    target.write("signal " + layer_name +
                 "_data: pixel_array ("
                 + layer_name + "_OUT_SIZE - 1 downto 0);\n")
    target.write("signal " + layer_name + "_dv\t: std_logic;\n")
    target.write("signal " + layer_name + "_fv\t: std_logic;\n")


def WriteInputSignal(target, layer_name, next_layer_name):
    target.write("signal " + layer_name + "_data: pixel_array(" +
                 next_layer_name + "_IN_SIZE-1 downto 0);\n")
    target.write("signal " + layer_name + "_dv\t: std_logic;\n")
    target.write("signal " + layer_name + "_fv\t: std_logic;\n")


def InstanceConvLayer(target, layer_name, previous_layer_name, use_relu_activation=False, use_tanh_activation=False):
    target.write(layer_name + ": ConvLayer\n")
    target.write("generic map (\n")
    target.write("  BITWIDTH   => BITWIDTH,\n")
    target.write("  PROD_WIDTH    => PROD_WIDTH,\n")
    target.write("  IMAGE_WIDTH  => " + layer_name + "_IMAGE_WIDTH,\n")
    target.write("  KERNEL_SIZE  => " + layer_name + "_KERNEL_SIZE,\n")
    target.write("  NB_IN_FLOWS  => " + layer_name + "_IN_SIZE,\n")
    target.write("  NB_OUT_FLOWS => " + layer_name + "_OUT_SIZE,\n")
    if use_relu_activation:
        target.write("  USE_RELU_ACTIVATION => true,\n")
    else:
        target.write("  USE_RELU_ACTIVATION => false,\n")
    if use_tanh_activation:
        target.write("  USE_TANH_ACTIVATION => true,\n")
    else:
        target.write("  USE_TANH_ACTIVATION => false,\n")
    target.write("  KERNEL_VALUE => " + layer_name + "_KERNEL_VALUE,\n")
    target.write("  BIAS_VALUE   => " + layer_name + "_BIAS_VALUE\n")
    target.write(")\n")
    target.write("port map (\n")
    target.write("  clk      => clk,\n")
    target.write("  reset_n  => reset_n,\n")
    target.write("  enable   => enable,\n")
    target.write("  in_data  => " + previous_layer_name + "_data,\n")
    target.write("  in_dv    => " + previous_layer_name + "_dv,\n")
    target.write("  in_fv    => " + previous_layer_name + "_fv,\n")
    target.write("  out_data => " + layer_name + "_data,\n")
    target.write("  out_dv   => " + layer_name + "_dv,\n")
    target.write("  out_fv   => " + layer_name + "_fv\n")
    target.write(");\n\n")


def InstancePoolLayer(target, layer_name, previous_layer_name):
    target.write(layer_name + " : PoolLayer\n")
    target.write("generic map (\n")
    target.write("  BITWIDTH   => BITWIDTH,\n")
    target.write("  IMAGE_WIDTH  => " + layer_name + "_IMAGE_WIDTH,\n")
    target.write("  KERNEL_SIZE  => " + layer_name + "_KERNEL_SIZE,\n")
    target.write("  NB_OUT_FLOWS => " + layer_name + "_OUT_SIZE\n")
    target.write(")\n")
    target.write("port map (\n")
    target.write("  clk      => clk,\n")
    target.write("  reset_n  => reset_n,\n")
    target.write("  enable   => enable,\n")
    target.write("  in_data  => " + previous_layer_name + "_data,\n")
    target.write("  in_dv    => " + previous_layer_name + "_dv,\n")
    target.write("  in_fv    => " + previous_layer_name + "_fv,\n")
    target.write("  out_data => " + layer_name + "_data,\n")
    target.write("  out_dv   => " + layer_name + "_dv,\n")
    target.write("  out_fv   => " + layer_name + "_fv\n")
    target.write(");\n\n")


def InstanceInputLayer(target, layer_name, next_layer_name, input_bitwidth):
    target.write("InputLayer_i : InputLayer\n")
    target.write("generic map (\n")
    target.write("  BITWIDTH      => BITWIDTH,\n")
    target.write("  INPUT_BIT_WIDTH => " + str(input_bitwidth) + ",\n")
    target.write("  NB_OUT_FLOWS    => " + next_layer_name + "_IN_SIZE\n")
    target.write(")\n")
    target.write("port map (\n")
    target.write("  clk      => clk,\n")
    target.write("  reset_n  => reset_n,\n")
    target.write("  enable   => enable,\n")
    target.write("  in_data  => in_data,\n")
    target.write("  in_dv    => in_dv,\n")
    target.write("  in_fv    => in_fv,\n")
    target.write("  out_data => " + layer_name + "_data,\n")
    target.write("  out_dv   => " + layer_name + "_dv,\n")
    target.write("  out_fv   => " + layer_name + "_fv\n")
    target.write("  );\n\n")


def InstanceDynInputLayer(target, layer_name, next_layer_name, input_bitwidth):
    target.write("DynInputLayer_i : DynInputLayer\n")
    target.write("generic map (\n")
    target.write("  BITWIDTH      => BITWIDTH,\n")
    target.write("  INPUT_BIT_WIDTH => " + str(input_bitwidth) + ",\n")
    target.write("  NB_OUT_FLOWS    => " + next_layer_name + "_IN_SIZE\n")
    target.write(")\n")
    target.write("port map (\n")
    target.write("  clk      => clk,\n")
    target.write("  reset_n  => reset_n,\n")
    target.write("  enable   => enable,\n")
    target.write("  in_data  => in_data,\n")
    target.write("  in_dv    => in_dv,\n")
    target.write("  in_fv    => in_fv,\n")
    target.write("  out_data => " + layer_name + "_data,\n")
    target.write("  out_dv   => " + layer_name + "_dv,\n")
    target.write("  out_fv   => " + layer_name + "_fv\n")
    target.write("  );\n\n")


def InstanceDisplayLayer(target, previous_layer_name):
    target.write("DisplayLayer_i: DisplayLayer\n")
    target.write("  generic map(\n")
    target.write("  BITWIDTH => BITWIDTH,\n")
    target.write("  NB_IN_FLOWS => " + previous_layer_name + "_OUT_SIZE\n")
    target.write("  )\n")
    target.write("  port map(\n")
    target.write("  in_data  => " + previous_layer_name + "_data,\n")
    target.write("  in_dv    => " + previous_layer_name + "_dv,\n")
    target.write("  in_fv    => " + previous_layer_name + "_fv,\n")
    target.write("  sel      => select_i,\n")
    target.write("  out_data => out_data,\n")
    target.write("  out_dv   => out_dv,\n")
    target.write("  out_fv   => out_fv\n")
    target.write(");\n")


def InstanceDynOutputLayer(target, previous_layer_name):
    target.write("DynOutputLayer_i: DynOutputLayer\n")
    target.write("  generic map(\n")
    target.write("  BITWIDTH => BITWIDTH,\n")
    target.write("  NB_IN_FLOWS => " + previous_layer_name + "_OUT_SIZE\n")
    target.write("  )\n")
    target.write("  port map(\n")
    target.write("  in_data  => " + previous_layer_name + "_data,\n")
    target.write("  in_dv    => " + previous_layer_name + "_dv,\n")
    target.write("  in_fv    => " + previous_layer_name + "_fv,\n")
    target.write("  out_data => out_data,\n")
    target.write("  out_dv   => out_dv,\n")
    target.write("  out_fv   => out_fv\n")
    target.write(");\n")


def WriteLibs(target, block_id):
    target.write("-- this file is automatically generated by DOSA to instantiate the VHDL4CNN library --\n")
    target.write("library ieee;\n")
    target.write("  use ieee.std_logic_1164.all;\n")
    target.write("  use ieee.numeric_std.all;\n")
    target.write("library work;\n")
    target.write("  use work.bitwidths_b{}.all;\n".format(block_id))
    target.write("  use work.cnn_types.all;\n")  # independent across Haddoc2
    target.write("  use work.params_b{}.all;\n".format(block_id))


def WriteEntity(target, block_id, first_layer_name):
    target.write("entity cnn_process_b{} is\n".format(block_id))
    target.write("generic(\n")
    target.write("  BITWIDTH  : integer := GENERAL_BITWIDTH;\n")
    target.write("  IMAGE_WIDTH : integer := " + first_layer_name + "_IMAGE_WIDTH\n")
    target.write(");\n")
    target.write("port(\n")
    target.write("  clk      : in  std_logic;\n")
    target.write("  reset_n  : in  std_logic;\n")
    target.write("  enable   : in  std_logic;\n")
    # target.write("  select_i : in  std_logic_vector(31 downto 0);\n")
    target.write("  in_data  : in  std_logic_vector(INPUT_BIT_WIDTH-1 downto 0);\n")
    target.write("  in_dv    : in  std_logic;\n")
    target.write("  in_rdy   : out std_logic;\n")
    target.write("  in_fv    : in  std_logic;\n")
    target.write("  out_data : out std_logic_vector(OUTPUT_BITWIDTH-1 downto 0);\n")
    target.write("  out_dv   : out std_logic;\n")
    target.write("  out_rdy  : in  std_logic;\n")
    target.write("  out_fv   : out std_logic\n")
    target.write("  );\n")
    target.write("end entity;\n\n")


def WriteArchitecutreHead(target, block_id):
    target.write("architecture STRUCTURAL of cnn_process_b{} is\n".format(block_id))


def WriteArchitectureEnd(target):
    target.write("-- to mimic AXIS, tready always 1\n in_rdy <= '1';\n")
    target.write("end architecture;\n")

