#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
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
    WriteInputLayerComponent(target)
    WriteDisplayLayerComponent(target)
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
    target.write("  out_data : out pixel_array(0 to NB_OUT_FLOWS-1);\n")
    target.write("  out_dv   : out std_logic;\n")
    target.write("  out_fv   : out std_logic\n")
    target.write(");\n")
    target.write("end component InputLayer;\n\n")


def WriteConvlayerComponent(target):
    target.write("component ConvLayer\n")
    target.write("generic (\n")
    target.write("  BITWIDTH   : integer;\n")
    target.write("  IMAGE_WIDTH  : integer;\n")
    target.write("  SUM_WIDTH    : integer;\n")
    target.write("  KERNEL_SIZE  : integer;\n")
    target.write("  NB_IN_FLOWS  : integer;\n")
    target.write("  NB_OUT_FLOWS : integer;\n")
    target.write("  KERNEL_VALUE : pixel_matrix;\n")
    target.write("  BIAS_VALUE   : pixel_array\n")
    target.write(");\n")
    target.write("port (\n")
    target.write("  clk      : in  std_logic;\n")
    target.write("  reset_n  : in  std_logic;\n")
    target.write("  enable   : in  std_logic;\n")
    target.write("  in_data  : in  pixel_array(0 to NB_IN_FLOWS - 1);\n")
    target.write("  in_dv    : in  std_logic;\n")
    target.write("  in_fv    : in  std_logic;\n")
    target.write("  out_data : out pixel_array(0 to NB_OUT_FLOWS - 1);\n")
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
    target.write("  in_data  : in  pixel_array(0 to NB_IN_FLOWS-1);\n")
    target.write("  in_dv    : in  std_logic;\n")
    target.write("  in_fv    : in  std_logic;\n")
    target.write("  sel      : in  std_logic_vector(31 downto 0);\n")
    target.write("  out_data : out std_logic_vector(BITWIDTH-1 downto 0);\n")
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
    target.write("  in_data  : in  pixel_array(0 to NB_OUT_FLOWS - 1);\n")
    target.write("  in_dv    : in  std_logic;\n")
    target.write("  in_fv    : in  std_logic;\n")
    target.write("  out_data : out pixel_array(0 to NB_OUT_FLOWS - 1);\n")
    target.write("  out_dv   : out std_logic;\n")
    target.write("  out_fv   : out std_logic\n")
    target.write(");\n")
    target.write("end component PoolLayer;\n")


def WriteLayerSignal(target, layer_name):
    target.write("signal " + layer_name +
                 "_data: pixel_array (0 to "
                 + layer_name + "_OUT_SIZE - 1);\n")
    target.write("signal " + layer_name + "_dv\t: std_logic;\n")
    target.write("signal " + layer_name + "_fv\t: std_logic;\n")


def WriteInputSignal(target, layer_name, next_layer_name):
    target.write("signal " + layer_name + "_data: pixel_array(0 to " +
                 next_layer_name + "_IN_SIZE-1);\n")
    target.write("signal " + layer_name + "_dv\t: std_logic;\n")
    target.write("signal " + layer_name + "_fv\t: std_logic;\n")


def InstanceConvLayer(target, layer_name, previous_layer_name):
    target.write(layer_name + ": ConvLayer\n")
    target.write("generic map (\n")
    target.write("  BITWIDTH   => BITWIDTH,\n")
    target.write("  SUM_WIDTH    => SUM_WIDTH,\n")
    target.write("  IMAGE_WIDTH  => " + layer_name + "_IMAGE_WIDTH,\n")
    target.write("  KERNEL_SIZE  => " + layer_name + "_KERNEL_SIZE,\n")
    target.write("  NB_IN_FLOWS  => " + layer_name + "_IN_SIZE,\n")
    target.write("  NB_OUT_FLOWS => " + layer_name + "_OUT_SIZE,\n")
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


def WriteLibs(target):
    target.write("library ieee;\n")
    target.write("  use ieee.std_logic_1164.all;\n")
    target.write("  use ieee.numeric_std.all;\n")
    target.write("library work;\n")
    target.write("  use work.bitwidths.all;\n")
    target.write("  use work.cnn_types.all;\n")
    target.write("  use work.params.all;\n")


def WriteEntity(target):
    target.write("entity cnn_process is\n")
    target.write("generic(\n")
    target.write("  BITWIDTH  : integer := GENERAL_BITWIDTH;\n")
    target.write("  IMAGE_WIDTH : integer := CONV1_IMAGE_WIDTH\n")
    target.write(");\n")
    target.write("port(\n")
    target.write("  clk      : in std_logic;\n")
    target.write("  reset_n  : in std_logic;\n")
    target.write("  enable   : in std_logic;\n")
    target.write("  select_i : in std_logic_vector(31 downto 0);\n")
    target.write("  in_data  : in std_logic_vector(INPUT_BIT_WIDTH-1 downto 0);\n")
    target.write("  in_dv    : in std_logic;\n")
    target.write("  in_fv    : in std_logic;\n")
    target.write("  out_data : out std_logic_vector(BITWIDTH-1 downto 0);\n")
    target.write("  out_dv   : out std_logic;\n")
    target.write("  out_fv   : out std_logic\n")
    target.write("  );\n")
    target.write("end entity;\n\n")


def WriteArchitecutreHead(target):
    target.write("architecture STRUCTURAL of cnn_process is\n")


def WriteArchitectureEnd(target):
    target.write("end architecture;\n")

