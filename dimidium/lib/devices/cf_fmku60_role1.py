#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: July 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Device data for the cF FMKU60 module with Role config 1
#  *        MUST IMPLEMENT: get_performance_dict(), get_roofline_dict(), get_resource_dict()
#  *

from dimidium.lib.units import *
from dimidium.lib.devices.dosa_device import DosaBaseHw

# FPGA specs
# UltraScale KU0600
# Speedgrade -2
freq_fpga_mhz = 156.0
freq_fpga_ghz = 0.156
freq_fpga = freq_fpga_mhz * megaU  # Hz
clk_fpga = 6.4  # ns
us_dsp48_s2_fmax_g = 0.661  # Ghz
ku060_num_dsp = 2760.0
dsp_flop_s = 4.0 * us_dsp48_s2_fmax_g
us_dsp48_s2_gflops = ku060_num_dsp * dsp_flop_s  # 4 FLOPs per DSP per cycle, 2760 DSPs per FPGA

cF_all_dsp48_gflops = 4.0 * ku060_num_dsp * freq_fpga_ghz
cF_1_dsp48_gflops = 4.0 * freq_fpga_ghz
cF_bigRole_dsp48_gflops = 1028.0 * 4.0 * freq_fpga_ghz

cF_mantle_dsp48_gflops = 938.0 * 4.0 * freq_fpga_ghz

# DRAM bandwidth
b_s_fpga_ddr_gBs = 10.0  # 10GB/s (one memory bank of FMKU60)

# b_s_mantle_ddr_gBs = 75.5/8  # based on Xilinx measurements

# BRAM bandwidth
fpga_brams = 1080
big_role_brams = 351
b_s_fpga_bram_Bs = (big_role_brams * 72 / 8) / (
            1 / freq_fpga)  # 1080 BRAMs with 72 bit write per cycle each, Bytes/s
b_s_fpga_bram_gBs = b_s_fpga_bram_Bs / gigaU

# small_role_brams = 306
# b_s_mantle_bram_gBs = ((small_role_brams * 72/8) / (1/freq_fpga) ) / gigaU

# LUTRAM bandwidth (distributed RAM)
fpga_lutram_available_B = (9180 * 2 * 8) * 8  # 146880 available LUTRAMs, 64bit/8Byte each, Bytes
big_role_lutram_available_B = 52640.0
small_role_lutram_available_B = 47040.0
b_s_fpga_lutram_Bs = big_role_lutram_available_B / (1 / freq_fpga)  # Bytes/s
b_s_fpga_lutram_gBs = b_s_fpga_lutram_Bs / gigaU

# b_s_mantle_lutram_gBs = (small_role_lutram_available_B / (1/freq_fpga)) / gigaU

# network bandwidth
b_s_fpga_eth_gBs = 10.0 / 8.0  # 10Gbe

# b_s_mantle_eth_gBs = 9.87 / 8.0


class CfThemisto1(DosaBaseHw):

    def __init__(self, hw_type, name):
        super().__init__(hw_type, name)

    def get_performance_dict(self):
        ret = {'fpga_freq_Hz': freq_fpga, 'dsp48_gflops': cF_bigRole_dsp48_gflops,
           'bw_ddr4_gBs': b_s_fpga_ddr_gBs, 'bw_bram_gBs': b_s_fpga_bram_gBs,
           'bw_netw_gBs': b_s_fpga_eth_gBs, 'bw_lutram_gBs': b_s_fpga_lutram_gBs,
           'type': 'fpga'}
        return ret

    def get_roofline_dict(self):
        ret = {'sweet_spot': 0.081}
        return ret

    def get_resource_dict(self):
        return

