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

import dimidium.lib.singleton as dosa_singleton
from dimidium.lib.units import *
from dimidium.backend.devices.dosa_device import DosaHwClasses, DosaBaseHw
from dimidium.backend.buildTools.cFBuild1 import cFBuild1


class CfThemisto1(DosaBaseHw):

    def __init__(self, name):
        super().__init__(DosaHwClasses.FPGA_xilinx, 'cF_FMKU60_Themisto_1', name, [cFBuild1])
        self.initialized = False
        self.global_main_body_tmpl_path = None
        self.global_main_head_tmpl_path = None
        self.part_string = 'xcku060-ffva1156-2-i'
        self.clock_period_ns = 6.4
        self.clock_period_s = self.clock_period_ns * 1e-9
        self.cf_mod_type = 'FMKU60'
        self.cf_sra = 'Themisto'

    def _gen_numbers(self):
        if self.initialized:
            return
        config_dosa_flops_per_dsp_xilinx_fpgas = dosa_singleton.config.dtype.flops_per_dsp_xilinx_fpgas
        self.config_dosa_kappa = dosa_singleton.config.dtype.dosa_kappa

        # FPGA specs
        # UltraScale KU0600
        # Speedgrade -2
        freq_fpga_mhz = 156.0
        freq_fpga_ghz = 0.156
        self.freq_fpga = freq_fpga_mhz * megaU  # Hz
        clk_fpga = 6.4  # ns
        us_dsp48_s2_fmax_g = 0.661  # Ghz
        ku060_num_dsp = 2760.0
        # dsp_flop_s = 4.0 * us_dsp48_s2_fmax_g
        dsp_flop_s = config_dosa_flops_per_dsp_xilinx_fpgas * us_dsp48_s2_fmax_g
        us_dsp48_s2_gflops = ku060_num_dsp * dsp_flop_s  # 4 FLOPs per DSP per cycle, 2760 DSPs per FPGA

        # cF_all_dsp48_gflops = 4.0 * ku060_num_dsp * freq_fpga_ghz
        cF_all_dsp48_gflops = config_dosa_flops_per_dsp_xilinx_fpgas * ku060_num_dsp * freq_fpga_ghz
        # cF_1_dsp48_gflops = 4.0 * freq_fpga_ghz
        cF_1_dsp48_gflops = config_dosa_flops_per_dsp_xilinx_fpgas * freq_fpga_ghz
        # cF_bigRole_dsp48_gflops = 1028.0 * 4.0 * freq_fpga_ghz
        big_role_dsps = 1028.0
        role6_dsps = 1106
        self.cF_bigRole_dsp48_gflops = role6_dsps * config_dosa_flops_per_dsp_xilinx_fpgas * freq_fpga_ghz

        # cF_mantle_dsp48_gflops = 938.0 * 4.0 * freq_fpga_ghz
        cF_mantle_dsp48_gflops = 938.0 * config_dosa_flops_per_dsp_xilinx_fpgas * freq_fpga_ghz

        # DRAM bandwidth
        self.b_s_fpga_ddr_gBs = 10.0  # 10GB/s (one memory bank of FMKU60)

        # b_s_mantle_ddr_gBs = 75.5/8  # based on Xilinx measurements

        # BRAM bandwidth
        fpga_brams = 1080
        # big_role_brams = 351
        role6_brams = 348
        b_s_fpga_bram_Bs = (role6_brams * 72 / 8) / (
                1 / self.freq_fpga)  # 1080 BRAMs with 72 bit write per cycle each, Bytes/s
        self.b_s_fpga_bram_gBs = b_s_fpga_bram_Bs / gigaU

        # small_role_brams = 306
        # b_s_mantle_bram_gBs = ((small_role_brams * 72/8) / (1/freq_fpga) ) / gigaU

        # LUTRAM bandwidth (distributed RAM)
        fpga_lutram_available_B = (9180 * 2 * 8) * 8  # 146880 available LUTRAMs, 64bit/8Byte each, Bytes
        # big_role_lutram_available_B = 52640.0
        # small_role_lutram_available_B = 47040.0
        role6_lutram_available = 53400
        role6_lutram_available_inBytes = role6_lutram_available * 8
        b_s_fpga_lutram_Bs = role6_lutram_available_inBytes / (1 / self.freq_fpga)  # Bytes/s
        self.b_s_fpga_lutram_gBs = b_s_fpga_lutram_Bs / gigaU

        # TODO: flip flops?
        #  --> but are rather used internally?
        role6_ff_available = 203200

        # b_s_mantle_lutram_gBs = (small_role_lutram_available_B / (1/freq_fpga)) / gigaU

        # network bandwidth
        self.b_s_fpga_eth_gBs = 10.0 / 8.0  # 10Gbe
        # b_s_mantle_eth_gBs = 9.87 / 8.0

        # utilization
        total_flops_dsps = self.cF_bigRole_dsp48_gflops * gigaU * self.config_dosa_kappa
        role6_luts_available = 101600
        total_flops_luts = (role6_luts_available / dosa_singleton.config.utilization.xilinx_luts_to_dsp_factor) \
                           * config_dosa_flops_per_dsp_xilinx_fpgas * freq_fpga_ghz * self.config_dosa_kappa
        self.total_flops_hw = total_flops_luts + total_flops_dsps
        total_bytes_bram = (role6_brams * 36 * kiloU) / 8  # 36Kb RAMs
        total_bytes_lutram = role6_lutram_available_inBytes / \
                             dosa_singleton.config.utilization.xilinx_lutram_to_bram_factor
        self.total_bytes_hw = total_bytes_bram + total_bytes_lutram

        self.initialized = True
        return

    def get_performance_dict(self):
        self._gen_numbers()
        max_gflops = self.cF_bigRole_dsp48_gflops * self.config_dosa_kappa
        ret = {'fpga_freq_Hz': self.freq_fpga, 'dsp48_gflops': max_gflops,
               'bw_dram_gBs': self.b_s_fpga_ddr_gBs, 'bw_bram_gBs': self.b_s_fpga_bram_gBs,
               'bw_netw_gBs': self.b_s_fpga_eth_gBs, 'bw_lutram_gBs': self.b_s_fpga_lutram_gBs,
               # 'gflops_limit': max_gflops,
               'type': str(self.hw_class)}
        return ret

    def get_roofline_dict(self):
        self._gen_numbers()
        # ret = {'sweet_spot': 0.081}  # old DSP calc
        # ret = {'sweet_spot': 0.1566}  # for BRAM
        ret = {'sweet_spot': 0.0100}  # for LUTRAM
        return ret

    def get_resource_dict(self):
        self._gen_numbers()
        ret = {'total_flops': self.total_flops_hw, 'total_on_chip_memory_bytes': self.total_bytes_hw}
        return ret

    def get_max_flops(self):
        self._gen_numbers()
        return self.cF_bigRole_dsp48_gflops * gigaU * self.config_dosa_kappa

    def get_comm_latency_s(self):
        return 0.1 * mikroU

    def get_hw_utilization_tuple(self, flops, bake_in_params_bytes):
        self._gen_numbers()
        share_flops = float(flops/self.total_flops_hw) * dosa_singleton.config.utilization.dosa_mu
        share_memory = float(bake_in_params_bytes/self.total_bytes_hw) * dosa_singleton.config.utilization.dosa_mu
        return share_flops, share_memory

