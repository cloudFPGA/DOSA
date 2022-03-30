#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Aug 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Base Roofline class for DOSA devices
#  *        (including 3D Roofline)
#  *
#  *

import numpy as np
from enum import Enum

from dimidium.lib.util import rf_attainable_performance, BrickImplTypes, rf_calc_sweet_spot
import dimidium.lib.units as units
from dimidium.middleend.archGen.DosaContract import DosaContract

config_global_rf_ylim_min = 0.01
# config_global_rf_ylim_max = 100000
config_global_rf_ylim_max = 10000

__deactivated_bw_value__ = -1


class RooflineRegionsOiPlane(Enum):
    IN_HOUSE = 0
    ABOVE_TOP = 1
    ABOVE_NETWORK = 2
    ABOVE_DRAM = 3
    ABOVE_BRAM = 4


class RooflinePlane(Enum):
    OI_PLANE = 0
    UTIL_PLANE = 1


class RooflineRegionsEfficPlane(Enum):
    IN_HOUSE = 0
    ABOVTE_TOP = 1
    ABOVE_100 = 2


# compare roofline regions and return the better: 1 or 2, if equal also 1 will be returned
def get_rightmost_roofline_region(r1: RooflineRegionsOiPlane, r2: RooflineRegionsOiPlane, ignore_network=False):
    if r1 == RooflineRegionsOiPlane.IN_HOUSE or r2 == RooflineRegionsOiPlane.ABOVE_TOP:
        return 1
    if r2 == RooflineRegionsOiPlane.IN_HOUSE or r1 == RooflineRegionsOiPlane.ABOVE_TOP:
        return 2
    if not ignore_network:
        if (r1 == RooflineRegionsOiPlane.ABOVE_NETWORK or r1 == RooflineRegionsOiPlane.ABOVE_DRAM) \
                and (r2 == RooflineRegionsOiPlane.ABOVE_DRAM or r2 == RooflineRegionsOiPlane.ABOVE_BRAM):
            return 1
    else:
        if (r1 == RooflineRegionsOiPlane.ABOVE_DRAM) \
                and (r2 == RooflineRegionsOiPlane.ABOVE_DRAM or r2 == RooflineRegionsOiPlane.ABOVE_BRAM):
            return 1
    if r1 == RooflineRegionsOiPlane.ABOVE_BRAM and r2 == RooflineRegionsOiPlane.ABOVE_BRAM:
        return 1
    if r1 == RooflineRegionsOiPlane.ABOVE_TOP and r2 == RooflineRegionsOiPlane.ABOVE_TOP:
        return 1
    return 2


class DosaRoofline(object):

    # operational intensity vector
    oi_list_small = np.arange(0.01, 1, 0.01)
    oi_list_middle = np.arange(1, 1500, 0.1)
    oi_list_big = np.arange(1501, 10100, 1)
    oi_list = np.concatenate((oi_list_small, oi_list_middle, oi_list_big))
    utt_list = np.concatenate((oi_list_small, oi_list_middle, oi_list_big))

    # Perf boundaries
    ylim_min = config_global_rf_ylim_min * units.gigaU
    ylim_max = config_global_rf_ylim_max * units.gigaU

    def __init__(self, rl_F=0, net_bw_B=0, dram_bw_B=0,
                 bram_bw_B=__deactivated_bw_value__,
                 # lutram_bw_B=__deactivated_bw_value__
                 ):
        self.roof_F = rl_F
        self.net_bw_B = net_bw_B
        self.dram_bw_B = dram_bw_B
        self.bram_bw_B = bram_bw_B
        # self.lutram_bw_B = lutram_bw_B
        # sweet spots
        self._cache_valid_ = False
        self.sp_net = __deactivated_bw_value__
        self.sp_dram = __deactivated_bw_value__
        self.sp_bram = __deactivated_bw_value__
        self.utility_total_flops = None
        self.utility_total_bytes = None
        self.sp_effic_compute = __deactivated_bw_value__
        self.sp_effic_memory = __deactivated_bw_value__

    def get_region_OIPlane(self, oi_FB, req_perf_F) -> RooflineRegionsOiPlane:
        if oi_FB < 0.001 or req_perf_F < 0.001:
            return RooflineRegionsOiPlane.IN_HOUSE
        # TODO: do not assume DRAM BW is always higher than network BW?
        if req_perf_F >= self.roof_F:
            return RooflineRegionsOiPlane.ABOVE_TOP
        # if not self._cache_valid_:
        #     self.update_sweet_spots()
        ap_net = rf_attainable_performance(oi_FB, self.roof_F, self.net_bw_B)
        ap_dram = rf_attainable_performance(oi_FB, self.roof_F, self.dram_bw_B)
        if self.bram_bw_B != __deactivated_bw_value__:
            ap_bram = rf_attainable_performance(oi_FB, self.roof_F, self.bram_bw_B)
            if req_perf_F >= ap_bram:
                return RooflineRegionsOiPlane.ABOVE_BRAM
        if req_perf_F > ap_dram:
            return RooflineRegionsOiPlane.ABOVE_DRAM
        if req_perf_F > ap_net:
            return RooflineRegionsOiPlane.ABOVE_NETWORK
        return RooflineRegionsOiPlane.IN_HOUSE

    def get_region_OIPlane_iter_based(self, oi_iter, req_iter, contracts) -> RooflineRegionsOiPlane:
        if oi_iter < 0.001 or req_iter < 0.001:
            return RooflineRegionsOiPlane.IN_HOUSE
        cl = []
        if isinstance(contracts, list):
            cl = contracts
        elif isinstance(contracts, DosaContract):
            cl = [contracts]
        else:
            print("Can't determine Roofline for this argument {}. STOP".format(contracts))
            exit(1)
        min_iter = float('inf')
        total_comp_share = 0.0
        total_mem_share = 0.0
        for contr in cl:
            if contr.iter_hz < min_iter:
                min_iter = contr.iter_hz
            total_comp_share += contr.comp_util_share
            total_mem_share += contr.mem_util_share
        max_util = max(total_comp_share, total_mem_share)
        max_iter = (1.0/max_util) * min_iter
        # TODO: do not assume DRAM BW is always higher than network BW?
        if req_iter > max_iter:
            return RooflineRegionsOiPlane.ABOVE_TOP
        # if not self._cache_valid_:
        #     self.update_sweet_spots()
        ap_net = rf_attainable_performance(oi_iter, max_iter, self.net_bw_B)
        ap_dram = rf_attainable_performance(oi_iter, max_iter, self.dram_bw_B)
        if self.bram_bw_B != __deactivated_bw_value__:
            ap_bram = rf_attainable_performance(oi_iter, max_iter, self.bram_bw_B)
            if req_iter >= ap_bram:
                return RooflineRegionsOiPlane.ABOVE_BRAM
        if req_iter > ap_dram:
            return RooflineRegionsOiPlane.ABOVE_DRAM
        if req_iter > ap_net:
            return RooflineRegionsOiPlane.ABOVE_NETWORK
        return RooflineRegionsOiPlane.IN_HOUSE

    def get_max_perf_at_oi_iter_based(self, oi_iter, contracts, ignore_net=False, ignore_bram=False):
        if oi_iter < 0.001:
            oi_iter = 0.01
        cl = []
        if isinstance(contracts, list):
            cl = contracts
        elif isinstance(contracts, DosaContract):
            cl = [contracts]
        else:
            print("Can't determine Roofline for this argument {}. STOP".format(contracts))
            exit(1)
        min_iter = float('inf')
        total_comp_share = 0.0
        total_mem_share = 0.0
        for contr in cl:
            if contr.iter_hz < min_iter:
                min_iter = contr.iter_hz
            total_comp_share += contr.comp_util_share
            total_mem_share += contr.mem_util_share
        max_util = max(total_comp_share, total_mem_share)
        max_iter = (1.0/max_util) * min_iter
        ap_net = rf_attainable_performance(oi_iter, max_iter, self.net_bw_B)
        ap_dram = rf_attainable_performance(oi_iter, max_iter, self.dram_bw_B)
        if self.bram_bw_B != __deactivated_bw_value__:
            ap_bram = rf_attainable_performance(oi_iter, max_iter, self.bram_bw_B)
        else:
            ap_bram = -1
            ignore_bram = True
        if ap_net <= ap_dram and (ap_net <= ap_bram or ignore_bram) and not ignore_net:
            return ap_net
        if (ap_dram <= ap_net or ignore_net) and (ap_dram <= ap_bram or ignore_bram):
            return ap_dram
        return ap_bram

    def get_max_perf_at_oi(self, oi_FB, ignore_net=False, ignore_bram=False):
        if oi_FB < 0.001:
            oi_FB = 0.01
        ap_net = rf_attainable_performance(oi_FB, self.roof_F, self.net_bw_B)
        ap_dram = rf_attainable_performance(oi_FB, self.roof_F, self.dram_bw_B)
        if self.bram_bw_B != __deactivated_bw_value__:
            ap_bram = rf_attainable_performance(oi_FB, self.roof_F, self.bram_bw_B)
        else:
            ap_bram = -1
            ignore_bram = True
        if ap_net <= ap_dram and (ap_net <= ap_bram or ignore_bram) and not ignore_net:
            return ap_net
        if (ap_dram <= ap_net or ignore_net) and (ap_dram <= ap_bram or ignore_bram):
            return ap_dram
        return ap_bram

    def get_ap_OiPlane(self, oi_FB, consider_type=BrickImplTypes.ENGINE):
        ap_net = rf_attainable_performance(oi_FB, self.roof_F, self.net_bw_B)
        ap_dram = rf_attainable_performance(oi_FB, self.roof_F, self.dram_bw_B)
        if consider_type == BrickImplTypes.STREAM:
            return ap_net
        res_ap = min(ap_net, ap_dram)
        if self.bram_bw_B != __deactivated_bw_value__:
            ap_bram = rf_attainable_performance(oi_FB, self.roof_F, self.bram_bw_B)
            res_ap = min(res_ap, ap_bram)
        # if self.lutram_bw_B != __deactivated_bw_value__:
        #     ap_lutram = rf_attainable_performance(oi, self.roof_F, self.lutram_bw_B)
        #     res_ap = min(res_ap, ap_lutram)
        return res_ap

    def from_perf_dict(self, perf_dict):
        if 'dsp48_gflops' in perf_dict:
            self.roof_F = perf_dict['dsp48_gflops'] * units.gigaU
        else:
            self.roof_F = perf_dict['cpu_gflops'] * units.gigaU
        self.net_bw_B = perf_dict['bw_netw_gBs'] * units.gigaU
        self.dram_bw_B = perf_dict['bw_dram_gBs'] * units.gigaU
        if 'bw_bram_gBs' in perf_dict:
            self.bram_bw_B = perf_dict['bw_bram_gBs'] * units.gigaU
        else:
            self.bram_bw_B = __deactivated_bw_value__
        self._cache_valid_ = False
        
    def from_resource_dict(self, res_dict):
        # should be the same like roof_F, but may change?
        self.utility_total_flops = res_dict['total_flops']  
        self.utility_total_bytes = res_dict['total_on_chip_memory_bytes']
        self._cache_valid_ = False

    def update_sweet_spots(self):
        self.sp_net = rf_calc_sweet_spot(self.oi_list, self.roof_F, self.net_bw_B)
        self.sp_dram = rf_calc_sweet_spot(self.oi_list, self.roof_F, self.dram_bw_B)
        if self.bram_bw_B != __deactivated_bw_value__:
            self.sp_bram = rf_calc_sweet_spot(self.oi_list, self.roof_F, self.bram_bw_B)
        self.sp_effic_memory = rf_calc_sweet_spot(self.utt_list, self.utility_total_bytes, 1)
        self.sp_effic_compute = rf_calc_sweet_spot(self.utt_list, self.utility_total_flops, 1)
        self._cache_valid_ = True

    # def get_region_EfficPlane(self, effic_FB, req_perf_F) -> RooflineRegionsEfficPlane:
    #     if effic_FB < 0.001 or req_perf_F < 0.001:
    #         return RooflineRegionsEfficPlane.IN_HOUSE
    #     if req_perf_F >= self.utility_total_flops:
    #         return RooflineRegionsEfficPlane.ABOVE_TOP
    #     # if not self._cache_valid_:
    #     #     self.update_sweet_spots()
    #     ap_comp = rf_attainable_performance(effic_FB, self.utility_total_flops, 1)
    #     ap_mem = rf_attainable_performance(effic_FB, self.utility_total_bytes, 1)
    #     if (req_perf_F > ap_comp) or (req_perf_F > ap_mem):
    #         return RooflineRegionsEfficPlane.ABOVE_100
    #     return RooflineRegionsEfficPlane.IN_HOUSE

    # def get_max_perf_at_effic(self, effic_FB):
    #     if effic_FB < 0.001:
    #         effic_FB = 0.01
    #     ap_comp = rf_attainable_performance(effic_FB, self.utility_total_flops, 1)
    #     ap_mem = rf_attainable_performance(effic_FB, self.utility_total_bytes, 1)
    #     res_ap = min(ap_mem, ap_comp)
    #     return res_ap

    # def get_ap_EfficPlane(self, effic_FB):
    #     return self.get_max_perf_at_effic(effic_FB)


