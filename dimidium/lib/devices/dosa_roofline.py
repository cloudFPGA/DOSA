#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Aug 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Base Roofline class for DOSA devices
#  *
#  *

import numpy as np
from enum import Enum

from dimidium.lib.util import rf_attainable_performance, BrickImplTypes, rf_calc_sweet_spot
import dimidium.lib.units as units

__deactivated_bw_value__ = -1


class RooflineRegions(Enum):
    IN_HOUSE = 0
    ABOVE_TOP = 1
    ABOVE_NETWORK = 2
    ABOVE_DRAM = 3
    ABOVE_BRAM = 4


class DosaRoofline(object):

    # operational intensity vector
    oi_list_small = np.arange(0.01, 1, 0.01)
    oi_list_middle = np.arange(1, 1500, 0.1)
    oi_list_big = np.arange(1501, 10100, 1)
    oi_list = np.concatenate((oi_list_small, oi_list_middle, oi_list_big))

    # Perf boundaries
    ylim_min = 0.01 * units.gigaU
    ylim_max = 100000 * units.gigaU

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

    def get_region(self, oi_FB, req_perf_F) -> RooflineRegions:
        # TODO: do not assume DRAM BW is always higher than network BW?
        if req_perf_F >= self.roof_F:
            return RooflineRegions.ABOVE_TOP
        # if not self._cache_valid_:
        #     self.update_sweet_spots()
        ap_net = rf_attainable_performance(oi_FB, self.roof_F, self.net_bw_B)
        ap_dram = rf_attainable_performance(oi_FB, self.roof_F, self.dram_bw_B)
        if self.bram_bw_B != __deactivated_bw_value__:
            ap_bram = rf_attainable_performance(oi_FB, self.roof_F, self.bram_bw_B)
            if req_perf_F >= ap_bram:
                return RooflineRegions.ABOVE_BRAM
        if req_perf_F > ap_dram:
            return RooflineRegions.ABOVE_DRAM
        if req_perf_F > ap_net:
            return RooflineRegions.ABOVE_NETWORK
        return RooflineRegions.IN_HOUSE

    def get_ap(self, oi_FB, consider_type=BrickImplTypes.ENGINE):
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
            self.roof_F = perf_dict['dsl48_gflops'] * units.gigaU
        else:
            self.roof_F = perf_dict['cpu_gflops'] * units.gigaU
        self.net_bw_B = perf_dict['bw_netw_gBs'] * units.gigaU
        self.dram_bw_B = perf_dict['bw_dram_gBs'] * units.gigaU
        if 'bw_bram_gBs' in perf_dict:
            self.bram_bw_B = perf_dict['bw_bram_gBs'] * units.gigaU
        else:
            self.bram_bw_B = __deactivated_bw_value__
        self._cache_valid_ = False

    def update_sweet_spots(self):
        self.sp_net = rf_calc_sweet_spot(self.oi_list, self.roof_F, self.net_bw_B)
        self.sp_dram = rf_calc_sweet_spot(self.oi_list, self.roof_F, self.dram_bw_B)
        if self.bram_bw_B != __deactivated_bw_value__:
            self.sp_bram = rf_calc_sweet_spot(self.oi_list, self.roof_F, self.bram_bw_B)
        self._cache_valid_ = True

