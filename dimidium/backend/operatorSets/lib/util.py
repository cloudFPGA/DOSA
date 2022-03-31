#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Mar 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Library for utility functions for OSGs
#  *
#  *
import math

import dimidium.lib.singleton as dosa_singleton


def get_avg_util_dict_bytes_based(entries, consider_paramB=False, consider_ops_num=False,  consider_outB=False):
    lutlog_total = 0
    lutmem_total = 0
    register_total = 0
    bram_total = 0
    dsps_total = 0
    wrapper_lutlog_total = 0
    wrapper_lutmem_total = 0
    wrapper_register_total = 0
    wrapper_bram_total = 0
    wrapper_dsps_total = 0
    bytes_total = 0
    latency_total = 0
    # e_cnt = 0
    ops_cnt = 0
    # last_latency = 0
    input_bytes_for_latency = 0
    for e in entries:
        if consider_paramB:
            bytes_total += e['paramB']
        else:
            bytes_total += e['inpB']
        if consider_outB:
            bytes_total += e['outB']
        if e['latency_lim_per_tensor_cycl'] > 0:
            latency_total += e['latency_lim_per_tensor_cycl']
            input_bytes_for_latency += e['inpB']
            # last_latency = e['latency_lim_per_tensor_cycl']
        # else:
        #     latency_total += last_latency
        #     # e_cnt += len(e['ops'])
        lutlog_total += e['LUTLOG']
        lutmem_total += e['LUTMEM']
        register_total += e['Registers']
        bram_total += e['BRAM']
        dsps_total += e['DSPs']
        wrapper_lutlog_total += e['wrapper']['LUTLOG']
        wrapper_lutmem_total += e['wrapper']['LUTMEM']
        wrapper_register_total += e['wrapper']['Registers']
        wrapper_bram_total += e['wrapper']['BRAM']
        wrapper_dsps_total += e['wrapper']['DSPs']
        ops_cnt += len(e['ops'])
    divider = bytes_total
    if consider_ops_num:
        divider *= ops_cnt
    ret_util_dict = {'LUTLOG': lutlog_total / divider, 'LUTMEM': lutmem_total / divider,
                     'Registers': register_total / divider, 'BRAM': bram_total / divider,
                     'DSPs': dsps_total / divider,
                     # 'latency_lim_per_tensor_cycl': math.ceil(latency_total / ops_cnt),
                     'latency_lim_per_tensor_cycl': latency_total / input_bytes_for_latency,
                     'wrapper': {
                         'LUTLOG': wrapper_lutlog_total / divider,
                         'LUTMEM': wrapper_lutmem_total / divider,
                         'Registers': wrapper_register_total / divider,
                         'BRAM': wrapper_bram_total / divider,
                         'DSPs': wrapper_dsps_total / divider
                     }}
    return ret_util_dict


def get_share_of_FPGA_resources(res_dict, util_dict):
    share_dict = {}
    share_dict['LUTLOG'] = (util_dict['LUTLOG'] / res_dict['LUTLOG']) * dosa_singleton.config.utilization.dosa_mu_comp
    share_dict['LUTMEM'] = (util_dict['LUTMEM'] / res_dict['LUTMEM']) * dosa_singleton.config.utilization.dosa_mu_mem
    share_dict['Registers'] = (util_dict['Registers'] / res_dict['Registers']) * dosa_singleton.config.utilization.dosa_mu_mem
    share_dict['BRAM'] = (util_dict['BRAM'] / res_dict['BRAM']) * dosa_singleton.config.utilization.dosa_mu_mem
    share_dict['DSPs'] = (util_dict['DSPs'] / res_dict['DSPs']) * dosa_singleton.config.utilization.dosa_mu_comp
    # highest_share = 0.0
    # for k in share_dict:
    #     e = share_dict[k]
    #     if e > highest_share:
    #         highest_share = e
    # return share_dict, e
    return share_dict


