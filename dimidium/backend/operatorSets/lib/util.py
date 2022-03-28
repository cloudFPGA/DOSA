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


def get_avg_util_dict_bytes_based(entries, consider_paramB=False):
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
    e_cnt = 0
    for e in entries:
        bytes_total += e['inpB']
        if consider_paramB:
            bytes_total += e['paramB']
        if e['latency_lim_per_tensor_cycl'] > 0:
            latency_total += e['latency_lim_per_tensor_cycl']
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
        e_cnt += len(e['ops'])
    ret_util_dict = {'LUTLOG': lutlog_total / bytes_total, 'LUTMEM': lutmem_total / bytes_total,
                     'Registers': register_total / bytes_total, 'BRAM': bram_total / bytes_total,
                     'DSPs': register_total / bytes_total,
                     'latency_lim_per_tensor_cycl': math.ceil(latency_total / e_cnt),
                     'wrapper': {
                         'LUTLOG': wrapper_lutlog_total / bytes_total,
                         'LUTMEM': wrapper_lutmem_total / bytes_total,
                         'Registers': wrapper_register_total / bytes_total,
                         'BRAM': wrapper_bram_total / bytes_total,
                         'DSPs': wrapper_dsps_total / bytes_total
                     }}
    return ret_util_dict


def get_share_of_FPGA_resources(res_dict, util_dict):
    share_dict = {}
    share_dict['LUTLOG'] = util_dict['LUTLOG'] / res_dict['LUTLOG']
    share_dict['LUTMEM'] = util_dict['LUTMEM'] / res_dict['LUTMEM']
    share_dict['Registers'] = util_dict['Registers'] / res_dict['Registers']
    share_dict['BRAM'] = util_dict['BRAM'] / res_dict['BRAM']
    share_dict['DSPs'] = util_dict['DSPs'] / res_dict['DSPs']
    # highest_share = 0.0
    # for k in share_dict:
    #     e = share_dict[k]
    #     if e > highest_share:
    #         highest_share = e
    # return share_dict, e
    return share_dict


