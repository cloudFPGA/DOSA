#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Jun 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Utility functions for DOSA/dimidium
#  *
#  *


def convert_oi_list_for_plot(dpl, default_to_ignore=1.0):
    cmpl_list = []
    uinp_list = []
    detail_list = []
    total_flops = 0
    total_uinp_B = 0
    total_param_B = 0
    for l in dpl:
        e = dpl[l]
        cmpl = e['cmpl']
        if cmpl == default_to_ignore:
            continue
        uinp = e['uinp']
        name = e['name']
        layer = e['layer']
        cn = {'name': name + "_" + layer + "_engine", 'oi': cmpl}
        un = {'name': name + "_" + layer + "_stream", 'oi': uinp}
        total_flops += e['flop']
        total_param_B += e['parB']
        total_uinp_B += e['inpB']
        cmpl_list.append(cn)
        uinp_list.append(un)
        detail_list.append(e)
    total = {'flops': total_flops, 'para_B': total_param_B, 'uinp_B': total_uinp_B}
    return cmpl_list, uinp_list, total, detail_list



