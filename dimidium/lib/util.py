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

import re


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
        if cmpl == default_to_ignore or cmpl == 0:
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


# based on: https://www.oreilly.com/library/view/python-cookbook/0596001673/ch03s15.html
def multiple_replce(text, repdict):
    # Create a regular expression from all of the dictionary keys
    regex = re.compile("|".join(map(re.escape, repdict.keys())))

    # For each match, look up the corresponding value in the dictionary
    return regex.sub(lambda match: repdict[match.group(0)], text)


# based on: https://stackoverflow.com/questions/65542170/how-to-replace-all-occurence-of-string-in-a-nested-dict
def replace_deep(dicttoreplace, repdict):
    if isinstance(dicttoreplace, str):
        return multiple_replce(dicttoreplace, repdict)
    elif isinstance(dicttoreplace, dict):
        return {multiple_replce(k, repdict): replace_deep(v, repdict) for k, v in dicttoreplace.items()}
    elif isinstance(dicttoreplace, list):
        return [replace_deep(v, repdict) for v in dicttoreplace]
    else:
        # nothing to do?
        return dicttoreplace

