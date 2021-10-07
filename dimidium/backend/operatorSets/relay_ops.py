#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Oct 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Collection of relay op calls to be used by OSGs
#  *
#  *

import tvm.relay as relay
from inspect import getmembers, isfunction
import copy

# relay ops: https://github.com/apache/tvm/blob/main/python/tvm/relay/op/__init__.py
# structure is similar

osg_not_available_key = "[DOSA:OSG:ERROR] OPERATION NOT AVAILABLE."

op = {}


# better in function (?)
def init_ops(debug=False):
    tup_list = []
    # from .reduce import *
    tup_list.extend(getmembers(relay.op.reduce, isfunction))
    # from .tensor import *
    tup_list.extend(getmembers(relay.op.tensor, isfunction))
    # from .transform import *
    tup_list.extend(getmembers(relay.op.transform, isfunction))
    # from .algorithm import *
    tup_list.extend(getmembers(relay.op.algorithm, isfunction))
    for tup in tup_list:
        f_name = tup[0]
        if debug:
            if f_name in op:
                print("[DOSA:OSG:WARNING] relay op {} already in op dictionary.".format(f_name))
        op[f_name] = osg_not_available_key

    # from . import vm
    tup_list = getmembers(relay.op.vm, isfunction)
    tmp_dict = {}
    for tup in tup_list:
        f_name = tup[0]
        if debug:
            if f_name in tmp_dict:
                print("[DOSA:OSG:WARNING] relay op {} already in tmp_dict.".format(f_name))
        tmp_dict[f_name] = osg_not_available_key
    op['vm'] = tmp_dict
    # from . import nn
    tup_list = getmembers(relay.op.nn, isfunction)
    tmp_dict = {}
    for tup in tup_list:
        f_name = tup[0]
        if debug:
            if f_name in tmp_dict:
                print("[DOSA:OSG:WARNING] relay op {} already in tmp_dict.".format(f_name))
        tmp_dict[f_name] = osg_not_available_key
    op['nn'] = tmp_dict
    # # from . import annotation
    # tup_list = getmembers(relay.op.annotation, isfunction)
    # tmp_dict = {}
    # for tup in tup_list:
    #     f_name = tup[0]
    #     if debug:
    #         if f_name in tmp_dict:
    #             print("[DOSA:OSG:WARNING] relay op {} already in tmp_dict.".format(f_name))
    #     tmp_dict[f_name] = osg_not_available_key
    # op['annotation'] = tmp_dict
    # from . import memory
    tup_list = getmembers(relay.op.memory, isfunction)
    tmp_dict = {}
    for tup in tup_list:
        f_name = tup[0]
        if debug:
            if f_name in tmp_dict:
                print("[DOSA:OSG:WARNING] relay op {} already in tmp_dict.".format(f_name))
        tmp_dict[f_name] = osg_not_available_key
    op['memory'] = tmp_dict
    # from . import image
    tup_list = getmembers(relay.op.image, isfunction)
    tmp_dict = {}
    for tup in tup_list:
        f_name = tup[0]
        if debug:
            if f_name in tmp_dict:
                print("[DOSA:OSG:WARNING] relay op {} already in tmp_dict.".format(f_name))
        tmp_dict[f_name] = osg_not_available_key
    op['image'] = tmp_dict
    # from . import vision
    tup_list = getmembers(relay.op.vision, isfunction)
    tmp_dict = {}
    for tup in tup_list:
        f_name = tup[0]
        if debug:
            if f_name in tmp_dict:
                print("[DOSA:OSG:WARNING] relay op {} already in tmp_dict.".format(f_name))
        tmp_dict[f_name] = osg_not_available_key
    op['vision'] = tmp_dict
    # # from . import op_attrs
    # tup_list = getmembers(relay.op.op_attrs, isfunction)
    # tmp_dict = {}
    # for tup in tup_list:
    #     f_name = tup[0]
    #     if debug:
    #         if f_name in tmp_dict:
    #             print("[DOSA:OSG:WARNING] relay op {} already in tmp_dict.".format(f_name))
    #     tmp_dict[f_name] = osg_not_available_key
    # op['op_attrs'] = tmp_dict
    # from . import random
    tup_list = getmembers(relay.op.random, isfunction)
    tmp_dict = {}
    for tup in tup_list:
        f_name = tup[0]
        if debug:
            if f_name in tmp_dict:
                print("[DOSA:OSG:WARNING] relay op {} already in tmp_dict.".format(f_name))
        tmp_dict[f_name] = osg_not_available_key
    op['random'] = tmp_dict


def get_op_dict_copy():
    return copy.deepcopy(op)


# automatic ops init?
if len(op) < 1:
    init_ops()

