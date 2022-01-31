#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Oct 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Base class for DOSA Operator Set Generators (OSGs)
#  *
#  *

import abc

from dimidium.backend.devices.dosa_device import DosaHwClasses
from dimidium.backend.buildTools.BaseBuild import BaseBuild
from dimidium.lib.util import deep_update, BrickImplTypes

# to init relay_ops
import dimidium.backend.operatorSets.relay_ops as relay_ops


class BaseOSG(metaclass=abc.ABCMeta):

    def __init__(self, name, device_classes: [DosaHwClasses], framework_path, impl_types: [BrickImplTypes]):
        self.name = name
        self.device_classes = device_classes
        self.framework_path = framework_path
        self.possible_impl_types = impl_types
        self.relay2osg = relay_ops.op
        # self.relay2osg = {}
        # init with all False
        # self.relay2osg = {x: False for x in relay_ops.op}
        self.relay2osg = deep_update(relay_ops.get_op_dict_copy(), False)
        self.dosaHwTypes = []
        self.priority = -1
        self.priority_internal = -1
        self.suggested_max_block_length = 10

    def __repr__(self):
        return "OSG({}, for {})".format(self.name, self.device_classes)

    def select_dosa_hw_types(self, classes_dict):
        self.dosaHwTypes = []
        for hc in classes_dict:
            if hc in self.device_classes:
                new_possible_hc = classes_dict[hc]
                for nhc in new_possible_hc:
                    if nhc not in self.dosaHwTypes:
                        self.dosaHwTypes.append(nhc)

    @abc.abstractmethod
    def init(self, dosa_hw_classes_dict, priority_internal):
        print("[DOSA:OSG:ERROR] NOT YET IMPLEMENTED.")

    def annotate_brick(self, brick_node):
        supported_once = False
        for op in brick_node.local_op_iter_gen():
            op_supported = self.check_op(op.op_call)
            if op_supported:
                supported_once = True
                op.add_possible_osg(self)
        if supported_once:
            brick_node.add_available_osg(self)

    def check_op(self, op_str):
        """checks if the given relay op is supported by this OSG and returns a boolean"""
        op_str_list = op_str.split('.')
        if len(op_str_list) == 1:
            if op_str_list[0] not in relay_ops.op:
                print("[DOSA:OSG:ERROR] {} is not a valid relay op.".format(op_str_list))
                return False
            if op_str_list[0] in self.relay2osg:
                if callable(self.relay2osg[op_str_list[0]]):
                    return True
                else:
                    return self.relay2osg[op_str_list[0]]
            return False
        elif len(op_str_list) == 2:
            if op_str_list[0] not in relay_ops.op:
                print("[DOSA:OSG:ERROR] {} is not a valid relay op.".format(op_str_list))
                return False
            if op_str_list[1] not in relay_ops.op[op_str_list[0]]:
                print("[DOSA:OSG:ERROR] {} is not a valid relay op.".format(op_str_list))
                return False
            if op_str_list[0] in self.relay2osg:
                if op_str_list[1] in self.relay2osg[op_str_list[0]]:
                    if callable(self.relay2osg[op_str_list[0]][op_str_list[1]]):
                        return True
                    else:
                        return self.relay2osg[op_str_list[0]][op_str_list[1]]
            return False
        else:
            print("[DOSA:OSG:ERROR] {} is not a valid relay op.".format(op_str_list))
            return False

    def _get_osg_func(self, op_call):
        if 'nn.' in op_call[0:3]:
            func = self.relay2osg['nn'][op_call[3:]]
        else:
            func = self.relay2osg[op_call]
        return func

    @abc.abstractmethod
    def build_block(self, arch_block, build_tool):
        print("[DOSA:OSG:ERROR] NOT YET IMPLEMENTED.")

    @abc.abstractmethod
    def build_container(self, container, build_tool):
        print("[DOSA:OSG:ERROR] NOT YET IMPLEMENTED.")

    # @abc.abstractmethod
    # def generate_brick(self, brick_node):
    #     print("[DOSA:OSG:ERROR] NOT YET IMPLEMENTED.")

    # @abc.abstractmethod
    # def generate_bricks(self, brick_nodes):
    #     print("[DOSA:OSG:ERROR] NOT YET IMPLEMENTED.")

    # @abc.abstractmethod
    # def comm_wrap_brick(self, todo):
    #     print("[DOSA:OSG:ERROR] NOT YET IMPLEMENTED.")

    @abc.abstractmethod
    def estimate_flops_brick(self, brick_node):
        print("[DOSA:OSG:ERROR] NOT YET IMPLEMENTED.")


class UndecidedOSG(BaseOSG):
    def init(self, dosa_hw_classes_dict, priority_internal):
        # self.select_dosa_hw_types(dosa_hw_classes_dict)
        # should not be initialized
        pass

    def build_block(self, arch_block):
        pass

    def build_container(self, container):
        pass

    def generate_brick(self, brick_node):
        pass

    def generate_bricks(self, brick_nodes):
        pass

    def comm_wrap_brick(self, todo):
        pass

    def estimate_flops_brick(self, brick_node):
        pass


placeholderOSG = UndecidedOSG('OSG_placholder', [DosaHwClasses.UNDECIDED], "/none/", [])


def sort_osg_list(osg_list, use_internal_prio=True):
    osgs_by_priority = {}
    for osg in osg_list:
        osg_prio = osg.priority_internal
        if not use_internal_prio:
            osg_prio = osg.priority
        if osg_prio in osgs_by_priority.keys():
            osgs_by_priority[osg_prio].append(osg)
        else:
            osgs_by_priority[osg_prio] = [osg]
    osgs_sorted = sorted(osgs_by_priority)
    if not use_internal_prio:
        osgs_sorted.reverse()  # reverse, since 0 is lowest external prio
    ret_list = []
    for prio in osgs_sorted:
        osg_list = osgs_by_priority[prio]
        # if len(osg_list) > 1:
        # just use any order?
        ret_list.extend(osg_list)
    return ret_list

