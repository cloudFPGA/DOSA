#  /*******************************************************************************
#   * Copyright 2019 -- 2024 IBM Corporation
#   *
#   * Licensed under the Apache License, Version 2.0 (the "License");
#   * you may not use this file except in compliance with the License.
#   * You may obtain a copy of the License at
#   *
#   *     http://www.apache.org/licenses/LICENSE-2.0
#   *
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS,
#   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   * See the License for the specific language governing permissions and
#   * limitations under the License.
#  *******************************************************************************/
#

#  *
#  *                       cloudFPGA
#  *    =============================================
#  *     Created: Oct 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Base class for DOSA Operator Set Generators (OSGs)
#  *
#  *

import sys
import abc
from collections import Counter


from gradatim.backend.devices.dosa_device import DosaHwClasses
from gradatim.backend.buildTools.BaseBuild import BaseBuild
from gradatim.lib.util import deep_update, BrickImplTypes
from gradatim.lib.dosa_dtype import DosaDtype

# to init relay_ops
import gradatim.backend.operatorSets.relay_ops as relay_ops
from gradatim.middleend.archGen.BrickContract import BrickContract


__default_osg_dict_value__ = (False, None)


class BaseOSG(metaclass=abc.ABCMeta):
    # _pseudo_infinity_ = int(sys.maxsize/1000)
    _pseudo_infinity_ = 65535

    def __init__(self, name, device_classes: [DosaHwClasses], supported_dtypes: [DosaDtype],
                 impl_types: [BrickImplTypes]):
        self.name = name
        self.device_classes = device_classes
        self.possible_impl_types = impl_types
        self.supported_dtypes = supported_dtypes
        # self.relay2osg = relay_ops.op
        # self.relay2osg = {}
        # init with all False
        # self.relay2osg = {x: False for x in relay_ops.op}
        self.relay2osg = deep_update(relay_ops.get_op_dict_copy(), __default_osg_dict_value__)
        self.dosaHwTypes = []
        self.priority = -1
        self.priority_internal = -1
        self.suggested_max_block_length = 10
        self.pipeline_tensor_store = 0
        self.supports_op_padding = False

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

    def annotate_brick(self, brick_node, target_hw, filter_impl_types=None, return_instead_annotate=False):
        supported_complete = True
        contr_list = [[]]
        if target_hw.hw_class in self.device_classes and \
                brick_node.used_dtype in self.supported_dtypes:
            for impl_type in self.possible_impl_types:
                if filter_impl_types is not None and impl_type != filter_impl_types:
                    continue
                for op in brick_node.local_op_iter_gen():
                    op_c = self.annotate_op(op, target_hw, impl_type, dont_annotate=return_instead_annotate)
                    if op_c is not None:
                        if isinstance(op_c, list):
                            # contr_list.extend(op_c)
                            for i in range(0, len(op_c)):
                                if i >= len(contr_list):
                                    contr_list.append([])
                                contr_list[i].append(op_c[i])
                        else:
                            contr_list[0].append(op_c)
                    else:
                        supported_complete = False
                if supported_complete:
                    all_length = len(contr_list[0])
                    for cl in contr_list:
                        if len(cl) < all_length:
                            # TODO: instead: make combinations?
                            #  but then: maybe some options are not compatible with each other?
                            continue
                        brick_contr = BrickContract(brick_node, target_hw, self, impl_type, cl)
                        if return_instead_annotate:
                            return brick_contr
                        brick_node.add_possible_contract(brick_contr)
        return None

    def annotate_op(self, op, target_hw, impl_type, dont_annotate=False):
        """checks if the given relay op is supported by this OSG, creates a contract and returns it"""
        is_supported, op_info = self.get_op_info(op.op_call)
        if not is_supported:
            return None
        osg_func, get_contr_func = op_info
        if (not callable(osg_func)) and (not isinstance(osg_func, bool) or (not osg_func)):
            return None
        list_of_contr = get_contr_func(op, target_hw, impl_type)
        if list_of_contr is not None and not dont_annotate:
            if isinstance(list_of_contr, list):
                for poc in list_of_contr:
                    op.add_possible_contract(poc)
            else:
                op.add_possible_contract(list_of_contr)
        return list_of_contr

    def get_op_info(self, op_str):
        """checks if the given relay op is supported by this OSG and returns it's context"""
        op_str_list = op_str.split('.')
        if len(op_str_list) == 1:
            if op_str_list[0] not in relay_ops.op:
                print("[DOSA:OSG:ERROR] {} is not a valid relay op.".format(op_str_list))
                return False, None
            if op_str_list[0] in self.relay2osg:
                return True, self.relay2osg[op_str_list[0]]
            return False, None
        elif len(op_str_list) == 2:
            if op_str_list[0] not in relay_ops.op:
                print("[DOSA:OSG:ERROR] {} is not a valid relay op.".format(op_str_list))
                return False, None
            if op_str_list[1] not in relay_ops.op[op_str_list[0]]:
                print("[DOSA:OSG:ERROR] {} is not a valid relay op.".format(op_str_list))
                return False, None
            if op_str_list[0] in self.relay2osg:
                if op_str_list[1] in self.relay2osg[op_str_list[0]]:
                    return True, self.relay2osg[op_str_list[0]][op_str_list[1]]
            return False, None
        else:
            print("[DOSA:OSG:ERROR] {} is not a valid relay op.".format(op_str_list))
            return False, None

    def get_costs_of_contract_extension(self, contract, add_brick, target_hw):
        """assumes a contract exists and states what it costs additionally to extend it and the new performance"""
        assert contract.osg == self
        if contract.impl_type == BrickImplTypes.STREAM:
            # it costs linearly?
            new_brick_contr = self.annotate_brick(add_brick, target_hw, return_instead_annotate=True)
            if new_brick_contr is None:
                return -1, -1, -1
            return new_brick_contr.comp_util_share, new_brick_contr.mem_util_share, \
                   min(contract.iter_hz, new_brick_contr.iter_hz)
        elif contract.impl_type == BrickImplTypes.ENGINE:
            return self._get_dyn_costs(contract, add_brick, target_hw)
        return -1, -1, -1

    def _get_dyn_costs(self, contract, add_brick, target_hw):
        """will be overwritten by engine OSGs"""
        print("[DOSA:OSG:ERROR] SHOULD NOT BE REACHED.")
        return -1, -1, -1

    def _get_osg_func(self, op_call):
        if 'nn.' in op_call[0:3]:
            func, _ = self.relay2osg['nn'][op_call[3:]]
        else:
            func, _ = self.relay2osg[op_call]
        return func

    def _get_contr_func(self, op_call):
        if 'nn.' in op_call[0:3]:
            _, func = self.relay2osg['nn'][op_call[3:]]
        else:
            _, func = self.relay2osg[op_call]
        return func

    @abc.abstractmethod
    def build_block(self, arch_block, build_tool, selected_contracts):
        print("[DOSA:OSG:ERROR] NOT YET IMPLEMENTED.")

    @abc.abstractmethod
    def build_container(self, container, build_tool, selected_contracts):
        print("[DOSA:OSG:ERROR] NOT YET IMPLEMENTED.")

    def get_ir_coverage(self):
        # from collections import Counter
        # Counter(d.values())
        # from itertools import chain
        # counter_dict = Counter(chain(*d.values()))
        # for mixed nested dictionaries, above is not working
        # tmp_str = str(self.relay2osg)
        # not_covered = tmp_str.count("(False, None)")
        total_entries = 0
        not_covered = 0
        for v in self.relay2osg.values():
            if type(v) == dict:
                total_entries += len(v)
                tmp_cnt = Counter(v.values())
                not_covered += tmp_cnt[__default_osg_dict_value__]
            else:
                total_entries += 1
                if v == __default_osg_dict_value__:
                    not_covered += 1
        ret = {'osg': self.name, 'total_entries': total_entries, 'not_covered': not_covered,
               'coverage': float(1-(not_covered/total_entries))}
        return ret


class UndecidedOSG(BaseOSG):
    def init(self, dosa_hw_classes_dict, priority_internal):
        # self.select_dosa_hw_types(dosa_hw_classes_dict)
        # should not be initialized
        pass

    def build_block(self, arch_block, build_tool, selected_contracts):
        pass

    def build_container(self, container, build_tool, selected_contracts):
        pass

    # def generate_brick(self, brick_node):
    #     pass

    # def generate_bricks(self, brick_nodes):
    #     pass

    # def comm_wrap_brick(self, todo):
    #     pass

    # def estimate_flops_brick(self, brick_node):
    #     pass


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


def filter_osg_list(osg_list, osg_allowlist):
    filtered_list = []
    for o in osg_list:
        removed = True
        for wn in osg_allowlist:
            if wn in o.name:
                filtered_list.append(o)
                removed = False
        if removed:
            print(f'[DOSA:OSG:INFO] Removing OSG "{o.name}" due to user constraint.')
    return filtered_list


def get_coverege_multiple_osgs(osg_list):
    merge_ops = deep_update(relay_ops.get_op_dict_copy(), 'not_covered')
    combined_names = ''
    for o in osg_list:
        combined_names += o.name
        combined_names += '_'
        for k in o.relay2osg:
            v = o.relay2osg[k]
            if type(v) == dict:
                for kk in o.relay2osg[k]:
                    vv = o.relay2osg[k][kk]
                    if vv != __default_osg_dict_value__:
                        merge_ops[k][kk] = 'covered'
            else:
                if v != __default_osg_dict_value__:
                    merge_ops[k] = 'covered'
    total_entries = 0
    not_covered = 0
    for v in merge_ops.values():
        if type(v) == dict:
            total_entries += len(v)
            tmp_cnt = Counter(v.values())
            not_covered += tmp_cnt['not_covered']
        else:
            total_entries += 1
            if v == 'not_covered':
                not_covered += 1

    ret = {'osg': combined_names[:-1], 'total_entries': total_entries, 'not_covered': not_covered,
           'coverage': float(1 - (not_covered / total_entries))}
    return ret

