#  /*******************************************************************************
#   * Copyright 2019 -- 2023 IBM Corporation
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
#  *     Created: Jan 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Base class for DOSA communication tools
#  *
#  *

import abc

from gradatim.backend.devices.dosa_device import DosaHwClasses


class BaseCommLib(metaclass=abc.ABCMeta):

    def __init__(self, name, device_classes: [DosaHwClasses]):
        self.name = name
        self.device_classes = device_classes
        self.dosaHwTypes = []
        self.priority = -1
        self.priority_internal = -1

    def __repr__(self):
        return "CommLib({}, for {})".format(self.name, self.device_classes)

    def select_dosa_hw_types(self, classes_dict):
        self.dosaHwTypes = []
        for hc in classes_dict:
            if hc in self.device_classes:
                new_possible_hc = classes_dict[hc]
                for nhc in new_possible_hc:
                    if nhc not in self.dosaHwTypes:
                        self.dosaHwTypes.append(nhc)

    def init(self, dosa_hw_classes_dict, priority_internal):
        self.priority_internal = priority_internal
        self.select_dosa_hw_types(dosa_hw_classes_dict)

    @abc.abstractmethod
    def build(self, comm_plan, build_tool):
        print("[DOSA:CommLib:ERROR] NOT YET IMPLEMENTED.")


class UndecidedCommLib(BaseCommLib):

    def build(self, comm_plan, build_tool):
        pass


placeholderCommLib = UndecidedCommLib('Placeholder Comm Lib', [DosaHwClasses.UNDECIDED])


def sort_commLib_list(comm_lib_list, use_internal_prio=True):
    comm_libs_by_priority = {}
    for comm_lib in comm_lib_list:
        comm_lib_prio = comm_lib.priority_internal
        if not use_internal_prio:
            comm_lib_prio = comm_lib.priority
        if comm_lib_prio in comm_libs_by_priority.keys():
            comm_libs_by_priority[comm_lib_prio].append(comm_lib)
        else:
            comm_libs_by_priority[comm_lib_prio] = [comm_lib]
    comm_libs_sorted = sorted(comm_libs_by_priority)
    if not use_internal_prio:
        comm_libs_sorted.reverse()  # reverse, since 0 is lowest external prio
    ret_list = []
    for prio in comm_libs_sorted:
        comm_lib_list = comm_libs_by_priority[prio]
        # if len(comm_lib_list) > 1:
        # just use any order?
        ret_list.extend(comm_lib_list)
    return ret_list

