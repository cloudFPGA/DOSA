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
#  *     Created: Oct 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Built in device library for DOSA arch gen
#  *
#  *

from gradatim.backend.devices.dosa_device import DosaHwClasses, DosaBaseHw
from gradatim.backend.devices.cf_fmku60_role1 import CfThemisto1
from gradatim.backend.devices.cF_infinity import CfPseudoFPGA
from gradatim.backend.devices.vcpu_dummy import VcpuDummy

cF_FMKU60_Themisto_1 = CfThemisto1('cF_FMKU60_Themisto-Role_1')
cF_Infinity_1 = CfPseudoFPGA('cF_Infinity_FPGA')
vCPU_x86 = VcpuDummy('CPU_dummy_x86-1')

# TODO: make types and classes extendable

types = [vCPU_x86, cF_FMKU60_Themisto_1, cF_Infinity_1]

# types_str = ['vCPU_x86', 'cF_FMKU60_Themisto_1']
# types_dict = {'vCPU_x86': vCPU_x86, 'cF_FMKU60_Themisto_1': cF_FMKU60_Themisto_1}
types_str = []
types_dict = {}
for e in types:
    types_str.append(e.type_str)
    types_dict[e.type_str] = e

# fallback_hw = ['vCPU_x86']
# TODO: not necessary?
# fallback_hw = [vCPU_x86.type_str]

classes_dict = {}
classes_dict[DosaHwClasses.UNDECIDED] = []
classes_dict[DosaHwClasses.CPU_generic] = [vCPU_x86]
classes_dict[DosaHwClasses.FPGA_generic] = [cF_FMKU60_Themisto_1, cF_Infinity_1]
classes_dict[DosaHwClasses.CPU_x86] = [vCPU_x86]
classes_dict[DosaHwClasses.FPGA_xilinx] = [cF_FMKU60_Themisto_1, cF_Infinity_1]

# dosa_devices = {'types': types, 'types_str': types_str, 'types_dict': types_dict, 'classes_dict': classes_dict}

