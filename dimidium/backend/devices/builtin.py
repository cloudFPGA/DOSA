#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Oct 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Built in device library for DOSA arch gen
#  *
#  *

from dimidium.backend.devices.dosa_device import DosaHwClasses, DosaBaseHw
from dimidium.backend.devices.cf_fmku60_role1 import CfThemisto1
from dimidium.backend.devices.vcpu_dummy import VcpuDummy

cF_FMKU60_Themisto_1 = CfThemisto1('cF_FMKU60_Themisto-Role_1')
vCPU_x86 = VcpuDummy('CPU_dummy_x86-1')

# TODO: make types and classes extendable

types = [vCPU_x86, cF_FMKU60_Themisto_1]

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
classes_dict[DosaHwClasses.FPGA_generic] = [cF_FMKU60_Themisto_1]
classes_dict[DosaHwClasses.CPU_x86] = [vCPU_x86]
classes_dict[DosaHwClasses.FPGA_xilinx] = [cF_FMKU60_Themisto_1]

# dosa_devices = {'types': types, 'types_str': types_str, 'types_dict': types_dict, 'classes_dict': classes_dict}

