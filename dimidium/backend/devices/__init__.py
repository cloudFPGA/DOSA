#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: July 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Device library for DOSA arch gen
#  *
#  *

from dimidium.backend.devices.dosa_device import DosaHwClasses, DosaBaseHw
import dimidium.backend.devices.cf_fmku60_role1
import dimidium.backend.devices.vcpu_dummy

cF_FMKU60_Themisto_1 = cf_fmku60_role1.CfThemisto1('cF_FMKU60_Themisto-Role_1')
vCPU_x86 = vcpu_dummy.VcpuDummy('CPU_dummy_x86-1')

types = [vCPU_x86, cF_FMKU60_Themisto_1]

# types_str = ['vCPU_x86', 'cF_FMKU60_Themisto_1']
# types_dict = {'vCPU_x86': vCPU_x86, 'cF_FMKU60_Themisto_1': cF_FMKU60_Themisto_1}
types_str = []
types_dict = {}
for e in types:
    types_str.append(e.type_str)
    types_dict[e.type_str] = e

# fallback_hw = ['vCPU_x86']
fallback_hw = [vCPU_x86.type_str]

