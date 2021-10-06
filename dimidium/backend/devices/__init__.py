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

from dimidium.backend.devices.dosa_device import DosaBaseHw
import dimidium.backend.devices.cf_fmku60_role1
import dimidium.backend.devices.vcpu_dummy

cF_FMKU60_Themisto_1 = cf_fmku60_role1.CfThemisto1('fpga', 'cF_FMKU60_Themisto-Role_1')
vCPU_x86 = vcpu_dummy.VcpuDummy('cpu', 'CPU_dummy_x86-1')

types = ['vCPU_x86', 'cF_FMKU60_Themisto_1']

types_dict = {'vCPU_x86': vCPU_x86, 'cF_FMKU60_Themisto_1': cF_FMKU60_Themisto_1}

fallback_hw = ['vCPU_x86']

