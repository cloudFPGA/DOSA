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

import dimidium.lib.devices.cf_fmku60_role1 as cF_FMKU60_Themisto_1
import dimidium.lib.devices.vcpu_dummy as vCPU_x86

types = ['vCPU_x86', 'cF_FMKU60_Themisto_1']

types_dict = {'vCPU_x86': vCPU_x86, 'cF_FMKU60_Themisto_1': cF_FMKU60_Themisto_1}

fallback_hw = ['vCPU_x86']

