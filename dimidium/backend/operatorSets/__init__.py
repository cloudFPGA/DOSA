#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Oct 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Operator Set Generator (OSG) library for DOSA arch gen
#  *
#  *

from dimidium.backend.devices.dosa_device import DosaHwClasses
from dimidium.backend.buildTools.BaseBuild import BaseBuild
from dimidium.backend.operatorSets.BaseOSG import UndecidedOSG, BaseOSG
# from dimidium.backend.operatorSets.relay_ops import init_ops, op


placeholderOSG = UndecidedOSG('OSG_placholder', DosaHwClasses.UNDECIDED, "/none/", BaseBuild('dummy'))

# add all available OSGs here

availableOSGs = []

# # automatic ops init?
# if len(op) < 1:
#     init_ops()

