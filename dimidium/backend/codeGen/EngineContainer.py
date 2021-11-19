#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Nov 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Class to help the generation of Engine blocks
#  *
#  *

import dimidium.lib.singleton as dosa_singleton
from dimidium.middleend.archGen.ArchBlock import ArchBlock


class EngineContainer(object):

    def __init__(self, arch_block):
        self.block_ref = arch_block