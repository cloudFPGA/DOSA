#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Jan 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Library with communication frameworks for DOSA code generation
#  *
#  *

from dimidium.backend.commLibs.ZrlmpiCommLib import ZrlmpiCommLib

# add all available CommLibs here

comm_lib_zrlmpi = ZrlmpiCommLib()

builtin_comm_libs = [comm_lib_zrlmpi]

