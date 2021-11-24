#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Mar 2020
#  *     Authors: FAB, WEI, NGL
#  *
#  *     Description:
#  *        XPR/TCL PROJECT SPECIFIC settings; 
#  *        Generic file to extract ENVIRONMENT variables
#  *        for the ROLE
#  *

# 1. cFDK settings
source $env(cFpRootDir)/cFDK/SRA/LIB/tcl/xpr_settings.tcl

#-------------------------------------------------------------------------------
# 2. Role and PR specific settings
set currDir      [pwd]
set rootDir      [file dirname [pwd]] 
set hdlDir       ${rootDir}/hdl
set hlsDir       ${rootDir}/hls
set ipDir        ${rootDir}/ip
set simDir       ${rootDir}/sim
set tclDir       ${rootDir}/tcl
set xdcDir       ${rootDir}/xdc
set xprDir       ${rootDir}/xpr

#-------------------------------------------------------------------------------
# Xilinx Project Settings
set xprName      "role${brdPartName}_${cFpSRAtype}"

set topName      ${usedRoleType}
set topFile      "Role.vhdl"


#-- IPs Managed by this Xilinx (Xpr) Project
set ipXprDir     ${ipDir}/managed_ip_project
set ipXprName    "managed_ip_project"
set ipXprFile    [file join ${ipXprDir} ${ipXprName}.xpr ]

#-----------------------------------

