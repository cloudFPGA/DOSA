# *****************************************************************************
# *                            cloudFPGA
# *            All rights reserved -- Property of IBM
# *----------------------------------------------------------------------------
# * Created : Mar 03 2018
# * Authors : Francois Abel
# * 
# * Description : This file is typically sourced from another TCL script in
# *   in order to set the main project variables when working in the Project
# *   Mode of the Vivado design flow.
# * 
# * Synopsis : source <this_file>
# * 
# ******************************************************************************

#-------------------------------------------------------------------------------
# Xilinx Project Settings
#-------------------------------------------------------------------------------
set xprName      "roleFMKU60_Themisto"
set xilPartName  "xcku060-ffva1156-2-i"

set topName      "Role_Themisto"
set topFile      "Role.vhdl"


#-------------------------------------------------------------------------------
# TOP  Project Settings  
#-------------------------------------------------------------------------------
set currDir      [pwd]
set rootDir      [file dirname [pwd]] 
set hdlDir       ${rootDir}/hdl
set hlsDir       ${rootDir}/hls
set ipDir        ${rootDir}/ip
set simDir       ${rootDir}/sim
set tclDir       ${rootDir}/tcl
set xdcDir       ${rootDir}/xdc
set xprDir       ${rootDir}/xpr

#-- IPs Managed by this Xilinx (Xpr) Project
set ipXprDir     ${ipDir}/managed_ip_project
set ipXprName    "managed_ip_project"
set ipXprFile    [file join ${ipXprDir} ${ipXprName}.xpr ]

#-- IPs Managed by the Shell-Role-Architecture (Sra) Project  
set ipSraDir     ${rootDir}/../../IP
set ipSraXprName "managed_ip_project"
set ipSraXprDir  ${ipSraDir}/${ipSraXprName}
set ipSraXprFile [ file join ${ipSraXprDir} ${ipSraXprName}.xpr ]

#-------------------------------------------------------------------------------
# TRACE, DEBUG and OTHER Settings and Variables  
#-------------------------------------------------------------------------------
set OK         0
set KO         1

set gDbgLvl    3

set OS         [string toupper [lindex $tcl_platform(os) 0]]

set BLACK      0 
set RED        1 
set GREEN      2 
set YELLOW     3
set BLUE       4 
set MAGENTA    5 
set CYAN       6 
set WHITE      7


#-------------------------------------------------------------------------------
# proc my_puts { text {fg -1} }
#  A procedure that adds a specific prompt and that changes the color of the 
#  text to display by calling the 'tput utility of Linux. 
#  :param text the text to display as a string.
#  :param fg   the foreground color to set (0=Black, 1=Red, 2=Green, 3=Yellow, 
#                                          4=Blue, 5=Magenta, 6=Cyan, 7=White).
#  :return nothing
#
#  See https://linux.101hacks.com/ps1-examples/prompt-color-using-tput/
#-------------------------------------------------------------------------------
proc my_puts { text {fg -1} } {
    set prompt "<cloudFPGA> "
    if { ${fg} != -1  && ${fg} >= 0 && ${fg} <= 7 } {
        set color ${fg}
    } else {
        set color ${::BLUE}
    }
    if { ${::OS} eq "WINDOWS" } {
        set coloredText "${prompt}${text}"
    } else {
        set coloredText [exec tput setaf ${color}]${prompt}${text}[exec tput sgr0]
    }
    puts ${coloredText}
}

#-------------------------------------------------------------------------------
# proc my_err_puts { text }
#  A procedure to display an error message colored in "RED". 
#  :param text the error message to display as a string.
#  :return nothing
#-------------------------------------------------------------------------------
proc my_err_puts { text } {
    my_puts "ERROR: ${text}" ${::RED} 
}

#-------------------------------------------------------------------------------
# proc my_warn_puts { text }
#  A procedure to display a warning message colored in "MAGENTA". 
#  :param text the warning message to display as a string.
#  :return nothing
#-------------------------------------------------------------------------------
proc my_warn_puts { text } {
    my_puts "WARNING: ${text}" ${::MAGENTA} 
}

#-------------------------------------------------------------------------------
# proc my_info_puts { text }
#  A procedure to display an information message colored in "GREEN". 
#  :param text the info message to display as a string.
#  :return nothing
#-------------------------------------------------------------------------------
proc my_info_puts { text } {
    my_puts "INFO: ${text}" ${::MAGENTA} 
}

#-------------------------------------------------------------------------------
# proc my_dbg_trace { text dbgLvl }
#  A procedure to display a debug message colored as a function of the global 
#  debug level set in this file and the debug level requested by the caller. 
#  :param text   the text to display as a string.
#  :param dbgLvl the current debug level used by the caller.
#  :return nothing
#-------------------------------------------------------------------------------
proc my_dbg_trace { text dbgLvl } {
    if { ${::gDbgLvl} == 0 } { 
        return 
    } else {
        switch ${dbgLvl} {
            1 { if  { ${::gDbgLvl} >= 1 } { my_puts "<D1> ${text}" ${::CYAN}   } }
            2 { if  { ${::gDbgLvl} >= 2 } { my_puts "<D2> ${text}" ${::GREEN}  } }
            3 { if  { ${::gDbgLvl} >= 3 } { my_puts "<D3> ${text}" ${::YELLOW} } }
            default { my_puts "<UNKNOWN DEBUG LEVEL> ${text}"                  }
        }
    }
}





