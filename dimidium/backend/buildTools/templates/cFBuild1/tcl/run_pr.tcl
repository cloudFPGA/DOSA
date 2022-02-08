# *
# * Copyright 2016 -- 2022 IBM Corporation
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *

# ******************************************************************************
# *                            cloudFPGA
# *-----------------------------------------------------------------------------
# * Created : Mar 03 2018
# * Authors : Francois Abel
# * 
# * Description : A Tcl script that creates the TOP level project in so-called
# *   "Project Mode" of the Vivado design flow.
# * 
# * Synopsis : vivado -mode batch -source <this_file> [-notrace]
# *                               [-log     <log_file_name>]
# *                               [-tclargs [script_arguments]]
# *
# * Reference documents:
# *  - UG939 / Lab3 / Scripting the Project Mode.
# *  - UG835 / All  / Vivado Design Suite Tcl Guide. 
# ******************************************************************************

package require cmdline


# Set the Global Settings used by the SHELL Project
#-------------------------------------------------------------------------------
#source xpr_settings.tcl
source xpr_settings_role.tcl

# Set the Local Settings used by this Script
#-------------------------------------------------------------------------------
set dbgLvl_1         1
set dbgLvl_2         2
set dbgLvl_3         3


################################################################################
#                                                                              #
#                        *     *                                               #
#                        **   **    **       *    *    *                       #
#                        * * * *   *  *      *    **   *                       #
#                        *  *  *  *    *     *    * *  *                       #
#                        *     *  ******     *    *  * *                       #
#                        *     *  *    *     *    *   **                       #
#                        *     *  *    *     *    *    *                       #
#                                                                              #
################################################################################

set force 0

#-------------------------------------------------------------------------------
# Parsing of the Command Line
#  Note: All the strings after the '-tclargs' option are considered as TCL
#        arguments to this script and are passed on to TCL 'argc' and 'argv'.
#-------------------------------------------------------------------------------
if { $argc > 0 } {
    my_dbg_trace "There are [ llength $argv ] TCL arguments to this script." ${dbgLvl_1}
    set i 0
    foreach arg $argv {
        my_dbg_trace "  argv\[$i\] = $arg " ${dbgLvl_2}
        set i [ expr { ${i} + 1 } ]
    }
    # Step-1: Processing of '$argv' using the 'cmdline' package
    set options {
        { h     "Display the help information." }
        { force    "Continue, even if an old project will be deleted."}
    }
    set usage "\nUSAGE: Vivado -mode batch -source ${argv0} -notrace -tclargs \[OPTIONS] \nOPTIONS:"
    
    array set kvList [ cmdline::getoptions argv ${options} ${usage} ]
    
    # Process the arguments
    foreach { key value } [ array get kvList ] {
        my_dbg_trace "KEY = ${key} | VALUE = ${value}" ${dbgLvl_2}
        if { ${key} eq "h"  && ${value} eq 1} {
            puts "${usage} \n";
            return ${::OK}
        }
        if { ${key} eq "force" && ${value} eq 1 } { 
          set force 1
          my_dbg_trace "Setting force to \'1\' " ${dbgLvl_1}
        }
    }
}


my_puts "################################################################################"
my_puts "##"
my_puts "##  CREATING PROJECT: ${xprName}  "
my_puts "##"
my_puts "##    Vivado Version is ${VIVADO_VERSION}  "
my_puts "##"
my_puts "################################################################################"
my_puts "Start at: [clock format [clock seconds] -format {%T %a %b %d %Y}] \n"

# Always start from the root directory
#-------------------------------------------------------------------------------
catch { cd ${rootDir} }

#===============================================================================
# Create Xilinx Project
#===============================================================================

#create_project -in_memory -part ${xilPartName} ${xprDir}/${xprName}.log

# Create the Xilinx project
#-------------------------------------------------------------------------------
if { [ file exists ${xprDir} ] != 1 } {
    file mkdir ${xprDir}
}
create_project ${xprName} ${xprDir} -force



my_dbg_trace "Done with create_project." ${dbgLvl_1}

# Set Project Properties
#-------------------------------------------------------------------------------
#set obj [ get_projects ${xprName} ]
set obj [ current_project ]

set_property -name "part"            -value ${xilPartName} -objects ${obj} -verbose
set_property -name "target_language" -value "VHDL"         -objects ${obj} -verbose

set_property -name "default_lib"                -value "xil_defaultlib"       -objects ${obj}
set_property -name "dsa.num_compute_units"      -value "60"                   -objects ${obj}
set_property -name "ip_cache_permissions"       -value "read write"           -objects ${obj}
set_property -name "part"                       -value "xcku060-ffva1156-2-i" -objects ${obj}
set_property -name "simulator_language"         -value "Mixed"                -objects ${obj}
set_property -name "sim.ip.auto_export_scripts" -value "1"                    -objects ${obj}

set_property -name "ip_output_repo"             -value "${xprDir}/${xprName}/${xprName}.cache/ip" -objects ${obj}

if { [format "%.1f" ${VIVADO_VERSION}] == 2017.4 } {
  my_dbg_trace "Enabling the use of deprecated PRAGMAs." ${dbgLvl_2};
  set_property verilog_define {USE_DEPRECATED_DIRECTIVES=true} [ current_fileset ]
  set_property generic        {gVivadoVersion=2017}            [ current_fileset ]
}

my_dbg_trace "Done with set project properties." ${dbgLvl_1}


# Create IP directory and set IP repository paths
#-------------------------------------------------------------------------------
set obj [ get_filesets sources_1 ]
if { [ file exists ${ipDir} ] != 1 } {
    my_puts "Creating a managed IP directory: \'${ipDir}\' "
    file mkdir ${ipDir}
} else {
    my_dbg_trace "Setting ip_repo_paths to ${ipDir}" ${dbgLvl_1}
    set_property ip_repo_paths [ concat ${ipDir} ${hlsDir} ] [current_project]
}

# Rebuild user ip_repo's index before adding any source files
#-------------------------------------------------------------------------------
update_ip_catalog -rebuild

# Create 'sources_1' fileset (if not found)
#-------------------------------------------------------------------------------
if { [ string equal [ get_filesets -quiet sources_1 ] "" ] } {
  create_fileset -srcset sources_1
}

# Set 'sources_1' fileset object an add *ALL* the HDL Source Files from the HLD
#  Directory (Recursively) 
#-------------------------------------------------------------------------------
set obj   [ get_filesets sources_1 ]


#set files [ list "[ file normalize "${hdlDir}" ]" ]
##OBSOLETE  add_files -fileset ${obj} ${hdlDir}
#add_files -fileset ${obj} ${files}
#my_dbg_trace "HDL files: ${files} " ${dbgLvl_2}

# Add HDL Source Files for the ROLE and turn VHDL-2008 mode on
#---------------------------------------------------------------------------
add_files  ${hdlDir}
set roleVhdlList [ glob -nocomplain ${hdlDir}/*.vhd* ]
if { $roleVhdlList ne "" } {
  set_property file_type {VHDL 2008} [ get_files [ file normalize ${hdlDir}/*.vhd* ] ]
}
update_compile_order -fileset sources_1

my_dbg_trace "Done with adding HDL files..." ${dbgLvl_1}


# Add *ALL* the User-based IPs (i.e. VIVADO- as well HLS-based) needed for the ROLE. 
#---------------------------------------------------------------------------
set ipList [ glob -nocomplain ${ipDir}/ip_user_files/ip/* ]
if { $ipList ne "" } {
    foreach ip $ipList {
        set ipName [file tail ${ip} ]
        add_files ${ipDir}/${ipName}/${ipName}.xci
        my_dbg_trace "Done with add_files for ROLE: ${ipDir}/${ipName}/${ipName}.xci" 2
    }
}

update_ip_catalog
my_dbg_trace "Done with update_ip_catalog for the ROLE" ${dbgLvl_1}




# Set 'sources_1' fileset properties
#-------------------------------------------------------------------------------
set obj [ get_filesets sources_1 ]
set_property -name "top"      -value ${topName}           -objects ${obj} -verbose
set_property -name "top_file" -value ${hdlDir}/${topFile} -objects ${obj} -verbose


# Create 'constrs_1' fileset (if not found)
#-------------------------------------------------------------------------------
if { [ string equal [ get_filesets -quiet constrs_1 ] "" ] } {
  create_fileset -constrset constrs_1
}


## set the current synth run
#current_run -synthesis [get_runs synth_1]



my_puts "################################################################################"
my_puts "##  DONE WITH PROJECT CREATION "
my_puts "################################################################################"
my_puts "End at: [clock format [clock seconds] -format {%T %a %b %d %Y}] \n"

my_puts "################################################################################"
my_puts "##"
my_puts "##  RUN SYNTHESIS: ${xprName}  in OOC"
my_puts "##"
my_puts "################################################################################"
my_puts "Start at: [clock format [clock seconds] -format {%T %a %b %d %Y}] \n"

#launch_runs synth_1
#wait_on_run synth_1

#synth ip cores
set ipList [ glob -nocomplain ${ipDir}/ip_user_files/ip/* ]
        if { $ipList ne "" } {
            foreach ip $ipList {
                set ipName [file tail ${ip} ]
                synth_ip [get_files ${ipDir}/${ipName}/${ipName}.xci] -force
                my_dbg_trace "Done with SYNTHESIS of IP Core: ${ipDir}/${ipName}/${ipName}.xci" 2
            }
        }

synth_design -mode out_of_context -top $topName -part ${xilPartName}

#-jobs 8

my_puts "################################################################################"
my_puts "##  DONE WITH SYNTHESIS RUN; WRITE FILES TO .dcp"
my_puts "################################################################################"
my_puts "End at: [clock format [clock seconds] -format {%T %a %b %d %Y}] \n"

write_checkpoint -force ${topName}_OOC.dcp

# Close project
#-------------------------------------------------------------------------------
 close_project




