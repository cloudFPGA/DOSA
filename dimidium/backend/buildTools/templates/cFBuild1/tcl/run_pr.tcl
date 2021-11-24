# ******************************************************************************
# *                            cloudFPGA
# *            All rights reserved -- Property of IBM
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
my_puts "################################################################################"
my_puts "Start at: [clock format [clock seconds] -format {%T %a %b %d %Y}] \n"

# Always start from the root directory
#-------------------------------------------------------------------------------
catch { cd ${rootDir} }

# Check if the Xilinx Project Already Exists
#-------------------------------------------------------------------------------
#if { [ file exists ${xprDir}/${xprName}.xpr ] == 1 && ! ${force} } {
#    my_warn_puts "The project \'${xprName}.xpr\' already exists!"
#    my_warn_puts "You are about to delete the project directory: '${xprDir}\' "
#    my_warn_puts "\t Are you sure (Y/N) ? "
#    flush stdout
#    set kbdIn [ gets stdin ]
#    scan ${kbdIn} "%s" keyPressed
#    if { [ string toupper ${keyPressed} ] ne "Y" } {
#        my_puts "OK, go it. This script (\'${argv0}\') will be aborted now."
#        my_puts "Bye.\n" 
#        exit 0
#    }
#}
#
## Clean Previous Xilinx Project Directory
##-------------------------------------------------------------------------------
#file delete -force ${xprDir}
#file mkdir ${xprDir}

# # Check if the Managed IP Project Already Exists
# #-------------------------------------------------------------------------------
# if { ! [ file exists ${ipXprDir}/${ipXprName}.xpr ] } {
#     my_info_puts "The managed IP project \'${ipXprName}.xpr\' does not exist yet!"
#     my_info_puts "Do you want to generate the IP cores used by the SHELL now (Y/N) ? "
#     flush stdout
#     set kbdIn [ gets stdin ]
#     scan ${kbdIn} "%s" keyPressed
#     if { [ string toupper ${keyPressed} ] eq "Y" } {
#         set rc [ source ${tclDir}/create_ip_cores.tcl ]
#         if { ${rc} != ${OK} } {
#             my_puts("")
#             my_warn_puts "Failed to generate the IPs cores used by the SHELL!"
#             my_warn_puts "  (Number IPs that failed to be generated = ${rc})"
#             my_puts("")
#         } else {
#             my_puts "##  Done with IP cores generation."
#         }
#     } else {
#         my_puts "OK, go it. The generation of the IP cores will be skipped."
#         my_puts("")
#    }
# }

#===============================================================================
# Create Xilinx Project
#===============================================================================
#create_project ${xprName} ${xprDir} -part ${xilPartName} 
create_project -in_memory -part ${xilPartName} ${xprDir}/${xprName}.log
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

#set_property -name "ip_output_repo"             -value "${xprDir}/${xprName}/${xprName}.cache/ip" -objects ${obj}

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
set files [ list "[ file normalize "${hdlDir}" ]" ]
#OBSOLETE  add_files -fileset ${obj} ${hdlDir}
add_files -fileset ${obj} ${files}
my_dbg_trace "Done with adding HDL files.." ${dbgLvl_1}

#OBSOLETE # Update Compile Order
#OBSOLETE #-------------------------------------------------------------------------------
#OBSOLETE update_compile_order -fileset sources_1

# Set 'sources_1' fileset file properties for remote files
#-------------------------------------------------------------------------------
# set file "$origin_dir/../hdl/roleFlash.vhdl"
# set file [file normalize $file]
# set file_obj [get_files -of_objects [get_filesets sources_1] [list "*$file"]]
# set_property -name "file_type" -value "VHDL" -objects $file_obj

# Set 'sources_1' fileset file properties for local files
# None

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

# Turn VHDL-2008 mode on 
#-------------------------------------------------------------------------------
set_property file_type {VHDL 2008} [ get_files *.vhd* ]


# Create 'constrs_1' fileset (if not found)
#-------------------------------------------------------------------------------
if { [ string equal [ get_filesets -quiet constrs_1 ] "" ] } {
  create_fileset -constrset constrs_1
}

# Set 'constrs_1' fileset object and Add/Import constrs file and set constraint
#  file properties
#-------------------------------------------------------------------------------
#set obj [ get_filesets constrs_1 ]
#set dir "[ file normalize "${xdcDir}" ]" 
#my_dbg_trace "Set \'constrs_1\': dir   = ${dir}" ${dbgLvl_3}  
#set files [ add_files -fileset ${obj} ${dir} ]
#my_dbg_trace "Set \'constrs_1\': files = ${files}" ${dbgLvl_3}  
#set file_obj [ get_files -of_objects [ get_filesets constrs_1 ] [ list "$dir/*" ] ] 
#my_dbg_trace "Set \'constrs_1\': file_obj = ${file_obj}" ${dbgLvl_3}  
#set_property -name "file_type" -value "XDC" -objects ${file_obj}

#[TODO]set_property used_in_synthesis false [get_files ${xdcDir}/${xprName}_timg.xdc]
#[TODO] set_property used_in_synthesis false [get_files ${xdcDir}/${xprName}_pins.xdc]
#my_dbg_trace "Done with adding XDC files." ${dbgLvl_1}



# Create 'sim_1' fileset (if not found)
#-------------------------------------------------------------------------------
#if {[string equal [get_filesets -quiet sim_1] ""]} {
#  create_fileset -simset sim_1
#}
## Set 'sim_1' fileset object
#set obj [get_filesets sim_1]
## [TODO] Empty (no sources present)
## Set 'sim_1' fileset properties
#set obj [get_filesets sim_1]
#set_property -name "top" -value ${topName} -objects $obj
#

# Create 'synth_1' run (if not found)
#-------------------------------------------------------------------------------
#set year [ lindex [ split [ version -short ] "." ] 0 ]  
#if { [ string equal [ get_runs -quiet synth_1 ] ""] } {
#    create_run -name synth_1 -part ${xilPartName} -flow {Vivado Synthesis ${year}} -strategy "Vivado Synthesis Defaults" -constrset constrs_1
#} else {
#  set_property strategy "Vivado Synthesis Defaults" [ get_runs synth_1 ]
#    set_property flow "Vivado Synthesis ${year}" [ get_runs synth_1 ]
#}
#set obj [ get_runs synth_1 ]
#set_property set_report_strategy_name 1 ${obj}
#set_property report_strategy {Vivado Synthesis Default Reports} ${obj}
#set_property set_report_strategy_name 0 ${obj}
## Create 'synth_1_synth_report_utilization_0' report (if not found)
#if { [ string equal [ get_report_configs -of_objects [get_runs synth_1] synth_1_synth_report_utilization_0] "" ] } {
#  create_report_config -report_name synth_1_synth_report_utilization_0 -report_type report_utilization:1.0 -steps synth_design -runs synth_1
#}
#set obj [get_report_configs -of_objects [get_runs synth_1] synth_1_synth_report_utilization_0]
#if { ${obj} != "" } {
#
#}
#set obj [get_runs synth_1]
#set_property -name "part" -value ${xilPartName} -objects ${obj}
#set_property -name "strategy" -value "Vivado Synthesis Defaults" -objects ${obj}
#
#set_property -name "mode" -value "out_of_context" -objects ${obj}
#
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

# Launch Vivado' GUI
#-------------------------------------------------------------------------------
#catch { cd ${xprDir} }
#start_gui




