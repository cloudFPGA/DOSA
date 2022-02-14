# *****************************************************************************
# *                            cloudFPGA
# *            All rights reserved -- Property of IBM
# *----------------------------------------------------------------------------
# * Created : Jun 2017
# * Authors : Burkhard Ringlein
# * 
# * Description : A Tcl script for the HLS batch syhthesis of the "Castor" SMC
# *   process used by the SHELL of a cloudFPGA module.
# *   project.
# * 
# * Synopsis : vivado_hls -f <this_file>
# *
# *
# * Reference documents:
# *  - UG902 / Ch.4 / High-Level Synthesis Reference Guide.
# *
# *-----------------------------------------------------------------------------
# * Modification History:
# ******************************************************************************

# User defined settings
#-------------------------------------------------
set projectName    "$env(dosaIpName)"
set solutionName   "solution1"
set xilPartName    "xcku060-ffva1156-2-i"

set ipName         ${projectName}
set ipDisplayName  "Wrapper for Haddoc2 Layers"
set ipDescription  "Application for cloudFPGA"
set ipVendor       "IBM"
set ipLibrary      "hls"
set ipVersion      "1.0"
set ipPkgFormat    "ip_catalog"

# Set Project Environment Variables  
#-------------------------------------------------
set currDir      [pwd]
set srcDir       ${currDir}/src
set tbDir        ${currDir}/tb
#set implDir      ${currDir}/${appName}_prj/${solutionName}/impl/ip 
#set repoDir      ${currDir}/../../ip


# Get targets out of env  
#-------------------------------------------------

set hlsSim $env(hlsSim)
set hlsCoSim $env(hlsCoSim)
set useWrapperTest $env(useWrapperTest)

# Open and Setup Project
#-------------------------------------------------
open_project  ${projectName}_prj
#set_top       haddoc_wrapper
set_top       ${projectName}


# library files
add_files ${srcDir}/../../lib/axi_utils.hpp -cflags "-Wno-attributes"
#add_files ${srcDir}/../../lib/interface_utils.hpp

if { $useWrapperTest } {
  add_files   ${srcDir}/haddoc_wrapper.cpp -cflags "-DWRAPPER_TEST -Wno-attributes"
  add_files   ${srcDir}/haddoc_wrapper.hpp -cflags "-DWRAPPER_TEST -Wno-attributes"
  add_files -tb tb/tb_haddoc2_wrapper.cpp -cflags "-DWRAPPER_TEST -Wno-attributes"
} else {
  add_files   ${srcDir}/haddoc_wrapper.cpp -cflags "-Wno-attributes"
  add_files   ${srcDir}/haddoc_wrapper.hpp -cflags "-Wno-attributes"
  add_files -tb tb/tb_haddoc2_wrapper.cpp -cflags "-Wno-attributes"
}


open_solution ${solutionName}

set_part      ${xilPartName}
create_clock -period 6.4 -name default

#catch {config_array_partition -maximum_size 4096}
catch {config_array_partition -maximum_size 4096}


# Run C Simulation and Synthesis
#-------------------------------------------------

if { $hlsSim} { 
  csim_design -compiler gcc -clean
} else {

  csynth_design
  
  if { $hlsCoSim} {
    cosim_design -compiler gcc -trace_level all 
  } else {
  
  # Export RTL
  #-------------------------------------------------
    export_design -rtl vhdl -format ${ipPkgFormat} -library ${ipLibrary} -display_name ${ipDisplayName} -description ${ipDescription} -vendor ${ipVendor} -version ${ipVersion}
  }
}

exit


