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
set ipDisplayName  "TIPS engine"
set ipDescription  "Tensor processor without Interlocked Pipeline Stages for DOSA"
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
set useTipsTest $env(useTipsTest)

# Open and Setup Project
#-------------------------------------------------
open_project  ${projectName}_prj
#set_top       haddoc_wrapper
set_top       ${projectName}


# library files
add_files ${srcDir}/../../lib/axi_utils.hpp -cflags "-Wno-attributes -std=c++0x"
add_files ${srcDir}/../../lib/interface_utils.hpp -cflags "-Wno-attributes -std=c++0x"

if { $useTipsTest } {
  add_files   ${srcDir}/tips.cpp -cflags "-DTIPS_TEST -Wno-attributes -std=c++0x"
  add_files   ${srcDir}/tips.hpp -cflags "-DTIPS_TEST -Wno-attributes -std=c++0x"
  add_files -tb tb/tb_tips.cpp -cflags   "-DTIPS_TEST -Wno-attributes -std=c++0x"
  add_files ${srcDir}/alu.hpp -cflags "-DTIPS_TEST -Wno-attributes -std=c++0x"
  add_files ${srcDir}/alu.cpp -cflags "-DTIPS_TEST -Wno-attributes -std=c++0x"
} else {
  add_files   ${srcDir}/tips.cpp -cflags "-Wno-attributes -std=c++0x"
  add_files   ${srcDir}/tips.hpp -cflags "-Wno-attributes -std=c++0x"
  add_files ${srcDir}/alu.hpp -cflags "-Wno-attributes -std=c++0x"
  add_files ${srcDir}/alu.cpp -cflags "-Wno-attributes -std=c++0x"
  add_files -tb tb/tb_tips.cpp -cflags "-Wno-attributes -std=c++0x"
}



open_solution ${solutionName}

set_part      ${xilPartName}
create_clock -period 6.4 -name default

catch {config_array_partition -maximum_size 4096}
#catch {config_array_partition -maximum_size 16384}


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


