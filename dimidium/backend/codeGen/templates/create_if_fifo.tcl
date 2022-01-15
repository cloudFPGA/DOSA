
#------------------------------------------------------------------------------  
# VIVADO-IP : FIFO Generator
#------------------------------------------------------------------------------
set ipModName "{DOSA_FMSTR_NAME}"
set ipName    "fifo_generator"
set ipVendor  "xilinx.com"
set ipLibrary "ip"
set ipVersion "13.2"
set ipCfgList [ list CONFIG.Performance_Options {{First_Word_Fall_Through}} CONFIG.Input_Data_Width {DOSA_FMSTR_BITWIDTH} CONFIG.Output_Data_Width {DOSA_FMSTR_BITWIDTH} \
                CONFIG.Input_Depth {DOSA_FMSTR_DEPTH} CONFIG.Output_Depth {DOSA_FMSTR_DEPTH} \
              ]

set rc [ my_customize_ip ${{ipModName}} ${{ipDir}} ${{ipVendor}} ${{ipLibrary}} ${{ipName}} ${{ipVersion}} ${{ipCfgList}} ]

if {{ ${{rc}} != ${{::OK}} }} {{ set nrErrors [ expr {{ ${{nrErrors}} + 1 }} ] }}

