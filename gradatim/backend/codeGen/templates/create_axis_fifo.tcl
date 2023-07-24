
#------------------------------------------------------------------------------  
# VIVADO-IP : FIFO Generator
#------------------------------------------------------------------------------
# TODO: maybe use different clock domains? So that the Role could be faster?

set ipModName "{DOSA_FMSTR_NAME}_tdata"
set ipName    "fifo_generator"
set ipVendor  "xilinx.com"
set ipLibrary "ip"
set ipVersion "13.2"
set ipCfgList [ list CONFIG.Performance_Options {{First_Word_Fall_Through}} CONFIG.Input_Data_Width {DOSA_FMSTR_BITWIDTH_TDATA} CONFIG.Output_Data_Width {DOSA_FMSTR_BITWIDTH_TDATA} \
                CONFIG.Input_Depth {DOSA_FMSTR_DEPTH} CONFIG.Output_Depth {DOSA_FMSTR_DEPTH} \
              ]

set rc [ my_customize_ip ${{ipModName}} ${{ipDir}} ${{ipVendor}} ${{ipLibrary}} ${{ipName}} ${{ipVersion}} ${{ipCfgList}} ]

if {{ ${{rc}} != ${{::OK}} }} {{ set nrErrors [ expr {{ ${{nrErrors}} + 1 }} ] }}


#------------------------------------------------------------------------------  
# VIVADO-IP : FIFO Generator
#------------------------------------------------------------------------------
# TODO: maybe use different clock domains? So that the Role could be faster?

set ipModName "{DOSA_FMSTR_NAME}_tkeep"
set ipName    "fifo_generator"
set ipVendor  "xilinx.com"
set ipLibrary "ip"
set ipVersion "13.2"
set ipCfgList [ list CONFIG.Performance_Options {{First_Word_Fall_Through}} CONFIG.Input_Data_Width {DOSA_FMSTR_BITWIDTH_TKEEP} CONFIG.Output_Data_Width {DOSA_FMSTR_BITWIDTH_TKEEP} \
                CONFIG.Input_Depth {DOSA_FMSTR_DEPTH} CONFIG.Output_Depth {DOSA_FMSTR_DEPTH} \
              ]

set rc [ my_customize_ip ${{ipModName}} ${{ipDir}} ${{ipVendor}} ${{ipLibrary}} ${{ipName}} ${{ipVersion}} ${{ipCfgList}} ]

if {{ ${{rc}} != ${{::OK}} }} {{ set nrErrors [ expr {{ ${{nrErrors}} + 1 }} ] }}


#------------------------------------------------------------------------------  
# VIVADO-IP : FIFO Generator
#------------------------------------------------------------------------------
# TODO: maybe use different clock domains? So that the Role could be faster?

set ipModName "{DOSA_FMSTR_NAME}_tlast"
set ipName    "fifo_generator"
set ipVendor  "xilinx.com"
set ipLibrary "ip"
set ipVersion "13.2"
set ipCfgList [ list CONFIG.Performance_Options {{First_Word_Fall_Through}} CONFIG.Input_Data_Width {DOSA_FMSTR_BITWIDTH_TLAST} CONFIG.Output_Data_Width {DOSA_FMSTR_BITWIDTH_TLAST} \
                CONFIG.Input_Depth {DOSA_FMSTR_DEPTH} CONFIG.Output_Depth {DOSA_FMSTR_DEPTH} \
              ]

set rc [ my_customize_ip ${{ipModName}} ${{ipDir}} ${{ipVendor}} ${{ipLibrary}} ${{ipName}} ${{ipVersion}} ${{ipCfgList}} ]

if {{ ${{rc}} != ${{::OK}} }} {{ set nrErrors [ expr {{ ${{nrErrors}} + 1 }} ] }}



