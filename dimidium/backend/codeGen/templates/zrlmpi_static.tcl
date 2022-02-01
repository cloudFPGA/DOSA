
#------------------------------------------------------------------------------  
# IBM-HSL-IP : ZRLMPI MPE core
#------------------------------------------------------------------------------
set ipModName "MessagePassingEngine"
set ipName    "mpe_main"
set ipVendor  "IBM"
set ipLibrary "hls"
set ipVersion "1.0"
set ipCfgList  [ list ]

set rc [ my_customize_ip ${ipModName} ${ipDir} ${ipVendor} ${ipLibrary} ${ipName} ${ipVersion} ${ipCfgList} ]

if { ${rc} != ${::OK} } { set nrErrors [ expr { ${nrErrors} + 1 } ] }


#------------------------------------------------------------------------------  
# VIVADO-IP : FIFO Generator
#------------------------------------------------------------------------------
set ipModName "FifoMpiData"
set ipName    "fifo_generator"
set ipVendor  "xilinx.com"
set ipLibrary "ip"
set ipVersion "13.2"
#set ipCfgList [ list CONFIG.Performance_Options {First_Word_Fall_Through} CONFIG.Input_Data_Width {73} CONFIG.Output_Data_Width {73} \
#                CONFIG.Input_Depth {8192} CONFIG.Output_Depth {8192} \
#              ]
#set ipCfgList [ list CONFIG.Performance_Options {First_Word_Fall_Through} CONFIG.Input_Data_Width {73} CONFIG.Output_Data_Width {73} \
#                CONFIG.Input_Depth {16384} CONFIG.Output_Depth {16384} \
#              ]
set ipCfgList [ list CONFIG.Performance_Options {First_Word_Fall_Through} CONFIG.Input_Data_Width {73} CONFIG.Output_Data_Width {73} \
                CONFIG.Input_Depth {2048} CONFIG.Output_Depth {2048} \
              ]

set rc [ my_customize_ip ${ipModName} ${ipDir} ${ipVendor} ${ipLibrary} ${ipName} ${ipVersion} ${ipCfgList} ]

if { ${rc} != ${::OK} } { set nrErrors [ expr { ${nrErrors} + 1 } ] }


#------------------------------------------------------------------------------  
# VIVADO-IP : FIFO Generator
#------------------------------------------------------------------------------
set ipModName "FifoMpiInfo"
set ipName    "fifo_generator"
set ipVendor  "xilinx.com"
set ipLibrary "ip"
set ipVersion "13.2"
#set ipCfgList [ list CONFIG.Performance_Options {First_Word_Fall_Through} CONFIG.Input_Data_Width {72} CONFIG.Output_Data_Width {72} \
#                CONFIG.Input_Depth {512} CONFIG.Output_Depth {512} \
#              ]
#set ipCfgList [ list CONFIG.Performance_Options {First_Word_Fall_Through} CONFIG.Input_Data_Width {72} CONFIG.Output_Data_Width {72} \
#                CONFIG.Input_Depth {1024} CONFIG.Output_Depth {1024} \
#              ]
set ipCfgList [ list CONFIG.Performance_Options {First_Word_Fall_Through} CONFIG.Input_Data_Width {72} CONFIG.Output_Data_Width {72} \
                CONFIG.Input_Depth {16} CONFIG.Output_Depth {16} \
              ]

set rc [ my_customize_ip ${ipModName} ${ipDir} ${ipVendor} ${ipLibrary} ${ipName} ${ipVersion} ${ipCfgList} ]

if { ${rc} != ${::OK} } { set nrErrors [ expr { ${nrErrors} + 1 } ] }


#------------------------------------------------------------------------------  
# VIVADO-IP : FIFO Generator
#------------------------------------------------------------------------------
set ipModName "FifoMpiFeedback"
set ipName    "fifo_generator"
set ipVendor  "xilinx.com"
set ipLibrary "ip"
set ipVersion "13.2"
#set ipCfgList [ list CONFIG.Performance_Options {First_Word_Fall_Through} CONFIG.Input_Data_Width {8} CONFIG.Output_Data_Width {8} \
#                CONFIG.Input_Depth {512} CONFIG.Output_Depth {512} \
#              ]
#set ipCfgList [ list CONFIG.Performance_Options {First_Word_Fall_Through} CONFIG.Input_Data_Width {8} CONFIG.Output_Data_Width {8} \
#                CONFIG.Input_Depth {1024} CONFIG.Output_Depth {1024} \
#              ]
set ipCfgList [ list CONFIG.Performance_Options {First_Word_Fall_Through} CONFIG.Input_Data_Width {8} CONFIG.Output_Data_Width {8} \
                CONFIG.Input_Depth {16} CONFIG.Output_Depth {16} \
              ]

set rc [ my_customize_ip ${ipModName} ${ipDir} ${ipVendor} ${ipLibrary} ${ipName} ${ipVersion} ${ipCfgList} ]

if { ${rc} != ${::OK} } { set nrErrors [ expr { ${nrErrors} + 1 } ] }


