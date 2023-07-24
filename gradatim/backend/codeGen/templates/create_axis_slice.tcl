
#------------------------------------------------------------------------------  
# VIVADO-IP : AXI Register Slice
#------------------------------------------------------------------------------
#  Signal Properties
#    [Yes] : Enable TREADY
#    [2]   : TDATA Width (bytes)
#    [No]  : Enable TSTRB
#    [No] : Enable TKEEP
#    [No] : Enable TLAST
#    [0]   : TID Width (bits)
#    [0]   : TDEST Width (bits)
#    [0]   : TUSER Width (bits)
#    [No]  : Enable ACLKEN
#------------------------------------------------------------------------------
set ipModName "{DOSA_FMSTR_NAME}"
set ipName    "axis_register_slice"
set ipVendor  "xilinx.com"
set ipLibrary "ip"
set ipVersion "1.1"
set ipCfgList  [ list CONFIG.TDATA_NUM_BYTES {DOSA_FMSTR_NUM_BYTES} \
                      CONFIG.HAS_TKEEP {DOSA_FMSTR_TKEEP_YES} \
                      CONFIG.HAS_TLAST {DOSA_FMSTR_TLAST_YES} ]

set rc [ my_customize_ip ${{ipModName}} ${{ipDir}} ${{ipVendor}} ${{ipLibrary}} ${{ipName}} ${{ipVersion}} ${{ipCfgList}} ]

if {{ ${{rc}} != ${{::OK}} }} {{ set nrErrors [ expr {{ ${{nrErrors}} + 1 }} ] }}


