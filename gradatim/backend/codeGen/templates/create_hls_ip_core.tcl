
#------------------------------------------------------------------------------  
# IBM-HSL-IP : {DOSA_FMSTR_DESCR}
#------------------------------------------------------------------------------
set ipModName "{DOSA_FMSTR_MOD_NAME}"
set ipName    "{DOSA_FMSTR_IP_NAME}"
set ipVendor  "IBM"
set ipLibrary "hls"
set ipVersion "1.0"
set ipCfgList  [ list ]

set rc [ my_customize_ip ${{ipModName}} ${{ipDir}} ${{ipVendor}} ${{ipLibrary}} ${{ipName}} ${{ipVersion}} ${{ipCfgList}} ]

if {{ ${{rc}} != ${{::OK}} }} {{ set nrErrors [ expr {{ ${{nrErrors}} + 1 }} ] }}


