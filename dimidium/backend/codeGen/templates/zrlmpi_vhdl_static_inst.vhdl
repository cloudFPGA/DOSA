

  sAPP_Fifo_MPIif_full_n <=  not sAPP_Fifo_MPIif_full;
  sAPP_Fifo_MPIdata_full_n <= not sAPP_Fifo_MPIdata_full;
  sFifo_APP_MPIdata_empty_n <= not sFifo_APP_MPIdata_empty;

  sMPE_Fifo_MPIFeB_full_n <= not sMPE_Fifo_MPIFeB_full;
  sFifo_APP_MPIFeB_empty_n <= not sFifo_APP_MPIFeB_empty;

  FIFO_IF_APP_MPE: FifoMpiInfo
  port map (
             clk     => {clk},
             srst    => {rst},
             din     => sAPP_Fifo_MPIif_din    ,
             full    => sAPP_Fifo_MPIif_full   ,
             wr_en   => sAPP_Fifo_MPIif_write  ,
             dout    => sFifo_MPE_MPIif_dout   ,
             empty   => sFifo_MPE_MPIif_empty  ,
             rd_en   => sFifo_MPE_MPIif_read
           );

  FIFO_DATA_APP_MPE: FifoMpiData
  port map (
             clk     => {clk},
             srst    => {rst},
             din     => sAPP_Fifo_MPIdata_din    ,
             full    => sAPP_Fifo_MPIdata_full   ,
             wr_en   => sAPP_Fifo_MPIdata_write  ,
             dout    => sFifo_MPE_MPIdata_dout   ,
             empty   => sFifo_MPE_MPIdata_empty  ,
             rd_en   => sFifo_MPE_MPIdata_read
           );

  FIFO_DATA_MPE_APP: FifoMpiData
  port map (
             clk     => {clk},
             srst    => {rst},
             din     => sMPE_Fifo_MPIdata_din     ,
             full    => sMPE_Fifo_MPIdata_full    ,
             wr_en   => sMPE_Fifo_MPIdata_write   ,
             dout    => sFifo_APP_MPIdata_dout    ,
             empty   => sFifo_APP_MPIdata_empty   ,
             rd_en   => sFifo_APP_MPIdata_read
           );

  FIFO_FEB_MPE_APP: FifoMpiFeedback
  port map (
             clk     => {clk},
             srst    => {rst},
             din     => sMPE_Fifo_MPIFeB_din     ,
             full    => sMPE_Fifo_MPIFeB_full    ,
             wr_en   => sMPE_Fifo_MPIFeB_write   ,
             dout    => sFifo_APP_MPIFeB_dout    ,
             empty   => sFifo_APP_MPIFeB_empty   ,
             rd_en   => sFifo_APP_MPIFeB_read
           );

  sFifo_MPE_MPIif_empty_n <=  not sFifo_MPE_MPIif_empty;
  sFifo_MPE_MPIdata_empty_n <= not sFifo_MPE_MPIdata_empty;
  sMPE_Fifo_MPIdata_full_n <= not sMPE_Fifo_MPIdata_full;

  sMetaInTlastAsVector_Udp(0) <= siNRC_Role_Udp_Meta_TLAST;
  soROLE_Nrc_Udp_Meta_TLAST <=  sMetaOutTlastAsVector_Udp(0);
  sDataInTlastAsVector_Udp(0) <= siNRC_Udp_Data_tlast;
  soNRC_Udp_Data_tlast <= sDataOutTlastAsVector_Udp(0);

  MPE: MessagePassingEngine
  port map (
             ap_clk              => {clk},
             ap_rst_n            => {rst_n},
        --ap_start            => piMMIO_Ly7_En,
             siTcp_data_TDATA    =>  siNRC_Udp_Data_tdata ,
             siTcp_data_TKEEP    =>  siNRC_Udp_Data_tkeep ,
             siTcp_data_TVALID   =>  siNRC_Udp_Data_tvalid,
             siTcp_data_TLAST    =>  sDataInTlastAsVector_Udp,
             siTcp_data_TREADY   =>  siNRC_Udp_Data_tready,
             siTcp_meta_TDATA    =>  siNRC_Role_Udp_Meta_TDATA  ,
             siTcp_meta_TVALID   =>  siNRC_Role_Udp_Meta_TVALID ,
             siTcp_meta_TREADY   =>  siNRC_Role_Udp_Meta_TREADY ,
             siTcp_meta_TKEEP    =>  siNRC_Role_Udp_Meta_TKEEP  ,
             siTcp_meta_TLAST    =>  sMetaInTlastAsVector_Udp,
             soTcp_data_TDATA    =>  soNRC_Udp_Data_tdata   ,
             soTcp_data_TKEEP    =>  soNRC_Udp_Data_tkeep   ,
             soTcp_data_TVALID   =>  soNRC_Udp_Data_tvalid  ,
             soTcp_data_TLAST    =>  sDataOutTlastAsVector_Udp,
             soTcp_data_TREADY   =>  soNRC_Udp_Data_tready  ,
             soTcp_meta_TDATA    =>  soROLE_Nrc_Udp_Meta_TDATA  ,
             soTcp_meta_TVALID   =>  soROLE_Nrc_Udp_Meta_TVALID ,
             soTcp_meta_TREADY   =>  soROLE_Nrc_Udp_Meta_TREADY ,
             soTcp_meta_TKEEP    =>  soROLE_Nrc_Udp_Meta_TKEEP  ,
             soTcp_meta_TLAST    =>  sMetaOutTlastAsVector_Udp  ,
             poROL_NRC_Rx_ports_V => poROL_Nrc_Udp_Rx_ports,
             piFMC_rank_V        =>  piFMC_ROLE_rank,
             piFMC_rank_V_ap_vld =>  '1',
             poMMIO_V            =>  sMPE_Debug,
             siMPIif_V_dout       => sFifo_MPE_MPIif_dout      ,
             siMPIif_V_empty_n    => sFifo_MPE_MPIif_empty_n   ,
             siMPIif_V_read       => sFifo_MPE_MPIif_read      ,
             soMPIFeB_V_din       => sMPE_Fifo_MPIFeB_din     ,
             soMPIFeB_V_full_n    => sMPE_Fifo_MPIFeB_full_n  ,
             soMPIFeB_V_write     => sMPE_Fifo_MPIFeB_write   ,
             siMPI_data_V_dout    => sFifo_MPE_MPIdata_dout    ,
             siMPI_data_V_empty_n => sFifo_MPE_MPIdata_empty_n ,
             siMPI_data_V_read    => sFifo_MPE_MPIdata_read    ,
             soMPI_data_V_din     => sMPE_Fifo_MPIdata_din     ,
             soMPI_data_V_full_n  => sMPE_Fifo_MPIdata_full_n  ,
             soMPI_data_V_write   => sMPE_Fifo_MPIdata_write
           );


