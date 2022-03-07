
  signal sAPP_Fifo_MPIif_din        : std_ulogic_vector(71 downto 0);
  signal sAPP_Fifo_MPIif_full_n     : std_ulogic;
  signal sAPP_Fifo_MPIif_full       : std_ulogic;
  signal sAPP_Fifo_MPIif_write      : std_ulogic;
  signal sAPP_Fifo_MPIdata_din      : std_ulogic_vector(72 downto 0);
  signal sAPP_Fifo_MPIdata_full_n   : std_ulogic;
  signal sAPP_Fifo_MPIdata_full     : std_ulogic;
  signal sAPP_Fifo_MPIdata_write    : std_ulogic;
  signal sFifo_APP_MPIdata_dout     : std_ulogic_vector(72 downto 0);
  signal sFifo_APP_MPIdata_empty_n  : std_ulogic;
  signal sFifo_APP_MPIdata_empty    : std_ulogic;
  signal sFifo_APP_MPIdata_read     : std_ulogic;

  signal sFifo_MPE_MPIif_dout       : std_ulogic_vector(71 downto 0);
  signal sFifo_MPE_MPIif_empty_n    : std_ulogic;
  signal sFifo_MPE_MPIif_empty      : std_ulogic;
  signal sFifo_MPE_MPIif_read       : std_ulogic;
  signal sFifo_MPE_MPIdata_dout     : std_ulogic_vector(72 downto 0);
  signal sFifo_MPE_MPIdata_empty_n  : std_ulogic;
  signal sFifo_MPE_MPIdata_empty    : std_ulogic;
  signal sFifo_MPE_MPIdata_read     : std_ulogic;
  signal sMPE_Fifo_MPIdata_din      : std_ulogic_vector(72 downto 0);
  signal sMPE_Fifo_MPIdata_full_n   : std_ulogic;
  signal sMPE_Fifo_MPIdata_full     : std_ulogic;
  signal sMPE_Fifo_MPIdata_write    : std_ulogic;

  signal sFifo_APP_MPIFeB_dout       : std_ulogic_vector(7 downto 0);
  signal sFifo_APP_MPIFeB_empty_n    : std_ulogic;
  signal sFifo_APP_MPIFeB_empty      : std_ulogic;
  signal sFifo_APP_MPIFeB_read       : std_ulogic;
  signal sMPE_Fifo_MPIFeB_din        : std_ulogic_vector(7 downto 0);
  signal sMPE_Fifo_MPIFeB_full_n     : std_ulogic;
  signal sMPE_Fifo_MPIFeB_full       : std_ulogic;
  signal sMPE_Fifo_MPIFeB_write      : std_ulogic;

  signal sMetaOutTlastAsVector_Udp : std_logic_vector(0 downto 0);
  signal sMetaInTlastAsVector_Udp  : std_logic_vector(0 downto 0);
  signal sDataOutTlastAsVector_Udp : std_logic_vector(0 downto 0);
  signal sDataInTlastAsVector_Udp  : std_logic_vector(0 downto 0);

  signal sMPE_Debug  : std_logic_vector(31 downto 0);

  component MessagePassingEngine is
    port (
           siTcp_data_TDATA : IN STD_LOGIC_VECTOR (63 downto 0);
           siTcp_data_TKEEP : IN STD_LOGIC_VECTOR (7 downto 0);
           siTcp_data_TLAST : IN STD_LOGIC_VECTOR (0 downto 0);
           siTcp_meta_TDATA : IN STD_LOGIC_VECTOR (63 downto 0);
           siTcp_meta_TKEEP : IN STD_LOGIC_VECTOR (7 downto 0);
           siTcp_meta_TLAST : IN STD_LOGIC_VECTOR (0 downto 0);
           soTcp_data_TDATA : OUT STD_LOGIC_VECTOR (63 downto 0);
           soTcp_data_TKEEP : OUT STD_LOGIC_VECTOR (7 downto 0);
           soTcp_data_TLAST : OUT STD_LOGIC_VECTOR (0 downto 0);
           soTcp_meta_TDATA : OUT STD_LOGIC_VECTOR (63 downto 0);
           soTcp_meta_TKEEP : OUT STD_LOGIC_VECTOR (7 downto 0);
           soTcp_meta_TLAST : OUT STD_LOGIC_VECTOR (0 downto 0);
           poROL_NRC_Rx_ports_V : OUT STD_LOGIC_VECTOR (31 downto 0);
           piFMC_rank_V : IN STD_LOGIC_VECTOR (31 downto 0);
           poMMIO_V : OUT STD_LOGIC_VECTOR (31 downto 0);
           siMPIif_V_dout : IN STD_LOGIC_VECTOR (71 downto 0);
           siMPIif_V_empty_n : IN STD_LOGIC;
           siMPIif_V_read : OUT STD_LOGIC;
           soMPIFeB_V_din : OUT STD_LOGIC_VECTOR (7 downto 0);
           soMPIFeB_V_full_n : IN STD_LOGIC;
           soMPIFeB_V_write : OUT STD_LOGIC;
           siMPI_data_V_dout : IN STD_LOGIC_VECTOR (72 downto 0);
           siMPI_data_V_empty_n : IN STD_LOGIC;
           siMPI_data_V_read : OUT STD_LOGIC;
           soMPI_data_V_din : OUT STD_LOGIC_VECTOR (72 downto 0);
           soMPI_data_V_full_n : IN STD_LOGIC;
           soMPI_data_V_write : OUT STD_LOGIC;
           ap_clk : IN STD_LOGIC;
           ap_rst_n : IN STD_LOGIC;
           poROL_NRC_Rx_ports_V_ap_vld : OUT STD_LOGIC;
           poMMIO_V_ap_vld : OUT STD_LOGIC;
           soTcp_meta_TVALID : OUT STD_LOGIC;
           soTcp_meta_TREADY : IN STD_LOGIC;
           soTcp_data_TVALID : OUT STD_LOGIC;
           soTcp_data_TREADY : IN STD_LOGIC;
           siTcp_data_TVALID : IN STD_LOGIC;
           siTcp_data_TREADY : OUT STD_LOGIC;
           siTcp_meta_TVALID : IN STD_LOGIC;
           siTcp_meta_TREADY : OUT STD_LOGIC;
           piFMC_rank_V_ap_vld : IN STD_LOGIC );
  end component;


  component FifoMpiData is
    port (
           clk    : in std_logic;
           srst   : in std_logic;
           din    : in std_logic_vector(72 downto 0);
           full   : out std_logic;
           wr_en  : in std_logic;
           dout   : out std_logic_vector(72 downto 0);
           empty  : out std_logic;
           rd_en  : in std_logic );
  end component;

  component FifoMpiInfo is
    port (
           clk    : in std_logic;
           srst   : in std_logic;
           din    : in std_logic_vector(71 downto 0);
           full   : out std_logic;
           wr_en  : in std_logic;
           dout   : out std_logic_vector(71 downto 0);
           empty  : out std_logic;
           rd_en  : in std_logic );
  end component;

  component FifoMpiFeedback is
    port (
           clk    : in std_logic;
           srst   : in std_logic;
           din    : in std_logic_vector(7 downto 0);
           full   : out std_logic;
           wr_en  : in std_logic;
           dout   : out std_logic_vector(7 downto 0);
           empty  : out std_logic;
           rd_en  : in std_logic );
  end component;



