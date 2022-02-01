#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Jan 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        ZRLMPI wrapper generation, based on templates
#  *
#  *
import math
import os
from pathlib import Path

from dimidium.backend.codeGen.CommunicationWrapper import CommunicationWrapper
from dimidium.middleend.archGen.CommPlan import CommPlan

__filedir__ = os.path.dirname(os.path.abspath(__file__))


class ZrlmpiWrapper(CommunicationWrapper):

    def __init__(self, node_id, out_dir_path, comm_plan: CommPlan):
        super().__init__(node_id, out_dir_path, comm_plan)
        self.templ_dir_path = os.path.join(__filedir__, 'templates/zrlmpi_wrapper/')
        self.if_bitwidth = 64
        self._add_spare_lines_ = 2
        self.ip_name = 'zrlmpi_wrapper'
        self.ip_mod_name = 'ZrlmpiWrapper_n{}'.format(node_id)

    def generate(self):
        # 0. copy 'static' files, dir structure
        os.system('cp {}/run_hls.tcl {}'.format(self.templ_dir_path, self.out_dir_path))
        os.system('cp {}/Makefile {}'.format(self.templ_dir_path, self.out_dir_path))
        os.system('mkdir -p {}/tb/'.format(self.out_dir_path))
        os.system('cp {}/tb/tb_zrlmpi_wrapper.cpp {}/tb/'.format(self.templ_dir_path, self.out_dir_path))
        os.system('mkdir -p {}/src/'.format(self.out_dir_path))
        os.system('cp {}/src/zrlmpi_common.* {}/src/'.format(self.templ_dir_path, self.out_dir_path))
        os.system('cp {}/src/zrlmpi_int.hpp {}/src/'.format(self.templ_dir_path, self.out_dir_path))
        # 0b) copy hls lib, overwrite if necessary
        os.system('mkdir -p {}/../lib/'.format(self.out_dir_path))
        os.system('cp {}/../lib/* {}/../lib/'.format(self.templ_dir_path, self.out_dir_path))
        # 1. analyzing comm plan
        longest_msg = self.comm_plan.get_longest_msg_bytes()
        longest_msg_lines = math.ceil(float(longest_msg) / float(self.if_bitwidth / 8)) + self._add_spare_lines_
        comm_plan_length = self.comm_plan.get_comm_instr_num()
        # 2. wrapper.hpp
        with open(os.path.join(self.templ_dir_path, 'src/zrlmpi_wrapper.hpp'), 'r') as in_file, \
                open(os.path.join(self.out_dir_path, 'src/zrlmpi_wrapper.hpp'), 'w') as out_file:
            for line in in_file.readlines():
                if 'DOSA_ADD_INTERFACE_DEFINES' in line:
                    outline = ''
                    outline += '#define DOSA_WRAPPER_INPUT_IF_BITWIDTH {}\n'.format(self.if_bitwidth)
                    outline += '#define DOSA_WRAPPER_OUTPUT_IF_BITWIDTH {}\n'.format(self.if_bitwidth)
                    outline += '#define DOSA_WRAPPER_BUFFER_FIFO_DEPTH_LINES {}\n'.format(int(longest_msg_lines))
                    outline += '#define DOSA_WRAPPER_PROG_LENGTH {}\n'.format(comm_plan_length)
                else:
                    outline = line
                out_file.write(outline)
        # 3. wrapper.cpp
        with open(os.path.join(self.templ_dir_path, 'src/zrlmpi_wrapper.cpp'), 'r') as in_file, \
                open(os.path.join(self.out_dir_path, 'src/zrlmpi_wrapper.cpp'), 'w') as out_file:
            for line in in_file.readlines():
                if 'DOSA_ADD_mpi_commands' in line:
                    outline = ''
                    instr_num = 0
                    for ie in self.comm_plan.get_comm_instr():
                        cmnd_macro = 'MPI_INSTR_RECV'
                        if ie['instr'] == 'send':
                            cmnd_macro = 'MPI_INSTR_SEND'
                        rank = ie['rank']
                        counts = ie['count']
                        repeat = ie['repeat']
                        outline += f'      mpiCommands[{instr_num}]          = {cmnd_macro};\n'
                        outline += f'      mpiRanks[{instr_num}]             = {rank};\n'
                        outline += f'      mpiCounts[{instr_num}]            = {counts};\n'
                        outline += f'      commandRepetitions[{instr_num}]   = {repeat};\n'
                        instr_num += 1
                    assert instr_num == comm_plan_length
                else:
                    outline = line
                out_file.write(outline)
        # 4. copy 'static' MPE2 files
        os.system('mkdir -p {}/../mpe2/'.format(self.out_dir_path))
        os.system('cp -R {}/../ZRLMPI/LIB/HW/hls/mpe2/* {}/../mpe2/'.format(self.templ_dir_path, self.out_dir_path))
        os.system('rm -f {}/../mpe2/src/zrlmpi_common.*'.format(self.out_dir_path))
        os.system('cp {}/../ZRLMPI/LIB/COMMON/zrlmpi_common.* {}/../mpe2/src/'.format(self.templ_dir_path,
                                                                                      self.out_dir_path))

    def get_tcl_lines_wrapper_inst(self, ip_description='ZRLMPI wrapper instantiation for this node'):
        template_lines = Path(os.path.join(__filedir__, 'templates/create_hls_ip_core.tcl')).read_text()
        new_tcl_lines = template_lines.format(DOSA_FMSTR_DESCR=ip_description, DOSA_FMSTR_MOD_NAME=self.ip_mod_name,
                                              DOSA_FMSTR_IP_NAME=self.ip_name)
        static_lines = Path(os.path.join(__filedir__, 'templates/zrlmpi_static.tcl')).read_text()
        new_tcl_lines += static_lines
        return new_tcl_lines

    def get_wrapper_vhdl_decl_lines(self):
        # we need to do the connections between wrapper and MPE with network ourselves
        static_lines = Path(os.path.join(__filedir__, 'templates/zrlmpi_vhdl_static_decl.vhdl')).read_text()
        decl = '\n'
        # decl += ('-- thanks to the fantastic and incredible Vivado HLS...we need vectors with (0 downto 0)\n' +
        #          'signal s{ip_mod_name}_siData_tlast_as_vector    : std_ulogic_vector(0 downto 0);\n' +
        #          'signal s{ip_mod_name}_soData_tlast_as_vector    : std_ulogic_vector(0 downto 0);\n')
        # decl += '\n'
        decl += ('component {ip_mod_name} is\n' +
                 '   port (\n' +
                 '       piFMC_to_ROLE_rank_V : IN STD_LOGIC_VECTOR (31 downto 0);\n' +
                 '       piFMC_to_ROLE_size_V : IN STD_LOGIC_VECTOR (31 downto 0);\n' +
                 '       siData_V_tdata_V_dout : IN STD_LOGIC_VECTOR ({in_bitwidth} downto 0);\n' +
                 '       siData_V_tdata_V_empty_n : IN STD_LOGIC;\n' +
                 '       siData_V_tdata_V_read : OUT STD_LOGIC;\n' +
                 '       siData_V_tkeep_V_dout : IN STD_LOGIC_VECTOR ({in_bitwidth_tkeep} downto 0);\n' +
                 '       siData_V_tkeep_V_empty_n : IN STD_LOGIC;\n' +
                 '       siData_V_tkeep_V_read : OUT STD_LOGIC;\n' +
                 '       siData_V_tlast_V_dout : IN STD_LOGIC_VECTOR (0 downto 0);\n' +
                 '       siData_V_tlast_V_empty_n : IN STD_LOGIC;\n' +
                 '       siData_V_tlast_V_read : OUT STD_LOGIC;\n' +
                 '       soData_V_tdata_V_din : OUT STD_LOGIC_VECTOR ({out_bitwidth} downto 0);\n' +
                 '       soData_V_tdata_V_full_n : IN STD_LOGIC;\n' +
                 '       soData_V_tdata_V_write : OUT STD_LOGIC;\n' +
                 '       soData_V_tkeep_V_din : OUT STD_LOGIC_VECTOR ({out_bitwidth_tkeep} downto 0);\n' +
                 '       soData_V_tkeep_V_full_n : IN STD_LOGIC;\n' +
                 '       soData_V_tkeep_V_write : OUT STD_LOGIC;\n' +
                 '       soData_V_tlast_V_din : OUT STD_LOGIC_VECTOR (0 downto 0);\n' +
                 '       soData_V_tlast_V_full_n : IN STD_LOGIC;\n' +
                 '       soData_V_tlast_V_write : OUT STD_LOGIC;\n' +
                 '       soMPIif_V_din : OUT STD_LOGIC_VECTOR (71 downto 0);\n' +
                 '       soMPIif_V_full_n : IN STD_LOGIC;\n' +
                 '       soMPIif_V_write : OUT STD_LOGIC;\n' +
                 '       siMPIFeB_V_dout : IN STD_LOGIC_VECTOR (7 downto 0);\n' +
                 '       siMPIFeB_V_empty_n : IN STD_LOGIC;\n' +
                 '       siMPIFeB_V_read : OUT STD_LOGIC;\n' +
                 '       soMPI_data_V_din : OUT STD_LOGIC_VECTOR (72 downto 0);\n' +
                 '       soMPI_data_V_full_n : IN STD_LOGIC;\n' +
                 '       soMPI_data_V_write : OUT STD_LOGIC;\n' +
                 '       siMPI_data_V_dout : IN STD_LOGIC_VECTOR (72 downto 0);\n' +
                 '       siMPI_data_V_empty_n : IN STD_LOGIC;\n' +
                 '       siMPI_data_V_read : OUT STD_LOGIC;\n' +
                 '       debug_out_V : IN STD_LOGIC_VECTOR (31 downto 0);\n' +
                 '       ap_clk : IN STD_LOGIC;\n' +
                 '       ap_rst : IN STD_LOGIC );\n' +
                 '   end component {ip_mod_name};\n')
        dyn_lines = decl.format(ip_mod_name=self.ip_mod_name, in_bitwidth=(self.if_bitwidth - 1),
                                in_bitwidth_tkeep=(int((self.if_bitwidth + 7) / 8) - 1),
                                out_bitwidth=(self.if_bitwidth - 1),
                                out_bitwidth_tkeep=(int((self.if_bitwidth + 7) / 8) - 1))
        ret = static_lines + dyn_lines
        return ret

    def get_vhdl_inst_tmpl(self):
        static_lines = Path(os.path.join(__filedir__, 'templates/zrlmpi_vhdl_static_inst.vhdl')).read_text()
        decl = '\n'
        # decl += ('s{ip_mod_name}_siData_tlast_as_vector(0) <= ;\n' +
        #          's{ip_mod_name}_soData_tlast_as_vector(0) <= ;\n')
        # decl += '\n'
        decl += ('[inst_name]: {ip_mod_name}\n' +
                 'port map (\n' +
                 '    piFMC_to_ROLE_rank_V         => piFMC_ROLE_rank,\n' +
                 '    piFMC_to_ROLE_rank_V_ap_vld  => \'1\',\n' +
                 '    piFMC_to_ROLE_size_V         => piFMC_ROLE_size,\n' +
                 '    piFMC_to_ROLE_size_V_ap_vld  => \'1\',\n' +
                 '    siData_V_tdata_V_dout =>     [in_sig_0]  ,\n' +
                 '    siData_V_tdata_V_empty_n =>  [in_sig_1_n],\n' +
                 '    siData_V_tdata_V_read =>     [in_sig_2]  ,\n' +
                 '    siData_V_tkeep_V_dout =>     [in_sig_3]  ,\n' +
                 '    siData_V_tkeep_V_empty_n =>  [in_sig_4_n],\n' +
                 '    siData_V_tkeep_V_read =>     [in_sig_5]  ,\n' +
                 '    siData_V_tlast_V_dout =>     [in_sig_6]  ,\n' +
                 '    siData_V_tlast_V_empty_n =>  [in_sig_7_n],\n' +
                 '    siData_V_tlast_V_read =>     [in_sig_8]  ,\n' +
                 '    soData_V_tdata_V_din =>      [out_sig_0]  ,\n' +
                 '    soData_V_tdata_V_full_n =>   [out_sig_1_n],\n' +
                 '    soData_V_tdata_V_write =>    [out_sig_2]  ,\n' +
                 '    soData_V_tkeep_V_din =>      [out_sig_3]  ,\n' +
                 '    soData_V_tkeep_V_full_n =>   [out_sig_4_n],\n' +
                 '    soData_V_tkeep_V_write =>    [out_sig_5]  ,\n' +
                 '    soData_V_tlast_V_din =>      [out_sig_6]  ,\n' +
                 '    soData_V_tlast_V_full_n =>   [out_sig_7_n],\n' +
                 '    soData_V_tlast_V_write =>    [out_sig_8]  ,\n' +
                 '    soMPIif_V_din                => sAPP_Fifo_MPIif_din       ,\n' +
                 '    soMPIif_V_full_n             => sAPP_Fifo_MPIif_full_n    ,\n' +
                 '    soMPIif_V_write              => sAPP_Fifo_MPIif_write     ,\n' +
                 '    siMPIFeB_V_dout              => sFifo_APP_MPIFeB_dout     ,\n' +
                 '    siMPIFeB_V_empty_n           => sFifo_APP_MPIFeB_empty_n  ,\n' +
                 '    siMPIFeB_V_read              => sFifo_APP_MPIFeB_read     ,\n' +
                 '    soMPI_data_V_din             => sAPP_Fifo_MPIdata_din     ,\n' +
                 '    soMPI_data_V_full_n          => sAPP_Fifo_MPIdata_full_n  ,\n' +
                 '    soMPI_data_V_write           => sAPP_Fifo_MPIdata_write   ,\n' +
                 '    siMPI_data_V_dout            => sFifo_APP_MPIdata_dout    ,\n' +
                 '    siMPI_data_V_empty_n         => sFifo_APP_MPIdata_empty_n ,\n' +
                 '    siMPI_data_V_read            => sFifo_APP_MPIdata_read    ,\n' +
                 # '    debug_out_V =>  open,\n' +
                 '    ap_clk =>  [clk],\n' +
                 '    ap_rst =>  [rst]\n' +  # no comma
                 ');\n')
        decl += '\n'
        inst = decl.format(ip_mod_name=self.ip_mod_name)
        # replace [] with {}
        dyn_lines = inst.replace('[', '{').replace(']', '}')
        inst_tmpl = static_lines + dyn_lines
        return inst_tmpl
