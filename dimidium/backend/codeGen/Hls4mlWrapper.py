#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Mar 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Hls4ml Wrapper generation, based on templates
#  *
#  *

import os
from pathlib import Path

__filedir__ = os.path.dirname(os.path.abspath(__file__))


class Hls4mlWrapper:

    def __init__(self, block_id, in_dims, out_dims, acc_in_bitw, acc_out_bitw, if_in_bitw, if_out_bitw, out_dir_path):
        self.templ_dir_path = os.path.join(__filedir__, 'templates/hls4ml_wrapper/')
        self.ip_name = 'hls4ml_wrapper_b{}'.format(block_id)
        self.ip_mod_name = 'Hls4mlWrapper_b{}'.format(block_id)
        self.block_id = block_id
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.acc_in_bitw = acc_in_bitw
        self.acc_out_bitw = acc_out_bitw
        self.if_in_bitw = if_in_bitw
        self.if_out_bitw = if_out_bitw
        self.out_dir_path = out_dir_path

    def generate(self):
        # 0. copy 'static' files, dir structure
        os.system('cp {}/run_hls.tcl {}'.format(self.templ_dir_path, self.out_dir_path))
        os.system('mkdir -p {}/tb/'.format(self.out_dir_path))
        os.system('mkdir -p {}/src/'.format(self.out_dir_path))
        os.system('cp {}/tb/tb_hls4ml_wrapper.cpp {}/tb/'.format(self.templ_dir_path, self.out_dir_path))
        # 0b) copy hls lib, overwrite if necessary
        os.system('mkdir -p {}/../lib/'.format(self.out_dir_path))
        os.system('cp {}/../lib/* {}/../lib/'.format(self.templ_dir_path, self.out_dir_path))
        # 1. Makefile
        static_skip_lines = [0, 3, 4]
        with open(os.path.join(self.templ_dir_path, 'Makefile'), 'r') as in_file, \
                open(os.path.join(self.out_dir_path, 'Makefile'), 'w') as out_file:
            cur_line_nr = 0
            for line in in_file.readlines():
                if cur_line_nr in static_skip_lines or cur_line_nr > static_skip_lines[-1]:
                    # do nothing, copy
                    outline = line
                else:
                    if cur_line_nr == 1:
                        outline = 'ipName ={}\n'.format(self.ip_name)
                    elif cur_line_nr == 2:
                        outline = '\n'
                out_file.write(outline)
                cur_line_nr += 1
        # 2. wrapper.hpp
        with open(os.path.join(self.templ_dir_path, 'src/hls4ml_wrapper.hpp'), 'r') as in_file, \
                open(os.path.join(self.out_dir_path, 'src/hls4ml_wrapper.hpp'), 'w') as out_file:
            skip_line = False
            continue_skip = False
            for line in in_file.readlines():
                if skip_line:
                    skip_line = False
                    continue
                if continue_skip:
                    if 'DOSA_REMOVE_STOP' in line:
                        continue_skip = False
                    continue
                if 'DOSA_ADD_ip_name_BELOW' in line:
                    outline = 'void {}(\n'.format(self.ip_name)
                    # skip next line
                    skip_line = True
                elif 'DOSA_REMOVE_START' in line:
                    continue_skip = True
                    continue
                elif 'DOSA_ADD_INTERFACE_DEFINES' in line:
                    outline = ''
                    outline += '#define DOSA_WRAPPER_INPUT_IF_BITWIDTH {}\n'.format(self.if_in_bitw)
                    outline += '#define DOSA_WRAPPER_OUTPUT_IF_BITWIDTH {}\n'.format(self.if_out_bitw)
                    outline += '#define DOSA_HLS4ML_INPUT_BITWIDTH {}\n'.format(self.acc_in_bitw)
                    outline += '#define DOSA_HLS4ML_OUTPUT_BITWIDTH {}\n'.format(self.acc_out_bitw)
                    in_frame_size = self.in_dims[1]
                    for i in range(2, len(self.in_dims)):
                        in_frame_size *= self.in_dims[i]
                    out_frame_size = self.out_dims[1]
                    for i in range(2, len(self.out_dims)):
                        out_frame_size *= self.out_dims[i]
                    outline += '#define CNN_INPUT_FRAME_SIZE {}\n'.format(in_frame_size)
                    outline += '#define CNN_OUTPUT_FRAME_SIZE {}\n'.format(out_frame_size)
                else:
                    outline = line
                out_file.write(outline)
        # 3. wrapper.cpp
        with open(os.path.join(self.templ_dir_path, 'src/hls4ml_wrapper.cpp'), 'r') as in_file, \
                open(os.path.join(self.out_dir_path, 'src/hls4ml_wrapper.cpp'), 'w') as out_file:
            skip_line = False
            for line in in_file.readlines():
                if skip_line:
                    skip_line = False
                    continue
                if 'DOSA_ADD_ip_name_BELOW' in line:
                    outline = 'void {}(\n'.format(self.ip_name)
                    # skip next line
                    skip_line = True
                else:
                    outline = line
                out_file.write(outline)

    def get_tcl_lines_wrapper_inst(self, ip_description='Hls4mlWrapper instantiation'):
        template_lines = Path(os.path.join(__filedir__, 'templates/create_hls_ip_core.tcl')).read_text()
        new_tcl_lines = template_lines.format(DOSA_FMSTR_DESCR=ip_description, DOSA_FMSTR_MOD_NAME=self.ip_mod_name,
                                              DOSA_FMSTR_IP_NAME=self.ip_name)
        # also adding hls4ml ArchBlock export
        new_tcl_lines += '\n\n'
        new_tcl_lines += template_lines.format(DOSA_FMSTR_DESCR='Hls4ml instantiation',
                                               # both 'names' must be different, apparently...
                                               DOSA_FMSTR_MOD_NAME='HLS4ML_ArchBlock_{}'.format(self.block_id),
                                               DOSA_FMSTR_IP_NAME='ArchBlock_{}'.format(self.block_id))
        return new_tcl_lines

    def get_wrapper_vhdl_decl_lines(self):
        # we need to do the connections between wrapper and haddoc ourselves
        decl = ('signal ss{ip_mod_name}_to_Hls4ml_b{block_id}_tdata  : std_ulogic_vector({acc_inw_sub} downto 0);\n' +
                'signal ss{ip_mod_name}_to_Hls4ml_b{block_id}_tready : std_ulogic;\n' +
                'signal ss{ip_mod_name}_to_Hls4ml_b{block_id}_tvalid : std_ulogic;\n'
                'signal ssHls4ml_b{block_id}_to_{ip_mod_name}_tdata  : std_ulogic_vector({acc_outw_sub} downto 0);\n' +
                'signal ssHls4ml_b{block_id}_to_{ip_mod_name}_tready : std_ulogic;\n' +
                'signal ssHls4ml_b{block_id}_to_{ip_mod_name}_tvalid : std_ulogic;\n')
        decl += '\n'
        decl += ('component HLS4ML_ArchBlock_{block_id} is\n' +
                 'port (\n' +
                 '    input_0_V_TDATA : IN STD_LOGIC_VECTOR ({acc_inw_sub} downto 0);\n' +
                 '    output_0_V_TDATA : OUT STD_LOGIC_VECTOR ({acc_outw_sub} downto 0);\n' +
                 '    const_size_in_1 : OUT STD_LOGIC_VECTOR (15 downto 0);\n' +
                 '    const_size_out_1 : OUT STD_LOGIC_VECTOR (15 downto 0);\n' +
                 '    ap_clk : IN STD_LOGIC;\n' +
                 '    ap_rst_n : IN STD_LOGIC;\n' +
                 '    const_size_in_1_ap_vld : OUT STD_LOGIC;\n' +
                 '    const_size_out_1_ap_vld : OUT STD_LOGIC;\n' +
                 '    ap_start : IN STD_LOGIC;\n' +
                 '    ap_done : OUT STD_LOGIC;\n' +
                 '    input_0_V_TVALID : IN STD_LOGIC;\n' +
                 '    input_0_V_TREADY : OUT STD_LOGIC;\n' +
                 '    output_0_V_TVALID : OUT STD_LOGIC;\n' +
                 '    output_0_V_TREADY : IN STD_LOGIC;\n' +
                 '    ap_ready : OUT STD_LOGIC;\n' +
                 '    ap_idle : OUT STD_LOGIC\n' +
                 '  );\n' +
                 'end component HLS4ML_ArchBlock_{block_id};\n')
        decl += '\n'
        decl += ('component {ip_mod_name} is\n' +
                 'port (\n' +
                 '    siData_V_tdata_V_dout : IN STD_LOGIC_VECTOR ({if_in_width_tdata} downto 0);\n' +
                 '    siData_V_tdata_V_empty_n : IN STD_LOGIC;\n' +
                 '    siData_V_tdata_V_read : OUT STD_LOGIC;\n' +
                 '    siData_V_tkeep_V_dout : IN STD_LOGIC_VECTOR ({if_in_width_tkeep} downto 0);\n' +
                 '    siData_V_tkeep_V_empty_n : IN STD_LOGIC;\n' +
                 '    siData_V_tkeep_V_read : OUT STD_LOGIC;\n' +
                 '    siData_V_tlast_V_dout : IN STD_LOGIC_VECTOR ({if_in_width_tlast} downto 0);\n' +
                 '    siData_V_tlast_V_empty_n : IN STD_LOGIC;\n' +
                 '    siData_V_tlast_V_read : OUT STD_LOGIC;\n' +
                 '    soData_V_tdata_V_din : OUT STD_LOGIC_VECTOR ({if_out_width_tdata} downto 0);\n' +
                 '    soData_V_tdata_V_full_n : IN STD_LOGIC;\n' +
                 '    soData_V_tdata_V_write : OUT STD_LOGIC;\n' +
                 '    soData_V_tkeep_V_din : OUT STD_LOGIC_VECTOR ({if_out_width_tkeep} downto 0);\n' +
                 '    soData_V_tkeep_V_full_n : IN STD_LOGIC;\n' +
                 '    soData_V_tkeep_V_write : OUT STD_LOGIC;\n' +
                 '    soData_V_tlast_V_din : OUT STD_LOGIC_VECTOR ({if_out_width_tlast} downto 0);\n' +
                 '    soData_V_tlast_V_full_n : IN STD_LOGIC;\n' +
                 '    soData_V_tlast_V_write : OUT STD_LOGIC;\n' +
                 '    soToHls4mlData_V_V_TDATA : OUT STD_LOGIC_VECTOR ({acc_inw_sub} downto 0);\n' +
                 '    siFromHls4mlData_V_V_TDATA : IN STD_LOGIC_VECTOR ({acc_outw_sub} downto 0);\n' +
                 '    debug_out_V : OUT STD_LOGIC_VECTOR (31 downto 0);\n' +
                 '    ap_clk : IN STD_LOGIC;\n' +
                 '    ap_rst_n : IN STD_LOGIC;\n' +
                 '    soToHls4mlData_V_V_TVALID : OUT STD_LOGIC;\n' +
                 '    soToHls4mlData_V_V_TREADY : IN STD_LOGIC;\n' +
                 '    debug_out_V_ap_vld : OUT STD_LOGIC;\n' +
                 '    siFromHls4mlData_V_V_TVALID : IN STD_LOGIC;\n' +
                 '    siFromHls4mlData_V_V_TREADY : OUT STD_LOGIC);\n'
                 'end component {ip_mod_name};\n')
        ret = decl.format(block_id=self.block_id, ip_mod_name=self.ip_mod_name,
                          if_in_width_tdata=(self.if_in_bitw - 1),
                          if_in_width_tkeep=(int((self.if_in_bitw + 7) / 8) - 1), if_in_width_tlast=0,
                          if_out_width_tdata=(self.if_out_bitw - 1),
                          if_out_width_tkeep=(int((self.if_out_bitw + 7) / 8) - 1),
                          if_out_width_tlast=0,
                          acc_inw_sub=(self.acc_in_bitw - 1), acc_outw_sub=(self.acc_out_bitw - 1))
        return ret

    def get_vhdl_inst_tmpl(self):
        decl = ('[inst_name]_wrapper: {ip_mod_name}\n' +
                'port map (\n' +
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
                '    soToHls4mlData_V_V_TDATA    =>  ss{ip_mod_name}_to_Hls4ml_b{block_id}_tdata ,\n' +
                '    soToHls4mlData_V_V_TVALID   =>  ss{ip_mod_name}_to_Hls4ml_b{block_id}_tvalid,\n' +
                '    soToHls4mlData_V_V_TREADY   =>  ss{ip_mod_name}_to_Hls4ml_b{block_id}_tready,\n' +
                '    siFromHls4mlData_V_V_TDATA  =>  ssHls4ml_b{block_id}_to_{ip_mod_name}_tdata ,\n' +
                '    siFromHls4mlData_V_V_TVALID =>  ssHls4ml_b{block_id}_to_{ip_mod_name}_tvalid,\n' +
                '    siFromHls4mlData_V_V_TREADY =>  ssHls4ml_b{block_id}_to_{ip_mod_name}_tready,\n' +
                # '    debug_out_V =>  open,\n' +
                '    ap_clk =>  [clk],\n' +
                '    ap_rst_n =>  [rst_n]\n' +
                # '    debug_out_V_ap_vld =>  \n' +  # no comma
                ');\n')
        decl += '\n'
        decl += ('[inst_name]: HLS4ML_ArchBlock_{block_id}\n' +
                 'port map (\n' +
                 '    input_0_V_TDATA   =>  ss{ip_mod_name}_to_Hls4ml_b{block_id}_tdata ,\n' +
                 '    input_0_V_TVALID  =>  ss{ip_mod_name}_to_Hls4ml_b{block_id}_tvalid,\n' +
                 '    input_0_V_TREADY  =>  ss{ip_mod_name}_to_Hls4ml_b{block_id}_tready,\n' +
                 '    output_0_V_TDATA  =>  ssHls4ml_b{block_id}_to_{ip_mod_name}_tdata ,\n' +
                 '    output_0_V_TVALID =>  ssHls4ml_b{block_id}_to_{ip_mod_name}_tvalid,\n' +
                 '    output_0_V_TREADY =>  ssHls4ml_b{block_id}_to_{ip_mod_name}_tready,\n' +
                 # '    const_size_in_1 => open,\n' +
                 # '    const_size_out_1 => open,\n' +
                 # '    const_size_in_1_ap_vld => open,\n' +
                 # '    const_size_out_1_ap_vld => open,\n' +
                 '    ap_clk => [clk],\n' +
                 '    ap_start => [enable],\n' +
                 '    ap_rst_n => [rst_n]\n' +  # no comma
                 # '    ap_done => open,\n' +
                 # '    ap_ready => open,\n' +
                 # '    ap_idle => open\n' +  # no comma
                 '  );\n')
        inst = decl.format(block_id=self.block_id, ip_mod_name=self.ip_mod_name)
        # replace [] with {}
        inst_tmpl = inst.replace('[', '{').replace(']', '}')
        return inst_tmpl

    def get_debug_lines(self):
        signal_lines = [
            'ss{ip_mod_name}_to_Hls4ml_b{block_id}_tdata'.format(block_id=self.block_id, ip_mod_name=self.ip_mod_name),
            'ss{ip_mod_name}_to_Hls4ml_b{block_id}_tvalid'.format(block_id=self.block_id, ip_mod_name=self.ip_mod_name),
            'ss{ip_mod_name}_to_Hls4ml_b{block_id}_tready'.format(block_id=self.block_id, ip_mod_name=self.ip_mod_name),
            'ssHls4ml_b{block_id}_to_{ip_mod_name}_tdata'.format(block_id=self.block_id, ip_mod_name=self.ip_mod_name),
            'ssHls4ml_b{block_id}_to_{ip_mod_name}_tvalid'.format(block_id=self.block_id, ip_mod_name=self.ip_mod_name),
            'ssHls4ml_b{block_id}_to_{ip_mod_name}_tready'.format(block_id=self.block_id, ip_mod_name=self.ip_mod_name),
        ]
        width_lines = [self.acc_in_bitw, 1, 1,  self.acc_out_bitw, 1, 1]
        assert len(signal_lines) == len(width_lines)
        tcl_tmpl_lines = []
        decl_tmpl_lines = []
        inst_tmpl_lines = []

        for i in range(len(signal_lines)):
            sn = signal_lines[i]
            sw = width_lines[i]
            tcl_l = 'CONFIG.C_PROBE{i}_WIDTH {{' + str(sw) + '}}\\\n'
            decl_l = '; probe{i}    : in  std_logic_vector( ' + str(sw - 1) + ' downto 0)\n'  # semicolon at begin
            if sw == 1:
                inst_l = ', probe{i}(0)   =>   ' + sn + '\n'  # comma at begin
            else:
                inst_l = ', probe{i}      =>   ' + sn + '\n'  # comma at begin
            tcl_tmpl_lines.append(tcl_l)
            decl_tmpl_lines.append(decl_l)
            inst_tmpl_lines.append(inst_l)

        return tcl_tmpl_lines, decl_tmpl_lines, inst_tmpl_lines
