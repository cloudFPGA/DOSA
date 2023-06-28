#  /*******************************************************************************
#   * Copyright 2019 -- 2023 IBM Corporation
#   *
#   * Licensed under the Apache License, Version 2.0 (the "License");
#   * you may not use this file except in compliance with the License.
#   * You may obtain a copy of the License at
#   *
#   *     http://www.apache.org/licenses/LICENSE-2.0
#   *
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS,
#   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   * See the License for the specific language governing permissions and
#   * limitations under the License.
#  *******************************************************************************/
#

#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: May 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Tips engine core generation, based on templates
#  *
#  *

import os
from pathlib import Path
import math

from dimidium.lib.util import bit_width_to_tkeep

__filedir__ = os.path.dirname(os.path.abspath(__file__))


class TipsCore:

    def __init__(self, block_id, general_bitw, if_in_bitw, if_out_bitw, out_dir_path,
                 program, op_rom, longest_input, longest_op0, longest_op1, longest_output,
                 dtype_string, accum_dtype_string, fractional_bitw, accum_bitw):
        self.templ_dir_path = os.path.join(__filedir__, 'templates/tips/')
        self.ip_name = 'tips_engine_b{}'.format(block_id)
        self.ip_mod_name = 'TipsEngine_b{}'.format(block_id)
        self.block_id = block_id
        self.general_bitw = general_bitw
        self.if_in_bitw = if_in_bitw
        self.if_out_bitw = if_out_bitw
        self.out_dir_path = out_dir_path
        self.program = program
        self.op_rom = op_rom
        self.addr_space_length = -1
        self.op_rom_string = ''
        self.longest_input = longest_input
        self.longest_op0 = longest_op0
        self.longest_op1 = longest_op1
        self.longest_output = longest_output
        self.dtype_string = dtype_string
        self.accum_dtype_string = accum_dtype_string
        self.fractional_bitw = fractional_bitw
        self.accum_bitw = accum_bitw
        self._init_op_ram()

    def _init_op_ram(self):
        self.op_rom_string = '  const usedDtype opStore[DOSA_TIPS_ADDR_SPACE_LENGTH] = {\n    '
        r_cnt = 0
        self.addr_space_length = 0
        for r in self.op_rom:
            for e in r:
                self.op_rom_string += e + ', '
                self.addr_space_length += 1
            self.op_rom_string += '// operand {}\n    '.format(r_cnt)
            r_cnt += 1
        # TODO: necessary?
        self.op_rom_string += '0,0,0,0,0,0,0,0 //fill-to-end\n'
        self.addr_space_length += 8
        self.op_rom_string += '  };\n'

    def generate(self):
        # 0. copy 'static' files, dir structure
        os.system('cp {}/run_hls.tcl {}'.format(self.templ_dir_path, self.out_dir_path))
        os.system('mkdir -p {}/tb/'.format(self.out_dir_path))
        os.system('mkdir -p {}/src/'.format(self.out_dir_path))
        os.system('cp {}/src/alu.* {}/src/'.format(self.templ_dir_path, self.out_dir_path))
        os.system('cp {}/tb/tb_tips.cpp {}/tb/'.format(self.templ_dir_path, self.out_dir_path))
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
        with open(os.path.join(self.templ_dir_path, 'src/tips.hpp'), 'r') as in_file, \
                open(os.path.join(self.out_dir_path, 'src/tips.hpp'), 'w') as out_file:
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
                elif 'DOSA_ADD_CORE_DEFINES' in line:
                    tkeep_general = bit_width_to_tkeep(self.general_bitw)
                    tkeep_width = max(math.ceil(math.log2(tkeep_general)), 1)
                    assert tkeep_general > 0
                    assert tkeep_width > 0
                    assert tkeep_general >= tkeep_width
                    assert self.general_bitw in [8, 16, 32, 64]
                    outline = ''
                    outline += '#define DOSA_WRAPPER_INPUT_IF_BITWIDTH {}\n'.format(self.if_in_bitw)
                    outline += '#define DOSA_WRAPPER_OUTPUT_IF_BITWIDTH {}\n'.format(self.if_out_bitw)
                    outline += 'typedef uint{}_t usedDtype;\n'.format(self.general_bitw)
                    outline += 'typedef {} quantDtype;\n'.format(self.dtype_string)
                    outline += '#define DEBUG_FRACTIONAL_BITS {}\n'.format(self.fractional_bitw)
                    outline += '#define QUANT_SCALE_BACK_VALUE {}\n'.format(int(math.pow(2, self.fractional_bitw)))
                    outline += '#define DOSA_TIPS_USED_BITWIDTH {}\n'.format(self.general_bitw)
                    outline += '#define DOSA_TIPS_USED_BITWIDTH_PARTIAL_MASK {}\n'.format(tkeep_general)
                    outline += '#define DOSA_TIPS_LONGEST_INPUT {}\n'.format(self.longest_input)
                    outline += '#define DOSA_TIPS_LONGEST_OP0 {}\n'.format(self.longest_op0)
                    outline += '#define DOSA_TIPS_LONGEST_OP1 {}\n'.format(self.longest_op1)
                    outline += '#define DOSA_TIPS_LONGEST_OUTPUT {}\n'.format(self.longest_output)
                    outline += '#define DOSA_TIPS_PROGRAM_LENGTH {}\n'.format(len(self.program))
                    outline += '#define DOSA_TIPS_ADDR_SPACE_LENGTH {}\n'.format(self.addr_space_length)
                    outline += 'typedef {} aluAccumDtype;\n'.format(self.accum_dtype_string)
                    outline += '#define DOSA_TIPS_ALU_ACCUM_BITWIDTH {}\n'.format(self.accum_bitw)
                    outline += 'const int alu_op_pipeline_ii = {};\n'.format(max(self.longest_op0, self.longest_op1,
                                                                                 self.longest_input, self.longest_output))
                else:
                    outline = line
                out_file.write(outline)
        # 3. wrapper.cpp
        with open(os.path.join(self.templ_dir_path, 'src/tips.cpp'), 'r') as in_file, \
                open(os.path.join(self.out_dir_path, 'src/tips.cpp'), 'w') as out_file:
            skip_line = False
            for line in in_file.readlines():
                if skip_line:
                    skip_line = False
                    continue
                if 'DOSA_ADD_ip_name_BELOW' in line:
                    outline = 'void {}(\n'.format(self.ip_name)
                    # skip next line
                    skip_line = True
                elif 'DOSA_ADD_program' in line:
                    outline = '  const TipsOp program[] = {\n'
                    i = 0
                    for pl in self.program:
                        outline += f'    [{i}] = '
                        indent = '          '
                        pll = pl.splitlines()
                        for ll in pll:
                            if ll != '}':
                                if '{' in ll:
                                    outline += ll + '\n'
                                else:
                                    outline += indent + ll + '\n'
                            else:
                                if i == (len(self.program) - 1):
                                    outline += indent + '}\n'
                                else:
                                    outline += indent + '},\n'
                        i += 1
                    outline += '  };\n'
                    assert i == len(self.program)
                elif 'DOSA_ADD_op_store' in line:
                    outline = self.op_rom_string
                else:
                    outline = line
                out_file.write(outline)
        return

    def get_tcl_lines_inst(self, ip_description='TIPS Engine instantiation'):
        template_lines = Path(os.path.join(__filedir__, 'templates/create_hls_ip_core.tcl')).read_text()
        new_tcl_lines = template_lines.format(DOSA_FMSTR_DESCR=ip_description, DOSA_FMSTR_MOD_NAME=self.ip_mod_name,
                                              DOSA_FMSTR_IP_NAME=self.ip_name)
        return new_tcl_lines

    def get_vhdl_decl_lines(self):
        decl = 'signal s{ip_mod_name}_debug    : std_ulogic_vector(79 downto 0);\n'
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
                 '    debug_out_V : OUT STD_LOGIC_VECTOR (79 downto 0);\n' +
                 '    ap_clk : IN STD_LOGIC;\n' +
                 '    ap_rst : IN STD_LOGIC;\n' +
                 '    debug_out_V_ap_vld : OUT STD_LOGIC );\n' +
                 'end component {ip_mod_name};\n')
        ret = decl.format(ip_mod_name=self.ip_mod_name,
                          if_in_width_tdata=(self.if_in_bitw - 1),
                          if_in_width_tkeep=(int((self.if_in_bitw + 7) / 8) - 1), if_in_width_tlast=0,
                          if_out_width_tdata=(self.if_out_bitw - 1),
                          if_out_width_tkeep=(int((self.if_out_bitw + 7) / 8) - 1),
                          if_out_width_tlast=0)
        return ret

    def get_vhdl_inst_tmpl(self):
        tmpl = ('[inst_name]: {ip_mod_name}\n' +
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
                '    debug_out_V =>  s{ip_mod_name}_debug,\n' +
                '    ap_clk =>  [clk],\n' +
                '    ap_rst =>  [rst]\n' +  # no comma
                ');\n')
        inst = tmpl.format(ip_mod_name=self.ip_mod_name)
        # replace [] with {}
        inst_tmpl = inst.replace('[', '{').replace(']', '}')
        return inst_tmpl

    def get_debug_lines(self):
        signal_lines = [
            's{ip_mod_name}_debug'.format(ip_mod_name=self.ip_mod_name)
        ]
        width_lines = [80]
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


