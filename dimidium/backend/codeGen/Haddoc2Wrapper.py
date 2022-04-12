#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Jan 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Haddoc2 Wrapper generation, based on templates
#  *
#  *

import os
from pathlib import Path
import math

from dimidium.lib.util import bit_width_to_tkeep

__filedir__ = os.path.dirname(os.path.abspath(__file__))


class Haddoc2Wrapper:

    def __init__(self, block_id, in_dims, out_dims, general_bitw, if_in_bitw, if_out_bitw, out_dir_path,
                 wrapper_flatten_op, haddoc_op_cnt, first_layer_name):
        self.templ_dir_path = os.path.join(__filedir__, 'templates/haddoc2_wrapper/')
        self.ip_name = 'haddoc_wrapper_b{}'.format(block_id)
        self.ip_mod_name = 'Haddoc2Wrapper_b{}'.format(block_id)
        self.block_id = block_id
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.general_bitw = general_bitw
        self.if_in_bitw = if_in_bitw
        self.if_out_bitw = if_out_bitw
        self.out_dir_path = out_dir_path
        self.wrapper_flatten_op = wrapper_flatten_op
        self.haddoc_op_cnt = haddoc_op_cnt
        self.first_layer_name = first_layer_name

    def generate_haddoc2_wrapper(self):
        # 0. copy 'static' files, dir structure
        os.system('cp {}/run_hls.tcl {}'.format(self.templ_dir_path, self.out_dir_path))
        os.system('mkdir -p {}/tb/'.format(self.out_dir_path))
        os.system('mkdir -p {}/src/'.format(self.out_dir_path))
        os.system('cp {}/tb/tb_haddoc2_wrapper.cpp {}/tb/'.format(self.templ_dir_path, self.out_dir_path))
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
        with open(os.path.join(self.templ_dir_path, 'src/haddoc_wrapper.hpp'), 'r') as in_file, \
                open(os.path.join(self.out_dir_path, 'src/haddoc_wrapper.hpp'), 'w') as out_file:
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
                    tkeep_general = bit_width_to_tkeep(self.general_bitw)
                    tkeep_width = max(math.ceil(math.log2(tkeep_general)), 1)
                    assert tkeep_general > 0
                    assert tkeep_width > 0
                    assert tkeep_general >= tkeep_width
                    outline = ''
                    outline += '#define DOSA_WRAPPER_INPUT_IF_BITWIDTH {}\n'.format(self.if_in_bitw)
                    outline += '#define DOSA_WRAPPER_OUTPUT_IF_BITWIDTH {}\n'.format(self.if_out_bitw)
                    outline += '#define DOSA_HADDOC_GENERAL_BITWIDTH {}\n'.format(self.general_bitw)
                    outline += '#define DOSA_HADDOC_GENERAL_BITWIDTH_TKEEP {}\n'.format(tkeep_general)
                    outline += '#define DOSA_HADDOC_GENERAL_BITWIDTH_TKEEP_WIDTH {}\n'.format(tkeep_width)
                    outline += '#define DOSA_HADDOC_INPUT_CHAN_NUM {}\n'.format(self.in_dims[1])
                    outline += '#define DOSA_HADDOC_OUTPUT_CHAN_NUM {}\n'.format(self.out_dims[1])
                    outline += '#define DOSA_HADDOC_INPUT_FRAME_WIDTH {}\n'.format(self.in_dims[2])
                    outline += '#define DOSA_HADDOC_OUTPUT_FRAME_WIDTH {}\n'.format(self.out_dims[2])
                    flatten_str = 'false'
                    if self.wrapper_flatten_op is not None:
                        flatten_str = 'true'
                    outline += '#define DOSA_HADDOC_OUTPUT_BATCH_FLATTEN {}\n'.format(flatten_str)
                    outline += '#define DOSA_HADDOC_LAYER_CNT {}\n'.format(self.haddoc_op_cnt)
                    enum_def = 'enum ToHaddocEnqStates {RESET0 = 0, WAIT_DRAIN'
                    for b in range(0, self.in_dims[1]):
                        enum_def += ', FILL_BUF_{}'.format(b)
                    enum_def += '};\n'
                    outline += enum_def
                    enum_def = 'enum FromHaddocDeqStates {RESET1 = 0'
                    for b in range(0, self.out_dims[1]):
                        enum_def += ', READ_BUF_{}'.format(b)
                    enum_def += '};\n'
                    outline += enum_def
                else:
                    outline = line
                out_file.write(outline)
        # 3. wrapper.cpp
        with open(os.path.join(self.templ_dir_path, 'src/haddoc_wrapper.cpp'), 'r') as in_file, \
                open(os.path.join(self.out_dir_path, 'src/haddoc_wrapper.cpp'), 'w') as out_file:
            skip_line = False
            for line in in_file.readlines():
                if skip_line:
                    skip_line = False
                    continue
                if 'DOSA_ADD_ip_name_BELOW' in line:
                    outline = 'void {}(\n'.format(self.ip_name)
                    # skip next line
                    skip_line = True
                elif 'DOSA_ADD_toHaddoc_buffer_param_decl' in line:
                    outline = ''
                    for b in range(0, self.in_dims[1]):
                        outline += '    stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >    &sToHaddocBuffer_chan{},\n'.format(
                            b)
                elif 'DOSA_ADD_enq_fsm' in line:
                    fsm_tmpl = '    case FILL_BUF_{b}:\n      if( !siData.empty() && !sToHaddocBuffer_chan{b}.full() )\n' + \
                               '      {{\n        if(genericEnqState(siData, sToHaddocBuffer_chan{b}, current_frame_bit_cnt,' + \
                               ' hangover_store, hangover_store_valid_bits))\n        {{\n          ' + \
                               'enqueueFSM = FILL_BUF_{b1};\n        }}\n      }}\n      break;\n'
                    outline = ''
                    for b in range(0, self.in_dims[1]):
                        b1 = b + 1
                        if b1 >= self.in_dims[1]:
                            b1 = 0
                        outline += fsm_tmpl.format(b=b, b1=b1)
                elif 'DOSA_ADD_toHaddoc_deq_buffer_drain' in line:
                    fsm_tmpl = '    if( !sToHaddocBuffer_chan{b}.empty() )\n    {{\n      sToHaddocBuffer_chan{b}.read();\n' + \
                               '      one_not_empty = true;\n    }}\n'
                    outline = ''
                    for b in range(0, self.in_dims[1]):
                        outline += fsm_tmpl.format(b=b)
                elif 'DOSA_ADD_toHaddoc_deq_if_clause' in line:
                    outline = '       '
                    for b in range(0, self.in_dims[1]):
                        outline += ' && !sToHaddocBuffer_chan{}.empty()'.format(b)
                    outline += '\n'
                elif 'DOSA_ADD_deq_flatten' in line:
                    fsm_tmpl = '        Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> tmp_read_{b} = ' + \
                               'sToHaddocBuffer_chan{b}.read();\n        cur_line_bit_cnt[{b}] = ' + \
                               'flattenAxisBuffer(tmp_read_{b}, combined_input[{b}], hangover_bits[{b}], ' + \
                               'hangover_bits_valid_bits[{b}]);\n'
                    outline = ''
                    for b in range(0, self.in_dims[1]):
                        outline += fsm_tmpl.format(b=b)
                # elif 'DOSA_ADD_from_haddoc_buffer_param_decl' in line:
                #    outline = ''
                #    for b in range(0, self.out_dims[1]):
                #        outline += '    ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> *chan{b}_buffer_0,\n' \
                #                   '    ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> *chan{b}_buffer_1,\n' \
                #            .format(b=b)
                # elif 'DOSA_ADD_from_haddoc_buffer_clear' in line:
                #     outline = ''
                #     for b in range(0, self.out_dims[1]):
                #         outline += '        chan{b}_buffer_0[i] = 0x0;\n' \
                #                    '        chan{b}_buffer_1[i] = 0x0;\n' \
                #             .format(b=b)
                elif 'DOSA_ADD_from_haddoc_stream_param_decl' in line:
                    outline = ''
                    for b in range(0, self.out_dims[1]):
                        outline += '  stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> >    &sFromHaddocBuffer_chan{b},\n' \
                            .format(b=b)
                elif 'DOSA_ADD_from_haddoc_stream_full_check' in line:
                    outline = '         '
                    for b in range(0, self.out_dims[1]):
                        outline += ' && !sFromHaddocBuffer_chan{b}.full()' \
                            .format(b=b)
                    outline += '\n'
                elif 'DOSA_ADD_from_haddoc_stream_write' in line:
                    outline = ''
                    for b in range(0, self.out_dims[1]):
                        outline += ('        sFromHaddocBuffer_chan{b}.write((ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH>) ' +
                                    '(input_data >> {b} * DOSA_HADDOC_GENERAL_BITWIDTH));\n') \
                            .format(b=b)
                elif 'DOSA_ADD_output_stream_param_decl' in line:
                    outline = ''
                    for b in range(0, self.out_dims[1]):
                        outline += '    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >    &sOutBuffer_chan{b},\n' \
                            .format(b=b)
                # elif 'DOSA_ADD_from_haddoc_stream_drain' in line:
                #     fsm_tmpl = ('    if(!sFromHaddocBuffer_chan{b}.empty())\n    {{\n' +
                #                 '      sFromHaddocBuffer_chan{b}.read();\n' +
                #                 '      not_empty = true;\n    }}\n')
                #     outline = ''
                #     for b in range(0, self.out_dims[1]):
                #         outline += fsm_tmpl.format(b=b)
                # elif 'DOSA_ADD_widen' in line:
                #     fsm_tmpl = ('    if(!sFromHaddocBuffer_chan{b}.empty() && !sOutBuffer_chan{b}.full())\n    {{\n' +
                #                 '      genericWiden(sFromHaddocBuffer_chan{b}, sOutBuffer_chan{b}, ' +
                #                 'current_frame_bit_cnt[{b}], current_line_read_pnt[{b}], hangover_store[{b}],' +
                #                 ' hangover_store_valid_bits[{b}]);\n    }}\n')
                #     outline = ''
                #     for b in range(0, self.out_dims[1]):
                #         outline += fsm_tmpl.format(b=b)
                elif 'DOSA_ADD_pFromHaddocWiden_X_declaration' in line:
                    template_lines = Path(os.path.join(__filedir__, 'templates/haddoc2_wrapper/src/pFromHaddocWiden_b'
                                                                    '.fstrtmpl')).read_text()
                    outline = ''
                    for b in range(0, self.out_dims[1]):
                        outline += template_lines.format(b=b)
                elif 'DOSA_ADD_out_stream_drain' in line:
                    fsm_tmpl = ('    if(!sOutBuffer_chan{b}.empty())\n    {{\n' +
                                '      sOutBuffer_chan{b}.read();\n' +
                                '      not_empty = true;\n    }}\n')
                    outline = ''
                    for b in range(0, self.out_dims[1]):
                        outline += fsm_tmpl.format(b=b)
                elif 'DOSA_ADD_from_haddoc_deq_buf_read' in line:
                    fsm_tmpl = ('\n    case READ_BUF_{b}:\n      if(!soData.full() && !sOutBuffer_chan{b}.empty())\n' +
                                '      {{\n        tmp_read_0 = sOutBuffer_chan{b}.read();\n' +
                                '        uint32_t bit_read = extractByteCnt(tmp_read_0) * 8;\n' +
                                '        current_frame_bit_cnt += bit_read;\n' +
                                '        if(current_frame_bit_cnt >= HADDOC_OUTPUT_FRAME_BIT_CNT ||' +
                                ' tmp_read_0.getTLast() == 1)\n        {{\n          current_frame_bit_cnt = 0x0;\n' +
                                '          dequeueFSM = READ_BUF_{b1};\n          //check for tlast after each frame\n' +
                                '          if(!DOSA_HADDOC_OUTPUT_BATCH_FLATTEN)\n          {{\n' +
                                '            tmp_read_0.setTLast(0b1);\n          }} else {{\n' +
                                '            tmp_read_0.setTLast(0b0);\n          }}\n        }}\n' +
                                '        soData.write(tmp_read_0);\n      }}\n      break;\n')
                    fsm_tmpl_last = (
                            '\n    case READ_BUF_{b}:\n      if(!soData.full() && !sOutBuffer_chan{b}.empty())\n' +
                            '      {{\n        tmp_read_0 = sOutBuffer_chan{b}.read();\n' +
                            '        uint32_t bit_read = extractByteCnt(tmp_read_0) * 8;\n' +
                            '        current_frame_bit_cnt += bit_read;\n' +
                            '        if(current_frame_bit_cnt >= HADDOC_OUTPUT_FRAME_BIT_CNT ||' +
                            ' tmp_read_0.getTLast() == 1)\n        {{\n          current_frame_bit_cnt = 0x0;\n' +
                            '          dequeueFSM = READ_BUF_0;\n          //in all cases\n' +
                            '          tmp_read_0.setTLast(0b1);\n        }}\n' +
                            '        soData.write(tmp_read_0);\n      }}\n      break;\n')
                    outline = ''
                    for b in range(0, self.out_dims[1] - 1):
                        outline += fsm_tmpl.format(b=b, b1=b + 1)
                    outline += fsm_tmpl_last.format(b=self.out_dims[1] - 1)
                # elif 'DOSA_ADD_from_haddoc_buffer_read' in line:
                #     outline = 'if( current_array_slot_pnt == 0 )\n        {\n'
                #     for b in range(0, self.out_dims[1]):
                #         outline += '          ' + 'chan{b}_buffer_0[current_array_write_pnt] = (' \
                #                                   'ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH>) ' \
                #                                   '(input_data >> {b} * DOSA_HADDOC_GENERAL_BITWIDTH);\n' \
                #             .format(b=b)
                #     outline += '        } else {\n'
                #     for b in range(0, self.out_dims[1]):
                #         outline += '          ' + 'chan{b}_buffer_1[current_array_write_pnt] = (' \
                #                                   'ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH>) ' \
                #                                   '(input_data >> {b} * DOSA_HADDOC_GENERAL_BITWIDTH);\n' \
                #             .format(b=b)
                #     outline += '        }\n'
                # elif 'DOSA_ADD_output_deq_read_switch_case_buff0' in line:
                #     outline = '            switch (cur_channel) {\n'
                #     for b in range(0, self.out_dims[1]):
                #         outline += '              case {b}: nv = chan{b}_buffer_0[cur_read_position]; break;\n'\
                #             .format(b=b)
                #     outline += '            }\n'
                # elif 'DOSA_ADD_output_deq_read_switch_case_buff1' in line:
                #     outline = '            switch (cur_channel) {\n'
                #     for b in range(0, self.out_dims[1]):
                #         outline += '              case {b}: nv = chan{b}_buffer_1[cur_read_position]; break;\n' \
                #             .format(b=b)
                #     outline += '            }\n'
                elif 'DOSA_ADD_haddoc_buffer_instantiation' in line:
                    fsm_tmpl = '  static stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> > sToHaddocBuffer_chan{b} ' + \
                               '("sToHaddocBuffer_chan{b}");\n  #pragma HLS STREAM variable=sToHaddocBuffer_chan{b}   ' + \
                               'depth=cnn_input_frame_size\n'
                    outline = '\n'
                    for b in range(0, self.in_dims[1]):
                        outline += fsm_tmpl.format(b=b)
                    outline += '\n'
                    fsm_tmpl = '  static stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> > ' + \
                               'sFromHaddocBuffer_chan{b} ("sFromHaddocBuffer_chan{b}");\n' + \
                               '  #pragma HLS STREAM variable=sFromHaddocBuffer_chan{b} depth=cnn_output_frame_size\n'
                    fsm_tmpl += '  static stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> > ' + \
                                'sOutBuffer_chan{b} ("sOutBuffer_chan{b}");\n' + \
                                '  #pragma HLS STREAM variable=sOutBuffer_chan{b} depth=cnn_output_frame_size\n'
                    for b in range(0, self.out_dims[1]):
                        outline += fsm_tmpl.format(b=b)
                    # fsm_tmpl = '  static ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> g_chan{b}_buffer_0[' \
                    #            'CNN_OUTPUT_FRAME_SIZE];\n' + \
                    #            '  #pragma HLS ARRAY_PARTITION variable=g_chan{b}_buffer_0 cyclic ' \
                    #            'factor=2*wrapper_output_if_haddoc_words_cnt_ceil\n ' + \
                    #            '  static ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> g_chan{b}_buffer_1[' \
                    #            'CNN_OUTPUT_FRAME_SIZE];\n' + \
                    #            '  #pragma HLS ARRAY_PARTITION variable=g_chan{b}_buffer_1 cyclic ' \
                    #            'factor=2*wrapper_output_if_haddoc_words_cnt_ceil\n '
                    # for b in range(0, self.out_dims[1]):
                    #     outline += fsm_tmpl.format(b=b)
                elif 'DOSA_ADD_toHaddoc_buffer_list' in line:
                    outline = '     '
                    for b in range(0, self.in_dims[1]):
                        outline += ' sToHaddocBuffer_chan{},'.format(b)
                    outline += '\n'
                # elif 'DOSA_ADD_from_haddoc_buffer_list' in line:
                #     outline = '     '
                #     for b in range(0, self.out_dims[1]):
                #         outline += ' g_chan{b}_buffer_0, g_chan{b}_buffer_1,'.format(b=b)
                #     outline += '\n'
                elif 'DOSA_ADD_from_haddoc_stream_list' in line:
                    outline = '     '
                    for b in range(0, self.out_dims[1]):
                        outline += ' sFromHaddocBuffer_chan{b},'.format(b=b)
                    outline += '\n'
                elif 'DOSA_ADD_from_haddoc_out_list' in line:
                    outline = '     '
                    for b in range(0, self.out_dims[1]):
                        outline += ' sOutBuffer_chan{b},'.format(b=b)
                    outline += '\n'
                elif 'DOSA_ADD_pFromHaddocWiden_X_instantiate' in line:
                    outline = ''
                    tmpl = '  pFromHaddocWiden_{b}(sFromHaddocBuffer_chan{b}, sOutBuffer_chan{b});\n'
                    for b in range(0, self.out_dims[1]):
                        outline += tmpl.format(b=b)
                    outline += '\n'
                else:
                    outline = line
                out_file.write(outline)

    def get_tcl_lines_wrapper_inst(self, ip_description='Haddoc2Wrapper instantiation'):
        template_lines = Path(os.path.join(__filedir__, 'templates/create_hls_ip_core.tcl')).read_text()
        new_tcl_lines = template_lines.format(DOSA_FMSTR_DESCR=ip_description, DOSA_FMSTR_MOD_NAME=self.ip_mod_name,
                                              DOSA_FMSTR_IP_NAME=self.ip_name)
        return new_tcl_lines

    def get_wrapper_vhdl_decl_lines(self):
        # we need to do the connections between wrapper and haddoc ourselves
        decl = ('signal s{ip_mod_name}_to_Haddoc_b{block_id}_data  : std_ulogic_vector({haddoc_in_width} downto 0);\n' +
                'signal s{ip_mod_name}_to_Haddoc_b{block_id}_dv    : std_ulogic;\n' +
                'signal s{ip_mod_name}_to_Haddoc_b{block_id}_rdy   : std_ulogic;\n' +
                'signal s{ip_mod_name}_to_Haddoc_b{block_id}_fv    : std_ulogic;\n'
                'signal sHaddoc_b{block_id}_to_{ip_mod_name}_data  : std_ulogic_vector({haddoc_out_width} downto 0);\n' +
                'signal sHaddoc_b{block_id}_to_{ip_mod_name}_dv    : std_ulogic;\n' +
                'signal sHaddoc_b{block_id}_to_{ip_mod_name}_rdy   : std_ulogic;\n' +
                'signal sHaddoc_b{block_id}_to_{ip_mod_name}_fv    : std_ulogic;\n')
        decl += '\n'
        decl += 'signal s{ip_mod_name}_debug    : std_ulogic_vector(63 downto 0);\n'
        decl += '\n'
        decl += ('component cnn_process_b{block_id} is\n' +
                 'generic(\n' +
                 '  BITWIDTH  : integer := GENERAL_BITWIDTH;\n' +
                 '  IMAGE_WIDTH : integer := {first_layer_name}_IMAGE_WIDTH\n' +
                 ');\n' +
                 'port(\n' +
                 '  clk      : in  std_logic;\n' +
                 '  reset_n  : in  std_logic;\n' +
                 '  enable   : in  std_logic;\n' +
                 '  in_data  : in  std_logic_vector(INPUT_BIT_WIDTH-1 downto 0);\n' +
                 '  in_dv    : in  std_logic;\n' +
                 '  in_rdy   : out std_logic;\n' +
                 '  in_fv    : in  std_logic;\n' +
                 '  out_data : out std_logic_vector(OUTPUT_BITWIDTH-1 downto 0);\n' +
                 '  out_dv   : out std_logic;\n' +
                 '  out_rdy  : in  std_logic;\n' +
                 '  out_fv   : out std_logic\n' +
                 '  );\n' +
                 'end component cnn_process_b{block_id};\n')
        decl += '\n'
        decl += ('-- thanks to the fantastic and incredible Vivado HLS...we need vectors with (0 downto 0)\n' +
                 # 'signal s{ip_mod_name}_to_Haddoc_b{block_id}_dv_as_vector    : std_ulogic_vector(0 downto 0);\n' +
                 'signal s{ip_mod_name}_to_Haddoc_b{block_id}_fv_as_vector    : std_ulogic_vector(0 downto 0);\n'
                 # 'signal sHaddoc_b{block_id}_to_{ip_mod_name}_dv_as_vector    : std_ulogic_vector(0 downto 0);\n' +
                 'signal sHaddoc_b{block_id}_to_{ip_mod_name}_fv_as_vector    : std_ulogic_vector(0 downto 0);\n')
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
                 # '    po_haddoc_data_valid_V : OUT STD_LOGIC_VECTOR (0 downto 0);\n' +
                 # '    po_haddoc_data_vector_V : OUT STD_LOGIC_VECTOR ({haddoc_in_width} downto 0);\n' +
                 '    po_haddoc_data_V_V_TDATA  : OUT STD_LOGIC_VECTOR ({haddoc_in_width} downto 0);\n' +
                 '    po_haddoc_data_V_V_TVALID : OUT STD_LOGIC;\n' +
                 '    po_haddoc_data_V_V_TREADY : IN  STD_LOGIC;\n' +
                 '    po_haddoc_frame_valid_V : OUT STD_LOGIC_VECTOR (0 downto 0);\n' +
                 # '    pi_haddoc_data_valid_V : IN STD_LOGIC_VECTOR (0 downto 0);\n' +
                 # '    pi_haddoc_data_vector_V : IN STD_LOGIC_VECTOR ({haddoc_out_width} downto 0);\n' +
                 '    pi_haddoc_data_V_V_TDATA  : IN  STD_LOGIC_VECTOR ({haddoc_out_width} downto 0);\n' +
                 '    pi_haddoc_data_V_V_TVALID : IN  STD_LOGIC;\n' +
                 '    pi_haddoc_data_V_V_TREADY : OUT STD_LOGIC;\n' +
                 '    pi_haddoc_frame_valid_V : IN STD_LOGIC_VECTOR (0 downto 0);\n' +
                 '    debug_out_V : OUT STD_LOGIC_VECTOR (63 downto 0);\n' +
                 '    ap_clk : IN STD_LOGIC;\n' +
                 '    ap_rst_n : IN STD_LOGIC;\n' +
                 # '    pi_haddoc_data_valid_V_ap_vld : IN STD_LOGIC;\n' +
                 # '    pi_haddoc_data_vector_V_ap_vld : IN STD_LOGIC;\n' +
                 # '    po_haddoc_data_valid_V_ap_vld : OUT STD_LOGIC;\n' +
                 '    po_haddoc_frame_valid_V_ap_vld : OUT STD_LOGIC;\n' +
                 # '    po_haddoc_data_vector_V_ap_vld : OUT STD_LOGIC;\n' +
                 '    debug_out_V_ap_vld : OUT STD_LOGIC );\n' +
                 'end component {ip_mod_name};\n')
        ret = decl.format(block_id=self.block_id, ip_mod_name=self.ip_mod_name,
                          if_in_width_tdata=(self.if_in_bitw - 1),
                          if_in_width_tkeep=(int((self.if_in_bitw + 7) / 8) - 1), if_in_width_tlast=0,
                          if_out_width_tdata=(self.if_out_bitw - 1),
                          if_out_width_tkeep=(int((self.if_out_bitw + 7) / 8) - 1),
                          if_out_width_tlast=0, haddoc_in_width=((self.general_bitw * self.in_dims[1]) - 1),
                          haddoc_out_width=((self.general_bitw * self.out_dims[1]) - 1),
                          first_layer_name=self.first_layer_name)
        return ret

    def get_vhdl_inst_tmpl(self):
        decl = ( # 's{ip_mod_name}_to_Haddoc_b{block_id}_dv <= s{ip_mod_name}_to_Haddoc_b{block_id}_dv_as_vector(0);\n' +
                's{ip_mod_name}_to_Haddoc_b{block_id}_fv <= s{ip_mod_name}_to_Haddoc_b{block_id}_fv_as_vector(0);\n' +
                # 'sHaddoc_b{block_id}_to_{ip_mod_name}_dv_as_vector(0) <=  sHaddoc_b{block_id}_to_{ip_mod_name}_dv;\n' +
                'sHaddoc_b{block_id}_to_{ip_mod_name}_fv_as_vector(0) <=  sHaddoc_b{block_id}_to_{ip_mod_name}_fv;\n')
        decl += '\n'
        decl += ('[inst_name]_wrapper: {ip_mod_name}\n' +
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
                 # '    po_haddoc_data_valid_V =>  s{ip_mod_name}_to_Haddoc_b{block_id}_dv_as_vector,\n' +
                 # '    po_haddoc_data_vector_V =>  s{ip_mod_name}_to_Haddoc_b{block_id}_data,\n' +
                 '    po_haddoc_data_V_V_TDATA =>  s{ip_mod_name}_to_Haddoc_b{block_id}_data,\n' +
                 '    po_haddoc_data_V_V_TVALID =>  s{ip_mod_name}_to_Haddoc_b{block_id}_dv,\n' +
                 '    po_haddoc_data_V_V_TREADy =>  s{ip_mod_name}_to_Haddoc_b{block_id}_rdy,\n' +
                 '    po_haddoc_frame_valid_V =>  s{ip_mod_name}_to_Haddoc_b{block_id}_fv_as_vector,\n' +
                 # '    pi_haddoc_data_valid_V =>  sHaddoc_b{block_id}_to_{ip_mod_name}_dv_as_vector,\n' +
                 # '    pi_haddoc_data_vector_V =>  sHaddoc_b{block_id}_to_{ip_mod_name}_data,\n' +
                 '    pi_haddoc_data_V_V_TDATA =>  sHaddoc_b{block_id}_to_{ip_mod_name}_data,\n' +
                 '    pi_haddoc_data_V_V_TVALID =>  sHaddoc_b{block_id}_to_{ip_mod_name}_dv,\n' +
                 '    pi_haddoc_data_V_V_TREADY =>  sHaddoc_b{block_id}_to_{ip_mod_name}_rdy,\n' +
                 '    pi_haddoc_frame_valid_V =>  sHaddoc_b{block_id}_to_{ip_mod_name}_fv_as_vector,\n' +
                 '    debug_out_V =>  s{ip_mod_name}_debug,\n' +
                 '    ap_clk =>  [clk],\n' +
                 '    ap_rst_n =>  [rst_n]\n' +  # no comma
                 # '    pi_haddoc_data_valid_V_ap_vld =>  \'1\' ,\n' +
                 # '    pi_haddoc_data_vector_V_ap_vld => \'1\'\n' +  # no comma
                 # '    po_haddoc_data_valid_V_ap_vld =>  open ,\n' +
                 # '    po_haddoc_frame_valid_V_ap_vld =>  open ,\n' +
                 # '    po_haddoc_data_vector_V_ap_vld =>  open,\n' +
                 # '    debug_out_V_ap_vld =>  \n' +  # no comma
                 ');\n')
        decl += '\n'
        decl += ('[inst_name]: cnn_process_b{block_id}\n' +
                 'generic map (\n' +
                 '  BITWIDTH  => {haddoc_general_bitw},\n' +
                 '  IMAGE_WIDTH => {haddoc_image_width}\n' +  # no comma
                 ')\n' +  # no semicolon
                 'port map (\n' +
                 '  clk      =>  [clk],\n' +
                 '  reset_n  =>  [rst_n],\n' +
                 '  enable   =>  [enable],\n' +
                 '  in_data  =>  s{ip_mod_name}_to_Haddoc_b{block_id}_data,\n' +
                 '  in_dv    =>  s{ip_mod_name}_to_Haddoc_b{block_id}_dv,\n' +
                 '  in_rdy   =>  s{ip_mod_name}_to_Haddoc_b{block_id}_rdy,\n' +
                 '  in_fv    =>  s{ip_mod_name}_to_Haddoc_b{block_id}_fv,\n' +
                 '  out_data =>  sHaddoc_b{block_id}_to_{ip_mod_name}_data,\n' +
                 '  out_dv   =>  sHaddoc_b{block_id}_to_{ip_mod_name}_dv,\n' +
                 '  out_rdy  =>  sHaddoc_b{block_id}_to_{ip_mod_name}_rdy,\n' +
                 '  out_fv   =>  sHaddoc_b{block_id}_to_{ip_mod_name}_fv\n' +  # no comma
                 '  );\n')
        inst = decl.format(block_id=self.block_id, ip_mod_name=self.ip_mod_name, haddoc_general_bitw=self.general_bitw,
                           haddoc_image_width=self.in_dims[2])
        # replace [] with {}
        inst_tmpl = inst.replace('[', '{').replace(']', '}')
        return inst_tmpl

    def get_debug_lines(self):
        signal_lines = [
            's{ip_mod_name}_to_Haddoc_b{block_id}_data'.format(block_id=self.block_id, ip_mod_name=self.ip_mod_name),
            's{ip_mod_name}_to_Haddoc_b{block_id}_dv'.format(block_id=self.block_id, ip_mod_name=self.ip_mod_name),
            's{ip_mod_name}_to_Haddoc_b{block_id}_rdy'.format(block_id=self.block_id, ip_mod_name=self.ip_mod_name),
            's{ip_mod_name}_to_Haddoc_b{block_id}_fv'.format(block_id=self.block_id, ip_mod_name=self.ip_mod_name),
            'sHaddoc_b{block_id}_to_{ip_mod_name}_data'.format(block_id=self.block_id, ip_mod_name=self.ip_mod_name),
            'sHaddoc_b{block_id}_to_{ip_mod_name}_dv'.format(block_id=self.block_id, ip_mod_name=self.ip_mod_name),
            'sHaddoc_b{block_id}_to_{ip_mod_name}_rdy'.format(block_id=self.block_id, ip_mod_name=self.ip_mod_name),
            'sHaddoc_b{block_id}_to_{ip_mod_name}_fv'.format(block_id=self.block_id, ip_mod_name=self.ip_mod_name),
            's{ip_mod_name}_debug'.format(ip_mod_name=self.ip_mod_name)
        ]
        width_lines = [(self.general_bitw * self.in_dims[1]), 1, 1, 1,
                       (self.general_bitw * self.out_dims[1]), 1, 1, 1,
                       64]
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
