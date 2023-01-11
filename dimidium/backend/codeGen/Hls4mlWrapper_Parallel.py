#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Mar 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Hls4ml Wrapper generation with parallel interface,
#  *        based on templates
#  *
#  *

import os
from pathlib import Path
import math

from dimidium.lib.util import bit_width_to_tkeep
from dimidium.backend.codeGen.WrapperInterfaces import InterfaceVectorFifo

__filedir__ = os.path.dirname(os.path.abspath(__file__))


def _ceil_to_next_byte_bitw(bitw):
    ret = int((bitw * 8 + 7) / 8)
    return ret


class Hls4mlWrapper_Parallel:

    def __init__(self, block_id, in_dims, out_dims, acc_in_bitw, acc_out_bitw, if_in_bitw, if_out_bitw, out_dir_path,
                 layer_cnt):
        self.templ_dir_path = os.path.join(__filedir__, 'templates/hls4ml_wrapper_parallel/')
        self.ip_name = 'hls4ml_wrapper_parallel_b{}'.format(block_id)
        self.ip_mod_name = 'Hls4mlWrapper_parallel_b{}'.format(block_id)
        self.block_id = block_id
        self.in_dims = in_dims
        self.out_dims = out_dims
        # self.acc_in_bitw = _ceil_to_next_byte_bitw(acc_in_bitw)
        # self.acc_out_bitw = _ceil_to_next_byte_bitw(acc_out_bitw)
        if len(in_dims) >= 2:
            self.acc_in_bitw = acc_in_bitw * in_dims[1]
        else:
            self.acc_in_bitw = acc_in_bitw * in_dims[0]
        if len(out_dims) >= 2:
            self.acc_out_bitw = acc_out_bitw * out_dims[1]
        else:
            self.acc_out_bitw = acc_out_bitw * out_dims[0]
        assert acc_out_bitw == acc_in_bitw  # TODO: support different data types per layer
        self.general_bitw = acc_in_bitw
        self.if_in_bitw = if_in_bitw
        self.if_out_bitw = if_out_bitw
        self.out_dir_path = out_dir_path
        self.layer_cnt = layer_cnt
        self.to_accel_fifo = InterfaceVectorFifo('b{}_wrapper_to_accel'.format(block_id), 0, None,
                                                 bitwidth=self.acc_in_bitw)
        self.from_accel_fifo = InterfaceVectorFifo('b{}_accel_to_wrapper'.format(block_id), 0, None,
                                                   bitwidth=self.acc_out_bitw)

    def generate(self):
        # 0. copy 'static' files, dir structure
        os.system('cp {}/run_hls.tcl {}'.format(self.templ_dir_path, self.out_dir_path))
        os.system('mkdir -p {}/tb/'.format(self.out_dir_path))
        os.system('mkdir -p {}/src/'.format(self.out_dir_path))
        os.system('cp {}/tb/tb_hls4ml_parallel_wrapper.cpp {}/tb/'.format(self.templ_dir_path, self.out_dir_path))
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
        with open(os.path.join(self.templ_dir_path, 'src/hls4ml_parallel_wrapper.hpp'), 'r') as in_file, \
                open(os.path.join(self.out_dir_path, 'src/hls4ml_parallel_wrapper.hpp'), 'w') as out_file:
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
                    assert (len(self.in_dims) == 2) or (len(self.in_dims) == 4)
                    assert (len(self.out_dims) == 2) or (len(self.out_dims) == 4)
                    outline = ''
                    outline += '#define DOSA_WRAPPER_INPUT_IF_BITWIDTH {}\n'.format(self.if_in_bitw)
                    outline += '#define DOSA_WRAPPER_OUTPUT_IF_BITWIDTH {}\n'.format(self.if_out_bitw)
                    outline += '#define DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH {}\n'.format(self.general_bitw)
                    outline += '#define DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP {}\n'.format(tkeep_general)
                    outline += '#define DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP_WIDTH {}\n'.format(tkeep_width)
                    if len(self.in_dims) == 4:
                        outline += '#define DOSA_HLS4ML_PARALLEL_INPUT_CHAN_NUM {}\n'.format(self.in_dims[1])
                    else:
                        outline += '#define DOSA_HLS4ML_PARALLEL_INPUT_CHAN_NUM {}\n'.format(1)
                    if len(self.out_dims) == 4:
                        outline += '#define DOSA_HLS4ML_PARALLEL_OUTPUT_CHAN_NUM {}\n'.format(self.out_dims[1])
                    else:
                        outline += '#define DOSA_HLS4ML_PARALLEL_OUTPUT_CHAN_NUM {}\n'.format(1)
                    frame_width = self.in_dims[1]
                    if len(self.in_dims) == 4:
                        frame_width = self.in_dims[2] * self.in_dims[3]
                    # outline += '#define DOSA_HLS4ML_PARALLEL_INPUT_FRAME_WIDTH {}\n'.format(frame_width)
                    outline += '#define CNN_INPUT_FRAME_SIZE {}\n'.format(frame_width)
                    frame_width = self.out_dims[1]
                    if len(self.out_dims) == 4:
                        frame_width = self.out_dims[2] * self.out_dims[3]
                    # outline += '#define DOSA_HLS4ML_PARALLEL_OUTPUT_FRAME_WIDTH {}\n'.format(frame_width)
                    outline += '#define CNN_OUTPUT_FRAME_SIZE {}\n'.format(frame_width)
                    flatten_str = 'false'
                    # if self.wrapper_flatten_op is not None:
                    if len(self.out_dims) != 4:
                        flatten_str = 'true'
                    outline += '#define DOSA_HLS4ML_PARALLEL_OUTPUT_BATCH_FLATTEN {}\n'.format(flatten_str)
                    outline += '#define DOSA_HLS4ML_PARALLEL_LAYER_CNT {}\n'.format(self.layer_cnt)
                    # outline += '#define DOSA_HLS4ML_PARALLEL_VALID_WAIT_CNT {}\n'.format(self.initial_delay)
                    # TODO
                    outline += '#define DOSA_HLS4ML_PARALLEL_VALID_WAIT_CNT {}\n'.format(0)
                    enum_def = 'enum Tohls4ml_parallelEnqStates {RESET0 = 0, WAIT_DRAIN'
                    for b in range(0, self.in_dims[1]):
                        enum_def += ', FILL_BUF_{}'.format(b)
                    enum_def += '};\n'
                    outline += enum_def
                    enum_def = 'enum Fromhls4ml_parallelDeqStates {RESET1 = 0'
                    for b in range(0, self.out_dims[1]):
                        enum_def += ', READ_BUF_{}'.format(b)
                    enum_def += '};\n'
                    outline += enum_def
                else:
                    outline = line
                out_file.write(outline)
        # 3. wrapper.cpp
        with open(os.path.join(self.templ_dir_path, 'src/hls4ml_parallel_wrapper.cpp'), 'r') as in_file, \
                open(os.path.join(self.out_dir_path, 'src/hls4ml_parallel_wrapper.cpp'), 'w') as out_file:
            skip_line = False
            for line in in_file.readlines():
                if skip_line:
                    skip_line = False
                    continue
                if 'DOSA_ADD_ip_name_BELOW' in line:
                    outline = 'void {}(\n'.format(self.ip_name)
                    # skip next line
                    skip_line = True
                elif 'DOSA_ADD_tohls4ml_parallel_buffer_param_decl' in line:
                    outline = ''
                    for b in range(0, self.in_dims[1]):
                        outline += '    stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >    &sTohls4ml_parallelBuffer_chan{},\n'.format(
                            b)
                elif 'DOSA_ADD_demux_fsm' in line:
                    # if len(self.in_dims[1]) > 1:
                    # TODO
                    if self.in_dims[1] > 1:
                        fsm_tmpl = '    case FILL_BUF_{b}:\n' + \
                                   '      if( (!siData.empty() || hangover_present) && !sTohls4ml_parallelBuffer_chan{b}.full())\n' + \
                                   '      {{\n' + \
                                   '          if(hangover_present)\n' + \
                                   '          {{\n' + \
                                   '              tmp_read_0 = hangover_axis;\n' + \
                                   '          }} else {{\n' + \
                                   '              tmp_read_0 = siData.read();\n' + \
                                   '              new_bytes_cnt = extractByteCnt(tmp_read_0);\n' + \
                                   '        }}\n' + \
                                   '        hangover_present = false;\n' + \
                                   '        if((current_frame_byte_cnt + new_bytes_cnt) >= HLS4ML_PARALLEL_INPUT_FRAME_BYTE_CNT)\n' + \
                                   '        {{\n' + \
                                   '          uint32_t bytes_to_this_frame = HLS4ML_PARALLEL_INPUT_FRAME_BYTE_CNT - current_frame_byte_cnt;\n' + \
                                   '          int32_t bytes_to_next_frame = new_bytes_cnt - bytes_to_this_frame;\n' + \
                                   '          ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH> cur_input = tmp_read_0.getTData();\n' + \
                                   '          ap_uint<(DOSA_WRAPPER_INPUT_IF_BITWIDTH+7)/8> cur_tkeep = tmp_read_0.getTKeep();\n' + \
                                   '          ap_uint<(DOSA_WRAPPER_INPUT_IF_BITWIDTH+7)/8> this_tkeep = 0x0;\n' + \
                                   '          ap_uint<(DOSA_WRAPPER_INPUT_IF_BITWIDTH+7)/8> next_tkeep = 0x0;\n' + \
                                   '          for(uint32_t i = 0; i < WRAPPER_INPUT_IF_BYTES; i++)\n' + \
                                   '          {{\n' + \
                                   '            ap_uint<1> cur_tkeep_bit = (ap_uint<1>) (cur_tkeep >> i);\n' + \
                                   '            if(i < bytes_to_this_frame)\n' + \
                                   '            {{\n' + \
                                   '              this_tkeep |= ((ap_uint<(DOSA_WRAPPER_INPUT_IF_BITWIDTH+7)/8>) cur_tkeep_bit) << i;\n' + \
                                   '            }} else {{\n' + \
                                   '              next_tkeep |= ((ap_uint<(DOSA_WRAPPER_INPUT_IF_BITWIDTH+7)/8>) cur_tkeep_bit) << (i - bytes_to_this_frame);\n' + \
                                   '            }}\n' + \
                                   '          }}\n' + \
                                   '          Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> tmp_write_this =  Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH>(cur_input, this_tkeep, 0);\n' + \
                                   '          Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> tmp_write_next =  Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH>(cur_input >> (bytes_to_this_frame*8), next_tkeep, 0);\n' + \
                                   '          sTohls4ml_parallelBuffer_chan{b}.write(tmp_write_this);\n' + \
                                   '          if(bytes_to_next_frame > 0)\n' + \
                                   '          {{\n' + \
                                   '               hangover_present = true;\n' + \
                                   '               hangover_axis = tmp_write_next;\n' + \
                                   '               new_bytes_cnt = bytes_to_next_frame;\n' + \
                                   '          }}\n' + \
                                   '          current_frame_byte_cnt = 0x0;\n' + \
                                   '          enqueueFSM = FILL_BUF_{b1};\n' + \
                                   '        }} else {{\n' + \
                                   '          current_frame_byte_cnt += new_bytes_cnt;\n' + \
                                   '          tmp_read_0.setTLast(0);\n' + \
                                   '          sTohls4ml_parallelBuffer_chan{b}.write(tmp_read_0);\n' + \
                                   '        }}\n' + \
                                   '      }}\n' + \
                                   '      break;\n'
                        outline = ''
                        for b in range(0, self.in_dims[1]):
                            b1 = b + 1
                            if b1 >= self.in_dims[1]:
                                b1 = 0
                            outline += fsm_tmpl.format(b=b, b1=b1)
                    else:
                        outline = '    case FILL_BUF_0:\n' + \
                                  '      if( !siData.empty() && !sTohls4ml_parallelBuffer_chan0.full() )\n' + \
                                  '      {\n' + \
                                  '        Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> tmp_read_0 = siData.read();\n' + \
                                  '        tmp_read_0.setTLast(0);\n' + \
                                  '        sTohls4ml_parallelBuffer_chan.write(tmp_read_0);\n' + \
                                  '      }\n' + \
                                  '      break;\n'
                elif 'DOSA_ADD_pTohls4ml_parallelNarrow_X_declaration' in line:
                    template_lines = Path(
                        os.path.join(__filedir__, 'templates/hls4ml_wrapper_parallel/src/pToAccelNarrow_b'
                                                  '.fstrtmpl')).read_text()
                    outline = ''
                    for b in range(0, self.in_dims[1]):
                        outline += template_lines.format(b=b)
                elif 'DOSA_ADD_tohls4ml_parallel_deq_buffer_drain' in line:
                    fsm_tmpl = '    if( !sTohls4ml_parallelBuffer_chan{b}.empty() )\n    {{\n      sTohls4ml_parallelBuffer_chan{b}.read();\n' + \
                               '      one_not_empty = true;\n    }}\n'
                    outline = ''
                    for b in range(0, self.in_dims[1]):
                        outline += fsm_tmpl.format(b=b)
                elif 'DOSA_ADD_tohls4ml_parallel_pixelChain_param_decl' in line:
                    outline = ''
                    for b in range(0, self.in_dims[1]):
                        outline += '  stream<ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> >    &sTohls4ml_parallelPixelChain_chan{},\n'.format(
                            b)
                elif 'DOSA_ADD_tohls4ml_parallel_deq_pixelChain_drain' in line:
                    fsm_tmpl = '    if( !sTohls4ml_parallelPixelChain_chan{b}.empty() )\n    {{\n      sTohls4ml_parallelPixelChain_chan{b}.read();\n' + \
                               '      one_not_empty = true;\n    }}\n'
                    outline = ''
                    for b in range(0, self.in_dims[1]):
                        outline += fsm_tmpl.format(b=b)
                elif 'DOSA_ADD_tohls4ml_parallel_deq_if_clause' in line:
                    outline = '       '
                    for b in range(0, self.in_dims[1]):
                        # outline += ' && !sTohls4ml_parallelBuffer_chan{}.empty()'.format(b)
                        outline += ' && !sTohls4ml_parallelPixelChain_chan{}.empty()'.format(b)
                    outline += '\n'
                elif 'DOSA_ADD_deq_flatten' in line:
                    fsm_tmpl = '        pixel_array[{b}] = sTohls4ml_parallelPixelChain_chan{b}.read();\n'
                    outline = ''
                    for b in range(0, self.in_dims[1]):
                        outline += fsm_tmpl.format(b=b)
                elif 'DOSA_ADD_from_hls4ml_parallel_stream_param_decl' in line:
                    outline = ''
                    for b in range(0, self.out_dims[1]):
                        outline += '  stream<ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> >    &sFromhls4ml_parallelBuffer_chan{b},\n' \
                            .format(b=b)
                elif 'DOSA_ADD_from_hls4ml_parallel_stream_full_check' in line:
                    outline = '         '
                    for b in range(0, self.out_dims[1]):
                        outline += ' && !sFromhls4ml_parallelBuffer_chan{b}.full()' \
                            .format(b=b)
                    outline += '\n'
                elif 'DOSA_ADD_from_hls4ml_parallel_stream_write' in line:
                    outline = ''
                    for b in range(0, self.out_dims[1]):
                        outline += (
                                '        sFromhls4ml_parallelBuffer_chan{b}.write((ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH>) ' +
                                '(input_data >> {b} * DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH));\n') \
                            .format(b=b)
                elif 'DOSA_ADD_output_stream_param_decl' in line:
                    outline = ''
                    for b in range(0, self.out_dims[1]):
                        outline += '    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >    &sOutBuffer_chan{b},\n' \
                            .format(b=b)
                elif 'DOSA_ADD_pFromhls4ml_parallelWiden_X_declaration' in line:
                    template_lines = Path(
                        os.path.join(__filedir__, 'templates/hls4ml_wrapper_parallel/src/pFromAccelWiden_b'
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
                elif 'DOSA_ADD_from_hls4ml_parallel_deq_buf_read' in line:
                    fsm_tmpl = ('\n    case READ_BUF_{b}:\n      if(!soData.full() && !sOutBuffer_chan{b}.empty())\n' +
                                '      {{\n        tmp_read_0 = sOutBuffer_chan{b}.read();\n' +
                                '        uint32_t bit_read = extractByteCnt(tmp_read_0) * 8;\n' +
                                '        current_frame_bit_cnt += bit_read;\n' +
                                '        if(current_frame_bit_cnt >= HLS4ML_PARALLEL_OUTPUT_FRAME_BIT_CNT ||' +
                                ' tmp_read_0.getTLast() == 1)\n        {{\n          current_frame_bit_cnt = 0x0;\n' +
                                '          dequeueFSM = READ_BUF_{b1};\n          //check for tlast after each frame\n' +
                                '          if(!DOSA_HLS4ML_PARALLEL_OUTPUT_BATCH_FLATTEN)\n          {{\n' +
                                '            tmp_read_0.setTLast(0b1);\n          }} else {{\n' +
                                '            tmp_read_0.setTLast(0b0);\n          }}\n        }}\n' +
                                '        soData.write(tmp_read_0);\n      }}\n      break;\n')
                    fsm_tmpl_last = (
                            '\n    case READ_BUF_{b}:\n      if(!soData.full() && !sOutBuffer_chan{b}.empty())\n' +
                            '      {{\n        tmp_read_0 = sOutBuffer_chan{b}.read();\n' +
                            '        uint32_t bit_read = extractByteCnt(tmp_read_0) * 8;\n' +
                            '        current_frame_bit_cnt += bit_read;\n' +
                            '        if(current_frame_bit_cnt >= HLS4ML_PARALLEL_OUTPUT_FRAME_BIT_CNT ||' +
                            ' tmp_read_0.getTLast() == 1)\n        {{\n          current_frame_bit_cnt = 0x0;\n' +
                            '          dequeueFSM = READ_BUF_0;\n          //in all cases\n' +
                            '          tmp_read_0.setTLast(0b1);\n        }}\n' +
                            '        soData.write(tmp_read_0);\n      }}\n      break;\n')
                    outline = ''
                    for b in range(0, self.out_dims[1] - 1):
                        outline += fsm_tmpl.format(b=b, b1=b + 1)
                    outline += fsm_tmpl_last.format(b=self.out_dims[1] - 1)
                elif 'DOSA_ADD_hls4ml_parallel_buffer_instantiation' in line:
                    fsm_tmpl = '  static stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> > sTohls4ml_parallelBuffer_chan{b} ' + \
                               '("sTohls4ml_parallelBuffer_chan{b}");\n  #pragma HLS STREAM variable=sTohls4ml_parallelBuffer_chan{b}   ' + \
                               'depth=cnn_input_frame_size\n'
                    fsm_tmpl += '  static stream<ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> > ' + \
                                'sTohls4ml_parallelPixelChain_chan{b} ("sTohls4ml_parallelPixelChain_chan{b}");\n' + \
                                '  #pragma HLS STREAM variable=sTohls4ml_parallelPixelChain_chan{b} depth=2*cnn_input_frame_size\n'
                    outline = '\n'
                    for b in range(0, self.in_dims[1]):
                        outline += fsm_tmpl.format(b=b)
                    outline += '\n'
                    fsm_tmpl = '  static stream<ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> > ' + \
                               'sFromhls4ml_parallelBuffer_chan{b} ("sFromhls4ml_parallelBuffer_chan{b}");\n' + \
                               '  #pragma HLS STREAM variable=sFromhls4ml_parallelBuffer_chan{b} depth=cnn_output_frame_size\n'
                    fsm_tmpl += '  static stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> > ' + \
                                'sOutBuffer_chan{b} ("sOutBuffer_chan{b}");\n' + \
                                '  #pragma HLS STREAM variable=sOutBuffer_chan{b} depth=cnn_output_frame_size\n'
                    for b in range(0, self.out_dims[1]):
                        outline += fsm_tmpl.format(b=b)
                elif 'DOSA_ADD_tohls4ml_parallel_buffer_list' in line:
                    outline = '     '
                    for b in range(0, self.in_dims[1]):
                        outline += ' sTohls4ml_parallelBuffer_chan{},'.format(b)
                    outline += '\n'
                elif 'DOSA_ADD_pTohls4ml_parallelNarrow_X_instantiate' in line:
                    outline = ''
                    tmpl = '  pToAccelNarrow_{b}(sTohls4ml_parallelBuffer_chan{b}, sTohls4ml_parallelPixelChain_chan{b});\n'
                    for b in range(0, self.in_dims[1]):
                        outline += tmpl.format(b=b)
                    outline += '\n'
                elif 'DOSA_ADD_tohls4ml_parallel_pixelChain_list' in line:
                    outline = '     '
                    for b in range(0, self.in_dims[1]):
                        outline += ' sTohls4ml_parallelPixelChain_chan{b},'.format(b=b)
                    outline += '\n'
                elif 'DOSA_ADD_from_hls4ml_parallel_stream_list' in line:
                    outline = '     '
                    for b in range(0, self.out_dims[1]):
                        outline += ' sFromhls4ml_parallelBuffer_chan{b},'.format(b=b)
                    outline += '\n'
                elif 'DOSA_ADD_from_hls4ml_parallel_out_list' in line:
                    outline = '     '
                    for b in range(0, self.out_dims[1]):
                        outline += ' sOutBuffer_chan{b},'.format(b=b)
                    outline += '\n'
                elif 'DOSA_ADD_pFromhls4ml_parallelWiden_X_instantiate' in line:
                    outline = ''
                    tmpl = '  pFromAccelWiden_{b}(sFromhls4ml_parallelBuffer_chan{b}, sOutBuffer_chan{b});\n'
                    for b in range(0, self.out_dims[1]):
                        outline += tmpl.format(b=b)
                    outline += '\n'
                else:
                    outline = line
                out_file.write(outline)

    def get_tcl_lines_wrapper_inst(self, ip_description='Hls4mlWrapper instantiation'):
        template_lines = Path(os.path.join(__filedir__, 'templates/create_hls_ip_core.tcl')).read_text()
        new_tcl_lines = template_lines.format(DOSA_FMSTR_DESCR=ip_description, DOSA_FMSTR_MOD_NAME=self.ip_mod_name,
                                              DOSA_FMSTR_IP_NAME=self.ip_name)
        # also adding hls4ml ArchBlock export
        new_tcl_lines += '\n'
        new_tcl_lines += template_lines.format(DOSA_FMSTR_DESCR='Hls4ml instantiation',
                                               # both 'names' must be different, apparently...
                                               DOSA_FMSTR_MOD_NAME='HLS4ML_ArchBlock_{}'.format(self.block_id),
                                               DOSA_FMSTR_IP_NAME='ArchBlock_{}'.format(self.block_id))
        new_tcl_lines += '\n'
        # and adding connecting FIFO
        new_tcl_lines += self.to_accel_fifo.get_tcl_lines()
        new_tcl_lines += '\n'
        new_tcl_lines += self.from_accel_fifo.get_tcl_lines()
        return new_tcl_lines

    def get_wrapper_vhdl_decl_lines(self):
        # we need to do the connections between wrapper and hls4ml_parallel ourselves
        decl = ('component HLS4ML_ArchBlock_{block_id} is\n' +
                'port (\n' +
                '    ap_clk : IN STD_LOGIC;\n' +
                '    ap_rst : IN STD_LOGIC;\n' +
                '    input_0_V_dout : IN STD_LOGIC_VECTOR ({acc_inw_sub} downto 0);\n' +
                '    input_0_V_empty_n : IN STD_LOGIC;\n' +
                '    input_0_V_read : OUT STD_LOGIC;\n' +
                '    output_0_V_din : OUT STD_LOGIC_VECTOR ({acc_outw_sub} downto 0);\n' +
                '    output_0_V_full_n : IN STD_LOGIC;\n' +
                '    output_0_V_write : OUT STD_LOGIC;\n' +
                '    const_size_in_1 : OUT STD_LOGIC_VECTOR (15 downto 0);\n' +
                '    const_size_in_1_ap_vld : OUT STD_LOGIC;\n' +
                '    const_size_out_1 : OUT STD_LOGIC_VECTOR (15 downto 0);\n' +
                '    const_size_out_1_ap_vld : OUT STD_LOGIC\n' +  # no ; at the end
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
                 '    po_hls4ml_parallel_data_V_V_din : OUT STD_LOGIC_VECTOR ({acc_inw_sub} downto 0);\n' +
                 '    po_hls4ml_parallel_data_V_V_full_n : IN STD_LOGIC;\n' +
                 '    po_hls4ml_parallel_data_V_V_write : OUT STD_LOGIC;\n' +
                 '    pi_hls4ml_parallel_data_V_V_dout : IN STD_LOGIC_VECTOR ({acc_outw_sub} downto 0);\n' +
                 '    pi_hls4ml_parallel_data_V_V_empty_n : IN STD_LOGIC;\n' +
                 '    pi_hls4ml_parallel_data_V_V_read : OUT STD_LOGIC;\n' +
                 '    debug_out_V : OUT STD_LOGIC_VECTOR (63 downto 0);\n' +
                 '    ap_clk : IN STD_LOGIC;\n' +
                 '    ap_rst : IN STD_LOGIC;\n' +  # this time reset...why ever
                 '    debug_out_V_ap_vld : OUT STD_LOGIC);\n' +
                 'end component {ip_mod_name};\n')
        decl += '\n'
        decl += 'signal s{ip_mod_name}_debug    : std_ulogic_vector(63 downto 0);\n'
        decl += '\n'
        decl_filled = decl.format(block_id=self.block_id, ip_mod_name=self.ip_mod_name,
                                  if_in_width_tdata=(self.if_in_bitw - 1),
                                  if_in_width_tkeep=(int((self.if_in_bitw + 7) / 8) - 1), if_in_width_tlast=0,
                                  if_out_width_tdata=(self.if_out_bitw - 1),
                                  if_out_width_tkeep=(int((self.if_out_bitw + 7) / 8) - 1),
                                  if_out_width_tlast=0,
                                  acc_inw_sub=(self.acc_in_bitw - 1), acc_outw_sub=(self.acc_out_bitw - 1))
        ret = self.to_accel_fifo.get_vhdl_signal_declaration()
        ret += self.to_accel_fifo.get_vhdl_entity_declaration()
        ret += self.from_accel_fifo.get_vhdl_signal_declaration()
        ret += self.from_accel_fifo.get_vhdl_entity_declaration()
        ret += decl_filled
        return ret

    def get_vhdl_inst_tmpl(self):
        global_map_dict = {}
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
                '    po_hls4ml_parallel_data_V_V_din     =>  {out_sig_0},\n' +
                '    po_hls4ml_parallel_data_V_V_full_n  =>  {out_sig_1n},\n' +
                '    po_hls4ml_parallel_data_V_V_write   =>  {out_sig_2},\n' +
                '    pi_hls4ml_parallel_data_V_V_dout    =>  {in_sig_0},\n' +
                '    pi_hls4ml_parallel_data_V_V_empty_n =>  {in_sig_1n},\n' +
                '    pi_hls4ml_parallel_data_V_V_read    =>  {in_sig_2},\n' +
                '    debug_out_V =>  s{ip_mod_name}_debug,\n' +
                # '    debug_out_V =>  open,\n' +
                '    ap_clk =>  [clk],\n' +
                '    ap_rst =>  [rst]\n' +  # this time not rst_n...
                # '    debug_out_V_ap_vld =>  \n' +  # no comma
                ');\n')
        decl += '\n'
        our_signals = self.to_accel_fifo.get_vhdl_signal_dict()
        inst_tmpl = self.to_accel_fifo.get_vhdl_entity_inst_tmpl()
        map_dict = {'in_sig_0': our_signals['to_signals']['0'],
                    'in_sig_1_n': our_signals['to_signals']['1_n'],
                    'in_sig_1': our_signals['to_signals']['1'],
                    'in_sig_2': our_signals['to_signals']['2'],
                    'out_sig_0': our_signals['from_signals']['0'],
                    'out_sig_1_n': our_signals['from_signals']['1_n'],
                    'out_sig_1': our_signals['from_signals']['1'],
                    'out_sig_2': our_signals['from_signals']['2'],
                    'inst_name': self.to_accel_fifo.name + '_inst',
                    'clk': '[clk]',
                    'rst': '[rst]',
                    'rst_n': '[rst_n]',
                    'enable': '[enable]'
                    }
        global_map_dict['out_sig_0'] = map_dict['in_sig_0']
        global_map_dict['out_sig_1'] = map_dict['in_sig_1']
        global_map_dict['out_sig_1n'] = map_dict['in_sig_1_n']
        global_map_dict['out_sig_2'] = map_dict['in_sig_2']
        global_map_dict['a_in_sig_0'] = map_dict['out_sig_0']
        global_map_dict['a_in_sig_1'] = map_dict['out_sig_1']
        global_map_dict['a_in_sig_1n'] = map_dict['out_sig_1_n']
        global_map_dict['a_in_sig_2'] = map_dict['out_sig_2']
        decl += inst_tmpl.format_map(map_dict)
        our_signals = self.from_accel_fifo.get_vhdl_signal_dict()
        inst_tmpl = self.from_accel_fifo.get_vhdl_entity_inst_tmpl()
        map_dict = {'in_sig_0': our_signals['to_signals']['0'],
                    'in_sig_1_n': our_signals['to_signals']['1_n'],
                    'in_sig_1': our_signals['to_signals']['1'],
                    'in_sig_2': our_signals['to_signals']['2'],
                    'out_sig_0': our_signals['from_signals']['0'],
                    'out_sig_1_n': our_signals['from_signals']['1_n'],
                    'out_sig_1': our_signals['from_signals']['1'],
                    'out_sig_2': our_signals['from_signals']['2'],
                    'inst_name': self.from_accel_fifo.name + '_inst',
                    'clk': '[clk]',
                    'rst': '[rst]',
                    'rst_n': '[rst_n]',
                    'enable': '[enable]'
                    }
        global_map_dict['in_sig_0'] = map_dict['out_sig_0']
        global_map_dict['in_sig_1'] = map_dict['out_sig_1']
        global_map_dict['in_sig_1n'] = map_dict['out_sig_1_n']
        global_map_dict['in_sig_2'] = map_dict['out_sig_2']
        global_map_dict['a_out_sig_0'] = map_dict['in_sig_0']
        global_map_dict['a_out_sig_1'] = map_dict['in_sig_1']
        global_map_dict['a_out_sig_1n'] = map_dict['in_sig_1_n']
        global_map_dict['a_out_sig_2'] = map_dict['in_sig_2']
        decl += inst_tmpl.format_map(map_dict)
        decl += '\n'
        decl += ('[inst_name]: HLS4ML_ArchBlock_{block_id}\n' +
                 'port map (\n' +
                 '    input_0_V_dout    => {a_in_sig_0},\n' +
                 '    input_0_V_empty_n => {a_in_sig_1n},\n' +
                 '    input_0_V_read    => {a_in_sig_2},\n' +
                 '    output_0_V_din    => {a_out_sig_0},\n' +
                 '    output_0_V_full_n => {a_out_sig_1n},\n' +
                 '    output_0_V_write  => {a_out_sig_2},\n' +
                 # '    const_size_in_1 => open,\n' +
                 # '    const_size_out_1 => open,\n' +
                 # '    const_size_in_1_ap_vld => open,\n' +
                 # '    const_size_out_1_ap_vld => open,\n' +
                 '    ap_clk => [clk],\n' +
                 # '    ap_start => [enable],\n' +
                 '    ap_rst => [rst]\n' +  # no comma
                 # '    ap_done => open,\n' +
                 # '    ap_ready => open,\n' +
                 # '    ap_idle => open\n' +  # no comma
                 '  );\n')
        global_map_dict['block_id'] = self.block_id
        global_map_dict['ip_mod_name'] = self.ip_mod_name
        # inst = decl.format(block_id=self.block_id, ip_mod_name=self.ip_mod_name)
        inst = decl.format_map(global_map_dict)
        # replace [] with {}
        inst_tmpl = inst.replace('[', '{').replace(']', '}')
        return inst_tmpl

    def get_debug_lines(self):
        signal_lines = [
            's{ip_mod_name}_debug'.format(ip_mod_name=self.ip_mod_name)
        ]
        width_lines = [64]
        assert len(signal_lines) == len(width_lines)
        tcl_tmpl_lines = []
        decl_tmpl_lines = []
        inst_tmpl_lines = []
        tcl_1, decl_1, inst_1 = self.to_accel_fifo.get_debug_lines()
        tcl_2, decl_2, inst_2 = self.from_accel_fifo.get_debug_lines()
        tcl_tmpl_lines.extend(tcl_1)
        decl_tmpl_lines.extend(decl_1)
        inst_tmpl_lines.extend(inst_1)
        tcl_tmpl_lines.extend(tcl_2)
        decl_tmpl_lines.extend(decl_2)
        inst_tmpl_lines.extend(inst_2)

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


