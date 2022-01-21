#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Jan 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Haddoc2 Wrapper generation, based on templates
#  *        TODO: here or in operatorSets/lib/haddoc2?
#  *

import os


def generate_haddoc2_wrapper(block_name, in_dims, out_dims, general_bitw, if_in_bitw, if_out_bitw, out_dir_path,
                             wrapper_flatten_op, haddoc_op_cnt):
    filedir = os.path.dirname(os.path.abspath(__file__))
    templ_dir_path = os.path.join(filedir, 'templates/haddoc2_wrapper/')
    # 0. copy 'static' files, dir structure
    os.system('cp {}/run_hls.tcl {}'.format(templ_dir_path, out_dir_path))
    os.system('mkdir -p {}/tb/'.format(out_dir_path))
    os.system('mkdir -p {}/src/'.format(out_dir_path))
    os.system('cp {}/tb/tb_haddoc2_wrapper.cpp {}/tb/'.format(templ_dir_path, out_dir_path))
    # 1. Makefile
    static_skip_lines = [0, 3, 4]
    with open(os.path.join(templ_dir_path, 'Makefile'), 'r') as in_file, \
        open(os.path.join(out_dir_path, 'Makefile'), 'w') as out_file:
        cur_line_nr = 0
        for line in in_file.readlines():
            if cur_line_nr in static_skip_lines or cur_line_nr > static_skip_lines[-1]:
                # do nothing, copy
                outline = line
            else:
                if cur_line_nr == 1:
                    outline = 'ipName =haddoc_wrapper_{}\n'.format(block_name)
                elif cur_line_nr == 2:
                    outline = '\n'
            out_file.write(outline)
            cur_line_nr += 1
    # 2. wrapper.hpp
    with open(os.path.join(templ_dir_path, 'src/haddoc_wrapper.hpp'), 'r') as in_file, \
            open(os.path.join(out_dir_path, 'src/haddoc_wrapper.hpp'), 'w') as out_file:
        for line in in_file.readlines():
            if 'DOSA_ADD_INTERFACE_DEFINES' in line:
                outline = ''
                outline += '#define DOSA_WRAPPER_INPUT_IF_BITWIDTH {}\n'.format(if_in_bitw)
                outline += '#define DOSA_WRAPPER_OUTPUT_IF_BITWIDTH {}\n'.format(if_out_bitw)
                outline += '#define DOSA_HADDOC_GENERAL_BITWIDTH {}\n'.format(general_bitw)
                outline += '#define DOSA_HADDOC_INPUT_CHAN_NUM {}\n'.format(in_dims[1])
                outline += '#define DOSA_HADDOC_OUTPUT_CHAN_NUM {}\n'.format(out_dims[1])
                outline += '#define DOSA_HADDOC_INPUT_FRAME_WIDTH {}\n'.format(in_dims[2])
                outline += '#define DOSA_HADDOC_OUTPUT_FRAME_WIDTH {}\n'.format(out_dims[2])
                flatten_str = 'false'
                if wrapper_flatten_op is not None:
                    flatten_str = 'true'
                outline += '#define DOSA_HADDOC_OUTPUT_BATCH_FLATTEN {}\n'.format(flatten_str)
                outline += '#define DOSA_HADDOC_LAYER_CNT {}\n'.format(haddoc_op_cnt)
                enum_def = 'enum ToHaddocEnqStates {RESET0 = 0, WAIT_DRAIN'
                for b in range(0, in_dims[1]):
                    enum_def += ', FILL_BUF_{}'.format(b)
                enum_def += '};\n'
                outline += enum_def
            else:
                outline = line
            out_file.write(outline)
    # 2. wrapper.cpp
    with open(os.path.join(templ_dir_path, 'src/haddoc_wrapper.cpp'), 'r') as in_file, \
            open(os.path.join(out_dir_path, 'src/haddoc_wrapper.cpp'), 'w') as out_file:
        for line in in_file.readlines():
            if 'DOSA_ADD_toHaddoc_buffer_param_decl' in line:
                outline = ''
                for b in range(0, in_dims[1]):
                    outline += '    stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >    &sToHaddocBuffer_chan{},\n'.format(b)
            elif 'DOSA_ADD_enq_fsm' in line:
                fsm_tmpl = '    case FILL_BUF_{b}:\n      if( !siData.empty() && !sToHaddocBuffer_chan{b}.full() )\n' +\
                           '      {{\n        if(genericEnqState(siData, sToHaddocBuffer_chan{b}, current_frame_bit_cnt,'+\
                           ' hangover_store, hangover_store_valid_bits))\n        {{\n          '+\
                           'enqueueFSM = FILL_BUF_{b1};\n        }}\n      }}\n      break;\n'
                outline = ''
                for b in range(0, in_dims[1]):
                    b1 = b+1
                    if b1 >= in_dims[1]:
                        b1 = 0
                    outline += fsm_tmpl.format(b=b, b1=b1)
            elif 'DOSA_ADD_toHaddoc_deq_buffer_drain' in line:
                fsm_tmpl = '    if( !sToHaddocBuffer_chan{b}.empty() )\n    {{\n      sToHaddocBuffer_chan{b}.read();\n' +\
                           '      one_not_empty = true;\n    }}\n'
                outline = ''
                for b in range(0, in_dims[1]):
                    outline += fsm_tmpl.format(b=b)
            elif 'DOSA_ADD_toHaddoc_deq_if_clause' in line:
                outline = '       '
                for b in range(0, in_dims[1]):
                    outline += ' && !sToHaddocBuffer_chan{}.empty()'.format(b)
                outline += '\n'
            elif 'DOSA_ADD_deq_flatten' in line:
                fsm_tmpl = '        tmp_read_0 = sToHaddocBuffer_chan{b}.read();\n        cur_line_bit_cnt[{b}] = '+\
                           'flattenAxisBuffer(tmp_read_0, combined_input[{b}], hangover_bits[{b}], '+\
                           'hangover_bits_valid_bits[{b}]);\n'
                outline = ''
                for b in range(0, in_dims[1]):
                    outline += fsm_tmpl.format(b=b)
            elif 'DOSA_ADD_haddoc_buffer_instantiation' in line:
                fsm_tmpl = '  static stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> > sToHaddocBuffer_chan{b} '+\
                           '("sToHaddocBuffer_chan{b}");\n  #pragma HLS STREAM variable=sToHaddocBuffer_chan{b}   '+\
                           'depth=input_fifo_depth\n'
                outline = ''
                for b in range(0, in_dims[1]):
                    outline += fsm_tmpl.format(b=b)
            elif 'DOSA_ADD_toHaddoc_buffer_list' in line:
                outline = '     '
                for b in range(0, in_dims[1]):
                    outline += ' sToHaddocBuffer_chan{},'.format(b)
                outline += '\n'
            else:
                outline = line
            out_file.write(outline)
    return 0


