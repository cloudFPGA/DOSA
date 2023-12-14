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
#  *    =============================================
#  *     Created: Dec 2023
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Tools to convert a generated hls4ml kernel to an engine kernel
#  *
#  *

import os
from pathlib import Path


def parse_load_line(line):
    l1 = line.split('load_weights_from_txt<')
    l2 = l1[1].split(',')
    dtype = l2[0]
    l3 = l2[1].split('>(')
    size_1d = int(l3[0])
    variable_name = l3[1]
    return dtype, variable_name, size_1d


def parse_signature_line(line):
    l0 = line.split(' ')
    l1 = [l for l in l0 if len(l) > 0]
    dtype = l1[0]
    l2 = l1[1].split('[')
    variable_name = l2[0]
    size_1d = l2[1].split(']')[0]
    return dtype, variable_name, size_1d


def parse_typdef_line(line):
    l1 = line.split(' ')
    hls_type = l1[1]
    custom_type = l1[2].split(';')[0]
    return custom_type, hls_type


def parse_define_line(line):
    l1 = line.split(' ')
    define_name = l1[1]
    define_value = int(l1[2])
    return define_name, define_value


class Hls4mlToEngineConverter:

    def __init__(self, hls4ml_dir_path, hls4ml_project_name, build_tool):
        self.hls4ml_dir_path = hls4ml_dir_path
        self.hls4ml_project_name = hls4ml_project_name
        self.build_tool = build_tool
        self.detected_channels = None

    def convert_kernel(self):

        # parse custom types
        type_defines = 'defines.h'
        custom_types = {}
        custom_lengths = {}
        with open(os.path.join(self.hls4ml_dir_path, 'firmware', type_defines), 'r') as in_file:
            for line in in_file.readlines():
                if 'typedef' in line:
                    custom_type, hls_type = parse_typdef_line(line)
                    custom_types[custom_type] = hls_type
                elif '#define N' in line:
                    define_name, define_value = parse_define_line(line)
                    custom_lengths[define_name] = define_value

        # update main cpp
        old_name = f'{self.hls4ml_project_name}.cpp'
        new_name = f'{self.hls4ml_project_name}_new.cpp'
        outfile = []
        orig_signature_line = ''
        new_signature_lines = ''
        insert_index = None
        pragma_index = None
        orig_pragma_line = ''
        new_pragma_lines = ''
        channels = []
        with open(os.path.join(self.hls4ml_dir_path, 'firmware', old_name), 'r') as in_file:
            cur_line_nr = 0
            for line in in_file.readlines():
                outfile.append(line)
                if 'input_t input' in line:
                    dtype, variable_name, size_1ds = parse_signature_line(line)
                    try:
                        size_1d = int(size_1ds)
                    except:
                        size_1d = custom_lengths[size_1ds]
                    nd = {'dtype': dtype, 'name': variable_name, 'depth': size_1d, 'hls_type': custom_types[dtype]}
                    channels.append(nd)
                elif 'result_t' in line and insert_index is None:
                    dtype, variable_name, size_1ds = parse_signature_line(line)
                    try:
                        size_1d = int(size_1ds)
                    except:
                        size_1d = custom_lengths[size_1ds]
                    nd = {'dtype': dtype, 'name': variable_name, 'depth': size_1d, 'hls_type': custom_types[dtype]}
                    channels.append(nd)
                    insert_index = cur_line_nr
                    orig_signature_line = line
                elif 'load_weights_from_txt' in line:
                    dtype, variable_name, size_1d = parse_load_line(line)
                    nd = {'dtype': dtype, 'name': variable_name, 'depth': size_1d, 'hls_type': custom_types[dtype]}
                    channels.append(nd)
                    new_line = f'    {dtype} {variable_name}[{size_1d}],\n'
                    new_signature_lines += new_line
                    new_pragma = f'    #pragma HLS ARRAY_RESHAPE variable={variable_name} complete dim=0\n' \
                                 f'    #pragma HLS INTERFACE ap_fifo port={variable_name} name={variable_name}\n'
                    new_pragma_lines += new_pragma
                elif 'pragma HLS PIPELINE' in line and pragma_index is None:
                    pragma_index = cur_line_nr
                    orig_pragma_line = line
                cur_line_nr += 1

        self.detected_channels = channels
        new_signature_lines += orig_signature_line
        outfile[insert_index] = new_signature_lines
        new_pragma_lines += orig_pragma_line
        outfile[pragma_index] = new_pragma_lines

        # with open(os.path.join(self.hls4ml_dir_path, 'firmware', new_name), 'w') as out_file:
        with open(os.path.join(self.hls4ml_dir_path, 'firmware', old_name), 'w') as out_file:
            for outline in outfile:
                out_file.write(outline)

        # update signature of header
        old_name_h = f'{self.hls4ml_project_name}.h'
        new_name_h = f'{self.hls4ml_project_name}_new.h'
        with open(os.path.join(self.hls4ml_dir_path, 'firmware', old_name_h), 'r') as in_file, \
            open(os.path.join(self.hls4ml_dir_path, 'firmware', new_name_h), 'w') as out_file:
            for line in in_file.readlines():
                if 'result_t' in line:
                    outline = new_signature_lines
                else:
                    outline = line
                out_file.write(outline)
        os.system(f"mv {os.path.join(self.hls4ml_dir_path, 'firmware', new_name_h)} "
                  f"{os.path.join(self.hls4ml_dir_path, 'firmware', old_name_h)}")

        old_parameters = f'parameters.h'
        new_parameters = f'parameters_new.h'
        with open(os.path.join(self.hls4ml_dir_path, 'firmware', old_parameters), 'r') as in_file, \
                open(os.path.join(self.hls4ml_dir_path, 'firmware', new_parameters), 'w') as out_file:
            for line in in_file.readlines():
                if '#include "weights' in line:
                    continue
                else:
                    outline = line
                out_file.write(outline)
        os.system(f"mv {os.path.join(self.hls4ml_dir_path, 'firmware', new_parameters)} "
                  f"{os.path.join(self.hls4ml_dir_path, 'firmware', old_parameters)}")

        # filter function_instantiate pragmas from nnet
        os.system(f'/bin/bash -c "/usr/bin/sed -i \'/pragma HLS function_instantiate/d\' '
                  f'{os.path.join(self.hls4ml_dir_path, "firmware", "nnet_utils")}/*.h"')
        # pathlist = Path(os.path.join(self.hls4ml_dir_path, "firmware", "nnet_utils")).glob('*.h')
        # for nnet_file in pathlist:
        #     os.system(f"/usr/bin/sed -i '/pragma HLS function_instantiate/d' {nnet_file}")
        # return channels
