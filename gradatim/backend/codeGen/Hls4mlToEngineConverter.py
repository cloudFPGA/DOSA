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


def parse_load_line(line):
    l1 = line.split('load_weights_from_txt<')
    l2 = l1[1].split(',')
    dtype = l2[0]
    l3 = l2[1].split('>(')
    size_1d = int(l3[0])
    variable_name = l3[1]
    return dtype, variable_name, size_1d


class Hls4mlToEngineConverter:

    def __init__(self, hls4ml_dir_path, hls4ml_project_name, build_tool):
        self.hls4ml_dir_path = hls4ml_dir_path
        self.hls4ml_project_name = hls4ml_project_name
        self.build_tool = build_tool

    def convert_kernel(self):

        old_name = f'{self.hls4ml_project_name}.cpp'
        new_name = f'{self.hls4ml_project_name}_new.cpp'
        outfile = []
        orig_signature_line = ''
        new_signature_lines = ''
        insert_index = None
        with open(os.path.join(self.hls4ml_dir_path, old_name), 'r') as in_file:
            cur_line_nr = 0
            for line in in_file.readlines():
                outfile.append(line)
                if 'result_t' in line and insert_index is None:
                    insert_index = cur_line_nr
                    orig_signature_line = line
                elif 'load_weights_from_txt' in line:
                    dtype, variable_name, size_1d = parse_load_line(line)
                    new_line = f'    {dtype} {variable_name}[{size_1d}],\n'
                    new_signature_lines += new_line
                cur_line_nr += 1

        new_signature_lines += orig_signature_line
        outfile[insert_index] = new_signature_lines

        with open(os.path.join(self.hls4ml_dir_path, new_name), 'w') as out_file:
            for outline in out_file:
                out_file.write(outline)

        # TODO: remove include weights from parameters.h