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
#  *     Created: Mar 2023
#  *     Authors: NGL
#  *
#  *     Description:
#  *        ZRLMPI SW multi-node wrapper, based on templates
#  *
#  *
import math
import os
from pathlib import Path

import gradatim.lib.singleton as dosa_singleton
from gradatim.middleend.archGen.CommPlan import CommPlan
from gradatim.backend.codeGen.ZrlmpiSwApp import generate_quantized_node_root

__filedir__ = os.path.dirname(os.path.abspath(__file__))


class ZrlmpiSwMultiNodeApp:

    def __init__(self, node_id, out_dir_path, comm_plan: CommPlan):
        self.node_id = node_id
        self.out_dir_path = out_dir_path
        self.comm_plan = comm_plan
        self.templ_dir_path = os.path.abspath(os.path.join(__filedir__, 'templates/zrlmpi_multi_node_sw/'))
        self.gitmodule_dir_path = os.path.abspath(os.path.join(__filedir__, 'templates/ZRLMPI/'))

    def generate(self):
        # 1. copy ZRLMPI LIB
        os.system('mkdir -p {}/LIB'.format(self.out_dir_path))
        os.system('cp {}/*.?pp {}/LIB/'.format(self.gitmodule_dir_path + '/LIB/SW/', self.out_dir_path))
        os.system('cp {}/*.?pp {}/LIB/'.format(self.gitmodule_dir_path + '/LIB/COMMON/', self.out_dir_path))
        # 2. copy Makefile and python
        os.system('cp {}/Makefile {}/'.format(self.templ_dir_path, self.out_dir_path))
        os.system('cp {}/requirements.txt {}/'.format(self.templ_dir_path, self.out_dir_path))
        tmp_bricks_len = len(self.comm_plan.node.predecessors[-1].bricks)
        out_dims = self.comm_plan.node.predecessors[-1].bricks[tmp_bricks_len - 1].dims.out
        generate_quantized_node_root(f'{self.templ_dir_path}/dosa_root.py', f'{self.out_dir_path}/dosa_root.py',
                                     out_dims)
        # os.system('cp {}/dosa_root.py {}/'.format(self.templ_dir_path, self.out_dir_path))
        # get comm_plan data
        all_comm_instr = self.comm_plan.get_comm_instr()
        # assert len(comm_instr) == 2  # this ist just the root app
        comm_plan_len = len(all_comm_instr)
        # assert comm_plan_len % 2 == 0
        comm_instr_rank_wise = self.comm_plan.get_comm_instr_sorted()
        assert len(comm_instr_rank_wise) > 1
        # repetitions = 1 + dosa_singleton.config.backend.comm_message_pipeline_store
        # comm_plan_one_iteration_length = 2 * repetitions
        # all_iterations = 0
        all_node_0_instr = {}
        all_ranks = []
        total_send_repeat = 0
        total_recv_repeat = 0
        rank_pipeline_store = {}
        rank_min_in_repeat = {}
        rank_min_out_repeat = {}
        for cur_rank in comm_instr_rank_wise.keys():
            all_ranks.append(cur_rank)
            comm_instr = comm_instr_rank_wise[cur_rank]
            cur_send_cmds = []
            cur_recv_cmds = []
            all_send_cmds = []
            all_recv_cmds = []
            rank_send_repeat = 0
            rank_recv_repeat = 0
            last_instr = 'none'
            prog_i = 0
            for ci in comm_instr:
                if ci['instr'] == 'send':
                    if last_instr == 'send':
                        cur_send_cmds.append(ci)
                    else:
                        if len(cur_recv_cmds) > 0:
                            all_recv_cmds.append(cur_recv_cmds)
                            cur_recv_cmds = []
                        cur_send_cmds = [ci]
                        last_instr = 'send'
                    if ci['combine'] is None or ci['combine'] == 'start':
                        if prog_i < self.comm_plan.after_pipeline_full_instr_start:
                            rank_send_repeat += ci['repeat']
                else:
                    # recv_cmds.append(ci)
                    if last_instr == 'recv':
                        cur_recv_cmds.append(ci)
                    else:
                        if len(cur_send_cmds) > 0:
                            all_send_cmds.append(cur_send_cmds)
                            cur_send_cmds = []
                        cur_recv_cmds = [ci]
                        last_instr = 'recv'
                    if ci['combine'] is None or ci['combine'] == 'start':
                        if prog_i < self.comm_plan.after_pipeline_full_instr_start:
                            rank_recv_repeat += ci['repeat']
                prog_i += 1
            if len(cur_send_cmds) > 0:
                all_send_cmds.append(cur_send_cmds)
            if len(cur_recv_cmds) > 0:
                all_recv_cmds.append(cur_recv_cmds)
            node_0_instr = []
            assert len(all_send_cmds) == len(all_recv_cmds)
            for i in range(len(all_send_cmds)):
                node_0_instr.extend(all_send_cmds[i])
                node_0_instr.extend(all_recv_cmds[i])
            if rank_recv_repeat > total_recv_repeat:
                total_recv_repeat = rank_recv_repeat
            if rank_send_repeat > total_send_repeat:
                total_send_repeat = rank_send_repeat
            # empty pipeline store at the end...done by dosa_root
            all_node_0_instr[cur_rank] = node_0_instr
            rank_min_in_repeat[cur_rank] = rank_send_repeat
            rank_min_out_repeat[cur_rank] = rank_recv_repeat
            rank_pipeline_store[cur_rank] = rank_send_repeat - rank_recv_repeat
        pipeline_store = total_send_repeat - total_recv_repeat
        root_rank = all_ranks[0]
        len_fill_instr = self.comm_plan.after_pipeline_full_instr_start//len(all_ranks)
        # 3. generate dosa_infer.cpp
        with open(os.path.join(self.templ_dir_path, 'dosa_infer.cpp'), 'r') as in_file, \
                open(os.path.join(self.out_dir_path, 'dosa_infer.cpp'), 'w') as out_file:
            for line in in_file.readlines():
                if 'DOSA_ADD_mpi_config' in line:
                    outline = ''
                    for cur_rank in all_ranks:
                        node_0_instr = all_node_0_instr[cur_rank]
                        tmpl = '    mpiCommands[{i}]          = {instr};\n' + \
                               '    mpiRanks[{i}]             = {rank};\n' + \
                               '    mpiCounts[{i}]            = {count};\n' + \
                               '    byteCounts[{i}]           = {byte_cnt};\n' + \
                               '    commandRepetitions[{i}]   = {repeat};\n' + \
                               '    saveCurData[{i}]          = {save_cur_data};\n'
                        prog_i = 0
                        my_jump_addr = 0
                        outline += f'  if(rank == {cur_rank})\n  {{\n'
                        # if cur_rank == root_rank:
                        outline += '    //pipeline-FILL part\n'
                        skipped_instr = 0
                        for mi in node_0_instr:
                            # if cur_rank == root_rank or prog_i >= len_fill_instr:
                            instr = 'MPI_INSTR_SEND'
                            if mi['instr'] == 'recv':
                                instr = 'MPI_INSTR_RECV'
                            rank = mi['rank']
                            word_count = int((mi['count'] + 3) / 4)
                            repeat = mi['repeat']
                            save_cur_data = 'false'
                            if mi['instr'] == 'send' and mi['combine'] is not None and mi['combine'] != 'finish':
                                save_cur_data = 'true'
                            if repeat == 0:
                                skipped_instr += 1
                            else:
                                # if cur_rank != root_rank:
                                #     true_i -= len_fill_instr
                                outline += tmpl.format(i=prog_i, instr=instr, rank=rank, count=word_count, repeat=repeat,
                                                       save_cur_data=save_cur_data, byte_cnt=int(mi['count']))
                                prog_i += 1
                            if (prog_i + skipped_instr) == len_fill_instr:
                                outline += '    //pipeline-FULL part\n'
                                my_jump_addr = prog_i
                        # if cur_rank == root_rank:
                        outline += f'    my_prog_length = {prog_i};\n'
                        outline += f'    my_jump_addr = {my_jump_addr};\n'
                        # else:
                        #     outline += f'    my_prog_length = {prog_i - len_fill_instr};\n'
                        outline += '  }\n\n'
                    # add pipeline_depth and batch sizes
                    outline += '  //define batch behaviour\n'
                    for cur_rank in all_ranks:
                        tmpl = '  pipeline_depth[{i}] = {pip_depth};\n' + \
                               '  batch_input_size[{i}] = {min_input_batch};\n' +\
                               '  batch_output_size[{i}] = {min_output_batch};\n'
                        outline += tmpl.format(i=cur_rank, pip_depth=rank_pipeline_store[cur_rank],
                                               min_input_batch=rank_min_in_repeat[cur_rank],
                                               min_output_batch=rank_min_out_repeat[cur_rank])
                    outline += '\n'
                elif 'DOSA_ADD_MPI_barrier' in line:
                    root_rank = all_ranks[0]
                    other_ranks = all_ranks[1:]
                    outline = "  // custom (poor man's...) barrier (because we have only MPI_COMM_WORLD)\n"
                    outline += f'  if(rank == {root_rank})\n  {{\n    int ignore = 1;\n'
                    outline += '    printf("[DOSA:INFO] Trigger barrier...\\n");\n'
                    for otr in other_ranks:
                        outline += f'    MPI_Send(&ignore, 1, MPI_INTEGER, {otr}, rank, MPI_COMM_WORLD);\n'
                    outline += '  } else {\n    int ignore = 0;\n'
                    outline += '    printf("[DOSA:INFO] Waiting for barrier...\\n");\n'
                    outline += f'    MPI_Recv(&ignore, 1, MPI_INTEGER, {root_rank}, rank, MPI_COMM_WORLD, &status);\n'
                    outline += '    printf("[DOSA:INFO]\\t ...done. Start executing MPI commands...\\n");\n'
                    outline += '  }\n'
                else:
                    outline = line
                out_file.write(outline)
        # 4. copy dosa_infer.hpp
        with open(os.path.join(self.templ_dir_path, 'dosa_infer.hpp'), 'r') as in_file, \
                open(os.path.join(self.out_dir_path, 'dosa_infer.hpp'), 'w') as out_file:
            for line in in_file.readlines():
                if 'DOSA_ADD_APP_NODE_DEFINES' in line:
                    tmpl = ('#define DOSA_WRAPPER_MAX_PROG_LENGTH {total_l}\n' +
                            # '#define DOSA_MINIMAL_PROG_LENGTH {min_l}\n' +
                            # '#define DOSA_PIPELINE_STORE_DETPH {pip_store}\n' +
                            # '#define DOSA_MINIMAL_INPUT_NUM {min_in}\n' +
                            # '#define DOSA_MINIMAL_OUTPUT_NUM {min_out}\n' +
                            # '#define DOSA_COMM_PLAN_AFTER_FILL_JUMP {jump}\n' +
                            '#define DOSA_PIPELINE_FULL_BATCH_SIZE {full_pip}\n' +
                            '#define DOSA_MAX_PARALLEL_RANKS {max_ranks}\n')
                    # outline = tmpl.format(total_l=comm_plan_len, min_l=all_iterations)
                    outline = tmpl.format(total_l=comm_plan_len,
                                          # pip_store=pipeline_store,
                                          # min_in=total_send_repeat, min_out=total_recv_repeat,
                                          # jump=self.comm_plan.after_pipeline_full_instr_start,
                                          full_pip=dosa_singleton.config.backend.comm_message_interleaving,
                                          max_ranks=len(all_ranks))
                                            # min_l=comm_plan_len
                    outline += '// ATTENTION: currently, only batch-wise inference is supported\n'
                else:
                    outline = line
                out_file.write(outline)
        # 5. build shared lib
        os.system('cd {}; make lib'.format(self.out_dir_path))
        return 0
