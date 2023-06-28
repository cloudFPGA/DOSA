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
#  *     Created: Feb 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        ZRLMPI SW wrapper, based on templates
#  *
#  *
import math
import os
from pathlib import Path

import dimidium.lib.singleton as dosa_singleton
from dimidium.middleend.archGen.CommPlan import CommPlan

__filedir__ = os.path.dirname(os.path.abspath(__file__))


class ZrlmpiSwApp:

    def __init__(self, node_id, out_dir_path, comm_plan: CommPlan):
        self.node_id = node_id
        self.out_dir_path = out_dir_path
        self.comm_plan = comm_plan
        self.templ_dir_path = os.path.abspath(os.path.join(__filedir__, 'templates/zrlmpi_sw/'))
        self.gitmodule_dir_path = os.path.abspath(os.path.join(__filedir__, 'templates/ZRLMPI/'))

    def generate(self):
        # 1. copy ZRLMPI LIB
        os.system('mkdir -p {}/LIB'.format(self.out_dir_path))
        os.system('cp {}/*.?pp {}/LIB/'.format(self.gitmodule_dir_path + '/LIB/SW/', self.out_dir_path))
        os.system('cp {}/*.?pp {}/LIB/'.format(self.gitmodule_dir_path + '/LIB/COMMON/', self.out_dir_path))
        # 2. copy Makefile and python
        os.system('cp {}/Makefile {}/'.format(self.templ_dir_path, self.out_dir_path))
        os.system('cp {}/dosa_root.py {}/'.format(self.templ_dir_path, self.out_dir_path))
        # get comm_plan data
        comm_instr = self.comm_plan.get_comm_instr()
        # assert len(comm_instr) == 2  # this ist just the root app
        comm_plan_len = len(comm_instr)
        # assert comm_plan_len % 2 == 0
        assert len(self.comm_plan.get_comm_instr_sorted().keys()) == 1
        # repetitions = 1 + dosa_singleton.config.backend.comm_message_pipeline_store
        # comm_plan_one_iteration_length = 2 * repetitions
        # all_iterations = 0
        cur_send_cmds = []
        cur_recv_cmds = []
        all_send_cmds = []
        all_recv_cmds = []
        total_send_repeat = 0
        total_recv_repeat = 0
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
                        total_send_repeat += ci['repeat']
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
                        total_recv_repeat += ci['repeat']
            prog_i += 1
        if len(cur_send_cmds) > 0:
            all_send_cmds.append(cur_send_cmds)
        if len(cur_recv_cmds) > 0:
            all_recv_cmds.append(cur_recv_cmds)
        pipeline_store = total_send_repeat - total_recv_repeat
        node_0_instr = []
        assert len(all_send_cmds) == len(all_recv_cmds)
        for i in range(len(all_send_cmds)):
            node_0_instr.extend(all_send_cmds[i])
            node_0_instr.extend(all_recv_cmds[i])
        # empty pipeline store at the end...done by dosa_root
        # 3. generate dosa_infer.cpp
        with open(os.path.join(self.templ_dir_path, 'dosa_infer.cpp'), 'r') as in_file, \
                open(os.path.join(self.out_dir_path, 'dosa_infer.cpp'), 'w') as out_file:
            for line in in_file.readlines():
                if 'DOSA_ADD_mpi_config' in line:
                    outline = ''
                    # tmpl = '  mpiCommands[{i}]          = {c1};\n  mpiRanks[{i}]             = {r1};\n  mpiCounts[{i}]'+\
                    #        '            = {l1};\n  commandRepetitions[{i}]   = {t1};\n  mpiCommands[{i1}]          = '+\
                    #        '{c0};\n  mpiRanks[{i1}]             = {r0};\n  mpiCounts[{i1}]            = {l0};\n  '+\
                    #        'commandRepetitions[{i1}]   = {t0};\n'
                    # for i in range(0, comm_plan_len, 2):
                    #     ie0 = comm_instr[i]
                    #     ie1 = comm_instr[i+1]
                    #     c1 = 'MPI_INSTR_SEND'
                    #     if ie1['instr'] == 'recv':
                    #         c1 = 'MPI_INSTR_RECV'
                    #     c0 = 'MPI_INSTR_SEND'
                    #     # TODO: allow compute parallelization
                    #     assert ie0['combine'] is None
                    #     assert ie1['combine'] is None
                    #     if ie0['instr'] == 'recv':
                    #         c0 = 'MPI_INSTR_RECV'
                    #     # counts must be in byte4 size!
                    #     l0 = int((ie0['count'] + 3)/4)
                    #     l1 = int((ie1['count'] + 3)/4)
                    #     # outline = tmpl.format(r0=comm_instr[i+1]['rank'], sl=comm_instr[i+1]['count'],
                    #     #                       r1=comm_instr[i+0]['rank'], rl=comm_instr[i+0]['count'])
                    #     # outline += tmpl.format(i=i, i1=i+1, c1=c1, r1=ie1['rank'], l1=l1, t1=repetitions,
                    #     #                        c0=c0, r0=ie0['rank'], l0=l0, t0=repetitions)
                    #     outline += tmpl.format(i=i, i1=i+1, c1=c1, r1=ie1['rank'], l1=l1, t1=ie1['repeat'],
                    #                            c0=c0, r0=ie0['rank'], l0=l0, t0=ie0['repeat'])
                    #     # if all_iterations == 0:
                    #     #     # first pair
                    #     #     all_iterations = ie1['repeat'] + ie0['repeat']
                    tmpl = '  mpiCommands[{i}]          = {instr};\n' + \
                           '  mpiRanks[{i}]             = {rank};\n' + \
                           '  mpiCounts[{i}]            = {count};\n' + \
                           '  byteCounts[{i}]           = {byte_cnt};\n' + \
                           '  commandRepetitions[{i}]   = {repeat};\n' + \
                           '  saveCurData[{i}]          = {save_cur_data};\n'
                    prog_i = 0
                    # for si in send_cmds:
                    #     instr = 'MPI_INSTR_SEND'
                    #     rank = si['rank']
                    #     word_count = int((si['count'] + 3) / 4)
                    #     repeat = si['repeat']
                    #     save_cur_data = 'false'
                    #     if si['combine'] is not None and si['combine'] != 'finish':
                    #         save_cur_data = 'true'
                    #     outline += tmpl.format(i=prog_i, instr=instr, rank=rank, count=word_count, repeat=repeat,
                    #                            save_cur_data=save_cur_data)
                    #     prog_i += 1
                    # # prog_i continues...
                    # for ri in recv_cmds:
                    #     instr = 'MPI_INSTR_RECV'
                    #     rank = ri['rank']
                    #     word_count = int((ri['count'] + 3) / 4)
                    #     repeat = ri['repeat']
                    #     save_cur_data = 'false'  # always for recv
                    #     outline += tmpl.format(i=prog_i, instr=instr, rank=rank, count=word_count, repeat=repeat,
                    #                            save_cur_data=save_cur_data)
                    #     prog_i += 1
                    outline += '  //pipeline-FILL part\n'
                    skipped_instr = 0
                    for mi in node_0_instr:
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
                            outline += tmpl.format(i=prog_i, instr=instr, rank=rank, count=word_count, repeat=repeat,
                                                   save_cur_data=save_cur_data, byte_cnt=int(mi['count']))
                            prog_i += 1
                        if (prog_i + skipped_instr) == self.comm_plan.after_pipeline_full_instr_start:
                            outline += '  //pipeline-FULL part\n'
                else:
                    outline = line
                out_file.write(outline)
        # 4. copy dosa_infer.hpp
        with open(os.path.join(self.templ_dir_path, 'dosa_infer.hpp'), 'r') as in_file, \
                open(os.path.join(self.out_dir_path, 'dosa_infer.hpp'), 'w') as out_file:
            for line in in_file.readlines():
                if 'DOSA_ADD_APP_NODE_DEFINES' in line:
                    tmpl = ('#define DOSA_WRAPPER_PROG_LENGTH {total_l}\n' +
                            # '#define DOSA_MINIMAL_PROG_LENGTH {min_l}\n' +
                            '#define DOSA_PIPELINE_STORE_DETPH {pip_store}\n' +
                            '#define DOSA_MINIMAL_INPUT_NUM {min_in}\n' +
                            '#define DOSA_MINIMAL_OUTPUT_NUM {min_out}\n' +
                            '#define DOSA_COMM_PLAN_AFTER_FILL_JUMP {jump}\n' +
                            '#define DOSA_PIPELINE_FULL_BATCH_SIZE {full_pip}\n')
                    # outline = tmpl.format(total_l=comm_plan_len, min_l=all_iterations)
                    outline = tmpl.format(total_l=comm_plan_len, pip_store=pipeline_store,
                                          min_in=total_send_repeat, min_out=total_recv_repeat,
                                          jump=self.comm_plan.after_pipeline_full_instr_start,
                                          full_pip=dosa_singleton.config.backend.comm_message_interleaving)
                                            # min_l=comm_plan_len
                    outline += '// ATTENTION: currently, only batch-wise inference is supported\n'
                else:
                    outline = line
                out_file.write(outline)
        # 5. build shared lib
        os.system('cd {}; make lib'.format(self.out_dir_path))
        return 0
