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
        comm_plan_one_iteration_length = 2  # TODO: make dynamic?
        comm_plan_len = len(comm_instr)
        assert comm_plan_len % comm_plan_one_iteration_length == 0
        assert len(self.comm_plan.get_comm_instr_sorted().keys()) == 1
        repetitions = 1  # TODO: make dynamic
        # 3. copy dosa_infer.hpp
        with open(os.path.join(self.templ_dir_path, 'dosa_infer.hpp'), 'r') as in_file, \
                open(os.path.join(self.out_dir_path, 'dosa_infer.hpp'), 'w') as out_file:
            for line in in_file.readlines():
                if 'DOSA_ADD_APP_NODE_DEFINES' in line:
                    tmpl = '#define DOSA_WRAPPER_PROG_LENGTH {total_l}\n#define DOSA_MINIMAL_PROG_LENGTH {min_l}\n'
                    outline = tmpl.format(total_l=comm_plan_len, min_l=comm_plan_one_iteration_length)
                else:
                    outline = line
                out_file.write(outline)
        # 4. generate dosa_infer.cpp
        with open(os.path.join(self.templ_dir_path, 'dosa_infer.cpp'), 'r') as in_file, \
                open(os.path.join(self.out_dir_path, 'dosa_infer.cpp'), 'w') as out_file:
            for line in in_file.readlines():
                if 'DOSA_ADD_mpi_config' in line:
                    # tmpl = '  int first_node = {r0};\n  int send_mesage_length = {sl};\n  int last_node = {r1};\n  int ' \
                    #        'recv_mesage_length = {rl};\n '
                    tmpl = '  mpiCommands[{i}]          = {c1};\n  mpiRanks[{i}]             = {r1};\n  mpiCounts[{i}]'+\
                           '            = {l1};\n  commandRepetitions[{i}]   = {t1};\n  mpiCommands[{i1}]          = '+\
                           '{c0};\n  mpiRanks[{i1}]             = {r0};\n  mpiCounts[{i1}]            = {l0};\n  '+\
                           'commandRepetitions[{i1}]   = {t0};\n'
                    outline = ''
                    for i in range(0, comm_plan_len, 2):
                        ie0 = comm_instr[i]
                        ie1 = comm_instr[i+1]
                        c1 = 'MPI_INSTR_SEND'
                        if ie1['instr'] == 'recv':
                            c1 = 'MPI_INSTR_RECV'
                        c0 = 'MPI_INSTR_SEND'
                        if ie0['instr'] == 'recv':
                            c0 = 'MPI_INSTR_RECV'
                        # counts must be in byte4 size!
                        l0 = int((ie0['count'] + 3)/4)
                        l1 = int((ie1['count'] + 3)/4)
                        # outline = tmpl.format(r0=comm_instr[i+1]['rank'], sl=comm_instr[i+1]['count'],
                        #                       r1=comm_instr[i+0]['rank'], rl=comm_instr[i+0]['count'])
                        outline += tmpl.format(i=i, i1=i+1, c1=c1, r1=ie1['rank'], l1=l1, t1=repetitions,
                                               c0=c0, r0=ie0['rank'], l0=l0, t0=repetitions)
                else:
                    outline = line
                out_file.write(outline)
        # 5. build shared lib
        os.system('cd {}; make lib'.format(self.out_dir_path))
        return 0

