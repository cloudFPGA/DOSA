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
        # 3. copy dosa_infer.hpp
        os.system('cp {}/dosa_infer.hpp {}/'.format(self.templ_dir_path, self.out_dir_path))
        # 4. generate dosa_infer.cpp
        with open(os.path.join(self.templ_dir_path, 'dosa_infer.cpp'), 'r') as in_file, \
                open(os.path.join(self.out_dir_path, 'dosa_infer.cpp'), 'w') as out_file:
            for line in in_file.readlines():
                if 'DOSA_ADD_mpi_config' in line:
                    comm_instr = self.comm_plan.get_comm_instr()
                    assert len(comm_instr) == 2  # this ist just the root app
                    tmpl = '  int first_node = {r0};\n  int send_mesage_length = {sl};\n  int last_node = {r1};\n  int ' \
                           'recv_mesage_length = {rl};\n '
                    outline = tmpl.format(r0=comm_instr[1]['rank'], sl=comm_instr[1]['count'],
                                          r1=comm_instr[0]['rank'], rl=comm_instr[0]['count'])
                else:
                    outline = line
                out_file.write(outline)
        # 5. build shared lib
        os.system('cd {}; make lib'.format(self.out_dir_path))
        return 0

