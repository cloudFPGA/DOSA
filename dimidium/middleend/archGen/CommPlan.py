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
#  *     Created: Jan 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Class for the creation of communication plans
#  *
#  *
import math

import dimidium.lib.singleton as dosa_singleton
from dimidium.lib.util import my_lcm


def _get_repetition_split(total_repetition, set_size):
    ret = []
    if set_size == 1:
        ret = [int(total_repetition)]
        return  ret
    if total_repetition % set_size == 0:
        ret = [int(total_repetition/set_size)] * set_size
    else:
        ret = [int(total_repetition - ((total_repetition//set_size) * (set_size - 1)))]
        remaining = [int(total_repetition//set_size)] * (set_size - 1)
        ret.extend(remaining)
    return ret


class CommPlan:

    def __init__(self, for_node, pipeline_store_until_now):
        self.node = for_node
        self.comm_instr = []
        self.pipeline_store_until_now = pipeline_store_until_now
        self.after_pipeline_full_instr_start = -1
        self.transactions_per_iteration = 0
        self.update()

    def as_dict(self):
        ret = {'instructions': self.comm_instr, 'transactions_per_iteration': self.transactions_per_iteration}
        return ret

    def update(self):
        brick_keys = list(self.node.bricks.keys())
        brick_keys.sort()
        #  first and last could be the same, but unimportant
        first_brick = self.node.bricks[brick_keys[0]]
        last_brick = self.node.bricks[brick_keys[-1]]
        # TODO: using new rank scheme
        #  need also conditional instr, e.g. 'cond' as rank number?
        # if len(self.node.inp_ranks) == 0:
        #    number_of_in_conns = len(self.node.ranks)
        #    incomming_ranks =
        # else:
        # TODO: make dynamic:
        assert len(self.node.inp_ranks) > 0 and len(self.node.out_ranks) > 0
        # if len(self.node.out_ranks) == 0:
        #    number_of_out_conns = len(self.node.out_ranks)
        # else:
        number_of_in_conns = len(self.node.inp_ranks) * len(self.node.ranks)
        number_of_out_conns = len(self.node.out_ranks) * len(self.node.ranks)
        comms_per_rank_in = int(number_of_in_conns / len(self.node.ranks))
        repetition_of_in_ranks = int(number_of_in_conns / len(self.node.inp_ranks))
        comms_per_rank_out = int(number_of_out_conns / len(self.node.ranks))
        repetition_of_out_ranks = int(number_of_out_conns / len(self.node.out_ranks))
        incomming_ranks = self.node.inp_ranks * repetition_of_in_ranks
        outgoing_ranks = self.node.out_ranks * repetition_of_out_ranks
        # for inr in self.node.inp_ranks:
        #     tl = [inr] * repetition_of_in_ranks
        #     incomming_ranks.extend(tl)
        # for otr in self.node.out_ranks:
        #     tl = [otr] * repetition_of_out_ranks
        #     outgoing_ranks.extend(tl)
        in_cmd_rank_list = []
        out_cmd_rank_list = []
        for rid in self.node.ranks:
            tl = [rid] * comms_per_rank_in
            tm = [rid] * comms_per_rank_out
            in_cmd_rank_list.extend(tl)
            out_cmd_rank_list.extend(tm)
        assert len(incomming_ranks) == len(in_cmd_rank_list)
        assert len(outgoing_ranks) == len(out_cmd_rank_list)
        if (len(incomming_ranks) > len(outgoing_ranks)) and \
                (len(incomming_ranks) % len(outgoing_ranks) == 0):
            ef = int(len(incomming_ranks) / len(outgoing_ranks))
            # assert len(incomming_ranks) % len(outgoing_ranks) == 0
            outgoing_ranks *= ef
            out_cmd_rank_list *= ef
        elif (len(outgoing_ranks) > len(incomming_ranks)) and \
                (len(outgoing_ranks) % len(incomming_ranks) == 0):
            ef = int(len(outgoing_ranks) / len(incomming_ranks))
            # assert len(outgoing_ranks) % len(incomming_ranks) == 0
            incomming_ranks *= ef
            in_cmd_rank_list *= ef
        else:
            # use least common mutliple
            lcm = my_lcm(len(outgoing_ranks), len(incomming_ranks))
            ef_1 = int(lcm / len(incomming_ranks))
            incomming_ranks *= ef_1
            in_cmd_rank_list *= ef_1
            ef_2 = int(lcm / len(outgoing_ranks))
            outgoing_ranks *= ef_2
            out_cmd_rank_list *= ef_2
        assert len(incomming_ranks) == len(outgoing_ranks)
        in_cmd_rank_list.sort()
        out_cmd_rank_list.sort()
        assert in_cmd_rank_list == out_cmd_rank_list
        # incomming_ranks.sort()
        # outgoing_ranks.sort()

        in_msg_cnt = first_brick.input_bytes
        out_msg_cnt = last_brick.output_bytes
        # Pipeline fill part
        # in_repetition = 1 + dosa_singleton.config.backend.comm_message_pipeline_store
        # TODO: self.pipeline_store_until now is <0!
        total_in_repetition = 1 + dosa_singleton.config.backend.comm_message_pipeline_store - self.pipeline_store_until_now
        # out_repetition = 1 + dosa_singleton.config.backend.comm_message_pipeline_store
        total_out_repetition = 1 + dosa_singleton.config.backend.comm_message_pipeline_store - \
                         (self.pipeline_store_until_now + self.node.total_pipeline_store)
        assert total_in_repetition > 0
        assert total_out_repetition > 0
        total_in_batch_size = in_msg_cnt * total_in_repetition
        # total_out_batch_size = out_msg_cnt * total_out_repetition
        in_repeat_list = [total_in_repetition]
        out_repeat_list = [total_out_repetition]
        if total_in_batch_size > dosa_singleton.config.backend.comm_message_max_buffer_interleaving:
            max_in_repeat = max(1, dosa_singleton.config.backend.comm_message_max_buffer_interleaving // in_msg_cnt)
            in_repeat_list = []
            out_repeat_list = []
            if max_in_repeat < (self.node.total_pipeline_store + 1):
                max_in_repeat = self.node.total_pipeline_store + 1
            min_out_repeat = max(1, max_in_repeat-self.node.total_pipeline_store)
            considered_in_repeat = 0
            considered_out_repeat = 0
            while considered_in_repeat < total_in_repetition:
                new_in_repeat = max_in_repeat
                last_round = False
                if considered_in_repeat + max_in_repeat > total_in_repetition:
                    new_in_repeat = total_in_repetition - considered_in_repeat
                if considered_in_repeat + max_in_repeat >= total_in_repetition:
                    last_round = True
                new_out_repeat = min_out_repeat
                if ((considered_out_repeat + new_out_repeat) > total_out_repetition) or last_round:
                    new_out_repeat = total_out_repetition - considered_out_repeat
                in_repeat_list.append(new_in_repeat)
                out_repeat_list.append(new_out_repeat)
                considered_out_repeat += new_out_repeat
                considered_in_repeat += new_in_repeat
                assert new_out_repeat > 0
                assert new_in_repeat > 0
                assert considered_in_repeat >= (considered_out_repeat + self.node.total_pipeline_store)
            assert considered_in_repeat == total_in_repetition
            assert considered_out_repeat == total_out_repetition
        # since we always transfer tensor after tensor, in & out are independent from each other
        # TODO: we don't need to take care of output...?`
        # repetition_cycles = [(total_in_repetition, total_out_repetition)]
        repetition_cycles = zip(in_repeat_list, out_repeat_list)
        self.comm_instr = []
        self.transactions_per_iteration = 0
        for in_repetition, out_repetition in repetition_cycles:
            data_parallel_incoming_set_of_ranks = len(incomming_ranks)
            incoming_repetition_distribution = _get_repetition_split(in_repetition, data_parallel_incoming_set_of_ranks)
            for i in range(len(incomming_ranks)):
                # here, we assume strict streaming...so one message in, one out (with repetition)
                if len(incomming_ranks[i]) == 1:
                    # POTENTIAL DATA PARALLELISM BEFORE
                    # new_msg_in_dict = {'instr': 'recv', 'rank': incomming_ranks[i][0], 'count': in_msg_cnt,
                    #                    'repeat': in_repetition, 'cond': in_cmd_rank_list[i], 'combine': None}
                    new_msg_in_dict = {'instr': 'recv', 'rank': incomming_ranks[i][0], 'count': in_msg_cnt,
                                       'repeat': incoming_repetition_distribution[i], 'cond': in_cmd_rank_list[i], 'combine': None}
                    self.comm_instr.append(new_msg_in_dict)
                    # self.transactions_per_iteration += in_repetition
                else:
                    # MODEL PARALLELISM BEFORE
                    # make repetition explicit
                    combine_comp_parallel = []
                    # if self.node.node_id % 2 == 0:
                    #     # I'm even, wait for even first
                    #     cur_parallel_ranks = sorted(incomming_ranks[i], key=lambda x: (x % 2, x))
                    # else:
                    #     # I'm odd, wait for odd first
                    #     cur_parallel_ranks = sorted(incomming_ranks[i], key=lambda x: (not (x % 2), x))
                    cur_parallel_ranks = incomming_ranks[i]
                    partial_msg_cnt = math.ceil(in_msg_cnt/len(cur_parallel_ranks))
                    for j in range(len(cur_parallel_ranks)):
                        combine_str = 'continue'
                        if j == 0:
                            combine_str = 'start'
                        elif j == len(cur_parallel_ranks) - 1:
                            combine_str = 'finish'
                        new_msg_in_dict = {'instr': 'recv', 'rank': cur_parallel_ranks[j], 'count': partial_msg_cnt,
                                           'repeat': 1, 'cond': in_cmd_rank_list[i], 'combine': combine_str}
                        combine_comp_parallel.append(new_msg_in_dict)
                    for r in range(0, incoming_repetition_distribution[i]):
                        self.comm_instr.extend(combine_comp_parallel)
                    # self.transactions_per_iteration += in_repetition

                data_parallel_outgoing_set_of_ranks = len(outgoing_ranks)
                outgoing_repetition_distribution = _get_repetition_split(out_repetition,
                                                                         data_parallel_outgoing_set_of_ranks)
                if len(outgoing_ranks[i]) == 1:
                    # POTENTIAL DATA PARALLELISM AFTERWARDS
                    # new_msg_out_dict = {'instr': 'send', 'rank': outgoing_ranks[i][0], 'count': out_msg_cnt,
                    #                     'repeat': out_repetition, 'cond': out_cmd_rank_list[i], 'combine': None}
                    new_msg_out_dict = {'instr': 'send', 'rank': outgoing_ranks[i][0], 'count': out_msg_cnt,
                                        'repeat': outgoing_repetition_distribution[i], 'cond': out_cmd_rank_list[i], 'combine': None}
                    self.comm_instr.append(new_msg_out_dict)
                    # self.transactions_per_iteration += out_repetition
                else:
                    # MODEL PARALLELISM AFTERWARDS
                    # make repetition explicit
                    combine_comp_parallel = []
                    # if self.node.node_id % 2 == 0:
                    #     # I'm even, send to even first
                    #     cur_parallel_ranks = sorted(outgoing_ranks[i], key=lambda x: (x % 2, x))
                    # else:
                    #     # I'm odd, send to odd first
                    #     cur_parallel_ranks = sorted(outgoing_ranks[i], key=lambda x: (not (x % 2), x))
                    cur_parallel_ranks = outgoing_ranks[i]
                    # out_msg_cnt doesn't change
                    for j in range(len(cur_parallel_ranks)):
                        combine_str = 'continue'
                        if j == 0:
                            combine_str = 'start'
                        elif j == len(cur_parallel_ranks) - 1:
                            combine_str = 'finish'
                        new_msg_out_dict = {'instr': 'send', 'rank': cur_parallel_ranks[j], 'count': out_msg_cnt,
                                            'repeat': 1, 'cond': out_cmd_rank_list[i], 'combine': combine_str}
                        combine_comp_parallel.append(new_msg_out_dict)
                    for r in range(0, outgoing_repetition_distribution[i]):
                        self.comm_instr.extend(combine_comp_parallel)
                    # self.transactions_per_iteration += out_repetition
        # Pipeline full part
        self.after_pipeline_full_instr_start = len(self.comm_instr)
        total_in_repetition = max(1, dosa_singleton.config.backend.comm_message_interleaving)
        total_out_repetition = max(1, dosa_singleton.config.backend.comm_message_interleaving)
        assert total_in_repetition > 0
        assert total_out_repetition > 0
        total_in_batch_size = in_msg_cnt * total_in_repetition
        # assert total_in_batch_size < dosa_singleton.config.backend.comm_message_max_buffer_interleaving
        in_repeat_list = [total_in_repetition]
        out_repeat_list = [total_out_repetition]
        if total_in_batch_size > dosa_singleton.config.backend.comm_message_max_buffer_interleaving:
            max_in_repeat = max(1, dosa_singleton.config.backend.comm_message_max_buffer_interleaving // in_msg_cnt)
            in_repeat_list = []
            out_repeat_list = []
            min_out_repeat = max(1, total_out_repetition // max_in_repeat)
            considered_in_repeat = 0
            considered_out_repeat = 0
            while considered_in_repeat < total_in_repetition:
                new_in_repeat = max_in_repeat
                last_round = False
                if considered_in_repeat + max_in_repeat > total_in_repetition:
                    new_in_repeat = total_in_repetition - considered_in_repeat
                if considered_in_repeat + max_in_repeat >= total_in_repetition:
                    last_round = True
                new_out_repeat = min_out_repeat
                if ((considered_out_repeat + new_out_repeat) > total_out_repetition) or last_round:
                    new_out_repeat = total_out_repetition - considered_out_repeat
                in_repeat_list.append(new_in_repeat)
                out_repeat_list.append(new_out_repeat)
                considered_out_repeat += new_out_repeat
                considered_in_repeat += new_in_repeat
                assert new_out_repeat > 0
                assert new_in_repeat > 0
            assert considered_in_repeat == total_in_repetition
            assert considered_out_repeat == total_out_repetition
        # since we always transfer tensor after tensor, in & out are independent from each other
        # TODO: we don't need to take care of output...?`
        # repetition_cycles = [(total_in_repetition, total_out_repetition)]
        repetition_cycles = zip(in_repeat_list, out_repeat_list)
        for in_repetition, out_repetition in repetition_cycles:
            for i in range(len(incomming_ranks)):
                # here, we assume strict streaming...so one message in, one out (with repetition)
                if len(incomming_ranks[i]) == 1:
                    new_msg_in_dict = {'instr': 'recv', 'rank': incomming_ranks[i][0], 'count': in_msg_cnt,
                                       'repeat': in_repetition, 'cond': in_cmd_rank_list[i], 'combine': None}
                    self.comm_instr.append(new_msg_in_dict)
                    self.transactions_per_iteration += in_repetition
                else:
                    # make repetition explicit
                    combine_comp_parallel = []
                    # if self.node.node_id % 2 == 0:
                    #     # I'm even, wait for even first
                    #     cur_parallel_ranks = sorted(incomming_ranks[i], key=lambda x: (x % 2, x))
                    # else:
                    #     # I'm odd, wait for odd first
                    #     cur_parallel_ranks = sorted(incomming_ranks[i], key=lambda x: (not (x % 2), x))
                    cur_parallel_ranks = incomming_ranks[i]
                    partial_msg_cnt = math.ceil(in_msg_cnt/len(cur_parallel_ranks))
                    for j in range(len(cur_parallel_ranks)):
                        combine_str = 'continue'
                        if j == 0:
                            combine_str = 'start'
                        elif j == len(cur_parallel_ranks) - 1:
                            combine_str = 'finish'
                        new_msg_in_dict = {'instr': 'recv', 'rank': cur_parallel_ranks[j], 'count': partial_msg_cnt,
                                           'repeat': 1, 'cond': in_cmd_rank_list[i], 'combine': combine_str}
                        combine_comp_parallel.append(new_msg_in_dict)
                    for r in range(0, in_repetition):
                        self.comm_instr.extend(combine_comp_parallel)
                    self.transactions_per_iteration += in_repetition
                if len(outgoing_ranks[i]) == 1:
                    new_msg_out_dict = {'instr': 'send', 'rank': outgoing_ranks[i][0], 'count': out_msg_cnt,
                                        'repeat': out_repetition, 'cond': out_cmd_rank_list[i], 'combine': None}
                    self.comm_instr.append(new_msg_out_dict)
                    self.transactions_per_iteration += out_repetition
                else:
                    # make repetition explicit
                    combine_comp_parallel = []
                    # if self.node.node_id % 2 == 0:
                    #     # I'm even, send to even first
                    #     cur_parallel_ranks = sorted(outgoing_ranks[i], key=lambda x: (x % 2, x))
                    # else:
                    #     # I'm odd, send to odd first
                    #     cur_parallel_ranks = sorted(outgoing_ranks[i], key=lambda x: (not (x % 2), x))
                    cur_parallel_ranks = outgoing_ranks[i]
                    # out_msg_cnt doesn't change
                    for j in range(len(cur_parallel_ranks)):
                        combine_str = 'continue'
                        if j == 0:
                            combine_str = 'start'
                        elif j == len(cur_parallel_ranks) - 1:
                            combine_str = 'finish'
                        new_msg_out_dict = {'instr': 'send', 'rank': cur_parallel_ranks[j], 'count': out_msg_cnt,
                                            'repeat': 1, 'cond': out_cmd_rank_list[i], 'combine': combine_str}
                        combine_comp_parallel.append(new_msg_out_dict)
                    for r in range(0, out_repetition):
                        self.comm_instr.extend(combine_comp_parallel)
                    self.transactions_per_iteration += out_repetition
        # to get number iteration
        self.transactions_per_iteration /= total_in_repetition

        # in_nodes = self.node.predecessors
        # out_nodes = self.node.successors
        # if len(in_nodes) == 0 and dosa_singleton.config.backend.create_rank_0_for_io:
        #     new_msg_inst_dict = {'instr': 'recv', 'rank': 0, 'count': in_msg_cnt, 'repeat': in_repetition}
        #     self.comm_instr.append(new_msg_inst_dict)
        # else:
        #     for n in in_nodes:
        #         new_msg_inst_dict = {'instr': 'recv', 'rank': n.node_id, 'count': in_msg_cnt, 'repeat': in_repetition}
        #         self.comm_instr.append(new_msg_inst_dict)
        # if len(out_nodes) == 0 and dosa_singleton.config.backend.create_rank_0_for_io:
        #     new_msg_inst_dict = {'instr': 'send', 'rank': 0, 'count': out_msg_cnt, 'repeat': out_repetition}
        #     self.comm_instr.append(new_msg_inst_dict)
        # else:
        #     for n in out_nodes:
        #         new_msg_inst_dict = {'instr': 'send', 'rank': n.node_id, 'count': out_msg_cnt, 'repeat': out_repetition}
        #         self.comm_instr.append(new_msg_inst_dict)
        return 0

    def get_comm_instr(self):
        return self.comm_instr

    def get_comm_instr_sorted(self):
        sorted_dict = {}
        for instr in self.comm_instr:
            if instr['cond'] not in sorted_dict:
                sorted_dict[instr['cond']] = [instr]
            else:
                sorted_dict[instr['cond']].append(instr)
        return sorted_dict

    def get_longest_msg_bytes(self):
        longest_msg = 0
        for ie in self.comm_instr:
            if ie['count'] > longest_msg:
                longest_msg = ie['count']
        return longest_msg

    def get_comm_instr_num(self):
        return len(self.comm_instr)
