#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Jan 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Class for the creation of communication plans
#  *
#  *

import dimidium.lib.singleton as dosa_singleton
from dimidium.lib.util import my_lcm


class CommPlan:

    def __init__(self, for_node, pipeline_store_until_now):
        self.node = for_node
        self.comm_instr = []
        self.pipeline_store_until_now = pipeline_store_until_now
        self.update()

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
        # in_repetition = 1 + dosa_singleton.config.backend.comm_message_interleaving
        in_repetition = 1 + dosa_singleton.config.backend.comm_message_interleaving - self.pipeline_store_until_now
        # out_repetition = 1 + dosa_singleton.config.backend.comm_message_interleaving
        out_repetition = 1 + dosa_singleton.config.backend.comm_message_interleaving - \
                         (self.pipeline_store_until_now + self.node.total_pipeline_store)
        assert in_repetition > 0
        assert out_repetition > 0
        for i in range(len(incomming_ranks)):
            # here, we assume strict streaming...so one message in, one out (with repetition)
            new_msg_in_dict = {'instr': 'recv', 'rank': incomming_ranks[i], 'count': in_msg_cnt,
                               'repeat': in_repetition, 'cond': in_cmd_rank_list[i]}
            self.comm_instr.append(new_msg_in_dict)
            new_msg_out_dict = {'instr': 'send', 'rank': outgoing_ranks[i], 'count': out_msg_cnt,
                                'repeat': out_repetition, 'cond': out_cmd_rank_list[i]}
            self.comm_instr.append(new_msg_out_dict)

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
