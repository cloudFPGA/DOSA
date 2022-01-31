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


class CommPlan:

    def __init__(self, for_node):
        self.node = for_node
        self.comm_instr = []
        self.update()

    def update(self):
        brick_keys = list(self.node.bricks.keys())
        brick_keys.sort()
        #  first and last could be the same, but unimportant
        first_brick = self.node.bricks[brick_keys[0]]
        last_brick = self.node.bricks[brick_keys[-1]]
        in_nodes = self.node.predecessors
        out_nodes = self.node.successors
        in_msg_cnt = first_brick.input_bytes
        out_msg_cnt = last_brick.output_bytes
        in_repetition = 1 + dosa_singleton.config.backend.comm_message_interleaving
        out_repetition = 1 + dosa_singleton.config.backend.comm_message_interleaving

        if len(in_nodes) == 0 and dosa_singleton.config.backend.create_rank_0_for_io:
            new_msg_inst_dict = {'instr': 'recv', 'rank': 0, 'count': in_msg_cnt, 'repeat': in_repetition}
            self.comm_instr.append(new_msg_inst_dict)
        else:
            for n in in_nodes:
                new_msg_inst_dict = {'instr': 'recv', 'rank': n.node_id, 'count': in_msg_cnt, 'repeat': in_repetition}
                self.comm_instr.append(new_msg_inst_dict)
        if len(out_nodes) == 0 and dosa_singleton.config.backend.create_rank_0_for_io:
            new_msg_inst_dict = {'instr': 'send', 'rank': 0, 'count': out_msg_cnt, 'repeat': out_repetition}
            self.comm_instr.append(new_msg_inst_dict)
        else:
            for n in out_nodes:
                new_msg_inst_dict = {'instr': 'send', 'rank': n.node_id, 'count': out_msg_cnt, 'repeat': out_repetition}
                self.comm_instr.append(new_msg_inst_dict)
        return 0

    def get_comm_instr(self):
        return self.comm_instr

