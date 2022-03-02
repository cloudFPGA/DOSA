#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Jan 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Tools for the generation of Node2Node communication blocks
#  *
#  *

import abc

from dimidium.middleend.archGen.CommPlan import CommPlan


class CommunicationWrapper(metaclass=abc.ABCMeta):

    def __init__(self, node_id, out_dir_path, comm_plan: CommPlan):
        self.node_id = node_id
        self.out_dir_path = out_dir_path
        self.comm_plan = comm_plan

    @abc.abstractmethod
    def generate(self):
        print("[DOSA:WrapperGeneration:ERROR] NOT YET IMPLEMENTED.")

    @abc.abstractmethod
    def get_tcl_lines_wrapper_inst(self):
        print("[DOSA:WrapperGeneration:ERROR] NOT YET IMPLEMENTED.")

    @abc.abstractmethod
    def get_wrapper_vhdl_decl_lines(self):
        print("[DOSA:WrapperGeneration:ERROR] NOT YET IMPLEMENTED.")

    @abc.abstractmethod
    def get_vhdl_inst_tmpl(self):
        print("[DOSA:WrapperGeneration:ERROR] NOT YET IMPLEMENTED.")

    @abc.abstractmethod
    def get_debug_lines(self):
        """
        :return: tcl_template lines, vhdl_decl lines, vhdl_instantiation lines as list
                    all having {i} as template for probe number
        """
        print("[DOSA:WrapperGeneration:ERROR] NOT YET IMPLEMENTED.")


