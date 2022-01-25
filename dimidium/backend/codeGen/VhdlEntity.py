#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Jan 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Class representing a VHDL entity with architecture
#  *        for DOSA build tools
#  *
#  *
from dimidium.backend.codeGen.WrapperInterfaces import WrapperInterface


class VhdlEntity:

    def __init__(self, template_file=None):
        self.template_file = template_file
        self.signal_decls = []
        self.comp_decls = {}
        self.network_input_adapter_inst = None
        self.network_output_adapter_inst = None
        self.processing_comp_insts = {}
        self.next_proc_comp_cnt = 0
        self.add_tcl_valid = False
        self.add_tcl_lines = []

    def set_template(self, template_file):
        self.template_file = template_file

    def set_network_input_adapter(self, decl_lines, inst_template, if_types):
        self.comp_decls['network_input_adapter'] = decl_lines
        self.network_input_adapter_inst = {'inst_tmpl': inst_template, 'if_types': if_types}

    def set_network_output_adapter(self, decl_lines, inst_template, if_types):
        self.comp_decls['network_output_adapter'] = decl_lines
        self.network_output_adapter_inst = {'inst_tmpl': inst_template, 'if_types': if_types}

    def add_comp_decls(self, name, decl_lines):
        self.comp_decls[name] = decl_lines

    def add_signal_decls(self, decl_lines):
        self.signal_decls.append(decl_lines)

    def add_proc_comp_inst(self, decl_lines, inst_template, input_if: WrapperInterface, output_if=None):
        if output_if is not None:
            assert isinstance(output_if, WrapperInterface)

        self.comp_decls[self.next_proc_comp_cnt] = decl_lines
        inst_name = 'proc_comp_{}'.format(self.next_proc_comp_cnt)
        self.processing_comp_insts[self.next_proc_comp_cnt] = {'inst_tmpl': inst_template, 'input_if': input_if,
                                                               'output_if': output_if, 'name': inst_name}
        self.next_proc_comp_cnt += 1

    def write_file(self, target_path):
        # TODO
        #  first: generate what still needs to be generated (connection to network out adapter)
        #  then write vhdl, from top to bottom
        self.add_tcl_valid = True
        return 0

    def get_add_tcl_lines(self):
        if not self.add_tcl_valid:
            print("[DOSA:VhdlGen:ERROR] The data from this call is not yet valid (add_tcl_lines of VhdlEntity). STOP.")
            exit(1)
        return self.add_tcl_lines

