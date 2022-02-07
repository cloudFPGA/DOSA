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

from dimidium.backend.codeGen.WrapperInterfaces import WrapperInterface, InterfaceAxisFifo


class VhdlEntity:

    def __init__(self, template_file=None):
        self.template_file = template_file
        self.signal_decls = []
        self.comp_decls = {}
        self.network_adapter_inst = None
        # self.network_output_adapter_inst = None
        self.processing_comp_insts = {}
        self.next_proc_comp_cnt = 0
        self.add_tcl_valid = False
        self.add_tcl_lines = []
        self.lib_includes = {}

    def set_template(self, template_file):
        self.template_file = template_file

    def set_network_adapter(self, decl_lines, inst_template, if_types):
        self.comp_decls['network_adapter'] = decl_lines
        self.network_adapter_inst = {'inst_tmpl': inst_template, 'if_types': if_types}

    # def set_network_output_adapter(self, decl_lines, inst_template, if_types):
    #     self.comp_decls['network_output_adapter'] = decl_lines
    #     self.network_output_adapter_inst = {'inst_tmpl': inst_template, 'if_types': if_types}

    def add_comp_decls(self, name, decl_lines):
        self.comp_decls[name] = decl_lines

    def add_signal_decls(self, decl_lines):
        self.signal_decls.append(decl_lines)

    def add_lib_include(self, lib_name, use_lines: list):
        if lib_name in self.lib_includes:
            self.lib_includes[lib_name].extend(use_lines)
        else:
            self.lib_includes[lib_name] = use_lines

    def add_proc_comp_inst(self, arch_block, decl_lines, inst_template, input_if: WrapperInterface, output_if=None):
        if output_if is not None:
            assert isinstance(output_if, WrapperInterface)

        self.comp_decls[self.next_proc_comp_cnt] = decl_lines
        inst_name = 'proc_comp_{}'.format(self.next_proc_comp_cnt)
        self.processing_comp_insts[self.next_proc_comp_cnt] = {'inst_tmpl': inst_template, 'input_if': input_if,
                                                               'output_if': output_if, 'name': inst_name,
                                                               'arch_block': arch_block}
        self.next_proc_comp_cnt += 1

    def write_file(self, target_path, target_device):
        # check compatibility of input layer
        is_compatible = False
        for at in self.network_adapter_inst['if_types']:
            if isinstance(self.processing_comp_insts[0]['input_if'], at):
                is_compatible = True
        if not is_compatible:
            print("[DOSA:VhdlGen:ERROR] Can't connect first processing component to network input adapter (" +
                  "wrong interface type). STOP.")
            exit(-1)
        # 1. generate what still needs to be generated
        output_if = None
        if self.processing_comp_insts[self.next_proc_comp_cnt - 1]['output_if'] is None:
            last_brick = self.processing_comp_insts[self.next_proc_comp_cnt - 1]['arch_block'].brick_list[-1]
            out_type = self.network_adapter_inst['if_types'][0]  # just take the first one?
            output_if = out_type('output_node_end', last_brick.output_bw_Bs, target_device)
        else:
            output_if = self.processing_comp_insts[self.next_proc_comp_cnt - 1]['output_if']
        # get tcl from all interfaces --> no, is done by OSGs
        # for pci in self.processing_comp_insts.keys():
        #     pc = self.processing_comp_insts[pci]
        #     tcl_l = pc['input_if'].get_tcl_lines()
        # 2. add output_if tcl
        tcl_lines = output_if.get_tcl_lines()
        self.add_tcl_lines.append(tcl_lines)
        self.add_tcl_valid = True
        # 3. write vhdl, from top to bottom
        with open(self.template_file, 'r') as in_file, \
                open(target_path, 'w') as out_file:
            for line in in_file.readlines():
                if 'DOSA_ADD_library_includes' in line:
                    if len(self.lib_includes) > 0:
                        outline = '-- DOSA generated library includes\n'
                        for lk in self.lib_includes:
                            outline += 'library {};\n'.format(lk)
                            for ll in self.lib_includes[lk]:
                                outline += '  use {}.{};\n'.format(lk, ll)
                        outline += '\n'
                    else:
                        outline = '-- no further DOSA libraries'
                elif 'DOSA_ADD_decl_lines' in line:
                    outline = '  -- DOSA generated interface declarations\n'
                    for pci in self.processing_comp_insts.keys():
                        pc = self.processing_comp_insts[pci]
                        sig_decl = pc['input_if'].get_vhdl_signal_declaration()
                        outline += '\n' + sig_decl
                        if_decl = pc['input_if'].get_vhdl_entity_declaration()
                        outline += '\n' + if_decl
                    outline += '\n' + output_if.get_vhdl_signal_declaration()
                    outline += '\n' + output_if.get_vhdl_entity_declaration()
                    outline += '\n\n  -- DOSA generated processing components declarations\n'
                    for dk in self.comp_decls.keys():
                        comp_decl = self.comp_decls[dk]
                        outline += '\n' + comp_decl
                    outline += '\n'
                elif 'DOSA_ADD_inst_lines' in line:
                    outline = '  -- DOSA generated instantiations\n'
                    outline += '\n  -- Instantiate network adapter\n'
                    inst_tmpl = self.network_adapter_inst['inst_tmpl']
                    next_signals = self.processing_comp_insts[0]['input_if'].get_vhdl_signal_dict()
                    assert isinstance(self.processing_comp_insts[0]['input_if'],
                                      InterfaceAxisFifo)  # TODO: make dynamic
                    last_signals = output_if.get_vhdl_signal_dict()
                    assert isinstance(output_if, InterfaceAxisFifo)  # TODO: make also dynamic
                    in_map_dict = {'in_sig_0': last_signals['from_signals']['0'],
                                   'in_sig_1_n': last_signals['from_signals']['1_n'],
                                   'in_sig_1': last_signals['from_signals']['1'],
                                   'in_sig_2': last_signals['from_signals']['2'],
                                   'in_sig_3': last_signals['from_signals']['3'],
                                   'in_sig_4_n': last_signals['from_signals']['4_n'],
                                   'in_sig_4': last_signals['from_signals']['4'],
                                   'in_sig_5': last_signals['from_signals']['5'],
                                   'in_sig_6': last_signals['from_signals']['6'],
                                   'in_sig_7_n': last_signals['from_signals']['7_n'],
                                   'in_sig_7': last_signals['from_signals']['7'],
                                   'in_sig_8': last_signals['from_signals']['8'],
                                   'out_sig_0': next_signals['to_signals']['0'],
                                   'out_sig_1_n': next_signals['to_signals']['1_n'],
                                   'out_sig_1': next_signals['to_signals']['1'],
                                   'out_sig_2': next_signals['to_signals']['2'],
                                   'out_sig_3': next_signals['to_signals']['3'],
                                   'out_sig_4_n': next_signals['to_signals']['4_n'],
                                   'out_sig_4': next_signals['to_signals']['4'],
                                   'out_sig_5': next_signals['to_signals']['5'],
                                   'out_sig_6': next_signals['to_signals']['6'],
                                   'out_sig_7_n': next_signals['to_signals']['7_n'],
                                   'out_sig_7': next_signals['to_signals']['7'],
                                   'out_sig_8': next_signals['to_signals']['8'],
                                   'inst_name': 'DosaNetworkAdapter',
                                   'clk': 'piSHL_156_25Clk',
                                   'rst': 'piMMIO_Ly7_Rst',
                                   'rst_n': 'sResetApps_n',
                                   'enable': 'piMMIO_Ly7_En'
                                   }
                    # other signals are static for network adapter
                    new_inst = inst_tmpl.format_map(in_map_dict)
                    outline += '\n' + new_inst
                    for pci in self.processing_comp_insts.keys():
                        pc = self.processing_comp_insts[pci]
                        assert isinstance(pc['input_if'], InterfaceAxisFifo)  # TODO: make dynamic
                        if pci + 1 >= self.next_proc_comp_cnt:
                            next_if = output_if
                        else:
                            next_if = self.processing_comp_insts[pci + 1]['input_if']
                        outline += '\n  -- Instantiate processing component {}\n'.format(pci)
                        our_signals = pc['input_if'].get_vhdl_signal_dict()
                        # first, instantiate interface (if necessary)
                        inst_tmpl = pc['input_if'].get_vhdl_entity_inst_tmpl()
                        map_dict = {'in_sig_0': our_signals['to_signals']['0'],
                                    'in_sig_1_n': our_signals['to_signals']['1_n'],
                                    'in_sig_1': our_signals['to_signals']['1'],
                                    'in_sig_2': our_signals['to_signals']['2'],
                                    'out_sig_0': our_signals['from_signals']['0'],
                                    'out_sig_1_n': our_signals['from_signals']['1_n'],
                                    'out_sig_1': our_signals['from_signals']['1'],
                                    'out_sig_2': our_signals['from_signals']['2'],
                                    'in_sig_3': our_signals['to_signals']['3'],
                                    'in_sig_4_n': our_signals['to_signals']['4_n'],
                                    'in_sig_4': our_signals['to_signals']['4'],
                                    'in_sig_5': our_signals['to_signals']['5'],
                                    'out_sig_3': our_signals['from_signals']['3'],
                                    'out_sig_4_n': our_signals['from_signals']['4_n'],
                                    'out_sig_4': our_signals['from_signals']['4'],
                                    'out_sig_5': our_signals['from_signals']['5'],
                                    'in_sig_6': our_signals['to_signals']['6'],
                                    'in_sig_7_n': our_signals['to_signals']['7_n'],
                                    'in_sig_7': our_signals['to_signals']['7'],
                                    'in_sig_8': our_signals['to_signals']['8'],
                                    'out_sig_6': our_signals['from_signals']['6'],
                                    'out_sig_7_n': our_signals['from_signals']['7_n'],
                                    'out_sig_7': our_signals['from_signals']['7'],
                                    'out_sig_8': our_signals['from_signals']['8'],
                                    'inst_name': pc['name'] + '_input_if',
                                    'clk': 'piSHL_156_25Clk',
                                    'rst': 'piMMIO_Ly7_Rst',
                                    'rst_n': 'sResetApps_n',
                                    'enable': 'piMMIO_Ly7_En'
                                    }
                        new_inst = inst_tmpl.format_map(map_dict)
                        outline += '\n' + new_inst
                        # next, instantiate processing component
                        next_signals = next_if.get_vhdl_signal_dict()
                        map_dict = {'in_sig_0': our_signals['from_signals']['0'],
                                    'in_sig_1_n': our_signals['from_signals']['1_n'],
                                    'in_sig_1': our_signals['from_signals']['1'],
                                    'in_sig_2': our_signals['from_signals']['2'],
                                    'out_sig_0': next_signals['to_signals']['0'],
                                    'out_sig_1_n': next_signals['to_signals']['1_n'],
                                    'out_sig_1': next_signals['to_signals']['1'],
                                    'out_sig_2': next_signals['to_signals']['2'],
                                    'in_sig_3': our_signals['from_signals']['3'],
                                    'in_sig_4_n': our_signals['from_signals']['4_n'],
                                    'in_sig_4': our_signals['from_signals']['4'],
                                    'in_sig_5': our_signals['from_signals']['5'],
                                    'out_sig_3': next_signals['to_signals']['3'],
                                    'out_sig_4_n': next_signals['to_signals']['4_n'],
                                    'out_sig_4': next_signals['to_signals']['4'],
                                    'out_sig_5': next_signals['to_signals']['5'],
                                    'in_sig_6': our_signals['from_signals']['6'],
                                    'in_sig_7_n': our_signals['from_signals']['7_n'],
                                    'in_sig_7': our_signals['from_signals']['7'],
                                    'in_sig_8': our_signals['from_signals']['8'],
                                    'out_sig_6': next_signals['to_signals']['6'],
                                    'out_sig_7_n': next_signals['to_signals']['7_n'],
                                    'out_sig_7': next_signals['to_signals']['7'],
                                    'out_sig_8': next_signals['to_signals']['8'],
                                    'inst_name': pc['name'],
                                    'clk': 'piSHL_156_25Clk',
                                    'rst': 'piMMIO_Ly7_Rst',
                                    'rst_n': 'sResetApps_n',
                                    'enable': 'piMMIO_Ly7_En'
                                    }
                        inst_tmpl = pc['inst_tmpl']
                        new_inst = inst_tmpl.format_map(map_dict)
                        outline += '\n' + new_inst
                    # instantiate output interface (if necessary)
                    our_signals = output_if.get_vhdl_signal_dict()
                    inst_tmpl = output_if.get_vhdl_entity_inst_tmpl()
                    map_dict = {'in_sig_0': our_signals['to_signals']['0'],
                                'in_sig_1_n': our_signals['to_signals']['1_n'],
                                'in_sig_1': our_signals['to_signals']['1'],
                                'in_sig_2': our_signals['to_signals']['2'],
                                'out_sig_0': our_signals['from_signals']['0'],
                                'out_sig_1_n': our_signals['from_signals']['1_n'],
                                'out_sig_1': our_signals['from_signals']['1'],
                                'out_sig_2': our_signals['from_signals']['2'],
                                'in_sig_3': our_signals['to_signals']['3'],
                                'in_sig_4_n': our_signals['to_signals']['4_n'],
                                'in_sig_4': our_signals['to_signals']['4'],
                                'in_sig_5': our_signals['to_signals']['5'],
                                'out_sig_3': our_signals['from_signals']['3'],
                                'out_sig_4_n': our_signals['from_signals']['4_n'],
                                'out_sig_4': our_signals['from_signals']['4'],
                                'out_sig_5': our_signals['from_signals']['5'],
                                'in_sig_6': our_signals['to_signals']['6'],
                                'in_sig_7_n': our_signals['to_signals']['7_n'],
                                'in_sig_7': our_signals['to_signals']['7'],
                                'in_sig_8': our_signals['to_signals']['8'],
                                'out_sig_6': our_signals['from_signals']['6'],
                                'out_sig_7_n': our_signals['from_signals']['7_n'],
                                'out_sig_7': our_signals['from_signals']['7'],
                                'out_sig_8': our_signals['from_signals']['8'],
                                'inst_name': pc['name'] + '_input_if',
                                'clk': 'piSHL_156_25Clk',
                                'rst': 'piMMIO_Ly7_Rst',
                                'rst_n': 'sResetApps_n',
                                'enable': 'piMMIO_Ly7_En'
                                }
                    new_inst = inst_tmpl.format_map(map_dict)
                    outline += '\n' + new_inst
                    outline += '\n'
                else:
                    outline = line
                out_file.write(outline)
        return 0

    def get_add_tcl_lines(self):
        if not self.add_tcl_valid:
            print("[DOSA:VhdlGen:ERROR] The data from this call is not yet valid (add_tcl_lines of VhdlEntity). STOP.")
            exit(1)
        return self.add_tcl_lines
