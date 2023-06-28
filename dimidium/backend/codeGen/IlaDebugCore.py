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
#  *     Created: Mar 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Class representing a Xilinx ILA Core for PR usage
#  *        and a VHDL Top Entity
#  *
#  *

from dimidium.backend.codeGen.WrapperInterfaces import WrapperInterface, InterfaceAxisFifo

_valid_fifo_depth_ = [1024, 2048, 4096, 8192, 16384]
_probe_len_threshold_ = 90
_fifo_depth_lower_index_ = 1


class IlaDebugCore:

    def __init__(self, fifo_depth='auto'):
        self.tcl_templates = []
        self.decl_templates = []
        self.inst_templates = []
        self.tcl_start = '\n#------------------------------------------------------------------------------\n' + \
                         '# VIVADO-IP : ILA Core\n' + \
                         '#------------------------------------------------------------------------------\n' + \
                         'set ipModName "ila_dosa_role_0"\n' + \
                         'set ipName    "ila"\n' + \
                         'set ipVendor  "xilinx.com"\n' + \
                         'set ipLibrary "ip"\n' + \
                         'set ipVersion "6.2"\n' + \
                         'set ipCfgList [list  CONFIG.C_NUM_OF_PROBES {num_probes} \\\n' + \
                         '                     CONFIG.C_DATA_DEPTH {fifo_depth} \\\n'
        self.tcl_end = '              ]\n' + \
                       '\nset rc [ my_customize_ip ${ipModName} ${ipDir} ${ipVendor} ${ipLibrary} ${ipName} ${' \
                       'ipVersion} ${ipCfgList} ]\n' + \
                       '\nif { ${rc} != ${::OK} } { set nrErrors [ expr { ${nrErrors} + 1 } ] }\n\n'
        self.decl_start = '\n  component ila_dosa_role_0 is\n' + \
                          '    port (\n ' + \
                          '      clk : IN STD_LOGIC\n'  # semicolon always at the begin of next line
        self.decl_end = '   );\n' + \
                        '  end component ila_dosa_role_0;\n\n'
        self.inst_start = '\n  --################################################################################\n' + \
                          '  --  Debug Core instantiation\n' + \
                          '  --################################################################################\n' + \
                          '\n  DBG: ila_dosa_role_0\n' + \
                          '    port map (\n ' + \
                          '      clk => {clk}\n'  # comma always at the begin of next line
        self.inst_end = '    );\n\n'
        self.num_probes = 0
        if fifo_depth not in _valid_fifo_depth_ and fifo_depth != 'auto':
            print("[DOSA:DEBUG:ERROR] The debug_core fifo depth must be in {} or 'auto'. STOP."
                  .format(_valid_fifo_depth_))
            exit(1)
        self.fifo_depth = fifo_depth

    def add_new_probes(self, tcl_tmpl, decl_tmpl, inst_tmpl):
        assert (len(tcl_tmpl) == len(decl_tmpl)) and (len(decl_tmpl) == len(inst_tmpl))
        self.tcl_templates.extend(tcl_tmpl)
        self.decl_templates.extend(decl_tmpl)
        self.inst_templates.extend(inst_tmpl)
        self.num_probes += len(tcl_tmpl)

    def get_tcl_lines(self):
        indent = '                     '
        if self.fifo_depth == 'auto':
            if len(self.tcl_templates) > _probe_len_threshold_:
                self.fifo_depth = _valid_fifo_depth_[_fifo_depth_lower_index_]
            else:
                self.fifo_depth = _valid_fifo_depth_[_fifo_depth_lower_index_+1]
        tcl_lines = self.tcl_start.format(num_probes=self.num_probes, fifo_depth=self.fifo_depth)
        for i in range(self.num_probes):
            te = self.tcl_templates[i]
            tcl_lines += indent + te.format(i=i)
        tcl_lines += self.tcl_end
        return tcl_lines

    def get_vhdl_decl(self):
        indent = '      '
        decl_lines = self.decl_start
        for i in range(self.num_probes):
            de = self.decl_templates[i]
            decl_lines += indent + de.format(i=i)
        decl_lines += self.decl_end
        return decl_lines

    def get_vhdl_inst_tmpl(self):
        indent = '      '
        inst_lines = self.inst_start
        # clk is template
        for i in range(self.num_probes):
            ie = self.inst_templates[i]
            inst_lines += indent + ie.format(i=i)
        inst_lines += self.inst_end
        return inst_lines
