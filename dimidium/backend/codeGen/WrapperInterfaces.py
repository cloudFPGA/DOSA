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
#  *        Tools for the generation of wrapper interfaces
#  *
#  *

import os
import abc
from pathlib import Path


__bit_ber_byte__ = 8
__small_interface_bitwidth__ = 64
__medium_interface_bitwidth__ = 512
__large_interface_bitwidth__ = 2048
__available_interface_bitwidth__ = [__small_interface_bitwidth__, __medium_interface_bitwidth__,
                                    __large_interface_bitwidth__]

__wrapper_min_depth__ = 512
# wrapper_interface_default_depth = 32
wrapper_interface_default_depth = __wrapper_min_depth__
wrapper_default_interface_bitwidth = __small_interface_bitwidth__

__filedir__ = os.path.dirname(os.path.abspath(__file__))


def get_fifo_name(name):
    return 'Fifo_{}'.format(name)


class WrapperInterface(metaclass=abc.ABCMeta):

    def __init__(self, mod_name, bw_s, target_hw, depth=wrapper_interface_default_depth, bitwidth=None):
        self.name = get_fifo_name(mod_name)
        self.bw_s = bw_s
        self.depth = depth
        if self.depth < __wrapper_min_depth__:
            self.depth = __wrapper_min_depth__
        self.target_hw = target_hw
        if bitwidth is None:
            self.bitwidth = self._get_wrapper_interface_bitwidth()
        else:
            self.bitwidth = bitwidth

    def _get_wrapper_interface_bitwidth(self):
        for bit_w in __available_interface_bitwidth__:
            bw_Bs = (bit_w / __bit_ber_byte__) / self.target_hw.clock_period_s
            if self.bw_s <= bw_Bs:
                return bit_w
        default = __available_interface_bitwidth__[-1]
        print(("[DOSA:InterfaceGeneration:WARNING] can't satisfy required input bandwidth of {} B/s with available " +
               "interface options. Defaulting to {}.").format(self.bw_s, default))
        return default

    def get_if_name(self):
        return self.name

    def get_if_bitwidth(self):
        return self.bitwidth

    @abc.abstractmethod
    def get_tcl_lines(self):
        print("[DOSA:WrapperGeneration:ERROR] NOT YET IMPLEMENTED.")

    @abc.abstractmethod
    def get_vhdl_entity_declaration(self):
        print("[DOSA:WrapperGeneration:ERROR] NOT YET IMPLEMENTED.")

    @abc.abstractmethod
    def get_vhdl_signal_declaration(self):
        print("[DOSA:WrapperGeneration:ERROR] NOT YET IMPLEMENTED.")

    @abc.abstractmethod
    def get_vhdl_entity_inst_tmpl(self):
        """empty string if no instantiation necessary (e.g. for pure AXIS)"""
        print("[DOSA:WrapperGeneration:ERROR] NOT YET IMPLEMENTED.")

    @abc.abstractmethod
    def get_vhdl_signal_dict(self):
        """:return dict with {to_signals: {}, from_signals: {} }.
                    if no instantiation necessary, to_signals and from_signals contain the exact same entries
        """
        print("[DOSA:WrapperGeneration:ERROR] NOT YET IMPLEMENTED.")

    @abc.abstractmethod
    def get_debug_lines(self):
        """
        :return: tcl_template lines, vhdl_decl lines, vhdl_instantiation lines as list
                    all having {i} as template for probe number
        """
        print("[DOSA:WrapperGeneration:ERROR] NOT YET IMPLEMENTED.")


class InterfaceVectorFifo(WrapperInterface):

    def get_tcl_lines(self):
        template_lines = Path(os.path.join(__filedir__, 'templates/create_if_fifo.tcl')).read_text()
        new_tcl_lines = template_lines.format(DOSA_FMSTR_NAME=self.name,
                                              DOSA_FMSTR_BITWIDTH='{' + str(self.bitwidth) + '}',
                                              DOSA_FMSTR_DEPTH='{' + str(self.depth) + '}')
        return new_tcl_lines

    def get_vhdl_entity_declaration(self):
        decl = ('component {name} is\n    port (\n       clk    : in std_logic;\n        srst   : in std_logic;\n' +
                '       din    : in std_logic_vector({width} downto 0);\n       full   : out std_logic;\n' +
                '       wr_en  : in std_logic;\n        dout   : out std_logic_vector({width} downto 0);\n' +
                '       empty  : out std_logic;\n       rd_en  : in std_logic );\nend component;\n') \
            .format(name=self.name, width=(self.bitwidth-1))
        return decl

    def get_vhdl_signal_declaration(self):
        signal_dict = self.get_vhdl_signal_dict()
        map_dict = {'in_sig_0': signal_dict['to_signals']['0'],
                    'in_sig_1_n': signal_dict['to_signals']['1_n'],
                    'in_sig_1': signal_dict['to_signals']['1'],
                    'in_sig_2': signal_dict['to_signals']['2'],
                    'out_sig_0': signal_dict['from_signals']['0'],
                    'out_sig_1_n': signal_dict['from_signals']['1_n'],
                    'out_sig_1': signal_dict['from_signals']['1'],
                    'out_sig_2': signal_dict['from_signals']['2'],
                    'width': (self.bitwidth-1)
                    }
        decl = ('signal {in_sig_0}    : std_ulogic_vector({width} downto 0);\n' +
                'signal {in_sig_1_n}  : std_ulogic;\n' +
                'signal {in_sig_1}    : std_ulogic;\n' +
                'signal {in_sig_2}    : std_ulogic;\n' +
                'signal {out_sig_0}    : std_ulogic_vector({width} downto 0);\n' +
                'signal {out_sig_1_n}  : std_ulogic;\n' +
                'signal {out_sig_1}    : std_ulogic;\n' +
                'signal {out_sig_2}    : std_ulogic;\n') \
            .format_map(map_dict)
        return decl

    def get_vhdl_entity_inst_tmpl(self):
        inst_tmpl = ('{in_sig_1_n} <= not {in_sig_1};\n' +
                     '{out_sig_1_n} <= not {out_sig_1};\n' +
                     '{inst_name}: ' + str(self.name) + '\n' +
                     'port map (\n' +
                     '           clk     => {clk},\n' +
                     '           srst    => {rst},\n' +
                     '           din     => {in_sig_0},\n' +
                     '           full    => {in_sig_1},\n' +
                     '           wr_en   => {in_sig_2},\n' +
                     '           dout    => {out_sig_0},\n' +
                     '           empty   => {out_sig_1},\n' +
                     '           rd_en   => {out_sig_2}\n' +
                     '         );\n')
        return inst_tmpl

    def get_vhdl_signal_dict(self):
        signal_dict = {'to_signals': {}, 'from_signals': {}}
        signal_dict['to_signals']['0'] = 'sTo{}_din'.format(self.name)
        signal_dict['to_signals']['1_n'] = 'sTo{}_full_n'.format(self.name)
        signal_dict['to_signals']['1'] = 'sTo{}_full'.format(self.name)
        signal_dict['to_signals']['2'] = 'sTo{}_write'.format(self.name)
        signal_dict['from_signals']['0'] = 'sFrom{}_dout'.format(self.name)
        signal_dict['from_signals']['1_n'] = 'sFrom{}_empty_n'.format(self.name)
        signal_dict['from_signals']['1'] = 'sFrom{}_empty'.format(self.name)
        signal_dict['from_signals']['2'] = 'sFrom{}_read'.format(self.name)
        return signal_dict

    def _get_vhdl_signal_width_dict(self):
        with_dict = {'to_signals': {}, 'from_signals': {}}
        with_dict['to_signals']['0']        = self.bitwidth
        with_dict['to_signals']['1_n']      = 1
        with_dict['to_signals']['1']        = 1
        with_dict['to_signals']['2']        = 1
        with_dict['from_signals']['0']      = self.bitwidth
        with_dict['from_signals']['1_n']    = 1
        with_dict['from_signals']['1']      = 1
        with_dict['from_signals']['2']      = 1
        return with_dict

    def get_debug_lines(self):
        signal_dict = self.get_vhdl_signal_dict()
        width_dict = self._get_vhdl_signal_width_dict()
        tcl_tmpl_lines = []
        decl_tmpl_lines = []
        inst_tmpl_lines = []

        for k in signal_dict['to_signals']:
            sn = signal_dict['to_signals'][k]
            sw = width_dict['to_signals'][k]
            tcl_l = 'CONFIG.C_PROBE{i}_WIDTH {{' + str(sw) + '}}\\\n'
            decl_l = '; probe{i}    : in  std_logic_vector( ' + str(sw - 1) + ' downto 0)\n'  # semicolon at begin
            if sw == 1:
                inst_l = ', probe{i}(0)   =>   ' + sn + '\n'  # comma at begin
            else:
                inst_l = ', probe{i}      =>   ' + sn + '\n'  # comma at begin
            tcl_tmpl_lines.append(tcl_l)
            decl_tmpl_lines.append(decl_l)
            inst_tmpl_lines.append(inst_l)

        for k in signal_dict['from_signals']:
            sn = signal_dict['from_signals'][k]
            sw = width_dict['from_signals'][k]
            tcl_l = 'CONFIG.C_PROBE{i}_WIDTH {{' + str(sw) + '}}\\\n'
            decl_l = '; probe{i}    : in  std_logic_vector( ' + str(sw - 1) + ' downto 0)\n'  # semicolon at begin
            if sw == 1:
                inst_l = ', probe{i}(0)   =>   ' + sn + '\n'  # comma at begin
            else:
                inst_l = ', probe{i}      =>   ' + sn + '\n'  # comma at begin
            tcl_tmpl_lines.append(tcl_l)
            decl_tmpl_lines.append(decl_l)
            inst_tmpl_lines.append(inst_l)

        return tcl_tmpl_lines, decl_tmpl_lines, inst_tmpl_lines


class InterfaceAxisFifo(WrapperInterface):

    def __init__(self, mod_name, bw_s, target_hw):
        super().__init__(mod_name, bw_s, target_hw)
        self._calc_bitw_ = -1
        self._calculate_bitws()

    def _calculate_bitws(self):
        self.tdata_bitw = self.bitwidth
        self.tkeep_bitw = int((self.bitwidth + 7) / 8)
        self.tlast_bitw = 1
        # the first fifo in a node might be re-scaled, so we might have to recalculate tkeep etc.
        self._calc_bitw_ = self.bitwidth

    def get_tcl_lines(self):
        if self._calc_bitw_ != self.bitwidth:
            self._calculate_bitws()
        template_lines = Path(os.path.join(__filedir__, 'templates/create_axis_fifo.tcl')).read_text()
        new_tcl_lines = template_lines.format(DOSA_FMSTR_NAME=self.name,
                                              DOSA_FMSTR_BITWIDTH_TDATA='{' + str(self.tdata_bitw) + '}',
                                              DOSA_FMSTR_BITWIDTH_TKEEP='{' + str(self.tkeep_bitw) + '}',
                                              DOSA_FMSTR_BITWIDTH_TLAST='{' + str(self.tlast_bitw) + '}',
                                              DOSA_FMSTR_DEPTH='{' + str(self.depth) + '}')
        return new_tcl_lines

    def get_vhdl_entity_declaration(self):
        if self._calc_bitw_ != self.bitwidth:
            self._calculate_bitws()
        single_decl = ('component {name} is\n' +
                       '    port (\n' +
                       '       clk    : in std_logic;\n' +
                       '       srst   : in std_logic;\n' +
                       '       din    : in std_logic_vector({width} downto 0);\n' +
                       '       full   : out std_logic;\n' +
                       '       wr_en  : in std_logic;\n' +
                       '       dout   : out std_logic_vector({width} downto 0);\n' +
                       '       empty  : out std_logic;\n' +
                       '       rd_en  : in std_logic );\n' +
                       'end component;\n')
        total_decl = single_decl.format(name=self.name + '_tdata', width=(self.tdata_bitw-1))
        total_decl += '\n'
        total_decl += single_decl.format(name=self.name + '_tkeep', width=(self.tkeep_bitw-1))
        total_decl += '\n'
        total_decl += single_decl.format(name=self.name + '_tlast', width=(self.tlast_bitw-1))
        # yes, results in (0 downto 0), but vivado_hls does this as well..
        return total_decl

    def get_vhdl_signal_declaration(self):
        if self._calc_bitw_ != self.bitwidth:
            self._calculate_bitws()
        signal_dict = self.get_vhdl_signal_dict()
        map_dict = {'in_sig_0': signal_dict['to_signals']['0'],
                    'in_sig_1_n': signal_dict['to_signals']['1_n'],
                    'in_sig_1': signal_dict['to_signals']['1'],
                    'in_sig_2': signal_dict['to_signals']['2'],
                    'out_sig_0': signal_dict['from_signals']['0'],
                    'out_sig_1_n': signal_dict['from_signals']['1_n'],
                    'out_sig_1': signal_dict['from_signals']['1'],
                    'out_sig_2': signal_dict['from_signals']['2'],
                    'width': (self.tdata_bitw-1),
                    'in_sig_3': signal_dict['to_signals']['3'],
                    'in_sig_4_n': signal_dict['to_signals']['4_n'],
                    'in_sig_4': signal_dict['to_signals']['4'],
                    'in_sig_5': signal_dict['to_signals']['5'],
                    'out_sig_3': signal_dict['from_signals']['3'],
                    'out_sig_4_n': signal_dict['from_signals']['4_n'],
                    'out_sig_4': signal_dict['from_signals']['4'],
                    'out_sig_5': signal_dict['from_signals']['5'],
                    'width_tkeep': (self.tkeep_bitw-1),
                    'in_sig_6': signal_dict['to_signals']['6'],
                    'in_sig_7_n': signal_dict['to_signals']['7_n'],
                    'in_sig_7': signal_dict['to_signals']['7'],
                    'in_sig_8': signal_dict['to_signals']['8'],
                    'out_sig_6': signal_dict['from_signals']['6'],
                    'out_sig_7_n': signal_dict['from_signals']['7_n'],
                    'out_sig_7': signal_dict['from_signals']['7'],
                    'out_sig_8': signal_dict['from_signals']['8'],
                    'width_tlast': (self.tlast_bitw-1)
                    }
        tmpl_decl = ('signal {in_sig_0}    : std_ulogic_vector({width} downto 0);\n' +
                     'signal {in_sig_1_n}  : std_ulogic;\n' +
                     'signal {in_sig_1}    : std_ulogic;\n' +
                     'signal {in_sig_2}    : std_ulogic;\n' +
                     'signal {out_sig_0}    : std_ulogic_vector({width} downto 0);\n' +
                     'signal {out_sig_1_n}  : std_ulogic;\n' +
                     'signal {out_sig_1}    : std_ulogic;\n' +
                     'signal {out_sig_2}    : std_ulogic;\n')
        tmpl_decl += '\n'
        tmpl_decl += ('signal {in_sig_3}    : std_ulogic_vector({width_tkeep} downto 0);\n' +
                      'signal {in_sig_4_n}  : std_ulogic;\n' +
                      'signal {in_sig_4}    : std_ulogic;\n' +
                      'signal {in_sig_5}    : std_ulogic;\n' +
                      'signal {out_sig_3}    : std_ulogic_vector({width_tkeep} downto 0);\n' +
                      'signal {out_sig_4_n}  : std_ulogic;\n' +
                      'signal {out_sig_4}    : std_ulogic;\n' +
                      'signal {out_sig_5}    : std_ulogic;\n')
        tmpl_decl += '\n'
        tmpl_decl += ('signal {in_sig_6}    : std_ulogic_vector({width_tlast} downto 0);\n' +
                      'signal {in_sig_7_n}  : std_ulogic;\n' +
                      'signal {in_sig_7}    : std_ulogic;\n' +
                      'signal {in_sig_8}    : std_ulogic;\n' +
                      'signal {out_sig_6}    : std_ulogic_vector({width_tlast} downto 0);\n' +
                      'signal {out_sig_7_n}  : std_ulogic;\n' +
                      'signal {out_sig_7}    : std_ulogic;\n' +
                      'signal {out_sig_8}    : std_ulogic;\n')
        decl = tmpl_decl.format_map(map_dict)
        return decl

    def get_vhdl_entity_inst_tmpl(self):
        if self._calc_bitw_ != self.bitwidth:
            self._calculate_bitws()
        inst_tmpl = ('{in_sig_1_n} <= not {in_sig_1};\n' +
                     '{out_sig_1_n} <= not {out_sig_1};\n' +
                     '{inst_name}_tdata: ' + str(self.name) + '_tdata' + '\n' +
                     'port map (\n' +
                     '           clk     => {clk},\n' +
                     '           srst    => {rst},\n' +
                     '           din     => {in_sig_0},\n' +
                     '           full    => {in_sig_1},\n' +
                     '           wr_en   => {in_sig_2},\n' +
                     '           dout    => {out_sig_0},\n' +
                     '           empty   => {out_sig_1},\n' +
                     '           rd_en   => {out_sig_2}\n' +
                     '         );\n')
        inst_tmpl += ('\n{in_sig_4_n} <= not {in_sig_4};\n' +
                      '{out_sig_4_n} <= not {out_sig_4};\n' +
                      '{inst_name}_tkeep: ' + str(self.name) + '_tkeep' + '\n' +
                      'port map (\n' +
                      '           clk     => {clk},\n' +
                      '           srst    => {rst},\n' +
                      '           din     => {in_sig_3},\n' +
                      '           full    => {in_sig_4},\n' +
                      '           wr_en   => {in_sig_5},\n' +
                      '           dout    => {out_sig_3},\n' +
                      '           empty   => {out_sig_4},\n' +
                      '           rd_en   => {out_sig_5}\n' +
                      '         );\n')
        inst_tmpl += ('\n{in_sig_7_n} <= not {in_sig_7};\n' +
                      '{out_sig_7_n} <= not {out_sig_7};\n' +
                      '{inst_name}_tlast: ' + str(self.name) + '_tlast' + '\n' +
                      'port map (\n' +
                      '           clk     => {clk},\n' +
                      '           srst    => {rst},\n' +
                      '           din     => {in_sig_6},\n' +
                      '           full    => {in_sig_7},\n' +
                      '           wr_en   => {in_sig_8},\n' +
                      '           dout    => {out_sig_6},\n' +
                      '           empty   => {out_sig_7},\n' +
                      '           rd_en   => {out_sig_8}\n' +
                      '         );\n')
        return inst_tmpl

    def get_vhdl_signal_dict(self):
        if self._calc_bitw_ != self.bitwidth:
            self._calculate_bitws()
        signal_dict = {'to_signals': {}, 'from_signals': {}}

        signal_dict['to_signals']['0'] = 'sTo{}_din'.format(self.name + '_tdata')
        signal_dict['to_signals']['1_n'] = 'sTo{}_full_n'.format(self.name + '_tdata')
        signal_dict['to_signals']['1'] = 'sTo{}_full'.format(self.name + '_tdata')
        signal_dict['to_signals']['2'] = 'sTo{}_write'.format(self.name + '_tdata')
        signal_dict['from_signals']['0'] = 'sTo{}_dout'.format(self.name + '_tdata')
        signal_dict['from_signals']['1_n'] = 'sTo{}_empty_n'.format(self.name + '_tdata')
        signal_dict['from_signals']['1'] = 'sTo{}_empty'.format(self.name + '_tdata')
        signal_dict['from_signals']['2'] = 'sTo{}_read'.format(self.name + '_tdata')

        signal_dict['to_signals']['3'] = 'sTo{}_din'.format(self.name + '_tkeep')
        signal_dict['to_signals']['4_n'] = 'sTo{}_full_n'.format(self.name + '_tkeep')
        signal_dict['to_signals']['4'] = 'sTo{}_full'.format(self.name + '_tkeep')
        signal_dict['to_signals']['5'] = 'sTo{}_write'.format(self.name + '_tkeep')
        signal_dict['from_signals']['3'] = 'sTo{}_dout'.format(self.name + '_tkeep')
        signal_dict['from_signals']['4_n'] = 'sTo{}_empty_n'.format(self.name + '_tkeep')
        signal_dict['from_signals']['4'] = 'sTo{}_empty'.format(self.name + '_tkeep')
        signal_dict['from_signals']['5'] = 'sTo{}_read'.format(self.name + '_tkeep')

        signal_dict['to_signals']['6'] = 'sTo{}_din'.format(self.name + '_tlast')
        signal_dict['to_signals']['7_n'] = 'sTo{}_full_n'.format(self.name + '_tlast')
        signal_dict['to_signals']['7'] = 'sTo{}_full'.format(self.name + '_tlast')
        signal_dict['to_signals']['8'] = 'sTo{}_write'.format(self.name + '_tlast')
        signal_dict['from_signals']['6'] = 'sTo{}_dout'.format(self.name + '_tlast')
        signal_dict['from_signals']['7_n'] = 'sTo{}_empty_n'.format(self.name + '_tlast')
        signal_dict['from_signals']['7'] = 'sTo{}_empty'.format(self.name + '_tlast')
        signal_dict['from_signals']['8'] = 'sTo{}_read'.format(self.name + '_tlast')
        return signal_dict

    def _get_vhdl_signal_width_dict(self):
        if self._calc_bitw_ != self.bitwidth:
            self._calculate_bitws()
        width_dict = {'to_signals': {}, 'from_signals': {}}

        width_dict['to_signals']['0']       = self.tdata_bitw
        width_dict['to_signals']['1_n']     = 1
        width_dict['to_signals']['1']       = 1
        width_dict['to_signals']['2']       = 1
        width_dict['from_signals']['0']     = self.tdata_bitw
        width_dict['from_signals']['1_n']   = 1
        width_dict['from_signals']['1']     = 1
        width_dict['from_signals']['2']     = 1

        width_dict['to_signals']['3']       = self.tkeep_bitw
        width_dict['to_signals']['4_n']     = 1
        width_dict['to_signals']['4']       = 1
        width_dict['to_signals']['5']       = 1
        width_dict['from_signals']['3']     = self.tkeep_bitw
        width_dict['from_signals']['4_n']   = 1
        width_dict['from_signals']['4']     = 1
        width_dict['from_signals']['5']     = 1

        width_dict['to_signals']['6']       = self.tlast_bitw + 0.1  # to allow right port map
        width_dict['to_signals']['7_n']     = 1
        width_dict['to_signals']['7']       = 1
        width_dict['to_signals']['8']       = 1
        width_dict['from_signals']['6']     = self.tlast_bitw + 0.1  # to allow right port map
        width_dict['from_signals']['7_n']   = 1
        width_dict['from_signals']['7']     = 1
        width_dict['from_signals']['8']     = 1
        return width_dict

    def get_debug_lines(self):
        if self._calc_bitw_ != self.bitwidth:
            self._calculate_bitws()
        signal_dict = self.get_vhdl_signal_dict()
        width_dict = self._get_vhdl_signal_width_dict()
        tcl_tmpl_lines = []
        decl_tmpl_lines = []
        inst_tmpl_lines = []

        for k in signal_dict['to_signals']:
            sn = signal_dict['to_signals'][k]
            sw = width_dict['to_signals'][k]
            tcl_l = 'CONFIG.C_PROBE{i}_WIDTH {{' + str(int(sw)) + '}}\\\n'
            decl_l = '; probe{i}    : in  std_logic_vector( ' + str(int(sw - 1)) + ' downto 0)\n'  # semicolon at begin
            if sw == 1:
                inst_l = ', probe{i}(0)   =>   ' + sn + '\n'  # comma at begin
            else:
                inst_l = ', probe{i}      =>   ' + sn + '\n'  # comma at begin
            tcl_tmpl_lines.append(tcl_l)
            decl_tmpl_lines.append(decl_l)
            inst_tmpl_lines.append(inst_l)

        for k in signal_dict['from_signals']:
            sn = signal_dict['from_signals'][k]
            sw = width_dict['from_signals'][k]
            tcl_l = 'CONFIG.C_PROBE{i}_WIDTH {{' + str(int(sw)) + '}}\\\n'
            decl_l = '; probe{i}    : in  std_logic_vector( ' + str(int(sw - 1)) + ' downto 0)\n'  # semicolon at begin
            if sw == 1:
                inst_l = ', probe{i}(0)   =>   ' + sn + '\n'  # comma at begin
            else:
                inst_l = ', probe{i}      =>   ' + sn + '\n'  # comma at begin
            tcl_tmpl_lines.append(tcl_l)
            decl_tmpl_lines.append(decl_l)
            inst_tmpl_lines.append(inst_l)

        return tcl_tmpl_lines, decl_tmpl_lines, inst_tmpl_lines

