#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Jan 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Tools for the generation of wrapper interfaces
#  *
#  *

import os
from pathlib import Path

from dimidium.backend.devices.dosa_device import DosaBaseHw

__bit_ber_byte_ = 8
__small_interface_bitwidth__ = 64
__medium_interface_bitwidth__ = 512
__large_interface_bitwidth__ = 2048
__available_interface_bitwidth__ = [__small_interface_bitwidth__, __medium_interface_bitwidth__,
                                    __large_interface_bitwidth__]


def get_wrapper_interface_bitwidth(input_bw_s, target_hw: DosaBaseHw):
    for bit_w in __available_interface_bitwidth__:
        bw_Bs = (bit_w/__bit_ber_byte_)/target_hw.clock_period_s
        if input_bw_s <= bw_Bs:
            return bit_w
    default = __available_interface_bitwidth__[-1]
    print(("[DOSA:InterfaceGeneration:WARNING] can't statisfy required input bandwidth of {} B/s with available " +
          "interface options. Defaulting to {}.").format(input_bw_s, default))
    return default


wrapper_interface_default_depth = 32


def get_tcl_lines_interface_fifo(name, bitwidth, depth):
    filedir = os.path.dirname(os.path.abspath(__file__))
    template_lines = Path(os.path.join(filedir, 'templates/create_if_fifo.tcl')).read_text()
    new_tcl_lines = template_lines.format(DOSA_FMSTR_NAME=name, DOSA_FMSTR_BITWIDTH='{'+str(bitwidth)+'}',
                                          DOSA_FMSTR_DEPTH='{'+str(depth)+'}')
    return new_tcl_lines


def get_fifo_name(name):
    return 'Fifo_{}'.format(name)


def get_tcl_lines_if_axis_fifo(name, bitwidth, depth):
    tdata_bitw = bitwidth
    tkeep_bitw = (bitwidth+7)/8
    tlast_bitw = 1
    filedir = os.path.dirname(os.path.abspath(__file__))
    template_lines = Path(os.path.join(filedir, 'templates/create_axis_fifo.tcl')).read_text()
    new_tcl_lines = template_lines.format(DOSA_FMSTR_NAME=name, DOSA_FMSTR_BITWIDTH_TDATA='{'+str(tdata_bitw)+'}',
                                          DOSA_FMSTR_BITWIDTH_TKEEP='{'+str(tkeep_bitw)+'}',
                                          DOSA_FMSTR_BITWIDTH_TLAST='{'+str(tlast_bitw)+'}',
                                          DOSA_FMSTR_DEPTH='{'+str(depth)+'}')
    return new_tcl_lines

