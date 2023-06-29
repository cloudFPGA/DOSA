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
#  *     Created: Dec 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Present current analysis as 3D/deep roofline
#  *
#  *

import json
import math
import itertools

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib as mpl

import dimidium.lib.singleton as dosa_singleton
from dimidium.lib.util import rf_attainable_performance, OptimizationStrategies, BrickImplTypes, rf_calc_sweet_spot
from dimidium.middleend.archGen.ArchDraft import ArchDraft
from dimidium.middleend.archGen.ArchNode import ArchNode
from dimidium.backend.devices.dosa_device import placeholderHw, DosaHwClasses

from dimidium.lib.units import *
# from dimidium.backend.devices.dosa_roofline import config_global_rf_ylim_min as __ylim_min__
__ylim_min__ = 1
from dimidium.backend.devices.dosa_roofline import config_global_rf_ylim_max as __ylim_max__


plt.rcParams.update({'figure.max_open_warning': 0})


# uses global variables...so better here instead of putting it in util
def set_size(w, h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


def turn_to_log(n, maxv=None, disable_min=False):
    if n <= 0.001:
        return 0
    if disable_min:
        return math.log10(n)
    ret = max(math.log10(n), 0.01)
    if maxv is None:
        return ret
    else:
        return min(ret, maxv)


def turn_to_log_np(n):
    return np.log10(n)


def draw_oi_list(ax, color, line_style, font_size, line_width, y_max, oi_list, x_min, x_max, z_order=10, y_min=0.1,
                 show_labels=True, print_debug=False):
    # text_height_values = [65, 120, 55, 100, 180]
    # text_height_values = [6.5, 12.0, 5.5, 10.0, 18.0]
    text_height_values = [0.65, 1.20, 0.55, 1.00, 1.80]
    th = itertools.cycle(text_height_values)
    for e in oi_list:
        e_oi = turn_to_log(e['oi'])
        if e_oi > x_max or e_oi < x_min:
            if print_debug:
                print("[DOSA:roofline] Warning: required OI {} of {} out of range, correcting it to borders."
                      .format(e_oi, e['name']))
            # continue
            if e_oi > x_max:
                e_oi = x_max
            else:
                e_oi = x_min
        ax.plot([e_oi, e_oi], [turn_to_log(e['mem_share'], maxv=10.1), turn_to_log(e['comp_share'], maxv=10.1)],
                [y_min, y_max], color=color,
                linestyle=line_style, linewidth=line_width, zorder=z_order)
        if show_labels:
            text_y_shift_factor = 1.0
            if len(e['name']) > 15:
                text_y_shift_factor = 50.0
            z_text = (turn_to_log(e['mem_share']) + turn_to_log(e['comp_share'])) / 2
            ax.text(x=turn_to_log(e['oi'] * 1.02), z=turn_to_log(4 + next(th) * text_y_shift_factor), y=z_text, s=e['name'], color=color,
                    fontsize=font_size, ha='left', va='top', rotation=90, zorder=z_order-1)


def draw_oi_marker(ax, color, marker, oi_list, x_min, x_max, z_order=8, print_debug=False):
    x = []
    y = []
    z = []
    for e in oi_list:
        e_oi = turn_to_log(e['oi'])
        if e_oi > x_max or e_oi < x_min:
            if print_debug:
                print("[DOSA:roofline] Warning: required OI {} of {} out of range, correcting it to borders."
                      .format(e_oi, e['name']))
            # continue
            if e_oi > x_max:
                e_oi = x_max
            else:
                e_oi = x_min
        x.append(e_oi)
        if not (__ylim_min__ < e['perf'] < __ylim_max__):
            if print_debug:
                print("[DOSA:roofline] Warning: required performance {} of {} out of range, correcting it."
                      .format(e['perf'], e['name']))
            if __ylim_min__ > e['perf']:
                y.append(__ylim_min__)
            else:
                y.append(__ylim_max__)
        else:
            y.append(e['perf'])
        z_pos = turn_to_log((e['mem_share'] + e['comp_share']) / 2)
        z.append(z_pos)
    # xs = [turn_to_log(e) for e in x]
    ys = [turn_to_log(e) for e in y]
    ax.scatter(xs=x, zs=ys, ys=z, marker=marker, color=color, zorder=z_order)


def generate_roofline_plt(arch_draft: ArchDraft, show_splits=False, show_labels=True, print_debug=False):
    unit = gigaU
    target_string = ""
    if arch_draft.strategy == OptimizationStrategies.THROUGHPUT:
        target_string = "{} sps".format(arch_draft.target_sps)
    elif arch_draft.strategy == OptimizationStrategies.LATENCY:
        target_string = "{} s/req".format(arch_draft.target_latency)
    else:
        target_string = "max {} nodes".format(arch_draft.target_resources)
    cmpl_list = []
    uinp_list = []
    cmpl_list2 = []
    uinp_list2 = []
    total_flops = 0
    total_uinp_B = 0
    total_param_B = 0
    total_mem_share_engine = 0
    total_comp_share_engine = 0
    total_mem_share_stream = 0
    total_comp_share_stream = 0
    for bb in arch_draft.brick_iter_gen():
        if bb.skip_in_roofline:
            continue
        cn = {'name': "{}_engine".format(bb.brick_uuid), 'oi': bb.oi_engine,
              'mem_share': bb.req_util_mem_engine*100, 'comp_share': bb.req_util_comp_engine*100}
        un = {'name': "{}_stream".format(bb.brick_uuid), 'oi': bb.oi_stream,
              'mem_share': bb.req_util_mem_stream*100, 'comp_share': bb.req_util_comp_stream*100}
        if bb.req_flops > 0:
            req_flop_u_e = bb.req_flops / unit
            req_flop_u_s = req_flop_u_e
        else:
            req_flop_u_e = bb.req_flops_engine / unit
            req_flop_u_s = bb.req_flops_stream / unit
        cn2 = {'name': "{}_engine".format(bb.brick_uuid), 'oi': bb.oi_engine, 'perf': req_flop_u_e,
               'mem_share': bb.req_util_mem_engine*100, 'comp_share': bb.req_util_comp_engine*100}
        un2 = {'name': "{}_stream".format(bb.brick_uuid), 'oi': bb.oi_stream, 'perf': req_flop_u_s,
               'mem_share': bb.req_util_mem_stream*100, 'comp_share': bb.req_util_comp_stream*100}
        total_flops += bb.flops
        total_uinp_B += bb.input_bytes
        total_param_B += bb.parameter_bytes
        total_mem_share_engine += bb.req_util_mem_engine*100
        total_comp_share_engine += bb.req_util_comp_engine*100
        total_mem_share_stream += bb.req_util_mem_stream*100
        total_comp_share_stream += bb.req_util_comp_stream*100
        if bb.selected_impl_type == BrickImplTypes.UNDECIDED or bb.selected_impl_type == BrickImplTypes.ENGINE:
            cmpl_list.append(cn)
            cmpl_list2.append(cn2)
        if bb.selected_impl_type == BrickImplTypes.UNDECIDED or bb.selected_impl_type == BrickImplTypes.STREAM:
            uinp_list.append(un)
            uinp_list2.append(un2)
    total = {'flops': total_flops, 'para_B': total_param_B, 'uinp_B': total_uinp_B,
             'comp_share_engine': total_comp_share_engine/arch_draft.get_total_nodes_cnt(),
             'comp_share_stream': total_comp_share_stream/arch_draft.get_total_nodes_cnt(),
             'mem_share_stream': total_mem_share_stream/arch_draft.get_total_nodes_cnt(),
             'mem_share_engine': total_mem_share_engine/arch_draft.get_total_nodes_cnt()}
    # print(total)
    plt_name = "'{}'\n(draft: {}, opt: {}, #nodes: {})".format(arch_draft.name, arch_draft.version,
                                                            str(arch_draft.strategy).split('.')[-1],
                                                            arch_draft.get_total_nodes_cnt())
    return draw_roofline(plt_name, arch_draft.batch_size, arch_draft.target_hw_set[0].get_performance_dict(),
                         arch_draft.target_hw_set[0].get_roofline_dict(), target_string, cmpl_list, uinp_list,
                         cmpl_list2, uinp_list2, total, show_splits, show_labels, print_debug)


def generate_roofline_for_node_plt(arch_node: ArchNode, parent_draft: ArchDraft, show_splits=True, show_labels=True,
                                   selected_only=False, print_debug=False):
    unit = gigaU
    target_string = ""
    if parent_draft.strategy == OptimizationStrategies.THROUGHPUT:
        target_string = "{} sps".format(parent_draft.target_sps)
    elif parent_draft.strategy == OptimizationStrategies.LATENCY:
        target_string = "{} s/req".format(parent_draft.target_latency)
    else:
        target_string = "max {} nodes".format(parent_draft.target_resources)
    cmpl_list = []
    uinp_list = []
    cmpl_list2 = []
    uinp_list2 = []
    total_flops = 0
    total_uinp_B = 0
    total_param_B = 0
    total_mem_share_engine = 0
    total_comp_share_engine = 0
    total_mem_share_stream = 0
    total_comp_share_stream = 0
    for bb in arch_node.local_brick_iter_gen():
        cn = {'name': "{}_{}_engine".format(bb.brick_uuid, bb.fn_label), 'oi': bb.oi_engine,
              'mem_share': bb.req_util_mem_engine*100, 'comp_share': bb.req_util_comp_engine*100}
        un = {'name': "{}_{}_stream".format(bb.brick_uuid, bb.fn_label), 'oi': bb.oi_stream,
              'mem_share': bb.req_util_mem_stream*100, 'comp_share': bb.req_util_comp_stream*100}
        if bb.req_flops > 0:
            req_flop_u_e = bb.req_flops / unit
            req_flop_u_s = req_flop_u_e
        else:
            req_flop_u_e = bb.req_flops_engine / unit
            req_flop_u_s = bb.req_flops_stream / unit
        cn2 = {'name': "{}_{}_engine".format(bb.brick_uuid, bb.fn_label), 'oi': bb.oi_engine, 'perf': req_flop_u_e,
               'mem_share': bb.req_util_mem_engine*100, 'comp_share': bb.req_util_comp_engine*100}
        un2 = {'name': "{}_{}_stream".format(bb.brick_uuid, bb.fn_label), 'oi': bb.oi_stream, 'perf': req_flop_u_s,
               'mem_share': bb.req_util_mem_stream*100, 'comp_share': bb.req_util_comp_stream*100}
        total_flops += bb.flops
        total_uinp_B += bb.input_bytes
        total_param_B += bb.parameter_bytes
        total_mem_share_engine += bb.req_util_mem_engine*100
        total_comp_share_engine += bb.req_util_comp_engine*100
        total_mem_share_stream += bb.req_util_mem_stream*100
        total_comp_share_stream += bb.req_util_comp_stream*100
        if selected_only:
            if bb.selected_impl_type == BrickImplTypes.ENGINE:
                cmpl_list.append(cn)
                cmpl_list2.append(cn2)
            elif bb.selected_impl_type == BrickImplTypes.STREAM:
                uinp_list.append(un)
                uinp_list2.append(un2)
        else:
            cmpl_list.append(cn)
            uinp_list.append(un)
            cmpl_list2.append(cn2)
            uinp_list2.append(un2)
    total = {'flops': total_flops, 'para_B': total_param_B, 'uinp_B': total_uinp_B,
             'comp_share_engine': total_comp_share_engine, 'comp_share_stream': total_comp_share_stream,
             'mem_share_stream': total_mem_share_stream, 'mem_share_engine': total_mem_share_engine}
    plt_name = "{}\n(draft: {}, node: {}, dpl: {}, opt: {})".format(parent_draft.name, parent_draft.version,
                                                                   arch_node.get_node_id(),
                                                                   arch_node.data_parallelism_level,
                                                                   str(parent_draft.strategy).split('.')[-1])
    perf_dict = arch_node.targeted_hw.get_performance_dict()
    rl_dict = arch_node.targeted_hw.get_roofline_dict()
    if arch_node.selected_hw_type != placeholderHw:
        perf_dict = arch_node.selected_hw_type.get_performance_dict()
        rl_dict = arch_node.selected_hw_type.get_roofline_dict()
    return draw_roofline(plt_name, parent_draft.batch_size, perf_dict,
                         rl_dict, target_string, cmpl_list, uinp_list,
                         cmpl_list2, uinp_list2, total, show_splits, show_labels, print_debug)


def draw_roofline(used_name, used_batch, perf_dict, roofline_dict, target_string, cmpl_list, uinp_list, cmpl_list2,
                  uinp_list2,
                  total, show_splits=True, show_labels=True, print_debug=False):
    # Arithmetic intensity vector
    oi_list_very_small = np.arange(0.001, 0.01, 0.001)
    oi_list_small = np.arange(0.01, 1, 0.01)
    oi_list_middle = np.arange(1, 1500, 1)
    oi_list_big = np.arange(1501, 10100, 100)
    # ai_list_t = [turn_to_log(e) for e in np.concatenate((ai_list_small, ai_list_middle, ai_list_big))]
    # oi_list = np.asarray(ai_list_t)
    oi_list_full = np.concatenate((oi_list_very_small, oi_list_small, oi_list_middle, oi_list_big))
    oi_list = turn_to_log_np(oi_list_full)
    utt_list_full = np.arange(0, 100, 1)
    utt_list = np.concatenate(([0], turn_to_log_np(np.arange(1, 100, 1))))
    # plots
    # fig, ax1 = plt.subplots()
    MY_SIZE = 26
    MY_SIZE_SMALL = MY_SIZE * 0.6
    MY_WIDTH = 1.6
    # line_style = 'dotted'
    line_style = 'solid'

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax2 = Axes3D(fig)

    # Attainable performance
    if perf_dict['type'] in [str(DosaHwClasses.FPGA_xilinx), str(DosaHwClasses.FPGA_generic)]:
        upper_limit_full = perf_dict['dsp48_gflops']
        upper_limit = turn_to_log(upper_limit_full)
        # p_fpga_ddr_max = np.asarray([rf_attainable_performance(x, upper_limit, perf_dict['bw_dram_gBs'])
        #                                             for x in oi_list])
        # p_fpga_bram_max = np.asarray([rf_attainable_performance(x, upper_limit, perf_dict['bw_bram_gBs'])
        #                                              for x in oi_list])
        # p_fpga_network_max = np.asarray([rf_attainable_performance(x, upper_limit, perf_dict['bw_netw_gBs'])
        #                                                 for x in oi_list])
        # p_fpga_lutram_max = np.asarray([rf_attainable_performance(x, upper_limit, perf_dict['bw_lutram_gBs'])
        #                                                for x in oi_list])

        # ax.plot(oi_list, p_fpga_lutram_max, color='tab:orange', linewidth=MY_WIDTH, label='current Role LUTRAM bandwidth', linestyle=line_style, zorder=1)
        # ax.plot_surface(X, Y, Z,
        # verts_lutram = [list(zip(oi_list, utt_list, p_fpga_lutram_max))]
        lutram_sp = turn_to_log(rf_calc_sweet_spot(oi_list_full, upper_limit_full, perf_dict['bw_lutram_gBs']),
                                disable_min=True)
        x = [oi_list[0], lutram_sp, lutram_sp, oi_list[0]]
        y = [utt_list[0], utt_list[0], utt_list[-1], utt_list[-1]]
        z = [0, upper_limit, upper_limit, 0]
        verts = [list(zip(x, y, z))]
        color = 'tab:orange'
        label = 'current Role LUTRAM bandwidth'
        # print(verts_lutram)
        l1 = Poly3DCollection(verts, color=color, linewidth=MY_WIDTH, label=label, linestyle=line_style, zorder=1)
        l1._edgecolors2d = l1._facecolor3d
        l1._facecolors2d = l1._facecolor3d
        ax.add_collection3d(l1)
        # ax.plot(oi_list, p_fpga_ddr_max, color='tab:red', linewidth=MY_WIDTH, label='current Role DRAM bandwidth',
        #         linestyle=line_style, zorder=1)
        bram_sp = turn_to_log(rf_calc_sweet_spot(oi_list_full, upper_limit_full, perf_dict['bw_bram_gBs']))
        x = [oi_list[0], bram_sp, bram_sp, oi_list[0]]
        verts = [list(zip(x, y, z))]
        color = 'tab:blue'
        label = 'current Role BRAM bandwidth'
        l1 = Poly3DCollection(verts, color=color, linewidth=MY_WIDTH, label=label, linestyle=line_style, zorder=1)
        l1._edgecolors2d = l1._facecolor3d
        l1._facecolors2d = l1._facecolor3d
        ax.add_collection3d(l1)
        # ax.plot(oi_list, p_fpga_bram_max, color='tab:blue', linewidth=MY_WIDTH, label='current Role BRAM bandwidth',
        #         linestyle=line_style, zorder=1)
        dram_sp = turn_to_log(rf_calc_sweet_spot(oi_list_full, upper_limit_full, perf_dict['bw_dram_gBs']))
        x = [oi_list[0], dram_sp, dram_sp, oi_list[0]]
        verts = [list(zip(x, y, z))]
        color = 'tab:red'
        label = 'current Role DRAM bandwidth'
        l1 = Poly3DCollection(verts, color=color, linewidth=MY_WIDTH, label=label, linestyle=line_style, zorder=1)
        l1._edgecolors2d = l1._facecolor3d
        l1._facecolors2d = l1._facecolor3d
        ax.add_collection3d(l1)
        # ax.plot(oi_list, p_fpga_network_max, color='tab:green', linewidth=MY_WIDTH, label='current Role network bandwidth',
        #         linestyle=line_style, zorder=1)
        net_sp = turn_to_log(rf_calc_sweet_spot(oi_list_full, upper_limit_full, perf_dict['bw_netw_gBs']))
        x = [oi_list[0], net_sp, net_sp, oi_list[0]]
        verts = [list(zip(x, y, z))]
        color = 'tab:green'
        label = 'current Role network bandwidth'
        l1 = Poly3DCollection(verts, color=color, linewidth=MY_WIDTH, label=label, linestyle=line_style, zorder=1)
        l1._edgecolors2d = l1._facecolor3d
        l1._facecolors2d = l1._facecolor3d
        ax.add_collection3d(l1)

        # sweet_spot = turn_to_log(roofline_dict['sweet_spot'])
        sweet_spot = lutram_sp
        color = 'darkmagenta'
        line_style = 'solid'  # otherwise we see the memory lines...
        x = [sweet_spot, oi_list[-1], oi_list[-1], sweet_spot]
        y = [utt_list[0], utt_list[0], utt_list[-1], utt_list[-1]]
        z = [upper_limit, upper_limit, upper_limit, upper_limit]
        verts_limit = [list(zip(x, y, z))]
        # print(verts_lutram)
        l1 = Poly3DCollection(verts_limit,
                              color=color, linewidth=MY_WIDTH * 1.2,
                              label='current Role peak performance', linestyle=line_style, zorder=3
                              )
        l1._edgecolors2d = l1._facecolor3d
        l1._facecolors2d = l1._facecolor3d
        ax.add_collection3d(l1)
        # ax.plot([sweet_spot, oi_list[-1]], [upper_limit, upper_limit], [0, 0], color=color, linestyle=line_style,
        #         linewidth=MY_WIDTH * 1.2, zorder=3)
        # sweet_spot_plane = [upper_limit for e in oi_list]
        # ax.plot_surface(X, Y, sweet_spot_plane)
        text = "{:.2f} GFLOPS/s theoretical DSP peak performance (for ROLE, {})" \
            .format(upper_limit, dosa_singleton.config.dtype.dosa_flops_explanation_str)
        # text_space = 100
        text_space = 10
        ax.text(x=sweet_spot, z=upper_limit + text_space, y=0, s=text, color=color, fontsize=MY_SIZE_SMALL)
    elif perf_dict['type'] in [str(DosaHwClasses.CPU_x86), str(DosaHwClasses.CPU_generic)]:
        upper_limit_full = perf_dict['cpu_gflops']
        upper_limit = turn_to_log(upper_limit_full)
        y = [utt_list[0], utt_list[0], utt_list[-1], utt_list[-1]]
        z = [0, upper_limit, upper_limit, 0]

        dram_sp = turn_to_log(rf_calc_sweet_spot(oi_list_full, upper_limit_full, perf_dict['bw_dram_gBs']))
        x = [oi_list[0], dram_sp, dram_sp, oi_list[0]]
        verts = [list(zip(x, y, z))]
        color = 'tab:red'
        label = 'current CPU DRAM bandwidth'
        l1 = Poly3DCollection(verts, color=color, linewidth=MY_WIDTH, label=label, linestyle=line_style, zorder=1)
        l1._edgecolors2d = l1._facecolor3d
        l1._facecolors2d = l1._facecolor3d
        ax.add_collection3d(l1)

        net_sp = turn_to_log(rf_calc_sweet_spot(oi_list_full, upper_limit_full, perf_dict['bw_netw_gBs']))
        x = [oi_list[0], net_sp, net_sp, oi_list[0]]
        verts = [list(zip(x, y, z))]
        color = 'tab:green'
        label = 'current CPU network bandwidth'
        l1 = Poly3DCollection(verts, color=color, linewidth=MY_WIDTH, label=label, linestyle=line_style, zorder=1)
        l1._edgecolors2d = l1._facecolor3d
        l1._facecolors2d = l1._facecolor3d
        ax.add_collection3d(l1)

        # sweet_spot = turn_to_log(roofline_dict['sweet_spot'])
        sweet_spot = dram_sp
        color = 'darkmagenta'
        line_style = 'solid'  # otherwise we see the memory lines...
        x = [sweet_spot, oi_list[-1], oi_list[-1], sweet_spot]
        y = [utt_list[0], utt_list[0], utt_list[-1], utt_list[-1]]
        z = [upper_limit, upper_limit, upper_limit, upper_limit]
        verts_limit = [list(zip(x, y, z))]
        # print(verts_lutram)
        l1 = Poly3DCollection(verts_limit,
                              color=color, linewidth=MY_WIDTH * 1.2,
                              label='current CPU peak performance', linestyle=line_style, zorder=3
                              )
        l1._edgecolors2d = l1._facecolor3d
        l1._facecolors2d = l1._facecolor3d
        ax.add_collection3d(l1)
        # ax.plot([sweet_spot, oi_list[-1]], [upper_limit, upper_limit], [0, 0], color=color, linestyle=line_style,
        #         linewidth=MY_WIDTH * 1.2, zorder=3)
        # sweet_spot_plane = [upper_limit for e in oi_list]
        # ax.plot_surface(X, Y, sweet_spot_plane)
        text = "{:.2f} GFLOPS/s theoretical CPU peak performance".format(upper_limit)
        # text_space = 100
        text_space = 10
        ax.text(x=sweet_spot, z=upper_limit + text_space, y=0, s=text, color=color, fontsize=MY_SIZE_SMALL)

    # custommarker = Path.circle()
    # color = 'darkturquoise'
    color = 'chocolate'
    # color2 = 'mediumspringgreen'
    color2 = 'firebrick'
    line_style = 'dashed'
    font_factor = 0.8
    marker1 = 'P'
    marker2 = 'D'

    draw_oi_list(ax, color, line_style, MY_SIZE * font_factor, MY_WIDTH * 1.2, upper_limit, cmpl_list,
                 oi_list[0], oi_list[-1], y_min=-0.1, show_labels=show_labels, print_debug=print_debug)
    draw_oi_list(ax, color2, line_style, MY_SIZE * font_factor, MY_WIDTH * 1.2, upper_limit, uinp_list,
                 oi_list[0], oi_list[-1], y_min=-0.1, show_labels=show_labels, print_debug=print_debug)

    draw_oi_marker(ax, color, marker1, cmpl_list2, oi_list[0], oi_list[-1], print_debug=print_debug)
    draw_oi_marker(ax, color2, marker2, uinp_list2, oi_list[0], oi_list[-1], print_debug=print_debug)
    marker1_text = 'req. perf. f. Engine arch. (w/ {}, batch {})'.format(target_string, used_batch)
    marker1_legend = mpl.lines.Line2D([], [], color=color, marker=marker1, linestyle='None', markersize=10,
                                      label=marker1_text)
    marker2_text = 'req. perf. f. Stream arch. (w/ {}, batch {})'.format(target_string, used_batch)
    marker2_legend = mpl.lines.Line2D([], [], color=color2, marker=marker2, linestyle='None', markersize=10,
                                      label=marker2_text)

    # color3 = 'orchid'
    color3 = 'aqua'
    oai_avg = turn_to_log(total['flops'] / (total['uinp_B'] + total['para_B']))
    zs = [turn_to_log(total['mem_share_engine'], maxv=10.1), turn_to_log(total['comp_share_engine'], maxv=10.1)]
    ax.plot([oai_avg, oai_avg], zs, [-0.1, upper_limit], color=color3, linestyle=line_style, linewidth=MY_WIDTH * 1.2,
            zorder=8)
    text = 'Engine avg.'
    z_text = (zs[0] + zs[1]) / 2
    ax.text(x=oai_avg * 1.02, z=1, y=z_text, s=text, color=color3, fontsize=MY_SIZE * font_factor, ha='left', va='top',
            rotation=90, zorder=8)
    if print_debug:
        print("[DOSA:roofline] Info: {} at {} ({}).".format(text, oai_avg, used_name))
    oai_avg2 = turn_to_log(total['flops'] / total['uinp_B'])
    zs2 = [turn_to_log(total['mem_share_stream'], maxv=10.1), turn_to_log(total['comp_share_stream'], maxv=10.1)]
    ax.plot([oai_avg2, oai_avg2], zs2, [-0.1, upper_limit], color=color3, linestyle=line_style,
            linewidth=MY_WIDTH * 1.2,
            zorder=8)
    text = 'Stream avg.'
    z_text = (zs2[0] + zs2[1]) / 2
    ax.text(x=oai_avg2 * 1.02, z=1, y=z_text, s=text, color=color3, fontsize=MY_SIZE * font_factor, ha='left', va='top',
            rotation=90, zorder=8)
    if print_debug:
        print("[DOSA:roofline] Info: {} at {} ({}).".format(text, oai_avg2, used_name))

    # if show_splits:
    #     # sweet_spot_index = p_fpga_network_max.index(upper_limit)
    #     # network_sweet_spot = math.floor(oi_list[sweet_spot_index])
    #     # upper_limit_list = [upper_limit for x in oi_list]
    #     alpha = 0.4
    #     # color = 'peru'
    #     color = 'tab:green'
    #     z_order = 0
    #     text = 'split w/ data parallelization'
    #     plt.fill_between(oi_list, p_fpga_network_max, upper_limit,
    #                      color=color, alpha=alpha, rasterized=True, zorder=z_order, label=text)
    #     # color = 'lightsalmon'
    #     color = 'tab:red'
    #     text = 'split w/ compute paral. (for Engines)'
    #     plt.fill_between(oi_list, p_fpga_ddr_max, upper_limit,
    #                      color=color, alpha=alpha, rasterized=True, zorder=z_order, label=text)
    #     # color = 'lightcoral'
    #     color = 'tab:blue'
    #     text = 'split w/ compute paral. (for Streams)'
    #     plt.fill_between(oi_list, p_fpga_bram_max, upper_limit,
    #                      color=color, alpha=alpha, rasterized=True, zorder=z_order, label=text)
    #     # color = 'goldenrod'
    #     color = 'darkmagenta'
    #     text = 'split w/ compute parallelization (in all cases)'
    #     plt.fill_between(oi_list, upper_limit, 1000000,
    #                      color=color, alpha=alpha, rasterized=True, zorder=z_order, label=text)

    # ax.xaxis.set_scale('log', base=10)
    # ax.yaxis.set_scale('log', base=10)
    ax.set_xticks(oi_list[::100])
    x_ticks_text = ['{:.2f}'.format(e) for e in oi_list_full[::100]]
    ax.set_xticklabels(x_ticks_text)
    ax.set_yticks(utt_list[::10])
    ax.set_yticklabels(utt_list_full[::10])
    # zticks_full = np.concatenate((np.asarray([__ylim_min__, 0.05, 0.1, 0.5]), np.arange(1, 2*upper_limit_full, 10)))
    # zticks_full = np.arange(1, 2*upper_limit_full, 10)
    zticks_full = np.concatenate((np.asarray([1, 10, 30, 50, 70, 90]), np.arange(100, 2*upper_limit_full, 50)))
    zticks = turn_to_log_np(zticks_full)
    ax.set_zticks(zticks)
    z_ticks_text = ['{:.2f}'.format(e) for e in zticks_full]
    ax.set_zticklabels(z_ticks_text)

    ax.set_xlim3d(oi_list[0], oi_list[-1])
    # ax.set_zlim3d(__ylim_min__, __ylim_max__)
    # ax.set_zlim3d(__ylim_min__, upper_limit*1.7)
    ax.set_zlim3d(zticks[0], zticks[-1])
    ax.set_ylim3d(utt_list[0], utt_list[-1])

    ax.set_xlabel('operational intensity (OI)\n[FLOPS/Byte]', fontsize=MY_SIZE)
    ax.set_zlabel('attainable performance\n[GFLOPS/s]', fontsize=MY_SIZE)
    ax.set_ylabel('utilization [%]\n(memory (bottom) ->\ncompute (top))', fontsize=MY_SIZE)

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(marker1_legend)
    handles.append(marker2_legend)
    title = "DOSA 3D Roofline for {}".format(used_name)
    # legend = plt.legend(handles=handles, ncol=3, fontsize=MY_SIZE, title=title)
    ax.legend(handles=handles, loc=0, ncol=3)
    # plt.grid(True, which="major", ls="-", color='0.89')
    # plt.tick_params(axis='both', which='both', labelsize=MY_SIZE)
    # plt.setp(legend.get_title(), fontsize=MY_SIZE * 1.2)
    plt.title(title, fontsize=MY_SIZE*1.2)

    # plt.subplots_adjust(top=0.8)
    return plt
