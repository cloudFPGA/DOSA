#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Jun 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Present current analysis as roofline
#  *
#  *

import json
import itertools
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.path import Path
from matplotlib.patches import Ellipse

from dimidium.lib.util import convert_oi_list_for_plot, ap
from dimidium.lib.archGen import calculate_required_performance

from dimidium.lib.units import *
__ylim_min__ = 0.01
__ylim_max__ = 100000


# https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units
# uses global variables...so better here instead of putting it in util
def set_size(w, h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


def draw_oi_list(plt, color, line_style, font_size, line_width, y_max, oi_list, z_order=5, y_min=0.1, show_labels=True):
    text_height_values = [65, 120, 55, 100, 180]
    th = itertools.cycle(text_height_values)
    for e in oi_list:
        plt.vlines(x=e['oi'], ymin=y_min, ymax=y_max, colors=color, linestyles=line_style, linewidth=line_width,
                   zorder=z_order)  # , label=e['name'])
        if show_labels:
            plt.text(x=e['oi']*1.02, y=next(th), s=e['name'], color=color, fontsize=font_size, ha='left', va='top',
                     rotation=90)


def draw_oi_marker(plt, color, marker, oi_list, z_order=8):
    x = []
    y = []
    for e in oi_list:
        x.append(e['oi'])
        if not (__ylim_min__ < e['perf'] < __ylim_max__):
            print("[DOSA:roofline] Warning: required performance {} out of range.".format(e['perf']))
            if __ylim_min__ > e['perf']:
                y.append(__ylim_min__)
            else:
                y.append(__ylim_max__)
        else:
            y.append(e['perf'])
    plt.scatter(x=x, y=y, marker=marker, color=color, zorder=z_order)


def generate_roofline_plt(detailed_analysis, target_sps, used_batch, used_name, perf_dict, roofline_dict,
                          show_splits=True, show_labels=True):

    # Arithmetic intensity vector
    ai_list_small = np.arange(0.01, 1, 0.01)
    ai_list_middle = np.arange(1, 1500, 0.1)
    ai_list_big = np.arange(1501, 10100, 1)
    ai_list = np.concatenate((ai_list_small, ai_list_middle, ai_list_big))

    # Attainable performance
    upper_limit = perf_dict['dsp48_gflops']
    p_fpga_ddr_max = [ap(x, upper_limit, perf_dict['bw_ddr4_gBs']) for x in ai_list]
    p_fpga_bram_max = [ap(x, upper_limit, perf_dict['bw_bram_gBs']) for x in ai_list]
    p_fpga_network_max = [ap(x, upper_limit, perf_dict['bw_netw_gBs']) for x in ai_list]
    p_fpga_lutram_max = [ap(x, upper_limit, perf_dict['bw_lutram_gBs']) for x in ai_list]

    # p_fpga_ddr_mantle = [ap(x, upper_limit, b_s_mantle_ddr_gBs) for x in ai_list]
    # p_fpga_bram_mantle = [ap(x, upper_limit, b_s_mantle_bram_gBs) for x in ai_list]
    # p_fpga_network_mantle = [ap(x, upper_limit, b_s_mantle_eth_gBs) for x in ai_list]
    # p_fpga_lutram_mantle = [ap(x, upper_limit, b_s_mantle_lutram_gBs) for x in ai_list]

    # plots
    # fig, ax1 = plt.subplots()
    MY_SIZE = 16
    MY_SIZE_SMALL = 15
    MY_WIDTH = 1.6
    # line_style = 'dotted'
    line_style = 'solid'

    plt.figure()
    plt.plot(ai_list, p_fpga_lutram_max, color='tab:orange', linewidth=MY_WIDTH, label='current Role LUTRAM bandwidth', linestyle=line_style, zorder=1)
    plt.plot(ai_list, p_fpga_ddr_max, color='tab:red', linewidth=MY_WIDTH, label='current Role DRAM bandwidth', linestyle=line_style, zorder=1)
    plt.plot(ai_list, p_fpga_bram_max, color='tab:blue', linewidth=MY_WIDTH, label='current Role BRAM bandwidth', linestyle=line_style, zorder=1)
    plt.plot(ai_list, p_fpga_network_max, color='tab:green', linewidth=MY_WIDTH, label='current Role network bandwidth', linestyle=line_style, zorder=1)

    # line_style = 'solid'
    # plt.plot(ai_list, p_fpga_lutram_mantle, color='bisque', linewidth=MY_WIDTH, label='Mantle LUTRAM bandwidth', linestyle=line_style, zorder=1)
    # plt.plot(ai_list, p_fpga_ddr_mantle, color='indianred', linewidth=MY_WIDTH, label='Mantle DRAM bandwidth', linestyle=line_style, zorder=1)
    # plt.plot(ai_list, p_fpga_bram_mantle, color='cornflowerblue', linewidth=MY_WIDTH, label='Mantle BRAM bandwidth', linestyle=line_style, zorder=1)
    # plt.plot(ai_list, p_fpga_network_mantle, color='palegreen', linewidth=MY_WIDTH, label='Mantle network bandwidth', linestyle=line_style, zorder=1)

    # color = 'tomato'
    # alpha=0.7
    # # rasterized to reduce size of PDF...
    # plt.fill_between(ai_list, p_fpga_lutram_mantle, p_fpga_lutram_max, color=color, alpha=alpha, rasterized=True, label='potential lost performance')
    # plt.fill_between(ai_list, p_fpga_ddr_mantle, p_fpga_ddr_max, color=color, alpha=alpha, rasterized=True)
    # plt.fill_between(ai_list, p_fpga_bram_mantle, p_fpga_bram_max, color=color, alpha=alpha, rasterized=True)
    # plt.fill_between(ai_list, p_fpga_network_mantle, p_fpga_network_max, color=color, alpha=alpha, rasterized=True)


    # mantle_sweet_spot = 0.0797
    # text = "Mantle reduced peak perf."
    # plt.hlines(y=cF_mantle_dsp48_gflops, xmin=mantle_sweet_spot, xmax=ai_list[-1], colors='orchid', linestyles=line_style,
    #           linewidth=MY_WIDTH, zorder=3, label=text)

    # plt.fill_between(np.arange(mantle_sweet_spot, ai_list[-1], 0.1), cF_bigRole_dsp48_gflops, cF_mantle_dsp48_gflops,
    #                 color='tomato', alpha=alpha, rasterized=True, zorder=2)

    sweet_spot = roofline_dict['sweet_spot']
    color = 'darkmagenta'
    line_style = 'solid'  # otherwise we see the memory lines...
    plt.hlines(y=upper_limit, xmin=sweet_spot, xmax=ai_list[-1], colors=color, linestyles=line_style, linewidth=MY_WIDTH*1.2, zorder=3)
    text = "{0:.2f} GFLOPS/s theoretical DSP peak performance (for ROLE)".format(upper_limit)
    plt.text(x=sweet_spot, y=upper_limit+100, s=text, color=color, fontsize=MY_SIZE_SMALL)

    # custommarker = Path.circle()
    # color = 'darkturquoise'
    color = 'chocolate'
    # color2 = 'mediumspringgreen'
    color2 = 'firebrick'
    line_style = 'dashed'
    font_factor = 0.8
    marker1 = 'P'
    marker2 = 'D'

    # marker_line = 65
    # oai = 0.17
    # plt.vlines(x=oai, ymin=0.1, ymax=cF_mantle_dsp48_gflops, colors=color, linestyles=line_style, linewidth=MY_WIDTH*1.2, zorder=5
    #            , label='OI of example domains')
    # text = 'Sparse\nMatrix\n\n\nSimple\nHash\nJoin'
    # plt.text(x=oai*1.1, y=marker_line-55, s=text, color=color, fontsize=MY_SIZE*font_factor, ha='left', va='top')
    #
    # marker_line = 105-20
    # oai = 0.5
    # plt.vlines(x=oai, ymin=0.1, ymax=cF_mantle_dsp48_gflops, colors=color, linestyles=line_style, linewidth=MY_WIDTH*1.2, zorder=5)
    # text = 'Stencil'
    # plt.text(x=oai*1.1, y=marker_line-55, s=text, color=color, fontsize=MY_SIZE*font_factor, ha='left', va='top')
    #
    # marker_line = 175-20
    # oai = 1.5
    # plt.vlines(x=oai, ymin=0.1, ymax=cF_mantle_dsp48_gflops, colors=color, linestyles=line_style, linewidth=MY_WIDTH*1.2, zorder=5)
    # text = '3D FFT'
    # plt.text(x=oai*1.1, y=marker_line-55, s=text, color=color, fontsize=MY_SIZE*font_factor, ha='left', va='top')
    #
    # marker_line = 56
    # oai = 5.2
    # plt.vlines(x=oai, ymin=0.1, ymax=cF_mantle_dsp48_gflops, colors=color, linestyles=line_style, linewidth=MY_WIDTH*1.2, zorder=5)
    # text = 'Dense\nMatrix'
    # plt.text(x=oai*1.1, y=marker_line-55, s=text, color=color, fontsize=MY_SIZE*font_factor, ha='left', va='top')
    #
    # marker_line = 58
    # oai = 16.1
    # plt.vlines(x=oai, ymin=0.1, ymax=cF_mantle_dsp48_gflops, colors=color, linestyles=line_style, linewidth=MY_WIDTH*1.2, zorder=5)
    # text = 'Particle Methods'
    # plt.text(x=oai*1.1, y=marker_line-55, s=text, color=color, fontsize=MY_SIZE*font_factor, ha='left', va='top')

    cmpl_list, uinp_list, total, detail_list = convert_oi_list_for_plot(detailed_analysis)
    draw_oi_list(plt, color, line_style, MY_SIZE*font_factor, MY_WIDTH*1.2, upper_limit, cmpl_list,
                 y_min=-0.1, show_labels=show_labels)
    draw_oi_list(plt, color2, line_style, MY_SIZE*font_factor, MY_WIDTH*1.2, upper_limit, uinp_list,
                 y_min=-0.1, show_labels=show_labels)

    annotated_list, cmpl_list2, uinp_list2 = calculate_required_performance(detail_list, target_sps, used_batch, unit=gigaU)
    draw_oi_marker(plt, color, marker1, cmpl_list2)
    draw_oi_marker(plt, color2, marker2, uinp_list2)
    marker1_text = 'req. perf. f. Engine arch. (w/ {} sps, batch {})'.format(target_sps, used_batch)
    marker1_legend = mpl.lines.Line2D([], [], color=color, marker=marker1, linestyle='None', markersize=10,
                                      label=marker1_text)
    marker2_text = 'req. perf. f. Stream arch. (w/ {} sps, batch {})'.format(target_sps, used_batch)
    marker2_legend = mpl.lines.Line2D([], [], color=color2, marker=marker2, linestyle='None', markersize=10,
                                      label=marker2_text)

    # color3 = 'orchid'
    color3 = 'aqua'
    oai_avg = total['flops'] / (total['uinp_B'] + total['para_B'])
    plt.vlines(x=oai_avg, ymin=-0.1, ymax=upper_limit, colors=color3, linestyles=line_style, linewidth=MY_WIDTH*1.2,
               zorder=8)
    text = 'Engine avg.'
    plt.text(x=oai_avg*1.02, y=1, s=text, color=color3, fontsize=MY_SIZE*font_factor, ha='left', va='top',
             rotation=90, zorder=8)
    print("[DOSA:roofline] Info: {} at {} ({}).".format(text, oai_avg, used_name))
    oai_avg2 = total['flops'] / total['uinp_B']
    plt.vlines(x=oai_avg2, ymin=-0.1, ymax=upper_limit, colors=color3, linestyles=line_style, linewidth=MY_WIDTH*1.2,
               zorder=8)
    text = 'Stream avg.'
    plt.text(x=oai_avg2*1.02, y=1, s=text, color=color3, fontsize=MY_SIZE*font_factor, ha='left', va='top',
             rotation=90, zorder=8)
    print("[DOSA:roofline] Info: {} at {} ({}).".format(text, oai_avg2, used_name))

    # plt.scatter(x=[oai_avg], y=[total['total_flops']*target_fps], marker=marker1, color=color3, zorder=6,
    #             label='req. perf. Engine avg.')
    # plt.scatter(x=[oai_avg2], y=[total['total_flops']*target_fps], marker=marker2, color=color3, zorder=6,
    #             label='req. perf. Stream avg.')

    if show_splits:
        # sweet_spot_index = p_fpga_network_max.index(upper_limit)
        # network_sweet_spot = math.floor(ai_list[sweet_spot_index])
        # upper_limit_list = [upper_limit for x in ai_list]
        alpha = 0.4
        # color = 'peru'
        color = 'tab:green'
        z_order = 0
        text = 'split w/ data parallelization'
        plt.fill_between(ai_list, p_fpga_network_max, upper_limit,
                         color=color, alpha=alpha, rasterized=True, zorder=z_order, label=text)
        # color = 'lightsalmon'
        color = 'tab:red'
        text = 'split w/ compute paral. (for Engines)'
        plt.fill_between(ai_list, p_fpga_ddr_max, upper_limit,
                         color=color, alpha=alpha, rasterized=True, zorder=z_order, label=text)
        # color = 'lightcoral'
        color = 'tab:blue'
        text = 'split w/ compute paral. (for Streams)'
        plt.fill_between(ai_list, p_fpga_bram_max, upper_limit,
                         color=color, alpha=alpha, rasterized=True, zorder=z_order, label=text)
        # color = 'goldenrod'
        color = 'darkmagenta'
        text = 'split w/ compute parallelization (in all cases)'
        plt.fill_between(ai_list, upper_limit, 1000000,
                         color=color, alpha=alpha, rasterized=True, zorder=z_order, label=text)

    plt.xscale('log', base=10)
    plt.yscale('log', base=10)
    # plt.ylim(0.1, 1500)
    plt.ylim(0.01, 100000)
    plt.xlim(ai_list[0], ai_list[-1])

    plt.xlabel('operational intensity (OI) [FLOPS/Byte]', fontsize=MY_SIZE)
    plt.ylabel('attainable performance [GFLOPS/s]', fontsize=MY_SIZE)

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(marker1_legend)
    handles.append(marker2_legend)
    title = "DOSA Roofline for '{}'".format(used_name)
    legend = plt.legend(handles=handles, ncol=3, bbox_to_anchor=(0, 1), loc='lower left', fontsize=MY_SIZE, title=title)
    plt.grid(True, which="major", ls="-", color='0.89')
    plt.tick_params(axis='both', which='both', labelsize=MY_SIZE)
    plt.setp(legend.get_title(), fontsize=MY_SIZE*1.2)

    # set_size(8, 5.5)
    plt.subplots_adjust(top=0.8)
    # plt.tight_layout()
    return plt


def show_roofline_plt(plt, blocking=True):
    # plt.savefig('roofline.pdf', dpi=300, format="pdf")
    # plt.savefig('roofline.png', dpi=300, format="png")
    plt.show(block=blocking)

