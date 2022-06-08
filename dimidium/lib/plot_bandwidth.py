#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Dec 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Present current analysis of bandwidths
#  *
#  *

import json
import itertools

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

from matplotlib.ticker import MaxNLocator

import dimidium.lib.singleton as dosa_singleton
from dimidium.lib.util import rf_attainable_performance, OptimizationStrategies, BrickImplTypes
from dimidium.middleend.archGen.ArchDraft import ArchDraft
from dimidium.middleend.archGen.ArchNode import ArchNode

from dimidium.lib.units import *
from dimidium.backend.devices.dosa_roofline import config_global_rf_ylim_min as __ylim_min__
from dimidium.backend.devices.dosa_roofline import config_global_rf_ylim_max as __ylim_max__

plt.rcParams.update({'figure.max_open_warning': 0})


# https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units
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


def generate_bandwidth_plt(arch_draft: ArchDraft, show_deubg=False):
    target_string = "target: "
    if arch_draft.strategy == OptimizationStrategies.THROUGHPUT:
        target_string += "{} sps".format(arch_draft.target_sps)
    elif arch_draft.strategy == OptimizationStrategies.LATENCY:
        target_string += "{} s/req".format(arch_draft.target_latency)
    else:
        target_string += "max {} nodes".format(arch_draft.target_resources)
    plt_name = "'{}' (draft: {}, opt: {}, #nodes: {})".format(arch_draft.name, arch_draft.version,
                                                            str(arch_draft.strategy).split('.')[-1],
                                                            arch_draft.get_total_nodes_cnt())

    id_list = []
    input_B_list = []
    output_B_list = []
    param_B_list = []
    xlist = []
    i = 0
    bricks_to_skip = []
    for bb in arch_draft.brick_iter_gen():
        if bb.skip_in_roofline:
            continue
        if bb in bricks_to_skip:
            continue
        id_list.append(bb.brick_uuid)
        input_B_list.append((bb.compute_parallelization_factor * bb.input_bw_Bs) / gigaU)
        output_B_list.append((bb.compute_parallelization_factor * bb.output_bw_Bs) / gigaU)
        param_B_list.append((bb.compute_parallelization_factor * bb.parameter_bytes) / kiloU)
        xlist.append(i)
        if bb.parallelized_bricks is not None:
            bricks_to_skip.extend(bb.parallelized_bricks)
        i += 1

    id_list = xlist
    MY_SIZE = 16
    MY_SIZE_SMALL = 15
    MY_WIDTH = 1.6
    # line_style = 'dotted'
    line_style = 'solid'
    alpha = 0.7

    # fig, ax1 = plt.subplots()
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.set_xlim(id_list[0], id_list[-1])
    color = 'tab:grey'
    # ax1.set_xlabel('ArchBrick Ids', fontsize=MY_SIZE)
    ax1.set_xlabel('computing operations', fontsize=MY_SIZE)
    ax1.set_ylabel('bandwidth in GB/s', fontsize=MY_SIZE)
    color = 'tab:blue'
    ln1 = ax1.fill_between(id_list, input_B_list, color=color, alpha=alpha, label="input bandwidth per operation",
                           linewidth=MY_WIDTH)
    color = 'tab:orange'
    ln2 = ax1.fill_between(id_list, output_B_list, color=color, alpha=alpha, label="output bandwidth per operation",
                           linewidth=MY_WIDTH)

    ax2 = ax1.twinx()
    color = 'tab:olive'
    ax2.set_ylabel('parameters in KB', fontsize=MY_SIZE)
    ln3 = ax2.fill_between(id_list, param_B_list, color=color, alpha=alpha, label="parameter per ArchBrick",
                           linewidth=MY_WIDTH)

    title = "DOSA bandwidth analysis for {}\n({})".format(plt_name, target_string)
    # handles, labels = plt.gca().get_legend_handles_labels()
    handles = [ln1, ln2, ln3]
    legend = plt.legend(handles=handles, ncol=3, bbox_to_anchor=(0, 1), loc='lower left', fontsize=MY_SIZE, title=title)
    # legend = plt.legend(ncol=3, bbox_to_anchor=(0, 1), loc='lower left', fontsize=MY_SIZE, title=title)
    plt.setp(legend.get_title(), multialignment='center')
    # lns = ln1+ln2+ln3
    # labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, loc=0)
    plt.grid(True, which="major", ls="-", color='0.89')
    plt.tick_params(axis='both', which='both', labelsize=MY_SIZE)
    # plt.axes().xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xticks(id_list)
    plt.setp(legend.get_title(), fontsize=MY_SIZE * 1.2)
    # plt.title(title, fontsize=MY_SIZE*1.2)
    plt.subplots_adjust(top=0.8)
    # plt.tight_layout()
    return plt
