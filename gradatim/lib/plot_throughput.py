#  /*******************************************************************************
#   * Copyright 2019 -- 2024 IBM Corporation
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
#  *     Created: May 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Present current analysis of throughput
#  *
#  *

import json
import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing

from matplotlib.ticker import MaxNLocator

import gradatim.lib.singleton as dosa_singleton
from gradatim.lib.util import rf_attainable_performance, OptimizationStrategies, BrickImplTypes
from gradatim.middleend.archGen.ArchDraft import ArchDraft
from gradatim.middleend.archGen.ArchNode import ArchNode

from gradatim.lib.units import *
from gradatim.backend.devices.dosa_roofline import config_global_rf_ylim_min as __ylim_min__
from gradatim.backend.devices.dosa_roofline import config_global_rf_ylim_max as __ylim_max__


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


def generate_throughput_plt_nodes(arch_draft: ArchDraft, show_deubg=False):
    target_string = "target: "
    if arch_draft.strategy == OptimizationStrategies.THROUGHPUT:
        target_string += "{} Ksps".format(arch_draft.target_sps/kiloU)
    elif arch_draft.strategy == OptimizationStrategies.LATENCY:
        target_string += "{} s/req".format(arch_draft.target_latency)
    else:
        target_string += "max {} nodes".format(arch_draft.target_resources)
    plt_name = "'{}' (draft: {}, opt: {}, #nodes: {})".format(arch_draft.name, arch_draft.version,
                                                            str(arch_draft.strategy).split('.')[-1],
                                                            arch_draft.get_total_nodes_cnt())

    id_list = []
    predicted_throughput_list = []
    max_throughput_list = []
    throughput_with_dpl = []
    prev_node = None
    prev_i = 0
    prev_id = -1
    parallel_factor_dict = {}
    xlist = []
    i = 0
    y_upper_lim = -1
    y_lower_lim = 2
    if arch_draft.target_sps/kiloU < y_lower_lim:
        y_upper_lim = arch_draft.target_sps/kiloU
    for nn in arch_draft.node_iter_gen():
        if nn.skip_in_roofline:
            continue
        # don't add up parallel nodes
        if prev_node in nn.parallel_nodes.values():
            # parallel_factor_dict[prev_id] = (len(nn.parallel_nodes), max_throughput_list[-1])
            # parallel_factor_dict[prev_i] = (len(nn.parallel_nodes), predicted_throughput_list[-1], prev_id)
            parallel_factor_dict[prev_i] = (len(nn.parallel_nodes), max_throughput_list[-1], prev_id)
            continue
        prev_node = nn
        prev_i = i
        xlist.append(i)
        prev_id = nn.node_id
        id_list.append(nn.node_id)
        predicted_throughput_list.append(float(nn.used_iter_hz / kiloU))
        max_throughput_list.append(float(nn.max_iter_hz / kiloU))
        throughput_with_dpl.append(float((nn.used_iter_hz / kiloU) * nn.data_parallelism_level))
        if (nn.used_iter_hz / kiloU) < y_lower_lim:
            y_lower_lim = float(nn.used_iter_hz / kiloU)
        if (nn.max_iter_hz/kiloU) > y_upper_lim:
            y_upper_lim = float(nn.max_iter_hz/kiloU)
        if ((nn.used_iter_hz / kiloU) * nn.data_parallelism_level) > y_upper_lim:
            y_upper_lim = float((nn.used_iter_hz / kiloU) * nn.data_parallelism_level)
        i += 1

    MY_SIZE = 26
    MY_SIZE_SMALL = MY_SIZE * 0.6
    MY_WIDTH = 1.6
    # line_style = 'dotted'
    line_style = 'solid'
    alpha = 0.7

    # xlist = [i for i in range(len(id_list))]
    # fig, ax1 = plt.subplots()
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.set_xlim(xlist[0], xlist[-1])
    ax1.set_ylim(y_lower_lim*0.5, y_upper_lim*20)
    ax1.set_yscale('log', base=10)
    color = 'tab:grey'
    ax1.set_xlabel('node Ids (FPGAs)', fontsize=MY_SIZE)
    ax1.set_ylabel('throughput in Ksamples/s (logarithmic)', fontsize=MY_SIZE)  # , color=color
    color = 'tab:blue'
    z = 5
    ln1 = ax1.fill_between(xlist, predicted_throughput_list, color=color, alpha=alpha,
                           label='implemented/predicted throughput per node', linewidth=MY_WIDTH, zorder=z)
    color = 'tab:orange'
    z = 3
    ln2 = ax1.fill_between(xlist, max_throughput_list, color=color, alpha=alpha,
                           label="theoretical throughput with 100% utilization per node", linewidth=MY_WIDTH, zorder=z)

    color = 'tab:olive'
    z = 4
    ln3 = ax1.fill_between(xlist, throughput_with_dpl, color=color, alpha=alpha,
                           label="implemented/predicted throughput with data-parallel nodes", linewidth=MY_WIDTH, zorder=z)

    color = 'tab:grey'
    z = 8
    for e in parallel_factor_dict:
        y = parallel_factor_dict[e][1] * 1.2
        pn = parallel_factor_dict[e][0]
        ni = parallel_factor_dict[e][2]
        text = 'Node {} has {} parallel nodes\n(with same throughput)'.format(ni, pn)
        plt.text(x=e, y=y, color=color, s=text, fontsize=MY_SIZE_SMALL, ha='left', va='bottom',
                 rotation=90, zorder=z)

    z = 10
    color = 'firebrick'
    plt.hlines(y=arch_draft.target_sps/kiloU, xmin=xlist[0], xmax=xlist[-1], colors=color, linestyles=line_style,
               linewidth=MY_WIDTH*1.2, zorder=z)
    text = 'user required ' + target_string
    plt.text(x=0.5, y=arch_draft.target_sps/kiloU*1.15, s=text, color=color,
             fontsize=MY_SIZE, zorder=z+1)

    title = "DOSA Throughput analysis (node-wise) for\n{}\n({})".format(plt_name, target_string)
    # handles, labels = plt.gca().get_legend_handles_labels()
    handles = [ln1, ln2, ln3]
    legend = plt.legend(handles=handles, ncol=2, bbox_to_anchor=(0, 1), loc='lower left', fontsize=MY_SIZE_SMALL, title=title)
    plt.setp(legend.get_title(), multialignment='center')
    # legend = plt.legend(ncol=3, bbox_to_anchor=(0, 1), loc='lower left', fontsize=MY_SIZE, title=title)
    # lns = ln1+ln2+ln3
    # labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, loc=0)
    plt.grid(True, which="major", ls="-", color='0.89')
    plt.tick_params(axis='both', which='both', labelsize=MY_SIZE)
    # plt.axes().xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xticks(xlist)
    ax1.set_xticklabels(id_list)
    plt.setp(legend.get_title(), fontsize=MY_SIZE)
    # plt.title(title, fontsize=MY_SIZE*1.2)
    plt.subplots_adjust(top=0.8)
    # plt.tight_layout()
    return plt


def generate_throughput_plt_bricks(arch_draft: ArchDraft, show_deubg=False):
    target_string = "target: "
    if arch_draft.strategy == OptimizationStrategies.THROUGHPUT:
        target_string += "{} Ksps".format(arch_draft.target_sps/kiloU)
    elif arch_draft.strategy == OptimizationStrategies.LATENCY:
        target_string += "{} s/req".format(arch_draft.target_latency)
    else:
        target_string += "max {} nodes".format(arch_draft.target_resources)
    plt_name = "'{}' (draft: {}, opt: {}, #nodes: {})".format(arch_draft.name, arch_draft.version,
                                                            str(arch_draft.strategy).split('.')[-1],
                                                            arch_draft.get_total_nodes_cnt())

    id_list = []
    predicted_throughput_list = []
    parallel_factor_dict = {}
    xlist = []
    i = 0
    y_upper_lim = -1
    y_lower_lim = 2
    bricks_to_skip = []
    if arch_draft.target_sps/kiloU < y_lower_lim:
        y_upper_lim = arch_draft.target_sps/kiloU
    for bb in arch_draft.brick_iter_gen():
        if bb.skip_in_roofline:
            continue
        if bb in bricks_to_skip:
            continue
        id_list.append(bb.brick_uuid)
        xlist.append(i)
        predicted_throughput_list.append(float(bb.iter_hz / kiloU))
        if (bb.iter_hz / kiloU) < y_lower_lim:
            y_lower_lim = float(bb.iter_hz / kiloU)
        if (bb.iter_hz/kiloU) > y_upper_lim:
            y_upper_lim = float(bb.iter_hz/kiloU)
        if bb.parallelized_bricks is not None:
            bricks_to_skip.extend(bb.parallelized_bricks)
            parallel_factor_dict[i] = (len(bb.parallelized_bricks), predicted_throughput_list[-1], bb.brick_uuid)
        i += 1

    MY_SIZE = 26
    MY_SIZE_SMALL = MY_SIZE * 0.6
    MY_WIDTH = 1.6
    # line_style = 'dotted'
    line_style = 'solid'
    alpha = 0.7
    id_list = xlist

    # xlist = [i for i in range(len(id_list))]
    # fig, ax1 = plt.subplots()
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.set_xlim(xlist[0], xlist[-1])
    ax1.set_ylim(y_lower_lim*0.5, y_upper_lim*20)
    ax1.set_yscale('log', base=10)
    color = 'tab:grey'
    ax1.set_xlabel('computing operations (i.e. layers)', fontsize=MY_SIZE)
    ax1.set_ylabel('throughput in Ksamples/s (logarithmic)', fontsize=MY_SIZE)  # , color=color
    color = 'tab:blue'
    z = 5
    ln1 = ax1.fill_between(xlist, predicted_throughput_list, color=color, alpha=alpha,
                           label='implemented/predicted throughput per Node', linewidth=MY_WIDTH, zorder=z)
    color = 'tab:orange'

    color = 'tab:grey'
    z = 8
    for e in parallel_factor_dict:
        y = parallel_factor_dict[e][1] * 1.2
        pn = parallel_factor_dict[e][0]
        ni = parallel_factor_dict[e][2]
        text = 'operation {} has {} parallel instances\n(with same throughput)'.format(ni, pn)
        plt.text(x=e, y=y, color=color, s=text, fontsize=MY_SIZE_SMALL, ha='left', va='bottom',
                 rotation=90, zorder=z)

    z = 10
    color = 'firebrick'
    plt.hlines(y=arch_draft.target_sps/kiloU, xmin=xlist[0], xmax=xlist[-1], colors=color, linestyles=line_style,
               linewidth=MY_WIDTH*1.2, zorder=z)
    text = 'user required ' + target_string
    plt.text(x=0.5, y=arch_draft.target_sps/kiloU*1.15, s=text, color=color,
             fontsize=MY_SIZE, zorder=z+1)

    title = "DOSA Throughput analysis (brick-wise) for\n{}\n({})".format(plt_name, target_string)
    # handles, labels = plt.gca().get_legend_handles_labels()
    handles = [ln1]
    legend = plt.legend(handles=handles, ncol=2, bbox_to_anchor=(0.5, 1), loc='lower center', fontsize=MY_SIZE_SMALL, title=title)
    plt.setp(legend.get_title(), multialignment='center')
    # legend = plt.legend(ncol=3, bbox_to_anchor=(0, 1), loc='lower left', fontsize=MY_SIZE, title=title)
    # lns = ln1+ln2+ln3
    # labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, loc=0)
    plt.grid(True, which="major", ls="-", color='0.89')
    plt.tick_params(axis='both', which='both', labelsize=MY_SIZE)
    # plt.axes().xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xticks(xlist)
    ax1.set_xticklabels(id_list)
    plt.setp(legend.get_title(), fontsize=MY_SIZE)
    # plt.title(title, fontsize=MY_SIZE*1.2)
    plt.subplots_adjust(top=0.8)
    # plt.tight_layout()
    return plt
