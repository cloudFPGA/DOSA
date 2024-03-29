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
    # MY_SIZE = 16
    MY_SIZE = 26
    MY_SIZE_SMALL = MY_SIZE * 0.6
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
    ax1.set_xlabel('computing operations (i.e. layers)', fontsize=MY_SIZE)
    ax1.set_ylabel('bandwidth in GB/s', fontsize=MY_SIZE)
    color = 'tab:blue'
    ln1 = ax1.fill_between(id_list, input_B_list, color=color, alpha=alpha, label="input bandwidth per operation",
                           linewidth=MY_WIDTH)
    color = 'tab:orange'
    ln2 = ax1.fill_between(id_list, output_B_list, color=color, alpha=alpha, label="output bandwidth per operation",
                           linewidth=MY_WIDTH)

    plt.tick_params(axis='both', which='both', labelsize=MY_SIZE)
    ax2 = ax1.twinx()
    color = 'tab:olive'
    ax2.set_ylabel('parameters in KB', fontsize=MY_SIZE)
    ln3 = ax2.fill_between(id_list, param_B_list, color=color, alpha=alpha, label="parameter per operation",  # ArchBrick
                           linewidth=MY_WIDTH)

    title = "DOSA bandwidth analysis for\n{}\n({})".format(plt_name, target_string)
    # handles, labels = plt.gca().get_legend_handles_labels()
    handles = [ln1, ln2, ln3]
    legend = plt.legend(handles=handles, ncol=3, bbox_to_anchor=(0, 1), loc='lower left', fontsize=MY_SIZE_SMALL, title=title)
    # legend = plt.legend(ncol=3, bbox_to_anchor=(0, 1), loc='lower left', fontsize=MY_SIZE, title=title)
    plt.setp(legend.get_title(), multialignment='center')
    # lns = ln1+ln2+ln3
    # labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, loc=0)
    plt.grid(True, which="major", ls="-", color='0.89')
    ax1.set_xticks(id_list)
    plt.tick_params(axis='both', which='both', labelsize=MY_SIZE)
    # plt.axes().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.setp(legend.get_title(), fontsize=MY_SIZE)
    # plt.title(title, fontsize=MY_SIZE*1.2)
    plt.subplots_adjust(top=0.8)
    # plt.tight_layout()
    return plt
