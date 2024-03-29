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
#  *     Created: July 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Class containing one version of a DOSA architectural draft
#  *
#  *
import copy
import os
import math
import json
import numpy as np

import gradatim.lib.singleton as dosa_singleton
from gradatim.lib.dosa_exceptions import DosaChangeArchType, DosaConstraintFail
from gradatim.backend.commLibs.BaseCommLib import BaseCommLib, placeholderCommLib
from gradatim.lib.util import OptimizationStrategies, BrickImplTypes, DosaRv
from gradatim.middleend.archGen.ArchFilter import MergeBrickContrFilter
from gradatim.middleend.archGen.archOpt import merge_bricks_pass
from gradatim.middleend.archGen.ArchNode import ArchNode
from gradatim.middleend.archGen.ArchBrick import ArchBrick
from gradatim.middleend.archGen.ArchOp import ArchOp
from gradatim.backend.devices.dosa_device import DosaBaseHw, placeholderHw, DosaHwClasses
from gradatim.backend.devices.builtin import vCPU_x86
from gradatim.backend.devices.dosa_roofline import RooflineRegionsOiPlane, get_rightmost_roofline_region
from gradatim.backend.operatorSets.BaseOSG import sort_osg_list
from gradatim.backend.operatorSets.osgs import osg_tvmCpu
from gradatim.middleend.archGen.BrickContract import BrickContract


__filedir__ = os.path.dirname(os.path.abspath(__file__))


class ArchDraft(object):

    # _bstr_fmt_ = "{:06}"
    # _bid_max_ = 99999

    def __init__(self, name, version, strategy: OptimizationStrategies, batch_size, sample_size, target_sps=-1,
                 target_latency=-1,
                 target_resources=-1, tvm_mod=None, tvm_params=None):
        self.name = name
        self.version = version
        self.strategy = strategy
        self.batch_size = batch_size
        self.target_sps = target_sps
        self.target_latency = target_latency
        self.target_resources = target_resources
        self.main_tvm_mod = tvm_mod
        self.main_tvm_params = tvm_params
        self.sample_size_B = sample_size
        self.nodes = {}
        self.nid_cnt = 0
        self.input_layer = None
        self.output_layer = None
        self.calc_throughput = -1
        self.calc_latency = -1
        self.calc_resources = -1
        self.target_hw_set = []
        self.fallback_hw_set = []
        self.tmp_notes = {}
        self.possible_hw_types = []
        self.all_selected_hw_types = []
        self.possible_comm_libs = []
        self.selected_comm_lib = placeholderCommLib
        self.available_osgs = []
        self.substract_client_nodes = 0
        self.total_perf_F = -1
        self.max_perf_iter_based = -1
        self.min_iter_hz = -1
        self.total_flops = 0
        self.total_parameters_bytes = 0
        self.max_brick_uuid = -1
        self.total_time_dse_seconds = 'N/A'

    def __repr__(self):
        return "ArchDraft({}, {}, {})".format(self.name, self.version, self.strategy)

    def as_dict(self):
        res = {'name': self.name, 'version': self.version, 'strategy': str(self.strategy),
               'batch_size': self.batch_size, 'target_sps': self.target_sps, 'target_latency': self.target_latency,
               'target_resources': self.target_resources,
               'total_implemented_perf_F': float(self.total_perf_F),
               'cluster_estimated_iter_hz': float(self.min_iter_hz),
               'total_flops': self.total_flops, 'total_parameter_bytes': self.total_parameters_bytes,
               # 'cluster_estimated_maximum_iter_hz': self.max_perf_iter_based,
               'input': str(self.input_layer), 'output': str(self.output_layer),
               'possible_hw_types': [], 'target_hw_set': [], 'fallback_hw_set': [],
               'possible_comm_libs': [], 'selected_comm_lib': repr(self.selected_comm_lib),
               'main_tvm_mod': str(self.main_tvm_mod)[:100], 'main_tvm_params': str(self.main_tvm_params)[:100],
               'total_nodes': len(self.nodes), 'total_time_DSE': self.total_time_dse_seconds,
               'nodes': {}}
        for thw in self.target_hw_set:
            tn = type(thw).__name__
            res['target_hw_set'].append(tn)
        for fhw in self.fallback_hw_set:
            fn = type(fhw).__name__
            res['fallback_hw_set'].append(fn)
        for ph in self.possible_hw_types:
            phs = repr(ph)
            res['possible_hw_types'].append(phs)
        for ni in self.nodes:
            n = self.nodes[ni]
            res['nodes'][ni] = n.as_dict()
        for pcl in self.possible_comm_libs:
            res['possible_comm_libs'].append(repr(pcl))
        return res

    def __str__(self):
        ret = self.as_dict()
        return json.dumps(ret, indent=2)

    def copy(self):
        """fast way to return an independent fresh copy (not all lists are copied)"""
        # to be faster than deepcopy
        # NOT overwriting __copy__
        nad = copy.copy(self)
        nad.main_tvm_mod = self.main_tvm_mod
        nad.main_tvm_params = self.main_tvm_params
        nad.nodes = {}
        for ni in self.nodes:
            nn = self.nodes[ni]
            nnn = nn.copy()  # not copy.copy...
            nad.nodes[ni] = nnn
        nad.target_hw_set = []
        for thw in self.target_hw_set:
            nthw = copy.copy(thw)
            nad.target_hw_set.append(nthw)
        nad.fallback_hw_set = []
        for thw in self.fallback_hw_set:
            nthw = copy.copy(thw)
            nad.fallback_hw_set.append(nthw)
        nad.tmp_notes = self.tmp_notes.copy()
        nad.possible_hw_types = []
        for thw in self.possible_hw_types:
            nthw = copy.copy(thw)
            nad.possible_hw_types.append(nthw)
        nad.all_selected_hw_types = []
        for thw in self.all_selected_hw_types:
            nthw = copy.copy(thw)
            nad.all_selected_hw_types.append(nthw)
        nad.possible_comm_libs = []
        for thw in self.possible_comm_libs:
            nthw = copy.copy(thw)
            nad.possible_comm_libs.append(nthw)
        nad.selected_comm_lib = placeholderCommLib
        return nad

    def add_node(self, node: ArchNode):
        # bstr = self._bstr_fmt_.format(self.bid_cnt)
        n_id = self.nid_cnt
        self.nid_cnt += 1
        # if self.bid_cnt > self._bid_max_:
        #    print("[DOSA:ArchDraft:ERROR] Brick Id overflow occurred!")
        node.set_node_id(n_id)
        self.nodes[n_id] = node

    def insert_node(self, node: ArchNode, new_id):
        if new_id > self.nid_cnt or new_id < 0:
            print("[DOSA:ArchDraft:ERROR] insert node with invalid new_id, skipping.")
            return
        for i in reversed(range(new_id, self.nid_cnt)):
            self.nodes[i + 1] = self.nodes[i]
            self.nodes[i + 1].set_node_id(i + 1)
        node.set_node_id(new_id)
        self.nodes[new_id] = node
        self.nid_cnt += 1

    def delete_node(self, id_to_delete):
        if id_to_delete > self.nid_cnt or id_to_delete < 0:
            print("[DOSA:ArchDraft:ERROR] delete node with invalid id, skipping.")
            return
        del self.nodes[id_to_delete]
        self.nid_cnt -= 1
        for i in range(id_to_delete, self.nid_cnt):
            self.nodes[i] = self.nodes[i + 1]
            del self.nodes[i + 1]
            self.nodes[i].set_node_id(i)

    # def update_node_links(self):
    #     # updates the successors and predecessors of the nodes
    #     last_ni = None
    #     for i in range(1, self.nid_cnt):
    #         self.nodes[i].predecessors = [i-1]
    #         self.nodes[i].successors = [i+1]

    def set_tvm_mod(self, tvm_mod):
        self.main_tvm_mod = tvm_mod

    def set_tvm_params(self, tvm_params):
        self.main_tvm_params = tvm_params

    def set_input_layer(self, in_dpl):
        self.input_layer = in_dpl

    def set_output_layer(self, out_dpl):
        self.output_layer = out_dpl

    def set_possible_comm_libs(self, poss_comm_libs: [BaseCommLib]):
        self.possible_comm_libs = poss_comm_libs

    def brick_iter_gen(self):
        for ni in self.nodes:
            nn = self.nodes[ni]
            for bi in nn.bricks:
                bb = nn.bricks[bi]
                yield bb

    def node_iter_gen(self):
        for ni in self.nodes:
            nn = self.nodes[ni]
            yield nn

    def op_iter_gen(self):
        for ni in self.nodes:
            nn = self.nodes[ni]
            for bi in nn.bricks:
                bb = nn.bricks[bi]
                for op in bb.ops:
                    oo = bb.ops[op]
                    yield oo

    def get_bricks_num(self):
        ret = 0
        for ni in self.nodes:
            ret += len(self.nodes[ni].bricks)
        return ret

    def add_possible_target_hw(self, nth: DosaBaseHw):
        self.target_hw_set.append(nth)

    def add_possible_fallback_hw(self, nfh: DosaBaseHw):
        self.fallback_hw_set.append(nfh)

    def get_total_nodes_cnt(self):
        total_nodes = 0
        for nn in self.node_iter_gen():
            total_nodes += nn.data_parallelism_level
        # if self.substract_node_0:
        #     total_nodes -= 1
        total_nodes -= self.substract_client_nodes
        return total_nodes

    def legalize(self, verbose=False, consider_switching_first=False, contract_look_ahead=0,
                 relax_utilization_check=False):
        # 0. bandwidth analysis (OSG not yet decided)
        #  including sorting of contracts based on strategy
        #  and filtering of contracts
        self.update_uuids()
        for nn in self.node_iter_gen():
            for lb in nn.local_brick_iter_gen():
                if self.strategy == OptimizationStrategies.RESOURCES:
                    lb.sort_contracts(by_utility=True)
                else:
                    lb.sort_contracts(by_utility=False)
                best_stream = lb.get_best_available_contract(filter_impl_type=BrickImplTypes.STREAM, consider_util=True,
                                                             filter_device=nn.targeted_hw,
                                                             consider_min_iter=lb.req_iter_hz)
                best_engine = lb.get_best_available_contract(filter_impl_type=BrickImplTypes.ENGINE, consider_util=True,
                                                             filter_device=nn.targeted_hw,
                                                             consider_min_iter=lb.req_iter_hz)
                if best_engine is None and best_stream is None:
                    # we have a problem...drop min ite requirement
                    if verbose:
                        print("[DOSA:archGen:INFO] No contract of Brick {} fulfills min_iter, relaxing requirement."
                              .format(lb.brick_uuid))
                    best_stream = lb.get_best_available_contract(filter_impl_type=BrickImplTypes.STREAM, consider_util=True,
                                                             filter_device=nn.targeted_hw)
                    best_engine = lb.get_best_available_contract(filter_impl_type=BrickImplTypes.ENGINE, consider_util=True,
                                                             filter_device=nn.targeted_hw)
                if best_engine is None:
                    lb.set_impl_type(BrickImplTypes.STREAM)
                    if verbose:
                        print("[DOSA:archGen:INFO] Setting ImplType of Brick {} to STREAM,".format(lb.brick_uuid) +
                              " since there are no Engine implementations available.")
                elif best_stream is None:
                    lb.set_impl_type(BrickImplTypes.ENGINE)
                    if verbose:
                        print("[DOSA:archGen:INFO] Setting ImplType of Brick {} to ENGINE,".format(lb.brick_uuid) +
                              " since there are no Stream implementations available.")
                else:
                    rr_engine = nn.roofline.get_region_OIPlane_iter_based(best_engine.iter_hz, lb.req_iter_hz,
                                                                          best_engine)
                    rr_stream = nn.roofline.get_region_OIPlane_iter_based(best_stream.iter_hz, lb.req_iter_hz,
                                                                          best_stream)
                    # compare roofline regions and return the better: 1 or 2, if equal also 1 will be returned
                    br = get_rightmost_roofline_region(rr_engine, rr_stream)
                    if br == 1:
                        lb.set_impl_type(BrickImplTypes.ENGINE)
                        if verbose:
                            print("[DOSA:archGen:INFO] Setting ImplType of Brick {} to ENGINE,".format(lb.brick_uuid) +
                                  " since there are no bandwidth limitations.")
                    else:
                        lb.set_impl_type(BrickImplTypes.STREAM)
                        if verbose:
                            print("[DOSA:archGen:INFO] Setting ImplType of Brick {} to STREAM,".format(lb.brick_uuid) +
                                  " since there are bandwidth limitations.")
                    # if rr_engine == RooflineRegionsOiPlane.ABOVE_BRAM or rr_engine == RooflineRegionsOiPlane.ABOVE_DRAM:
                    #     lb.set_impl_type(BrickImplTypes.STREAM)
                    #     if verbose:
                    #         print("[DOSA:archGen:INFO] Setting ImplType of Brick {} to STREAM,".format(lb.brick_uuid) +
                    #               " since there are bandwidth limitations.")
                # lb.update_possible_contracts()
        # ensure all are decided
        for bb in self.brick_iter_gen():
            assert bb.selected_impl_type != BrickImplTypes.UNDECIDED
        # 1. select best of possible contracts
        for nn in self.node_iter_gen():
            # last_osg = None
            skip_lbs = []
            assume_osg = None
            prev_contract = None
            # for lb in nn.local_brick_iter_gen():
            for bi in nn.bricks:
                lb = nn.bricks[bi]
                if lb in skip_lbs:
                    continue
                if self.strategy == OptimizationStrategies.RESOURCES:
                    lb.sort_contracts(by_utility=True, previous_contract=prev_contract)
                else:
                    lb.sort_contracts(by_utility=False, previous_contract=prev_contract)
                # lb.update_possible_contracts(consider_switching=True, assume_osg=last_osg)
                try:
                    # TODO: how to decide for another device later?
                    lb.update_possible_contracts(consider_switching=True, assume_osg=assume_osg,
                                                 assume_device=nn.targeted_hw)
                except DosaChangeArchType as int_e:
                    if verbose:
                        print("[DOSA:archGen:INFO] Setting ImplType of Brick {} to STREAM,".format(lb.brick_uuid) +
                              " since there are no possible Engine implementations available.")
                    # assert e.not_possible_brick_type == BrickImplTypes.ENGINE
                    assert len(int_e.args) >= 1 and int_e.args[0] == BrickImplTypes.ENGINE
                    lb.set_impl_type(BrickImplTypes.STREAM)
                    # sort again
                    if self.strategy == OptimizationStrategies.RESOURCES:
                        lb.sort_contracts(by_utility=True, previous_contract=prev_contract)
                    else:
                        lb.sort_contracts(by_utility=False, previous_contract=prev_contract)
                    # now, without a catch..
                    lb.update_possible_contracts(consider_switching=True, assume_osg=assume_osg,
                                                 assume_device=nn.targeted_hw)

                # TODO...how to best approx switching in the beginning?
                # lb.update_possible_contracts(consider_switching=consider_switching_first)
                # consider req_iter_hz per Brick to select contract
                selected_contract = None
                if contract_look_ahead > 0:
                    osgs_to_consider = lb.still_possible_osgs
                    osg_stats = {}
                    for o in osgs_to_consider:
                        co = lb.get_best_possible_contract(filter_osg=o)
                        comp_costs = co.comp_util_share
                        mem_costs = co.mem_util_share
                        if o != assume_osg:
                            comp_costs += co.switching_comp_share
                            mem_costs += co.switching_mem_share
                        osg_possible = True
                        for i in range(1, contract_look_ahead + 1):
                            lac = bi + i
                            if lac in nn.bricks:
                                nb = nn.bricks[lac]
                                # in look ahead mode...catching
                                try:
                                    nb.update_possible_contracts(consider_switching=False, assume_device=nn.targeted_hw)
                                except DosaChangeArchType as int_e:
                                    # since we are in look-ahead mode, we need to modify the impl type in order to take
                                    # advantage of look-ahead...
                                    if len(int_e.args) >= 1 and int_e.args[0] == BrickImplTypes.ENGINE:
                                        nb.set_impl_type(BrickImplTypes.STREAM)
                                        try:
                                            nb.update_possible_contracts(consider_switching=False,
                                                                         assume_device=nn.targeted_hw)
                                        except DosaChangeArchType as int_e:
                                            # again...we stop
                                            osg_possible = False
                                            break
                                    else:
                                        osg_possible = False
                                        break
                                nbpoc = nb.get_best_possible_contract(filter_osg=o)
                                if nbpoc is None:
                                    osg_possible = False
                                    break
                                if co.impl_type == BrickImplTypes.ENGINE and nbpoc.impl_type == BrickImplTypes.ENGINE:
                                    # both are engine, so we can consider shared costs...
                                    # TODO: use get_costs_of_contract_extension instead?
                                    pseudo_brick = ArchBrick()
                                    pseudo_brick.used_dtype = co.brick.used_dtype
                                    op_list = list(co.brick.ops.values())
                                    op_list.extend(list(nbpoc.brick.ops.values()))
                                    pseudo_brick.reconstruct_from_op_list(op_list)
                                    o.annotate_brick(pseudo_brick, co.device)
                                    new_contract = pseudo_brick.available_contracts[0]
                                    comp_costs += (new_contract.comp_util_share - co.comp_util_share)
                                    mem_costs += (new_contract.mem_util_share - co.mem_util_share)
                                else:
                                    comp_costs += nbpoc.comp_util_share
                                    mem_costs += nbpoc.mem_util_share
                            # save intermediate results
                            if self.strategy == OptimizationStrategies.RESOURCES:
                                if osg_possible and \
                                        comp_costs < dosa_singleton.config.utilization.dosa_xi \
                                        and mem_costs < dosa_singleton.config.utilization.dosa_xi:
                                    osg_stats[o] = (comp_costs, mem_costs, co, i)
                            else:
                                # we go for higher iterations?
                                if osg_possible:
                                    osg_stats[o] = (comp_costs, mem_costs, co, i)
                        # save final result
                        if self.strategy == OptimizationStrategies.RESOURCES:
                            if osg_possible and \
                                    comp_costs < dosa_singleton.config.utilization.dosa_xi \
                                    and mem_costs < dosa_singleton.config.utilization.dosa_xi:
                                osg_stats[o] = (comp_costs, mem_costs, co, contract_look_ahead)
                        else:
                            # we go for higher iterations?
                            if osg_possible:
                                osg_stats[o] = (comp_costs, mem_costs, co, contract_look_ahead)
                    best_osg = None
                    best_comp_costs = float('inf')
                    best_mem_costs = float('inf')
                    for osg in osg_stats:
                        osgs = osg_stats[osg]
                        # TODO: consider strategies
                        if osgs[0] < best_comp_costs and osgs[1] < best_mem_costs:
                            best_osg = osg
                            best_comp_costs = osgs[0]
                            best_mem_costs = osgs[1]
                            selected_contract = osgs[2]
                    if best_osg is not None:
                        assume_osg = best_osg
                        if verbose:
                            print('[DOSA:archGen:INFO] Assuming {} as OSG for next brick(s)...'.format(assume_osg.name))
                # for not-look ahead or other failures
                if selected_contract is None:
                    #  choosing contract that is above requirement with least resources
                    selected_contract = lb.get_best_sufficient_contract_with_least_resources(
                        consider_switching=consider_switching_first)
                if selected_contract is None:
                    print("[DOSA:archGen:ERROR] couldn't find any valid OSG for brick {} ({}). STOP."
                          .format(lb.brick_uuid, lb.ext_repr()))
                    # exit(1)
                    return DosaRv.ERROR
                # lb.set_osg(selected_contract.osg)
                # lb.selected_contract = selected_contract
                lb.set_contract(selected_contract)
                # last_osg = selected_contract.osg
                if verbose:
                    print('[DOSA:archGen:INFO] Selecting contract ({}) for brick {}.'.format(repr(selected_contract),
                                                                                             lb.brick_uuid))
                if not selected_contract.is_contract_to_be_merged:
                    prev_contract = selected_contract
        # ensure all are decided
        for bb in self.brick_iter_gen():
            # assert bb.selected_contract is not None
            if bb.selected_contract is None:
                return DosaRv.ERROR
        # TODO: merge engine bricks?
        #  also consider in contract selection the number of OSGs if node-numer is equal?
        #  consider required iter-hz
        merge_engine_bricks_filter = MergeBrickContrFilter()
        merge_bricks_pass(self, merge_engine_bricks_filter, verbose=verbose)
        # 2. split nodes based on selected contracts
        orig_nodes_handles = []
        for nn in self.node_iter_gen():
            orig_nodes_handles.append(nn)
        for nn in orig_nodes_handles:
            nn.update_used_perf_util_contr(prefer_engine=False, add_switching_costs=True)
            all_new_nodes = []
            cur_node = nn
            # while cur_node.used_comp_util_share > 1:
            while (cur_node.over_utilized_node \
                    or cur_node.used_comp_util_share > dosa_singleton.config.utilization.dosa_xi \
                    or cur_node.used_mem_util_share > dosa_singleton.config.utilization.dosa_xi) \
                    and cur_node.bid_cnt > 1:
                cur_comp_share = 0
                cur_mem_share = 0
                cur_osg = None
                for i in range(0, cur_node.bid_cnt):
                    cur_comp_share += cur_node.bricks[i].req_util_comp
                    cur_mem_share += cur_node.bricks[i].req_util_mem
                    if cur_node.bricks[i].tmp_osg != cur_osg:
                        cur_comp_share += cur_node.bricks[i].switching_comp_share
                        cur_mem_share += cur_node.bricks[i].switching_mem_share
                        cur_osg = cur_node.bricks[i].tmp_osg
                    # if cur_comp_share > 1.0 or cur_mem_share > 1.0:
                    # to use monolithic nodes as best as possible
                    # if (cur_node.over_utilized_node and i < 3 and (
                    #         cur_comp_share > (dosa_singleton.config.utilization.dosa_xi
                    #                           + dosa_singleton.config.utilization.dosa_xi_exception) or
                    #         cur_mem_share > (dosa_singleton.config.utilization.dosa_xi
                    #                          + dosa_singleton.config.utilization.dosa_xi_exception))) \
                    #         or \
                    #         ((not cur_node.over_utilized_node or i >= 3) and (
                    #                 cur_comp_share > dosa_singleton.config.utilization.dosa_xi or
                    #                 cur_mem_share > dosa_singleton.config.utilization.dosa_xi)):
                    if cur_comp_share > dosa_singleton.config.utilization.dosa_xi or \
                            cur_mem_share > dosa_singleton.config.utilization.dosa_xi:
                        old_i = i
                        if i == 0:
                            i = 1
                            # to use monolithic nodes as best as possible, more than one op if possible
                            for j in range(1, cur_node.bid_cnt):
                                cur_comp_share += cur_node.bricks[j].req_util_comp
                                cur_mem_share += cur_node.bricks[j].req_util_mem
                                if cur_node.bricks[j].tmp_osg != cur_osg:
                                    # only same OSG, TODO
                                    break
                                if cur_comp_share < dosa_singleton.config.utilization.dosa_xi_exception \
                                        and cur_mem_share < dosa_singleton.config.utilization.dosa_xi_exception:
                                    # so would fit
                                    i = j
                                else:
                                    break
                        verbose_msg = "local id of new first brick {}, {}; comp_share: {}, mem_share: {}"\
                            .format(old_i, cur_node.bricks[old_i].ext_repr(),
                                    cur_comp_share - cur_node.bricks[i].req_util_comp,
                                    cur_mem_share - cur_node.bricks[i].req_util_mem)
                        new_node = cur_node.split_horizontal(i)  # including update_used_perf_util
                        all_new_nodes.append(new_node)
                        if verbose:
                            print("[DOSA:archGen:INFO] Splitting node {} horizontally, ".format(cur_node.node_id) +
                                  "due to exceeded compute resource budget ({})."
                                  .format(verbose_msg))
                        cur_node = new_node
                        break
            if len(all_new_nodes) > 0:
                new_nid = nn.get_node_id() + 1
                for new_node in all_new_nodes:
                    self.insert_node(new_node, new_nid)
                    new_nid += 1
        assert len(self.nodes) == self.nid_cnt
        self.update_required_perf()  # to consider new latencies
        # 3. for each node: turn lone engine impls into streams
        #  (i.e. if the sequence is 1 engine, 2 stream, and 3 & 4 engine --> first engine doesn't make sense)
        #  in other words: ensure that all engine sets are bigger or equal 2
        #  no need to update req. perf or split nodes, is already considered in step 3
        # also check if 1/n-performance is enough
        self.update_uuids()
        for nn in self.node_iter_gen():
            cur_engine_set = []
            cur_engine_set_ops_cnt = 0
            turn_engine_to_stream_list = []
            highest_iter_req = -1
            lowest_iter_hz = float('inf')
            for bi in range(0, nn.bid_cnt):
                bb = nn.bricks[bi]
                if bb.selected_impl_type == BrickImplTypes.STREAM:
                    if cur_engine_set_ops_cnt < 2:
                        turn_engine_to_stream_list.extend(cur_engine_set)
                    else:
                        cur_engine_len = cur_engine_set_ops_cnt
                        if lowest_iter_hz / cur_engine_len < highest_iter_req:
                            print(('[DOSA:archGen:INFO] Engine set {} with len {} of node {} does not fulfill ' +
                                   'performance requirement: Combined iteration of {} while {} are required. Will be ' +
                                   'turned into Streams.').format(cur_engine_set, cur_engine_len, nn.node_id,
                                                                  lowest_iter_hz / cur_engine_len, highest_iter_req)
                                  )
                        turn_engine_to_stream_list.extend(cur_engine_set)
                        # else: is fine
                    cur_engine_set = []
                    highest_iter_req = -1
                    lowest_iter_hz = float('inf')
                    continue
                # else
                if bb.req_iter_hz > highest_iter_req:
                    highest_iter_req = bb.req_iter_hz
                if bb.iter_hz < lowest_iter_hz:
                    lowest_iter_hz = bb.iter_hz
                cur_engine_set.append(bi)
                cur_engine_set_ops_cnt += len(bb.ops)
            # last time
            if cur_engine_set_ops_cnt < 2:
                turn_engine_to_stream_list.extend(cur_engine_set)
            for bi in turn_engine_to_stream_list:
                bb = nn.bricks[bi]
                bb.set_impl_type(BrickImplTypes.STREAM)
                bb.update_possible_contracts(consider_switching=True)
                # consider req_iter_hz per Brick to select contract
                #  choosing contract that is above requirement with least resources
                selected_contract = bb.get_best_sufficient_contract_with_least_resources()
                if selected_contract is None:
                    print("[DOSA:archGen:INFO] Would turn ImplType of Brick {} to STREAM, but no contract for this is "
                          "available.".format(bb.brick_uuid))
                    bb.set_impl_type(BrickImplTypes.ENGINE)
                else:
                    # bb.set_osg(selected_contract.osg)
                    # bb.selected_contract = selected_contract
                    bb.set_contract(selected_contract)
                    if verbose:
                        print("[DOSA:archGen:INFO] Setting ImplType of Brick {} to STREAM,".format(bb.brick_uuid) +
                              " since it is an engine with only one operation.")
        # 4. compute parallelization if necessary
        orig_nodes_handles = []
        for nn in self.node_iter_gen():
            orig_nodes_handles.append(nn)
        for nn in orig_nodes_handles:
            needs_compute_utilization = False
            max_factor = 0
            for lb in nn.local_brick_iter_gen():
                if lb.needs_compute_parallelization:
                    needs_compute_utilization = True
                    if lb.compute_parallelization_factor > max_factor:
                        max_factor = lb.compute_parallelization_factor
            if needs_compute_utilization:
                if verbose:
                    print("[DOSA:archGen:INFO] Need to split computing operations of node {} with a factor {}."
                          .format(nn.node_id, max_factor))
                nn.needs_compute_parallelization = True
                nn.compute_parallelization_factor = max_factor
                # check if all are possible
                prev_lb = None
                used_factor = 1
                override_old_factor = False
                while used_factor != max_factor:
                    for lb in nn.local_brick_iter_gen():
                        if not lb.needs_compute_parallelization \
                                or lb.compute_parallelization_factor != max_factor:  # \
                            # or lb.local_brick_id != 0:
                            # TODO: allow also already splitted brick in the middle of a node?
                            # assert prev_lb is not None
                            # TODO: (reactive check later, now it will stop if not possible)
                            if prev_lb is None:
                                used_factor = lb.parallelize([lb.selected_contract], max_factor, with_inputs=False,
                                                             verbose=verbose, override_old_factor=override_old_factor)
                            else:
                                used_factor = lb.parallelize([lb.selected_contract], max_factor, with_inputs=True,
                                                             verbose=verbose, override_old_factor=override_old_factor)
                            if used_factor < max_factor:
                                print("[DOSA:archGen:ERROR] Node {} needs to be parallelized by factor {}, but "
                                      "brick {} can only be parallelized with a factor {} . STOP."
                                      .format(nn.node_id, max_factor, lb.brick_uuid, used_factor))
                                return DosaRv.ERROR
                        else:
                            # to stop while loop
                            used_factor = max_factor
                        if used_factor > max_factor:
                            max_factor = used_factor
                            if verbose:
                                print("[DOSA:archGen:INFO] Need to update the split factor of computing operations of "
                                      "node {} to use factor {}."
                                      .format(nn.node_id, max_factor))
                            used_factor = 1
                            override_old_factor = True
                        prev_lb = lb
                    if verbose and override_old_factor:
                        print(f"[DOSA:archGen:DEBUG] Convergence of split factors within node {nn.node_id}... "
                              f"used_factor: {used_factor}; max_factor: {max_factor};")
                    override_old_factor = True
                my_new_bricks = {}
                new_nodes = {}
                for i in range(0, nn.bid_cnt):
                    if len(nn.bricks[i].parallelized_bricks) != max_factor:
                        print("[DOSA:archGen:ERROR] Node {} needs to be parallelized by factor {}, but "
                              "brick {} can only be parallelized with a factor {} . STOP."
                              .format(nn.node_id, max_factor, i, len(nn.bricks[i].parallelized_bricks)))
                        return DosaRv.ERROR
                    for j in range(0, max_factor):
                        p_brick = nn.bricks[i].parallelized_bricks[j]
                        p_brick.available_contracts = []
                        p_brick.parallelized_bricks = nn.bricks[i].parallelized_bricks
                        p_brick.orig_tvm_node = nn.bricks[i].tvm_node
                        nn.bricks[i].selected_contract.osg.annotate_brick(p_brick,
                                                                          nn.bricks[i].selected_contract.device,
                                                                          filter_impl_types=nn.bricks[
                                                                              i].selected_impl_type)
                        p_brick.update_possible_contracts(consider_switching=True, force_no_split=True)
                        selected_contract = p_brick.get_best_sufficient_contract_with_least_resources()
                        if selected_contract is None:
                            print("[DOSA:archGen:ERROR] couldn't find any valid OSG for PARTIAL brick {} ({}). STOP."
                                  .format(p_brick.brick_uuid, p_brick.ext_repr()))
                            exit(1)
                        p_brick.set_contract(selected_contract)
                        if j == 0:
                            my_new_bricks[i] = p_brick
                        elif j in new_nodes.keys():
                            new_nodes[j].add_brick(p_brick)
                        else:
                            new_node = ArchNode(target_hw=nn.targeted_hw)
                            new_node.max_perf_F = nn.max_perf_F
                            new_node.roofline = nn.roofline
                            new_node.data_parallelism_level = nn.data_parallelism_level
                            new_node.predecessors = nn.predecessors
                            new_node.successors = nn.successors
                            new_node.needs_compute_parallelization = True
                            new_node.compute_parallelization_factor = max_factor
                            new_node.add_brick(p_brick)
                            new_nodes[j] = new_node
                nn.bid_cnt = len(my_new_bricks)
                nn.bricks = my_new_bricks
                # add pointers to all other companion nodes to all nodes
                # all_parallel_nodes = [nn]
                # all_parallel_nodes.extend(list(new_nodes))
                new_nodes[0] = nn
                for nni in new_nodes.keys():
                    nnn = new_nodes[nni]
                    nnn.parallel_nodes = new_nodes
                    nnn.parallel_nodes_index = nni
                # TODO: insert new nodes after current node in draft
                for j in range(1, max_factor):
                    self.insert_node(new_nodes[j], nn.node_id + j)
        # two different types of ranks: data-parallelism, compute parallelism --> in update uuids
        self.update_uuids()
        # 5. data parallelization if necessary
        # TODO: different strategy for resource optimization
        assert self.strategy != OptimizationStrategies.RESOURCES
        for nn in self.node_iter_gen():
            # only nodes with one brick should be affected
            need_to_split = False
            reasons_txt = []
            split_factor = 0
            for lb in nn.local_brick_iter_gen():
                # engine and stream
                rr = nn.roofline.get_region_OIPlane_iter_based(lb.selected_contract.oi_iter, lb.req_iter_hz,
                                                               lb.selected_contract)
                if rr == RooflineRegionsOiPlane.ABOVE_TOP or rr == RooflineRegionsOiPlane.ABOVE_BRAM:
                    nsf = lb.req_iter_hz / lb.selected_contract.iter_hz
                    if nsf > 1.0:
                        need_to_split = True
                        if nsf > split_factor:
                            reasons_txt.append('exceeded compute budget (caused by {} on brick {})'
                                               .format(repr(lb.selected_contract), repr(lb)))
                            split_factor = nsf
                if rr != RooflineRegionsOiPlane.IN_HOUSE:
                    ap_contr_iter = nn.roofline.get_max_perf_at_oi_iter_based(lb.selected_contract.oi_iter,
                                                                              lb.selected_contract)
                    ap_stream = nn.roofline.get_max_perf_at_oi(lb.oi_stream)
                    if rr == RooflineRegionsOiPlane.ABOVE_NETWORK \
                            and lb.selected_impl_type == BrickImplTypes.ENGINE \
                            and lb.req_flops > ap_stream:
                        nsf = lb.req_flops / ap_stream
                        if nsf > 1.0:
                            need_to_split = True
                            if nsf > split_factor:
                                reasons_txt.append('exceeded network bandwidth budget')
                                split_factor = nsf
                    else:
                        nsf = lb.req_iter_hz / ap_contr_iter
                        if nsf > 1.0:
                            need_to_split = True
                            if nsf > split_factor:
                                reasons_txt.append('exceeded bandwidth (DRAM or network) budget')
                                split_factor = nsf
            if need_to_split:
                split_factor_up = math.ceil(split_factor)
                if split_factor_up < 2:
                    split_factor_up = 2
                if split_factor_up > dosa_singleton.config.dse.max_vertical_split:
                    print("[DOSA:archGen:ERROR] Vertically split factor of {} is above limit {}."
                          .format(split_factor_up, dosa_singleton.config.dse.max_vertical_split))
                    raise DosaConstraintFail
                nn.split_vertical(factor=split_factor_up)  # including update of used perf
                if verbose:
                    print("[DOSA:archGen:INFO] Parallelize node {} vertically with factor {}, due to ({})."
                          .format(nn.node_id, split_factor_up, reasons_txt))
        # 6. merge sequential nodes (no data par, no twins, same targeted_hw) if possible, based on used_perf,
        #  (i.e move bricks after each other)
        for nn in self.node_iter_gen():
            nn.update_used_perf_util_contr(add_switching_costs=True)
        node_ids_to_delete = []
        for ni in range(0, len(self.nodes)):
            if ni in node_ids_to_delete:
                continue
            n1 = self.nodes[ni]
            if n1.data_parallelism_level > 1 or n1.compute_parallelization_factor > 1:
                continue
            if ni < (len(self.nodes) - 1):
                n2 = self.nodes[ni + 1]
                if n2.data_parallelism_level > 1 or n2.compute_parallelization_factor > 1:
                    continue
                if n1.targeted_hw == n2.targeted_hw:
                    # TODO: move bricks after each other?
                    #  does it make sense? just reduces the resource usage somewhere else?
                    # if (n1.used_perf_F + n2.used_perf_F) <= n1.max_perf_F:
                    # BETTER to be cautious...considering mu again
                    if (((n1.used_comp_util_share + n2.used_comp_util_share)
                         * dosa_singleton.config.utilization.dosa_mu_comp)
                        < dosa_singleton.config.utilization.dosa_xi) \
                            and (((n1.used_mem_util_share + n2.used_mem_util_share)
                                  * dosa_singleton.config.utilization.dosa_mu_comp)
                                 < dosa_singleton.config.utilization.dosa_xi):
                        # merge nodes totally
                        if verbose:
                            print("[DOSA:archGen:INFO] merging sequential, non-parallel nodes {} and {} totally."
                                  .format(n1.node_id, n2.node_id))
                        node_ids_to_delete.append(ni + 1)
                        for bb in n2.local_brick_iter_gen():
                            n1.add_brick(bb)
        node_ids_to_delete.reverse()
        for nd in node_ids_to_delete:
            self.delete_node(nd)
        # 7. update possible hw targets
        self.update_possible_hw_types()
        # 8. decide for hw, if targeted hw is possible, use this one
        #  if not, use other possible hw with largest roof_F
        #  if only smaller roof_F are available, use fallback hw
        #  (optimizing hw --> not part of legalizing)
        for nn in self.node_iter_gen():
            if nn.targeted_hw in nn.possible_hw_types:
                nn.selected_hw_type = nn.targeted_hw
            else:
                targeted_roof_F = nn.targeted_hw.get_roof_F()
                max_roof_F = 0
                selected_hw = None
                for phw in nn.possible_hw_types:
                    # if only smaller roof_F are available, use fallback hw
                    phw_roof_F = phw.get_roof_F()
                    if phw_roof_F >= targeted_roof_F:
                        if phw_roof_F > max_roof_F:
                            max_roof_F = phw_roof_F
                            selected_hw = phw
                if selected_hw is not None:
                    nn.selected_hw_type = selected_hw
                    print("[DOSA:archGen:INFO] Targeted hw {} not possible for node {}, selected {} instead."
                          .format(nn.targeted_hw, nn.node_id, nn.selected_hw_type))
                else:
                    if len(self.fallback_hw_set) > 0:
                        fallback_hw_found = False
                        for fhw in self.fallback_hw_set:
                            if fhw in nn.possible_hw_types:
                                # take first one, order represents priority
                                nn.selected_hw_type = fhw
                                fallback_hw_found = True
                                print(("[DOSA:archGen:WARNING] Targeted hw {} not possible for node {}, forced to use "
                                       + "fallback hw {} instead.").format(nn.targeted_hw, nn.node_id,
                                                                           nn.selected_hw_type))
                                break
                        if not fallback_hw_found:
                            print(("[DOSA:archGen:ERROR] Targeted hw {} not possible for node {}, failed to find " +
                                   "replacement or fallback. Impossible to legalize draft").format(nn.targeted_hw,
                                                                                                   nn.node_id))
                            return DosaRv.ERROR
        # ensure, all HW is decided and select them
        # TODO: obsolte, since we filter for targeted_hw?
        selected_hw_types = []
        for nn in self.node_iter_gen():
            assert nn.selected_hw_type != placeholderHw
            if nn.selected_hw_type not in selected_hw_types:
                selected_hw_types.append(nn.selected_hw_type)
        self.all_selected_hw_types = selected_hw_types
        # 9. create blocks
        self.update_uuids()
        for nn in self.node_iter_gen():
            nn.update_block_list()
        # 10. check for engine threshold
        # TODO
        # for nn in self.node_iter_gen():
        #     for ce in nn.engine_container_refs:
        #         if ce.resource_savings < dosa_singleton.config.middleend.engine_saving_threshold:
        #             for bb in ce.block_ref.brick_list:
        #                 bb.set_impl_type(BrickImplTypes.STREAM)
        #                 if verbose:
        #                     print("[DOSA:archGen:INFO] Setting ImplType of Brick {} to STREAM,".format(bb.brick_uuid) +
        #                           " since its resource savings ({}) are below threshold.".format(ce.resource_savings))
        # update blocks again
        # for nn in self.node_iter_gen():
        #     nn.update_block_list()
        # 11. update kernel uuids & req. perf
        # self.update_uuids()
        self.update_required_perf()
        self.total_perf_F = 0
        self.total_flops = 0
        self.total_parameters_bytes = 0
        self.max_perf_iter_based = float('inf')
        self.min_iter_hz = float('inf')
        for nn in self.node_iter_gen():
            # nn.update_used_perf_util()
            nn.update_used_perf_util_contr(add_switching_costs=True)
            self.total_perf_F += nn.used_perf_F
            n_iter_based = nn.data_parallelism_level * nn.max_perf_iter_based
            n_used_iter = nn.data_parallelism_level * nn.used_iter_hz
            if n_iter_based < self.max_perf_iter_based:
                self.max_perf_iter_based = n_iter_based
            if n_used_iter < self.min_iter_hz:
                self.min_iter_hz = n_used_iter
            # assert nn.used_comp_util_share < 1.1
            # TODO
            if nn.used_comp_util_share > 1:
                print("[DOSA:archGen:WARNING] Node {} has {} compute utilization...implementation may fail"
                      .format(nn.node_id, nn.used_comp_util_share))
            # assert nn.used_comp_util_share < 1.2
            # assert nn.used_mem_util_share < 1.2
            if (not relax_utilization_check) and (nn.used_comp_util_share > 1.2 or nn.used_mem_util_share > 1.2):
                print("[DOSA:archGen:ERROR] Node {} is over utilized (compute: {}, mem: {})."
                      .format(nn.node_id, nn.used_comp_util_share, nn.used_mem_util_share))
                raise DosaConstraintFail
        self.total_flops = 0
        self.total_parameters_bytes = 0
        for bb in self.brick_iter_gen():
            self.total_flops += bb.flops
            self.total_parameters_bytes += bb.parameter_bytes
        return DosaRv.OK

    def update_required_perf(self):
        if self.strategy == OptimizationStrategies.THROUGHPUT:
            if self.target_sps < 0:
                print(
                    "[DOSA:archGen:ERROR] Optimization strategy ({}) does not fit target numbers in constraint \
                    target_sps ({}). Stop."
                    .format(self.strategy, self.target_sps))
                return DosaRv.ERROR
            # optimizing towards throughput
            target_throughput = self.target_sps * self.sample_size_B
            # annotate input & output
            self.input_layer['inp_Bs'] = self.input_layer['inpB'] * self.target_sps
            self.input_layer['out_Bs'] = self.input_layer['outB'] * self.target_sps
            self.output_layer['inp_Bs'] = self.output_layer['inpB'] * self.target_sps
            self.output_layer['out_Bs'] = self.output_layer['outB'] * self.target_sps
            # annotate bricks
            # pipelined design: no communication latency
            for node in self.node_iter_gen():
                # take data parallelism into account
                local_data_par_level = node.data_parallelism_level
                for brick in node.local_brick_iter_gen():
                    # no need to differentiate between Stream an Engine here -> oi is separate
                    brick.input_bw_Bs = brick.input_bytes * (self.target_sps / local_data_par_level)
                    brick.output_bw_Bs = brick.output_bytes * (self.target_sps / local_data_par_level)
                    orig_req_flops = brick.flops * (self.target_sps / local_data_par_level)
                    brick.req_iter_hz = (self.target_sps / local_data_par_level)
                    brick.req_flops = orig_req_flops * brick.flops_conv_factor
                    # annotate also the latency
                    brick.req_latency = brick.flops / brick.req_flops
        elif self.strategy == OptimizationStrategies.LATENCY:
            # optimizing towards latency
            if self.target_latency < 0:
                print(
                    "[DOSA:archGen:ERROR] Optimization strategy ({}) does not fit target numbers in constraint \
                     target_latency ({}). Stop."
                    .format(self.strategy, self.target_latency))
                return DosaRv.ERROR
            # first, try with 1/N distribution
            # consider communication latency
            comm_latency = 0
            for nn in self.node_iter_gen():
                if nn.targeted_hw is not None:
                    cl1 = nn.targeted_hw.get_comm_latency_s()
                    # just consider it two times, don't consider successor
                    comm_latency += 2 * cl1
            compute_latency = self.target_latency - comm_latency
            if compute_latency < 0:
                print("[DOSA:archGen:ERROR] Communication latency ({}) is higher than target latency ({}). "
                      .format(comm_latency, self.target_latency) +
                      "Impossible to generate architecture. Stop.")
                return -1
            latency_per_brick = compute_latency / float(self.get_bricks_num())
            for node in self.node_iter_gen():
                # take data parallelism into account
                local_data_par_level = node.data_parallelism_level
                for brick in node.local_brick_iter_gen():
                    brick.req_latency = latency_per_brick
                    # calc_latency is depending on mode
                    # brick.req_perf_engine = (brick.oi_engine * brick.input_bytes) / latency_per_brick
                    # brick.req_perf_stream = (brick.oi_stream * brick.input_bytes) / latency_per_brick
                    orig_req_flops = brick.flops / (latency_per_brick * local_data_par_level)
                    brick.req_flops = orig_req_flops * brick.flops_conv_factor
                    # brick.req_iter_hz = brick.req_flops / brick.flops
                    brick.req_iter_hz = 1 / (latency_per_brick * local_data_par_level)
                    # TODO: calculate also input_bw_Bs and output_bw_Bs
        else:
            # optimizing towards resource footprint
            if self.target_resources < 0:
                print(
                    "[DOSA:archGen:ERROR] Optimization strategy ({}) does not fit target numbers in constraint \
                    target_resources ({}). Stop."
                    .format(self.strategy, self.target_resources))
                return DosaRv.ERROR
            # find max resources in flops
            max_resources = 0
            max_res_dev = "unknown"
            for d in self.target_hw_set:
                dr = d.get_max_flops()
                if dr > max_resources:
                    max_resources = dr
                    max_res_dev = d.name
            allowed_resources = max_resources * self.target_resources
            self.tmp_notes['max_res_dev'] = max_res_dev
            self.tmp_notes['allowed_resources'] = allowed_resources
            # first, try with 1/N distribution
            # ignore latency, data_par etc.
            resource_per_brick = allowed_resources / self.get_bricks_num()
            for brick in self.brick_iter_gen():
                orig_req_flops = resource_per_brick
                brick.req_flops = orig_req_flops * brick.flops_conv_factor
                # annotate also the latency
                brick.req_latency = brick.flops / brick.req_flops
                brick.req_iter_hz = brick.req_flops / brick.flops
        return DosaRv.OK

    def update_uuids(self, add_backlink=False):
        next_kuuid = 0
        next_buuid = 0
        next_bluuid = 0
        cur_rank = 0
        prev_nodes = []
        first_nodes = []
        first_nodes_done = False
        cur_nodes = []
        for nn in self.node_iter_gen():
            next_kuuid = nn.update_kernel_uuids(next_kuuid)
            next_buuid = nn.update_brick_uuids(next_buuid)
            next_bluuid = nn.update_block_ids(next_bluuid)
            new_ranks = [cur_rank]
            cur_rank += 1
            for c in range(1, nn.data_parallelism_level):
                new_ranks.append(cur_rank)
                cur_rank += 1
            nn.ranks = new_ranks
            if len(first_nodes) == 0:
                first_nodes.append(nn)
            elif nn.needs_compute_parallelization and not first_nodes_done:
                for fnn in first_nodes:
                    if fnn in nn.parallel_nodes.values():
                        first_nodes.append(nn)
                        break
            nn.inp_ranks = []
            nn.out_ranks = []
            nn.parallel_ranks = []
            appended = False
            if nn.needs_compute_parallelization:
                if len(cur_nodes) == 0:
                    cur_nodes = [nn]
                    appended = True
                else:
                    for cn in cur_nodes:
                        if cn in nn.parallel_nodes.values():
                            cur_nodes.append(nn)
                            appended = True
                            break
            if not nn.needs_compute_parallelization or not appended:
                # all_parallel_ranks = []
                for cn in cur_nodes:
                    # all_parallel_ranks.extend(pn.ranks)
                    # all_parallel_ranks.append(pn.ranks)  # list of list
                    for cnn in cur_nodes:
                        if cnn == cn:
                            continue
                        cn.parallel_ranks.append(cnn.ranks)  # list of list
                # for pn in prev_nodes:
                #     pn.parallel_ranks = all_parallel_ranks
                first_nodes_done = True
                prev_nodes = cur_nodes
                cur_nodes = [nn]
            for pn in prev_nodes:
                # nn.inp_ranks.extend(pn.ranks)
                # pn.out_ranks.extend(new_ranks)
                nn.inp_ranks.append(pn.ranks)
                pn.out_ranks.append(new_ranks)
        if add_backlink:
            for cn in cur_nodes:
                for fn in first_nodes:
                    cn.out_ranks.append(fn.ranks)
            for fn in first_nodes:
                for cn in cur_nodes:
                    fn.inp_ranks.append(cn.ranks)
        # transpose ranks
        for nn in self.node_iter_gen():
            nn.inp_ranks = np.array(nn.inp_ranks).T.tolist()
            nn.out_ranks = np.array(nn.out_ranks).T.tolist()
            nn.parallel_ranks = np.array(nn.parallel_ranks).T.tolist()
        self.max_brick_uuid = next_buuid - 1
        return

    # def update_possible_osgs(self):
    #     for nn in self.node_iter_gen():
    #         nn.update_possible_osgs()

    def update_possible_hw_types(self):
        cur_possible_hw_types = []
        for nn in self.node_iter_gen():
            nn.update_possible_hw_types()
            nn_phw = nn.possible_hw_types
            # add all possibilities
            for nn_pht in nn_phw:
                if nn_pht not in cur_possible_hw_types:
                    cur_possible_hw_types.append(nn_pht)
        # TODO: it is ok if different nodes have different hw
        # not_possible_hw_types = []
        # for nn in self.node_iter_gen():
        #     nn_phw = nn.possible_hw_types
        #     # now, remove all non-common options
        #     for cpht in cur_possible_hw_types:
        #         if cpht not in nn_phw:
        #             not_possible_hw_types.append(cpht)
        # not_possible_hw_types = list(set(not_possible_hw_types))
        # for npht in not_possible_hw_types:
        #     del cur_possible_hw_types[cur_possible_hw_types.index(npht)]
        self.possible_hw_types = cur_possible_hw_types

    def build(self, verbose=False):
        self.generate_communication()
        for nn in self.node_iter_gen():
            nn.build()
            build_folder_name = nn.build_tool.get_node_folder_name()
        # add to global cluster setup info
        self._generate_cluster_description()
        self._generate_extended_cluster_description(verbose=verbose)
        if dosa_singleton.config.backend.tmux_parallel_build > 0:
            self._generate_tmux_build_script()

    # def synth(self):
    #     for nn in self.node_iter_gen():
    #         nn.synth()

    def write_info(self, verbose=False):
        for nn in self.node_iter_gen():
            nn.build(only_folders=True)
        self._generate_cluster_description()
        self._generate_extended_cluster_description(verbose=verbose)

    def _generate_cluster_description(self):
        num_nodes = self.get_total_nodes_cnt()
        # if self.substract_node_0:
        #     num_nodes += 1
        num_nodes -= self.substract_client_nodes
        cluster_dict = {'name': self.name, 'total_nodes': num_nodes, 'nodes': []}
        #                 'num_cpu_clients': self.substract_client_nodes}
        # if dosa_singleton.config.backend.create_rank_0_for_io:
        #     cluster_dict['num_cpu_clients'] = len(self.nodes[0].ranks)
        for nn in self.node_iter_gen():
            nn_f = nn.build_tool.node_folder_name
            nn_ranks = nn.ranks
            n_hw = nn.selected_hw_type.name
            # ne = {nn_f: nn_ranks}
            ne = {'folder': nn_f, 'ranks': nn_ranks, 'type': n_hw}
            cluster_dict['nodes'].append(ne)
            # cluster_dict['nodes'][nn_f] = nn_ranks
        out_file = '{}/cluster.json'.format(dosa_singleton.config.global_build_dir)
        with open(out_file, 'w') as of:
            json.dump(cluster_dict, of, indent=4)
        out_file2 = '{}/generated_architecture.json'.format(dosa_singleton.config.global_build_dir)
        with open(out_file2, 'w') as of:
            of.write(str(self))

    def _generate_extended_cluster_description(self, verbose=False):
        cluster_dict = self.get_extended_cluster_description()
        out_file = '{}/arch_info.json'.format(dosa_singleton.config.global_build_dir)
        with open(out_file, 'w') as of:
            json.dump(cluster_dict, of, indent=4)
        if verbose:
            print("\n[VERBOSE] best draft found:")
            print(json.dumps(cluster_dict, indent=2))

    def get_extended_cluster_description(self):
        num_nodes = self.get_total_nodes_cnt()
        num_nodes -= self.substract_client_nodes
        # if self.substract_node_0:
        #     num_nodes += 1
        cluster_dict = {'name': self.name, 'total_flops': float(self.total_flops),
                        'total_parameter_bytes': int(self.total_parameters_bytes),
                        'predicted_performance_iter_hz': float(self.min_iter_hz),
                        'total_dse_time_s': self.total_time_dse_seconds,
                        'total_nodes': num_nodes, 'nodes': []}
        for nn in self.node_iter_gen():
            if nn.build_tool is not None:
                nn_f = nn.build_tool.node_folder_name
            else:
                nn_f = 'None'
            nn_ranks = nn.ranks
            n_hw = nn.selected_hw_type.name
            # ne = {nn_f: nn_ranks}
            node_dict = nn.as_dict()
            ne = {'folder': nn_f, 'ranks': nn_ranks, 'type': n_hw, 'blocks': node_dict['blocks'], 'bricks': {},
                  'estimations': node_dict['estimations']}
            for bb in nn.local_brick_iter_gen():
                bb_sum = bb.as_summary()
                ne['bricks'][bb.brick_uuid] = bb_sum
            cluster_dict['nodes'].append(ne)
            # cluster_dict['nodes'][nn_f] = nn_ranks
        return cluster_dict

    def _generate_tmux_build_script(self):
        os.system('cp {}/../../backend/buildTools/templates/cFBuild1/dosa_build.sh {}/'
                  .format(__filedir__, dosa_singleton.config.global_build_dir))
        tmux_tmpl = 'tmux new-window -t gradatim:{w} -n "{name}"\ntmux send-keys -t gradatim:{w} "{cmd}" C-m\n'
        cur_wi = 1
        cur_sleep_cnt = 0
        # TODO: make dynamic
        cmd_tmpl = 'cd {}; sleep {}h; ./sra build pr'
        cmd_tmpl_mono = 'cd {}; sleep {}h; ./sra build monolithic'
        cur_parallel_build = 0
        first_over_utilized = True
        first_mono_sleep_cnt = 0
        over_utilized_cnt = 0
        sleep_factor = 3
        with open(os.path.abspath(dosa_singleton.config.global_build_dir + '/dosa_build.sh'), 'a') as script_file:
            for nn in self.node_iter_gen():
                if nn.targeted_hw.hw_class in [DosaHwClasses.FPGA_generic, DosaHwClasses.FPGA_xilinx]:
                    nn_f = nn.build_tool.node_folder_name
                    nn_n = 'node_{:02d}'.format(nn.node_id)
                    # if nn.over_utilized_node:
                    #     if first_over_utilized:
                    #         cmd = cmd_tmpl_mono.format(nn_f, cur_sleep_cnt)
                    #         first_over_utilized = False
                    #         first_mono_sleep_cnt = cur_sleep_cnt
                    #     else:
                    #         if first_mono_sleep_cnt == cur_sleep_cnt:
                    #             cmd = cmd_tmpl_mono.format(nn_f, cur_sleep_cnt + sleep_factor)
                    #         else:
                    #             cmd = cmd_tmpl_mono.format(nn_f, cur_sleep_cnt)
                    #     over_utilized_cnt += 1
                    # else:
                    cmd = cmd_tmpl.format(nn_f, cur_sleep_cnt)
                    tmux_s = tmux_tmpl.format(w=cur_wi, cmd=cmd, name=nn_n)
                    script_file.write(tmux_s)
                    cur_wi += 1
                    cur_parallel_build += 1
                    if cur_parallel_build >= dosa_singleton.config.backend.tmux_parallel_build:
                        cur_parallel_build = 0
                        cur_sleep_cnt += sleep_factor
            script_file.write('\n\n')
        os.system('chmod +x {}/dosa_build.sh'.format(dosa_singleton.config.global_build_dir))

    def _add_node_0(self):
        node_0 = ArchNode(0, target_hw=vCPU_x86)
        node_0.selected_hw_type = vCPU_x86
        node_0.targeted_hw = vCPU_x86
        node_0.skip_in_roofline = True
        # add output brick first, so they match the input...output logic of CommPlan
        req_iter_hz = self.nodes[0].req_iter_hz  # should be equal to "entry node"...
        node_0.req_iter_hz = req_iter_hz
        output_brick = ArchBrick()
        output_brick.from_dpl_dict(self.output_layer)
        output_brick.skip_in_roofline = True
        output_op = ArchOp()
        output_op.is_pseudo_op = True
        output_contract_op = osg_tvmCpu._get_contr_offer(output_op, vCPU_x86, BrickImplTypes.ENGINE)
        output_contract_brick = BrickContract(output_brick, vCPU_x86, osg_tvmCpu, BrickImplTypes.ENGINE, [output_contract_op])
        output_brick.still_possible_contracts = [output_contract_brick]
        output_brick.available_contracts = [output_contract_brick]
        output_brick.selected_contract = output_contract_brick
        output_brick.still_possible_osgs = [osg_tvmCpu]
        output_brick.available_osgs = [osg_tvmCpu]
        output_brick.selected_osg = osg_tvmCpu
        output_brick.is_pseudo_brick = True
        output_brick.iter_hz = req_iter_hz
        output_brick.req_util_comp = 0.5
        output_brick.req_util_mem = 0.5
        node_0.add_brick(output_brick)
        input_brick = ArchBrick()
        input_brick.from_dpl_dict(self.input_layer)
        input_brick.skip_in_roofline = True
        input_op = ArchOp()
        input_op.is_pseudo_op = True
        input_contract_op = osg_tvmCpu._get_contr_offer(input_op, vCPU_x86, BrickImplTypes.ENGINE)
        input_contract_brick = BrickContract(input_brick, vCPU_x86, osg_tvmCpu, BrickImplTypes.ENGINE, [input_contract_op])
        input_brick.still_possible_contracts = [input_contract_brick]
        input_brick.available_contracts = [input_contract_brick]
        input_brick.selected_contract = input_contract_brick
        input_brick.still_possible_osgs = [osg_tvmCpu]
        input_brick.available_osgs = [osg_tvmCpu]
        input_brick.selected_osg = osg_tvmCpu
        input_brick.is_pseudo_brick = True
        input_brick.iter_hz = req_iter_hz
        input_brick.req_util_comp = 0.5
        input_brick.req_util_mem = 0.5
        node_0.add_brick(input_brick)
        self.insert_node(node_0, 0)
        self.substract_client_nodes = 1
        # add also with "inverted" successor/predecessor
        node_0.add_succ_node(self.nodes[1])
        node_0.add_pred_node(self.nodes[self.nid_cnt - 1])
        # update other nodes
        self.nodes[1].add_pred_node(node_0)
        self.nodes[self.nid_cnt - 1].add_succ_node(node_0)
        self.update_uuids(add_backlink=True)

    def generate_communication(self):
        # first, decide for communication lib
        # communication lib is "global" for the draft
        # for now, choose first possible
        for pcl in self.possible_comm_libs:
            possible = True
            for seldev in self.all_selected_hw_types:
                if seldev not in pcl.dosaHwTypes:
                    possible = False
                    break
            if possible:
                self.selected_comm_lib = pcl
                break
        if self.selected_comm_lib is placeholderCommLib:
            print('[DOSA:archGen:ERROR] Unable to find one common communication library. STOP.')
            exit(-1)
        # update node_id if necessary
        if dosa_singleton.config.backend.create_rank_0_for_io:
            self._add_node_0()
        # calculate pipeline effects
        draft_total_pipeline_store = 0
        prev_node = None
        for nn in self.node_iter_gen():
            # don't add up parallel nodes
            if prev_node not in nn.parallel_nodes.values():
                draft_total_pipeline_store += nn.total_pipeline_store
                prev_node = nn
        if dosa_singleton.config.backend.comm_message_pipeline_store < (draft_total_pipeline_store + 1):
            print("[DOSA:CommGen:INFO] Setting message interleaving to {}, due to higher pipeline storage within node."
                  .format(draft_total_pipeline_store + 1))
            dosa_singleton.config.backend.comm_message_pipeline_store = draft_total_pipeline_store + 1
        # if dosa_singleton.config.backend.create_rank_0_for_io:
        #    # so that CPU receives the right amount
        #    self.nodes[0].total_pipeline_store = total_pipeline_store
        # then, populate
        need_to_parallelize, nodes_to_parallelize, split_factors = self._populate_communication_generation(
            draft_total_pipeline_store)
        if need_to_parallelize and dosa_singleton.config.backend.allow_multiple_cpu_clients:
            assert len(split_factors) == len(nodes_to_parallelize)
            for i in range(len(nodes_to_parallelize)):
                pnn = nodes_to_parallelize[i]
                sf = split_factors[i]
                # if pnn.node_id == 0 and dosa_singleton.config.backend.create_rank_0_for_io:
                #     self.substract_client_nodes = sf
                pnn.split_vertical(factor=sf)
            # to get ranks right
            self.update_uuids(add_backlink=True)
            # no second round...
            ignore, also_ignore, dont_care = self._populate_communication_generation(draft_total_pipeline_store,
                                                                                     check_constraints=False)

    def _populate_communication_generation(self, draft_total_pipeline_store, check_constraints=True):
        pipeline_store_until_now = 0
        last_pipeline_store = 0
        prev_node = None
        need_to_parallelize = False
        nodes_to_parallelize = []
        split_factors = []
        for nn in self.node_iter_gen():
            if nn.node_id == 0 and dosa_singleton.config.backend.create_rank_0_for_io:
                # FIXME: find more elegant way...
                nn.total_pipeline_store = -1 * draft_total_pipeline_store
                nn.generate_communication(self.selected_comm_lib, draft_total_pipeline_store)
                nn.total_pipeline_store = 0
            else:
                # don't add up parallel nodes
                if prev_node not in nn.parallel_nodes.values():
                    # pipeline_store_until_now += nn.total_pipeline_store
                    pipeline_store_until_now += last_pipeline_store
                    prev_node = nn
                nn.generate_communication(self.selected_comm_lib, pipeline_store_until_now)
                last_pipeline_store = nn.total_pipeline_store
            # in all cases
            if check_constraints:
                check_dict = nn.check_connection_limit(verbose=True)
                if not check_dict['below_device-limit']:
                    need_to_parallelize = True
                    nodes_to_parallelize.append(nn)
                    split_factor_up = math.ceil(check_dict['split_factor'])
                    if split_factor_up < 2:
                        split_factor_up = 2
                    split_factors.append(split_factor_up)
                    print(f"[DOSA:CommGen:INFO] Annotate node {nn.node_id} to split vertically with factor "
                          f"{split_factor_up}, to fix transaction limitations (if enabled).")
        return need_to_parallelize, nodes_to_parallelize, split_factors

    def get_osg_coverage(self):
        osg_overage = {}
        osg_stats = {}
        total_ops = 0
        not_covered = 0
        for osg in self.available_osgs:
            osg_overage[osg.name] = 0
            osg_stats[osg.name] = {'count': 0, 'ratio': 0.0}
        for op in self.op_iter_gen():
            total_ops += 1
            already_considered = []
            for poc in op.possible_contracts:
                ogn = poc.osg.name
                if ogn not in already_considered:
                    osg_overage[ogn] += 1
                    already_considered.append(ogn)
        for osgn in osg_overage.keys():
            osg_stats[osgn]['count'] = osg_overage[osgn]
            osg_stats[osgn]['ratio'] = float(osg_overage[osgn]/total_ops)
        ret = {'draft': self.name, 'total_ops': total_ops, 'not_covered': not_covered,
               'dosa_coverage': float(1 - (not_covered / total_ops)),
               'osg_coverage': osg_stats}
        return ret

    def write_osg_coverage(self):
        coverage = self.get_osg_coverage()
        out_file = '{}/osg_statistic.json'.format(dosa_singleton.config.global_build_dir)
        with open(out_file, 'w') as of:
            json.dump(coverage, of, indent=4)

