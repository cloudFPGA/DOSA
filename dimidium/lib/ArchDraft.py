#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: July 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Class containing one version of a DOSA architectural draft
#  *
#  *

import json

from dimidium.lib.util import OptimizationStrategies
from dimidium.lib.ArchNode import ArchNode


class ArchDraft(object):

    # _bstr_fmt_ = "{:06}"
    # _bid_max_ = 99999

    def __init__(self, name, version, strategy: OptimizationStrategies, batch_size, target_sps=-1, target_latency=-1,
                 target_resources=-1, tvm_node=None):
        self.name = name
        self.version = version
        self.strategy = strategy
        self.batch_size = batch_size
        self.target_sps = target_sps
        self.target_latency = target_latency
        self.target_resources = target_resources
        self.main_tvm_handle = tvm_node
        self.nodes = {}
        self.nid_cnt = 0
        self.input_layer = None
        self.output_layer = None

    def __repr__(self):
        return "ArchDraft({}, {}, {})".format(self.name, self.version, self.strategy)

    def __str__(self):
        res = {'name': self.name, 'version': self.version, 'strategy': str(self.strategy),
               'batch_size': self.batch_size, 'target_sps': self.target_sps, 'target_latency': self.target_latency,
               'target_resources': self.target_resources,
               'input': str(self.input_layer), 'output': str(self.output_layer),
               'main_tvm_handle': str(self.main_tvm_handle)[:100], 'nodes': []}
        for ni in self.nodes:
            n = self.nodes[ni]
            res['nodes'].append(str(n))
        ret = {'ArchDraft': res}
        return json.dumps(ret, indent=2)

    def add_node(self, node: ArchNode):
        # bstr = self._bstr_fmt_.format(self.bid_cnt)
        n_id = self.nid_cnt
        self.nid_cnt += 1
        # if self.bid_cnt > self._bid_max_:
        #    print("[DOSA:ArchDraft:ERROR] Brick Id overflow occurred!")
        node.set_node_id(n_id)
        self.nodes[n_id] = node

    # def insert_node(self, node: ArchNode):
    #     """adds an ArchNode without overwriting it's node_id"""
    #     n_id = node.get_node_id()
    #     self.nodes[n_id] = node

    def set_tvm_handle(self, tvm_node):
        self.main_tvm_handle = tvm_node

    def set_input_layer(self, in_dpl):
        self.input_layer = in_dpl

    def set_output_layer(self, out_dpl):
        self.output_layer = out_dpl

