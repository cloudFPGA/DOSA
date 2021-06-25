#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Jun 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Visitor calculating OI for each node
#  *
#  *

import tvm

"""
UPDATE: we don't need to implement the pre/post order  visiting, this is done by TVM
"""

class RelayOiVisitor(object):

    _method_cache = None

    def __init__(self):
        self.oi_results = []

    def get_oi_results(self):
        return self.oi_results

    def visit(self, op: tvm.tir.Stmt):
        """ Visit an operation (i.e. Stmt)
        """

        if self._method_cache is None:
            self._method_cache = {}

        visitor = self._method_cache.get(op.__class__.__name__, None)
        if visitor is None:
            method = 'visit_' + op.__class__.__name__
            visitor = getattr(self, method, self.generic_visit)
            self._method_cache[op.__class__.__name__] = visitor

        return visitor(op)

    def generic_visit(self, op: tvm.tir.Stmt):
        """ Called if no explicit visitor function exists for a
            node. Implements preorder visiting of the node.
        """
        print("generic visit called for {}".format(op.__class__.__name__))
        # if hasattr(op, 'body'):
        for c in op:
            self.visit(c)
