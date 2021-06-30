#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Jun 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *       Library for calculating the OI for selected operations
#  *
#  *

import numpy as np
import math


class OiCalculator(object):

    def __init__(self, default_oi):
        self.default_oi = default_oi
        self._method_cache = None

    def calc(self, op_name, data_dim, param_dim, attrs, dtype):
        """
        Calculate OI for an operation
        """

        if self._method_cache is None:
            self._method_cache = {}

        # method_name = op.__class__.__name__
        method_name = op_name.split('.')[-1]
        visitor = self._method_cache.get(method_name, None)
        if visitor is None:
            method = 'calc_' + method_name
            # get method or default
            visitor = getattr(self, method, self.generic_calc)
            self._method_cache[method_name] = visitor

        return visitor(op_name, data_dim, param_dim, attrs, dtype)

    def generic_calc(self, op_name, data_dim, param_dim, attrs, dtype):
        """ Called if no explicit calculator function exists for an operation
        """
        print("generic calculation called for {}".format(op_name))
        return self.default_oi

