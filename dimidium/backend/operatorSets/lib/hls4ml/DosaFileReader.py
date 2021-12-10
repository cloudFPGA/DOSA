#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Nov 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Library for hls4ml code generation.
#  *        Please see the folder Readme.md for more information.
#  *
#  *

import numpy as np


class OsgDataReader(object):
    def __init__(self, config, data=None):
        self.config = config
        if data is None:
            self.data = {}
        else:
            self.data = data

    def add_data_entry(self, layer_name, var_name, ndarray: np.ndarray, data_format, overwrite=False):
        if layer_name in self.data and var_name in self.data[layer_name] and not overwrite:
            # don't overwrite data entries silently
            return False
        if layer_name not in self.data:
            self.data[layer_name] = {}
        if data_format != 'channels_last' and len(ndarray.shape) > 1:
            new_array = np.moveaxis(ndarray, 0, -1)
        else:
            new_array = ndarray
        self.data[layer_name][var_name] = new_array
        return True

    def add_data_dict(self, data_dict):
        for vn in data_dict['variables']:
            self.add_data_entry(data_dict['layer_name'], vn, data_dict['variables'][vn],
                                data_dict['data_format'][vn])

    def _find_data(self, layer_name, var_name):
        data = None
        if layer_name in self.data and var_name in self.data[layer_name]:
            data = self.data[layer_name][var_name]
        return data

    def get_weights_data(self, layer_name, var_name):
        """returns a ndarray (shape preserved)"""
        data = self._find_data(layer_name, var_name)
        if data is not None:
            # return data[()]
            return data
        else:
            return None

    def get_weights_shape(self, layer_name, var_name):
        """returns a tuple"""
        data = self._find_data(layer_name, var_name)
        if data is not None:
            return data.shape
        else:
            return None



