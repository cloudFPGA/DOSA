#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Mar 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Abstract class of implementation contracts
#  *
#  *

import abc


class DosaContract(metaclass=abc.ABCMeta):

    def __init__(self, device, osg, impl_type, iter_hz, comp_util_share, mem_util_share):
        self.device = device
        self.osg = osg
        self.impl_type = impl_type
        self.iter_hz = float(iter_hz)
        self.comp_util_share = comp_util_share
        self.mem_util_share = mem_util_share
        self.oi_iter = -1
        self.num_ops = -1


