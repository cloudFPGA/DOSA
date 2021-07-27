#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: July 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Base class for DOSA target devices
#  *
#  *

import abc

from dimidium.lib.units import *


class DosaBaseHw(metaclass=abc.ABCMeta):

    def __init__(self, hw_type, name):
        self.hw_type = hw_type
        self.name = name

    @abc.abstractmethod
    def get_performance_dict(self):
        print("[DOSA:DEVICES:ERROR] NOT YET IMPLEMENTED.")

    @abc.abstractmethod
    def get_roofline_dict(self):
        print("[DOSA:DEVICES:ERROR] NOT YET IMPLEMENTED.")

    @abc.abstractmethod
    def get_resource_dict(self):
        print("[DOSA:DEVICES:ERROR] NOT YET IMPLEMENTED.")

