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
from enum import Enum


class DosaHwClasses(Enum):
    UNDECIDED = 0
    CPU_generic = 1
    FPGA_generic = 2
    CPU_x86 = 3
    FPGA_xilinx = 4


class DosaBaseHw(metaclass=abc.ABCMeta):

    def __init__(self, hw_type: DosaHwClasses, type_str, name):
        self.hw_type = hw_type
        self.name = name
        self.type_str = type_str

    @abc.abstractmethod
    def get_performance_dict(self):
        print("[DOSA:DEVICES:ERROR] NOT YET IMPLEMENTED.")

    @abc.abstractmethod
    def get_roofline_dict(self):
        print("[DOSA:DEVICES:ERROR] NOT YET IMPLEMENTED.")

    @abc.abstractmethod
    def get_resource_dict(self):
        print("[DOSA:DEVICES:ERROR] NOT YET IMPLEMENTED.")

    @abc.abstractmethod
    def get_max_flops(self):
        print("[DOSA:DEVICES:ERROR] NOT YET IMPLEMENTED.")

    @abc.abstractmethod
    def get_comm_latency_s(self):
        print("[DOSA:DEVICES:ERROR] NOT YET IMPLEMENTED.")

