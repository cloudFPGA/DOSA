#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Oct 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Base class for DOSA Operator Set Generators (OSGs)
#  *
#  *

import abc
from dimidium.backend.devices.dosa_device import DosaHwClasses
from dimidium.middleend.archGen.ArchBrick import ArchBrick
from dimidium.backend.buildTools.BaseBuild import BaseBuild


class BaseOSG(metaclass=abc.ABCMeta):

    relay2osg = {}
    dosaHwTypes = []

    def __init__(self, name, device_class: DosaHwClasses, framework_path, build_tool: BaseBuild):
        self.name = name
        self.device_class = device_class
        self.framework_path = framework_path
        self.build_tool = build_tool

    @abc.abstractmethod
    def annotate_brick(self, brick_node: ArchBrick):
        print("[DOSA:OSG:ERROR] NOT YET IMPLEMENTED.")

    @abc.abstractmethod
    def generate_brick(self, brick_node: ArchBrick):
        print("[DOSA:OSG:ERROR] NOT YET IMPLEMENTED.")

    @abc.abstractmethod
    def comm_wrap_brick(self, todo):
        print("[DOSA:OSG:ERROR] NOT YET IMPLEMENTED.")

    @abc.abstractmethod
    def estimate_flops_brick(self, brick_node: ArchBrick):
        print("[DOSA:OSG:ERROR] NOT YET IMPLEMENTED.")


class UndecidedOSG(BaseOSG):
    def annotate_brick(self, brick_node: ArchBrick):
        pass

    def generate_brick(self, brick_node: ArchBrick):
        pass

    def comm_wrap_brick(self, todo):
        pass

    def estimate_flops_brick(self, brick_node: ArchBrick):
        pass



