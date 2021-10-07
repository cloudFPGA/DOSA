#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Oct 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Base classes for DOSA build tools
#  *
#  *

import abc


class BaseBuild(metaclass=abc.ABCMeta):

    def __init__(self, name):
        self.name = name


class BaseHwBuild(BaseBuild, metaclass=abc.ABCMeta):

    def __init__(self, name):
        super().__init__(name)


class BaseSwBuild(BaseBuild, metaclass=abc.ABCMeta):

    def __init__(self, name):
        super().__init__(name)

