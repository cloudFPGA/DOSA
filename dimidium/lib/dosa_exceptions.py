#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: May 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Library for exception handling within DOSA
#  *


class DosaInvalidAction(Exception):
    pass


class DosaImpossibleToProceed(Exception):
    pass


class DosaChangeArchType(Exception):
    pass


