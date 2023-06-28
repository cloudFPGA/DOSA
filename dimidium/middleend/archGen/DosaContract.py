#  /*******************************************************************************
#   * Copyright 2019 -- 2023 IBM Corporation
#   *
#   * Licensed under the Apache License, Version 2.0 (the "License");
#   * you may not use this file except in compliance with the License.
#   * You may obtain a copy of the License at
#   *
#   *     http://www.apache.org/licenses/LICENSE-2.0
#   *
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS,
#   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   * See the License for the specific language governing permissions and
#   * limitations under the License.
#  *******************************************************************************/
#

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


