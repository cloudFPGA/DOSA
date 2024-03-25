#  /*******************************************************************************
#   * Copyright 2019 -- 2024 IBM Corporation
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
#  *    =============================================
#  *     Created: Dec 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Utility library for OSGs for DOSA arch gen
#  *
#  *

from tvm.tir.expr import IntImm


def convert_IntImm_array(intImm_arr):
    ret_arr = []
    for e in intImm_arr:
        if isinstance(e, IntImm):
            ret_arr.append(e.value)
        else:
            inner_arr = convert_IntImm_array(e)
            ret_arr.append(inner_arr)
    return ret_arr

