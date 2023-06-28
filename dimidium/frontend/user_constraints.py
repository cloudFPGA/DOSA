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
#  *    =============================================
#  *     Created: Oct 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *       Library for parsing and checking user constraints
#  *
#  *

import math
import json

from dimidium.lib.util import OptimizationStrategies, config_bits_per_byte
from dimidium.lib.dosa_dtype import convert_tvmDtype_to_DosaDtype, DosaDtype
import dimidium.lib.singleton as dosa_singleton


__mandatory_user_keys__ = ['shape_dict', 'used_batch_n', 'name', 'target_sps', 'targeted_hw',
                           'target_resource_budget', 'arch_gen_strategy', 'fallback_hw', 'used_input_size_t',
                           'target_latency', 'quantization']
__optional_user_keys__ = ['overwrite_dtypes', 'osg_allowlist']
__arch_gen_strategies__ = ['performance', 'resources', 'default', 'latency', 'throughput']
# __valid_fallback_hws__ = ['None']
# __valid_fallback_hws__.extend(dosa_devices.fallback_hw)


def parse_uc_dict(path, dosa_devices):
    with open(path, 'r') as inp:
        user_constraints = json.load(inp)

    parsed_constraints = {}
    for k in __mandatory_user_keys__:
        if k not in user_constraints:
            print("ERROR: Mandatory key {} is missing in the constraints file {}. Stop.".format(k, path))
            exit(1)
        else:
            parsed_constraints[k] = user_constraints[k]

    for k in __optional_user_keys__:
        if k in user_constraints:
            parsed_constraints[k] = user_constraints[k]

    arch_target_devices = []
    for td in parsed_constraints['targeted_hw']:
        if td not in dosa_devices.types_str:
            print("ERROR: Target hardware {} is not supported. Stop.".format(td))
            exit(1)
        else:
            arch_target_devices.append(dosa_devices.types_dict[td])

    uags = parsed_constraints['arch_gen_strategy']
    if uags not in __arch_gen_strategies__:
        print("ERROR: Architecture optimization strategy {} is not supported. Stop.".format(uags))
        exit(1)

    arch_gen_strategy = OptimizationStrategies.DEFAULT
    if uags == 'performance':
        arch_gen_strategy = OptimizationStrategies.PERFORMANCE
    elif uags == 'resources':
        arch_gen_strategy = OptimizationStrategies.RESOURCES
    elif uags == 'latency':
        arch_gen_strategy = OptimizationStrategies.LATENCY
    elif uags == 'throughput':
        arch_gen_strategy = OptimizationStrategies.THROUGHPUT
    else:
        arch_gen_strategy = OptimizationStrategies.DEFAULT

    arch_fallback_hw = None
    if type(parsed_constraints['fallback_hw']) is list:
        for fhw in parsed_constraints['fallback_hw']:
            if fhw not in dosa_devices.types_str:
                print("ERROR: Fallback hardware {} is not supported. Stop.".format(fhw))
                exit(1)
        arch_fallback_hw = parsed_constraints['fallback_hw']
    else:
        if parsed_constraints['fallback_hw'] != "none":
            print("ERROR: Fallback hardware {} is not supported. Stop."
                  .format(parsed_constraints['fallback_hw']))
            exit(1)
        # arch_fallback_hw stays None

    used_batch = parsed_constraints['used_batch_n']
    used_in_size_t = parsed_constraints['used_input_size_t']
    sample_size_bit = used_in_size_t
    for inp_k in parsed_constraints['shape_dict']:
        inp_v = parsed_constraints['shape_dict'][inp_k]
        total_d = len(inp_v)
        if used_batch != inp_v[0]:
            print("ERROR: Batch sizes in constraint file contradict each other ({} != {}). Stop."
                  .format(used_batch, inp_v[0]))
            exit(1)
            # print("...trying to continue...")
        for i in range(1, total_d):  # exclude batch size
            d = inp_v[i]
            sample_size_bit *= d
    used_sample_size = math.ceil(sample_size_bit / config_bits_per_byte)
    parsed_constraints['used_sample_size'] = used_sample_size

    if parsed_constraints['quantization'] == 'none':
        parsed_constraints['do_quantization'] = False
    else:
        parsed_constraints['do_quantization'] = True
        low_dtype = convert_tvmDtype_to_DosaDtype(parsed_constraints['quantization'])
        if low_dtype == DosaDtype.UNKNOWN:
            print("ERROR: Quantization data type {} is not supported. Stop.".format(parsed_constraints['quantization']))
            exit(1)
        parsed_constraints['target_dtype'] = low_dtype

    # TODO: allow float as input?
    input_dtype_str = f'int{used_in_size_t}'
    parsed_constraints['input_dtype'] = convert_tvmDtype_to_DosaDtype(input_dtype_str)
    if 'overwrite_dtypes' in parsed_constraints:
        parsed_constraints['overwrite_imported_dtypes'] = True
        dosa_singleton.config.quant.overwrite_imported_dtypes = True
        data_dtype = convert_tvmDtype_to_DosaDtype(parsed_constraints['overwrite_dtypes']['data'])
        if data_dtype == DosaDtype.UNKNOWN:
            print('ERROR: Datatype {} to overwrite "data" data types is not supported. Stop.'.format(
                parsed_constraints['overwrite_dtypes']['data']))
            exit(1)
        dosa_singleton.config.quant.activation_dtype = data_dtype
        weights_dtype = convert_tvmDtype_to_DosaDtype(parsed_constraints['overwrite_dtypes']['weights'])
        if weights_dtype == DosaDtype.UNKNOWN:
            print('ERROR: Datatype {} to overwrite "weights" data types is not supported. Stop.'.format(
                parsed_constraints['overwrite_dtypes']['weights']))
            exit(1)
        dosa_singleton.config.quant.weight_dtype = weights_dtype
        # TODO
        dosa_singleton.config.quant.bias_dtype = weights_dtype
        if 'fixed_point_fraction_bits' in parsed_constraints['overwrite_dtypes']:
            parsed_constraints['overwrite_fixed_point_dtypes'] = True
            dosa_singleton.config.quant.fixed_point_fraction_bits = \
                int(parsed_constraints['overwrite_dtypes']['fixed_point_fraction_bits'])
            dosa_singleton.config.quant.overwrite_fixed_point_dtypes = True
            print("[DOSA:ConstraintParsing:WARNING] [NOT YET IMPLEMENTED] Custom fixed point fractional bits will be "
                  "ignored, due to unclear encoding.")
        else:
            parsed_constraints['overwrite_fixed_point_dtypes'] = False
        if 'accum_bits_factor' in parsed_constraints['overwrite_dtypes']:
            parsed_constraints['use_extra_accum_dtype'] = True
            # print("[DOSA:ConstraintParsing:WARNING] Accumulator factor is set to 2, because larger is not recommended.")
        else:
            parsed_constraints['use_extra_accum_dtype'] = False
        if weights_dtype != data_dtype:
            print('NOT YET IMPLEMENTED: For now, weights and data must have the same data type. Stop.')
            exit(1)
        if 'numbers_already_scaled' in parsed_constraints['overwrite_dtypes']:
            dosa_singleton.config.quant.numbers_already_scaled = \
                bool(parsed_constraints['overwrite_dtypes']['numbers_already_scaled'])
    else:
        parsed_constraints['overwrite_imported_dtypes'] = False
        parsed_constraints['overwrite_fixed_point_dtypes'] = False
        parsed_constraints['use_extra_accum_dtype'] = False

    osg_allowlist = None
    if 'osg_allowlist' in parsed_constraints:
        osg_allowlist = parsed_constraints['osg_allowlist']

    return parsed_constraints, arch_gen_strategy, arch_target_devices, arch_fallback_hw, osg_allowlist

