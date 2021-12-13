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

import os
import sys
import importlib

me_abs_dir = os.path.dirname(os.path.realpath(__file__))
# me_abs_file = os.path.abspath(os.path.realpath(__file__))
hls4ml_lib_path = os.path.abspath(me_abs_dir + '/../../../third_party_libs/hls4ml/')
sys.path.insert(0, hls4ml_lib_path)
# print(sys.path)

import dimidium.backend.third_party_libs.hls4ml.hls4ml
import dimidium.backend.third_party_libs.hls4ml.hls4ml.converters  # to initialize layer registration etc.
from dimidium.backend.third_party_libs.hls4ml.hls4ml.model.hls_model import HLSModel, HLSConfig
from dimidium.backend.third_party_libs.hls4ml.hls4ml.converters.keras_to_hls import get_supported_keras_layers
        # \, layer_handlers
from dimidium.backend.third_party_libs.hls4ml.hls4ml.converters.keras_to_hls import keras_to_hls, \
    get_supported_keras_layers, register_keras_layer_handler

layer_handlers = {}


def register_handlers():
    """based on https://github.com/fastmachinelearning/hls4ml/blob/master/hls4ml/converters/__init__.py"""
    #----------Layer handling register----------#
    # model_types = ['keras', 'pytorch', 'onnx']
    model_types = ['keras']
    hls4ml_converters_base_dir = hls4ml_lib_path + '/hls4ml/converters'
    hls4ml_converters_base_name = 'dimidium.backend.third_party_libs.hls4ml.hls4ml.converters'

    for model_type in model_types:
        # for module in os.listdir(os.path.dirname(__file__) + '/{}'.format(model_type)):
        for module in os.listdir(hls4ml_converters_base_dir + '/{}'.format(model_type)):
            if module == '__init__.py' or module[-3:] != '.py':
                continue
            try:
                # lib = importlib.import_module(__name__ + '.{}.'.format(model_type) + module[:-3])
                lib = importlib.import_module(hls4ml_converters_base_name + '.{}.'.format(model_type) + module[:-3])
                for name, func in list(lib.__dict__.items()):
                    # if 'func' is callable (i.e., function, class...)
                    # and has 'handles' attribute
                    # and is defined in this module (i.e., not imported)
                    if callable(func) and hasattr(func, 'handles') and func.__module__ == lib.__name__:
                        for layer in func.handles:

                            if model_type == 'keras':
                                register_keras_layer_handler(layer, func)
                                layer_handlers[layer] = func   # reproduce locally, to be sure
                            # elif model_type == 'pytorch':
                            #     register_pytorch_layer_handler(layer, func)
                            # elif model_type == 'onnx':
                            #     register_onnx_layer_handler(layer, func)

            except ImportError:
                continue


def dosa_to_hls(config, reader, model_arch):
    """based on: https://github.com/fastmachinelearning/hls4ml/blob/master/hls4ml/converters/keras_to_hls.py"""

    register_handlers()

    # This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []

    # print(model_arch)

    # Define layers to skip for conversion to HLS
    skip_layers = ['Dropout']
    # All supported layers
    # supported_layers = get_supported_keras_layers() + skip_layers
    supported_layers = list(layer_handlers.keys()) + skip_layers   # use local version, to be sure...

    # Map inputs of skipped and split (activation) layers
    inputs_map = {}

    # Loop through layers
    layer_counter = 0

    input_layers = None
    output_layers = None

    layer_config = None
    if model_arch['class_name'] == 'Sequential':
        print('Interpreting Sequential')
        layer_config = model_arch['config']
        if 'layers' in layer_config: # Newer Keras versions have 'layers' in 'config' key
            layer_config = layer_config['layers']
        # Sequential doesn't have InputLayer in TF < 2.3 (Keras 2.4.0)
        if layer_config[0]['class_name'] != 'InputLayer':
            input_layer = {}
            input_layer['name'] = 'input1'
            input_layer['class_name'] = 'InputLayer'
            input_layer['input_shape'] = layer_config[0]['config']['batch_input_shape'][1:]
            layer_list.append(input_layer)
            print('Input shape:', input_layer['input_shape'])
    elif model_arch['class_name'] in ['Model', 'Functional']: # TF >= 2.3 calls it 'Funcational' API
        print('Interpreting Model')
        layer_config = model_arch['config']['layers']
        input_layers = [ inp[0] for inp in model_arch['config']['input_layers'] ]
        output_layers = [ out[0] for out in model_arch['config']['output_layers'] ]

    # Get input shape and check for unsupported layer type
    for keras_layer in layer_config:
        if keras_layer['class_name'] not in supported_layers:
            raise Exception('ERROR: Unsupported layer type: {}'.format(keras_layer['class_name']))

    output_shapes = {}
    output_shape = None

    print('Topology:')
    for keras_layer in layer_config:
        if 'batch_input_shape' in keras_layer['config']:
            if 'inbound_nodes' in keras_layer and len(keras_layer['inbound_nodes']) > 0:
                input_shapes = [output_shapes[inbound_node[0]] for inbound_node in keras_layer['inbound_nodes'][0]]
            else:
                input_shapes = [keras_layer['config']['batch_input_shape']]
        else:
            if 'inbound_nodes' in keras_layer:
                input_shapes = [output_shapes[inbound_node[0]] for inbound_node in keras_layer['inbound_nodes'][0]]
            else:
                # Sequential model, so output_shape from the previous layer is still valid
                input_shapes = [output_shape]

        keras_class = keras_layer['class_name']

        if keras_class in skip_layers:
            if 'inbound_nodes' in keras_layer:
                name = keras_layer['config']['name']
                # Currently supported skipped layers have only one input
                parent_input = keras_layer['inbound_nodes'][0][0][0]
                # Skipped layers can follow each other (e.g., Dropout -> Flatten)
                inputs_map[name] = inputs_map.get(parent_input, parent_input)

            output_shapes[keras_layer['config']['name']] = input_shapes[0]

            continue

        if keras_class in supported_layers:
            layer_counter = layer_counter + 1

        #Extract inbound nodes
        if 'inbound_nodes' in keras_layer and len(keras_layer['inbound_nodes']) > 0:
            input_names = [ inputs_map.get(inp[0], inp[0]) for inp in keras_layer['inbound_nodes'][0] ]
        else:
            input_names = None

        layer, output_shape = layer_handlers[keras_class](keras_layer, input_names, input_shapes, reader, config)

        print('Layer name: {}, layer type: {}, input shapes: {}, output shape: {}'.format(layer['name'], layer['class_name'], input_shapes, output_shape))
        if keras_layer['class_name'] != 'InputLayer':
            layer['strategy'] = config['HLSConfig']['Model']['Strategy']
            # layer['ReuseFactor'] = keras_layer['ReuseFactor']
            layer['mult_limit'] = keras_layer['mult_limit']
            layer['loop_lim_outermost'] = keras_layer['loop_lim_outermost']
            layer['loop_lim_outer'] = keras_layer['loop_lim_outer']
            layer['loop_lim_inner'] = keras_layer['loop_lim_inner']
            layer['loop_lim_innermost'] = keras_layer['loop_lim_innermost']
        layer_list.append(layer)
        if 'activation' in layer and layer['class_name'] not in ['Activation', 'LeakyReLU', 'ThresholdedReLU', 'ELU', 'PReLU', 'Softmax', 'TernaryTanh']:# + qkeras_layers:
            act_layer = {}
            act_layer['name'] = layer['name'] + '_' + layer['activation']
            act_layer['activation'] = layer['activation']
            if 'activ_param' in layer:
                act_layer['activ_param'] = layer['activ_param']
                act_layer['class_name'] = layer['activation']
            elif layer['activation'] == 'softmax':
                act_layer['class_name'] = 'Softmax'
                act_layer['axis'] = -1
            else:
                act_layer['class_name'] = 'Activation'
            inputs_map[layer['name']] = act_layer['name']
            if output_layers is not None and layer['name'] in output_layers:
                output_layers = [act_layer['name'] if name == layer['name'] else name for name in output_layers]
            layer_list.append(act_layer)

        assert(output_shape is not None)

        output_shapes[layer['name']] = output_shape

    #################
    ## Generate HLS
    #################

    print('Creating HLS model')
    hls_model = HLSModel(config, reader, layer_list, input_layers, output_layers)
    return hls_model

