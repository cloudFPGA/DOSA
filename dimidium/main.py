#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Jun 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Main file for DOSA/dimidium flow
#  *
#  *

import sys
import json
import onnx
import numpy as np
import tvm
import tvm.relay as relay

from dimidium.lib.archGen import arch_gen
import dimidium.lib.plot_roofline as plot_roofline

__mandatory_keys__ = ['shape_dict']


def onnx_import(onnx_path, shape_dict, debug=False):
    onnx_model = onnx.load(onnx_path)
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
    if debug:
        print(mod.astext(show_meta_data=False))
    return mod, params


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("USAGE: {} ./path/to/nn.onnx ./path/to/constraint.json".format(sys.argv[0]))
        exit(1)

    onnx_path = sys.argv[1]
    const_path = sys.argv[2]

    with open(const_path, 'r') as inp:
        user_constraints = json.load(inp)

    for k in __mandatory_keys__:
        if k not in user_constraints:
            print("ERROR: Mandatory key {} is missing in the constraints file {}. stop.".format(k, const_path))
            exit(1)

    print("DOSA: Importing ONNX...")
    mod, params = onnx_import(onnx_path, user_constraints['shape_dict'])
    print("\t...done.\n")

    print("DOSA: Generating high-level architecture...")
    archDict = arch_gen(mod, params, debug=True)
    print("\t...done.\n")

    print("DOSA: Generating and showing roofline...")
    target_fps = user_constraints['target_fps']
    used_batch = user_constraints['used_batch_n']
    used_name = user_constraints['name']
    plt = plot_roofline.generate_roofline_plt(archDict['dpl'], target_fps, used_batch, used_name,
                                              show_splits=True, show_labels=True)
    plt2 = plot_roofline.generate_roofline_plt(archDict['fused_view'], target_fps, used_batch,
                                               used_name + " (optimized)",
                                               show_splits=True, show_labels=True)
    # plot_roofline.show_roofline_plt(plt, blocking=False) not necessary...
    plot_roofline.show_roofline_plt(plt2)
    print("\t...done.\n")

    print("\nDOSA finished successfully.\n")
