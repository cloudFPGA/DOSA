#!/bin/bash

# USAGE: ./gradatim/dosa.py ./path/to/dosa_config.json ./path/to/nn.onnx ./path/to/constraint.json ./path/to/build_dir [--no-roofline|--no-build|--only-stats|--only-coverage]
python3 ./gradatim/dosa.py $@

