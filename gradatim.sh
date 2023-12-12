#!/bin/bash

if [[ -z "$PYTHONPATH" ]]; then
  export PYTHONPATH=$(pwd)
fi
python3 ./gradatim/dosa.py $@

