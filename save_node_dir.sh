#!/bin/bash

# usage
if [[ $# -ne 2 ]]; then
  echo "USAGE:"
  echo "$0 <path/to/dir/to/save> <path/to/save/dir>"
  exit 1
fi

node_dir=$(basename $1)
build_dir_name=$(basename $(cd $1; cd .. ; pwd))
tstmp=$(date +"%Y-%m-%d_%H%M")
target_dir="$2/$build_dir_name/$node_dir/$tstmp/"

mkdir -p $target_dir/ROLE/
mkdir -p $target_dir/dcps/
cp -R $1/ROLE/* $target_dir/ROLE/
cp -R $1/dcps/* $target_dir/dcps/

