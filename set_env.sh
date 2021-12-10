#!/bin/bash

if [[ $0 == $BASH_SOURCE ]]; then
  echo "THIS script should be called with 'source /path/to/script.sh'"
  #exit 1
fi

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
# echo $DIR

export TVM_LOG_DEBUG=0

source $DIR/venv/bin/activate


