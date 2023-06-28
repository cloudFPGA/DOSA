#!/bin/bash
#  *
#  *                       cloudFPGA
#  *    =============================================
#  *     Created: Apr 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        script to start compilation of dosa nodes in parallel in tmux
#  *
#  *

[ -z "$XILINX_VIVADO" ] && echo "ERROR: XILINX_VIVADO MUST BE DEFINED as environment variable! (This is usually done by the xilinx settings scripts)." && exit 1;

# to suppress sra-tool tips
export SraToolHintWasShown=1


echo "tmux is starting...."
echo "attach with \`tmux at\`"

tmux new-session -d -s dimidium
tmux send-keys -t dimidium:0 "cat cluster.json" C-m

