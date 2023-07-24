/*******************************************************************************
 * Copyright 2019 -- 2023 IBM Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*******************************************************************************/


//  *
//  *                       cloudFPGA
//  *    =============================================
//  *     Created: Oct 2022
//  *     Authors: NGL
//  *
//  *     Description:
//  *        Template for a hls4ml wrapper using
//  *        the parallel IO interface
//  *

#ifndef _HLS4ML_PARALLEL_WRAPPER_H_
#define _HLS4ML_PARALLEL_WRAPPER_H_

#include <stdio.h>
#include <stdint.h>
#include "ap_int.h"
#include "ap_utils.h"
#include <hls_stream.h>

#include "../../lib/axi_utils.hpp"
#include "../../lib/interface_utils.hpp"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

using namespace hls;


//generated defines
//DOSA_REMOVE_START
#ifdef WRAPPER_TEST
#define DOSA_WRAPPER_INPUT_IF_BITWIDTH 64
#define DOSA_WRAPPER_OUTPUT_IF_BITWIDTH 64
#define DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH 8
#define DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP 1
#define DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP_WIDTH 1
#define DOSA_HLS4ML_PARALLEL_FROMACCEL_BITWIDTH 8
#define DOSA_HLS4ML_PARALLEL_FROMACCEL_BITWIDTH_TKEEP 1
#define DOSA_HLS4ML_PARALLEL_FROMACCEL_BITWIDTH_TKEEP_WIDTH 1
#define DOSA_HLS4ML_PARALLEL_INPUT_CHAN_NUM 3
#define DOSA_HLS4ML_PARALLEL_OUTPUT_CHAN_NUM 4
#define DOSA_HLS4ML_PARALLEL_INPUT_FRAME_WIDTH 10
#define CNN_INPUT_FRAME_SIZE (DOSA_HLS4ML_PARALLEL_INPUT_FRAME_WIDTH*DOSA_HLS4ML_PARALLEL_INPUT_FRAME_WIDTH)
#define DOSA_HLS4ML_PARALLEL_OUTPUT_FRAME_WIDTH 5
#define CNN_OUTPUT_FRAME_SIZE (DOSA_HLS4ML_PARALLEL_OUTPUT_FRAME_WIDTH*DOSA_HLS4ML_PARALLEL_OUTPUT_FRAME_WIDTH)
#define DOSA_HLS4ML_PARALLEL_OUTPUT_BATCH_FLATTEN true
#define DOSA_HLS4ML_PARALLEL_LAYER_CNT 3
#define DOSA_HLS4ML_PARALLEL_VALID_WAIT_CNT 25
//#define DOSA_HLS4ML_PARALLEL_INPUT_FRAME_LINE_CNT 13 //100+7/8
enum Tohls4ml_parallelEnqStates {RESET0 = 0, WAIT_DRAIN, FILL_BUF_0, FILL_BUF_1, FILL_BUF_2};
enum Fromhls4ml_parallelDeqStates {RESET1 = 0, READ_BUF_0, READ_BUF_1, READ_BUF_2, READ_BUF_3};
#endif
//LESSON LEARNED: ifdef/else for constants that affect INTERFACES does not work with vivado HLS...it uses the first occurence, apparently...

//DOSA_REMOVE_STOP
//DOSA_ADD_INTERFACE_DEFINES

//derived defines
#define DOSA_HLS4ML_PARALLEL_INPUT_BITDIWDTH (DOSA_HLS4ML_PARALLEL_INPUT_CHAN_NUM*DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH*CNN_INPUT_FRAME_SIZE)
#define DOSA_HLS4ML_PARALLEL_OUTPUT_BITDIWDTH (DOSA_HLS4ML_PARALLEL_OUTPUT_CHAN_NUM*DOSA_HLS4ML_PARALLEL_FROMACCEL_BITWIDTH*CNN_OUTPUT_FRAME_SIZE)
#define WRAPPER_INPUT_IF_BYTES ((DOSA_WRAPPER_INPUT_IF_BITWIDTH + 7)/ 8 )
#define WRAPPER_OUTPUT_IF_BYTES ((DOSA_WRAPPER_OUTPUT_IF_BITWIDTH + 7)/ 8 )
#define HLS4ML_PARALLEL_INPUT_FRAME_BIT_CNT (CNN_INPUT_FRAME_SIZE*DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH)
#define HLS4ML_PARALLEL_OUTPUT_FRAME_BIT_CNT (CNN_OUTPUT_FRAME_SIZE*DOSA_HLS4ML_PARALLEL_FROMACCEL_BITWIDTH)
#define HLS4ML_PARALLEL_INPUT_FRAME_BYTE_CNT ((HLS4ML_PARALLEL_INPUT_FRAME_BIT_CNT+7)/8)
#define HLS4ML_PARALLEL_OUTPUT_FRAME_BYTE_CNT ((HLS4ML_PARALLEL_OUTPUT_FRAME_BIT_CNT+7)/8)
#define WRAPPER_OUTPUT_IF_HLS4ML_PARALLEL_WORDS_CNT_CEIL ((DOSA_WRAPPER_OUTPUT_IF_BITWIDTH + DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH - 1)/ DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH)
#define HLS4ML_PARALLEL_AVG_LAYER_LATENCY (DOSA_HLS4ML_PARALLEL_VALID_WAIT_CNT)
//#define WRAPPTER_OUTPUT_TKEEP_PER_WORDS ( bitCntToTKeep(ap_uint<(DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+7)/8> DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH) )

//as constants for HLS pragmas
//const uint32_t cnn_input_frame_size = (CNN_INPUT_FRAME_SIZE);
//const uint32_t cnn_output_frame_size = (CNN_OUTPUT_FRAME_SIZE);
const uint32_t cnn_input_frame_size = (MAX(CNN_INPUT_FRAME_SIZE, 32)); //since we use it as stream depth
const uint32_t cnn_output_frame_size = (MAX(CNN_OUTPUT_FRAME_SIZE, 32)); //since we use it as tream depth
const uint32_t wrapper_output_if_hls4ml_parallel_words_cnt_ceil = (WRAPPER_OUTPUT_IF_HLS4ML_PARALLEL_WORDS_CNT_CEIL);


//independent defines
enum twoStatesFSM {RESET = 0, FORWARD};
enum threeStatesFSM {RESET3 = 0, FORWARD3, BACKLOG3};
//enum Fromhls4ml_parallelDeqStates {RESET1 = 0, WAIT_FRAME, READ_FRAME};
enum Fromhls4ml_parallelEnqStates {RESET2 = 0, CNT_UNTIL_VAILD, FORWARD2};



//DOSA_ADD_ip_name_BELOW
void hls4ml_parallel_wrapper_test(
    // ----- Wrapper Interface -----
    stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >   &siData,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >  &soData,
    // ----- hls4ml_parallel Interface -----
    stream<ap_uint<DOSA_HLS4ML_PARALLEL_INPUT_BITDIWDTH> > &po_hls4ml_parallel_data,
    stream<ap_uint<DOSA_HLS4ML_PARALLEL_OUTPUT_BITDIWDTH> > &pi_hls4ml_parallel_data,
    // ----- DEBUG IO ------
    ap_uint<64> *debug_out
    );

#endif

