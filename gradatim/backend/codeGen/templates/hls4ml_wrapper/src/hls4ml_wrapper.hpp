/*******************************************************************************
 * Copyright 2019 -- 2024 IBM Corporation
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
//  *     Created: Mar 2022
//  *     Authors: NGL
//  *
//  *     Description:
//  *        Template for a hls4ml wrapper
//  *

#ifndef _HLS4ML_WRAPPER_H_
#define _HLS4ML_WRAPPER_H_

#include <stdio.h>
#include <stdint.h>
#include "ap_int.h"
#include "ap_utils.h"
#include <hls_stream.h>

#include "../../lib/axi_utils.hpp"
#include "../../lib/interface_utils.hpp"

using namespace hls;

//generated defines
//DOSA_REMOVE_START
#ifdef WRAPPER_TEST
#define DOSA_WRAPPER_INPUT_IF_BITWIDTH 64
#define DOSA_WRAPPER_OUTPUT_IF_BITWIDTH 64
#define DOSA_HLS4ML_INPUT_BITWIDTH 8
#define DOSA_HLS4ML_OUTPUT_BITWIDTH 8
#define CNN_INPUT_FRAME_SIZE (10*10)
#define CNN_OUTPUT_FRAME_SIZE (5*5)
#endif
//LESSON LEARNED: ifdef/else for constants that affect INTERFACES is dangeorus with vivado HLS

//DOSA_REMOVE_STOP
//DOSA_ADD_INTERFACE_DEFINES

//derived defines
#define HLS4ML_INPUT_FRAME_BIT_CNT (CNN_INPUT_FRAME_SIZE*DOSA_HLS4ML_INPUT_BITWIDTH)
#define HLS4ML_OUTPUT_FRAME_BIT_CNT (CNN_OUTPUT_FRAME_SIZE*DOSA_HLS4ML_OUTPUT_BITWIDTH)
#define WRAPPER_INPUT_IF_BYTES ((DOSA_WRAPPER_INPUT_IF_BITWIDTH + 7)/ 8 )
#define WRAPPER_OUTPUT_IF_BYTES ((DOSA_WRAPPER_OUTPUT_IF_BITWIDTH + 7)/ 8 )

//as constants for HLS pragmas
const uint32_t cnn_input_frame_size = (CNN_INPUT_FRAME_SIZE);
const uint32_t cnn_output_frame_size = (CNN_OUTPUT_FRAME_SIZE);


//independent defines
enum twoStatesFSM {RESET = 0, FORWARD};



//DOSA_ADD_ip_name_BELOW
void hls4ml_wrapper_test(
    // ----- Wrapper Interface -----
    stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >   &siData,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >  &soData,
    // ----- hls4ml Interface -----
    stream<ap_uint<DOSA_HLS4ML_INPUT_BITWIDTH> >    &soToHls4mlData,
    stream<ap_uint<DOSA_HLS4ML_OUTPUT_BITWIDTH> >   &siFromHls4mlData,
    // ----- DEBUG IO ------
    ap_uint<32> *debug_out
);

#endif

