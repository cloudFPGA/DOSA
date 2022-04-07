//  *
//  *                       cloudFPGA
//  *     Copyright IBM Research, All Rights Reserved
//  *    =============================================
//  *     Created: Apr 2022
//  *     Authors: NGL
//  *
//  *     Description:
//  *        Template for a Tensor processor without
//  *          Interlocked Pipeline Stages (TIPS) engine
//  *

#ifndef _TIPS_ENGINE_H_
#define _TIPS_ENGINE_H_

#include <stdio.h>
#include <stdint.h>
#include "ap_int.h"
#include "ap_utils.h"
#include <hls_stream.h>

#include "../../lib/axi_utils.hpp"
//#include "../../lib/interface_utils.hpp"

using namespace hls;

//generated defines
//DOSA_REMOVE_START
#ifdef TIPS_TEST
#define DOSA_WRAPPER_INPUT_IF_BITWIDTH 64
#define DOSA_WRAPPER_OUTPUT_IF_BITWIDTH 64
#endif

//DOSA_REMOVE_STOP
//DOSA_ADD_INTERFACE_DEFINES

//derived defines

//as constants for HLS pragmas
const uint32_t cnn_input_frame_size = (CNN_INPUT_FRAME_SIZE);
const uint32_t cnn_output_frame_size = (CNN_OUTPUT_FRAME_SIZE);


//independent defines
enum twoStatesFSM {RESET = 0, FORWARD};



//DOSA_ADD_ip_name_BELOW
void tips_test(
    // ----- DOSA Interface -----
    stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >   &siData,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >  &soData,
    // ----- add potential DRAM Interface -----
    // ----- DEBUG out ------
    ap_uint<64> *debug_out
);

#endif

