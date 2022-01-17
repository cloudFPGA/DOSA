//  *
//  *                       cloudFPGA
//  *     Copyright IBM Research, All Rights Reserved
//  *    =============================================
//  *     Created: Jan 2022
//  *     Authors: NGL
//  *
//  *     Description:
//  *        Template for a Haddoc2 wrapper
//  *

#ifndef _HADDOC_WRAPPER_H_
#define _HADDOC_WRAPPER_H_

#include <stdio.h>
#include <stdint.h>
#include "ap_int.h"
#include "ap_utils.h"
#include <hls_stream.h>

#include "axi_utils.hpp"
#include "interface_utils.hpp"

using namespace hls;

#define HADDOC_AVG_LAYER_LATENCY 3

#ifdef WRAPPER_TEST
#define DOSA_WRAPPER_INPUT_IF_BITWIDTH 64
#define DOSA_WRAPPER_OUTPUT_IF_BITWIDTH 64
#define DOSA_HADDOC_GENERAL_BITWIDTH 9
#define DOSA_HADDOC_INPUT_CHAN_NUM 3
#define DOSA_HADDOC_OUTPUT_CHAN_NUM 3
#define DOSA_HADDOC_INPUT_BITDIWDTH (DOSA_HADDOC_INPUT_CHAN_NUM*DOSA_HADDOC_GENERAL_BITWIDTH)
#define DOSA_HADDOC_OUTPUT_BITDIWDTH (DOSA_HADDOC_OUTPUT_CHAN_NUM*DOSA_HADDOC_GENERAL_BITWIDTH)
#define DOSA_HADDOC_INPUT_FRAME_WIDTH 10
#define DOSA_HADDOC_OUTPUT_FRAME_WIDTH 5
#define DOSA_HADDOC_OUTPUT_FLATTEN_PRESENT true
#define DOSA_HADDOC_LAYER_CNT 3
enum ToHaddocEnqStates {RESET = 0, FILL_BUF_0, FILL_BUF_1, FILL_BUF_2};
#else
//DOSA_ADD_INTERFACE_DEFINES
#endif

#define WRAPPER_INPUT_IF_BYTES ((DOSA_WRAPPER_INPUT_IF_BITWIDTH + 7)/ 8 )
#define WRAPPER_OUTPUT_IF_BYTES ((DOSA_WRAPPER_OUTPUT_IF_BITWIDTH + 7)/ 8 )
#define HADDOC_INPUT_FRAME_BIT_CNT (DOSA_HADDOC_INPUT_FRAME_WIDTH*DOSA_HADDOC_INPUT_FRAME_WIDTH*DOSA_HADDOC_GENERAL_BITWIDTH)
#define HADDOC_OUTPUT_FRAME_BIT_CNT (DOSA_HADDOC_OUTPUT_FRAME_WIDTH*DOSA_HADDOC_OUTPUT_FRAME_WIDTH*DOSA_HADDOC_GENERAL_BITWIDTH)
enum ToHaddocDeqStates {RESET = 0, FORWARD};


void haddoc_wrapper(
    // ----- Wrapper Interface -----
    stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >   &siData,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >  &soData,
    // ----- Haddoc Interface -----
    ap_uint<1>                                *po_haddoc_data_valid,
    ap_uint<1>                                *po_haddoc_frame_valid,
    ap_uint<DOSA_HADDOC_INPUT_BITDIWDTH> >    *po_haddoc_data_vector,
    ap_uint<1>                                *pi_haddoc_data_valid,
    ap_uint<1>                                *pi_haddoc_frame_valid,
    ap_uint<DOSA_HADDOC_OUTPUT_BITDIWDTH> >   *pi_haddoc_data_vector,
    // ----- DEBUG IO ------
    ap_uint<32> *debug_out
);

#endif

