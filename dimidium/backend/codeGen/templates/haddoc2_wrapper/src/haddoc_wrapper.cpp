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

#include <stdio.h>
#include <stdint.h>
#include "ap_int.h"
#include "ap_utils.h"
#include <hls_stream.h>

#include "haddoc_wrapper.hpp"

using namespace hls;


void pToHaddocEnq(
    stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >   &siData,
#ifdef WRAPPER_TEST
  stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> >    &sBuffer_chan1,
  stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> >    &sBuffer_chan2,
  stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> >    &sBuffer_chan3
#else
    //DOSA_ADD_enq_declaration
#endif
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static ToHaddocEnqStates enqueueFSM = RESET;
#pragma HLS reset variable=enqueueFSM
  static ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH> overflow_data;
#pragma HLS reset variable=overflow_data
  static ap_uint<64> current_byte_cnt;
#pragma HLS reset variable=current_byte_cnt;
  //-- LOCAL VARIABLES ------------------------------------------------------

  switch(enqueueFSM)
  {
    default:
    case RESET:
      //TODO: necessary?
      overflow_data = 0x0;
      current_byte_cnt = 0x0;
      enqueueFSM = FILL_BUF_0;
      break;
#ifdef WRAPPER_TEST
    case FILL_BUF_0:
      if( !siData.empty() && !sBuffer_chan1.full() )
      {
        //TODO: consider overflow_data
        //read axis, determine byte count, fill in buffer,
        //write to overflow data
      }
      break;
#else
      //DOSA_ADD_enq_fsm
#endif;
  }


}



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
    )
{
  //-- DIRECTIVES FOR THE BLOCK ---------------------------------------------
#pragma HLS INTERFACE ap_ctrl_none port=return

#pragma HLS INTERFACE ap_fifo port=siData
#pragma HLS INTERFACE ap_fifo port=soData

#pragma HLS INTERFACE ap_ovld register port=po_haddoc_data_valid
#pragma HLS INTERFACE ap_ovld register port=po_haddoc_frame_valid
#pragma HLS INTERFACE ap_ovld register port=po_haddoc_data_vector
#pragma HLS INTERFACE ap_vld register port=pi_haddoc_data_valid
#pragma HLS INTERFACE ap_vld register port=pi_haddoc_frame_valid
#pragma HLS INTERFACE ap_vld register port=pi_haddoc_data_vector
#pragma HLS INTERFACE ap_ovld register port=debug_out


  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS DATAFLOW

  //-- STATIC VARIABLES (with RESET) ------------------------------------------

  //-- STATIC DATAFLOW VARIABLES ------------------------------------------
  const int fifo_depth = (DOSA_HADDOC_INPUT_FRAME_WIDTH*DOSA_HADDOC_INPUT_FRAME_WIDTH);
#ifdef WRAPPER_TEST
  static stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> > sBuffer_chan1 ("sBuffer_chan1");
  #pragma HLS STREAM variable=sBuffer_chan1   depth=fifo_depth
  static stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> > sBuffer_chan2 ("sBuffer_chan2");
  #pragma HLS STREAM variable=sBuffer_chan2   depth=fifo_depth
  static stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> > sBuffer_chan3 ("sBuffer_chan3");
  #pragma HLS STREAM variable=sBuffer_chan3   depth=fifo_depth
#else
  //DOSA_ADD_STREAM_INSTANTIATION
#endif
  static stream<NetworkWord>       sBuffer_Data   ("sBuffer_Data");
  static stream<NetworkMetaStream> sBuffer_Meta   ("sBuffer_Meta");
  static stream<NodeId>            sDstNode_sig   ("sDstNode_sig");

#pragma HLS STREAM variable=sBuffer_Data     depth=252
#pragma HLS STREAM variable=sBuffer_Meta     depth=32
#pragma HLS STREAM variable=sDstNode_sig     depth=1


}



