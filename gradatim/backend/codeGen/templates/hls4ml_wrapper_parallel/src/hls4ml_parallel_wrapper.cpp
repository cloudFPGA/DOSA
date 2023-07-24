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

#include <stdio.h>
#include <stdint.h>
#include "ap_int.h"
#include "ap_utils.h"
#include <hls_stream.h>
#include <cassert>

#include "hls4ml_parallel_wrapper.hpp"

using namespace hls;



void pTohls4ml_parallelDemux(
#ifdef WRAPPER_TEST
  stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >    &sTohls4ml_parallelBuffer_chan1,
  stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >    &sTohls4ml_parallelBuffer_chan2,
  stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >    &sTohls4ml_parallelBuffer_chan3,
#else
    //DOSA_ADD_tohls4ml_parallel_buffer_param_decl
#endif
    stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >   &siData,
    uint16_t *debug
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static Tohls4ml_parallelEnqStates enqueueFSM = RESET0;
#pragma HLS reset variable=enqueueFSM
  static uint32_t current_frame_byte_cnt = 0x0;
#pragma HLS reset variable=current_frame_byte_cnt
  static uint32_t wait_drain_cnt = cnn_input_frame_size;
#pragma HLS reset variable=wait_drain_cnt
  static bool hangover_present = false;
#pragma HLS reset variable=hangover_present
  static uint32_t new_bytes_cnt = 0x0;
#pragma HLS reset variable=new_bytes_cnt

  //-- STATIC VARIABLES ------------------------------------------------------
  static Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> hangover_axis;
  //-- LOCAL VARIABLES ------------------------------------------------------
  Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> tmp_read_0;

  switch(enqueueFSM)
  {
    default:
    case RESET0:
      //TODO: necessary?
      current_frame_byte_cnt = 0x0;
      wait_drain_cnt = cnn_input_frame_size;
      hangover_present = false;
      new_bytes_cnt = 0x0;
#ifndef __SYNTHESIS_
      wait_drain_cnt = 10;
#endif
      enqueueFSM = WAIT_DRAIN;
      break;

    case WAIT_DRAIN:
      printf("wait_drain_cnt = %d\n", wait_drain_cnt);
      if(wait_drain_cnt == 0)
      {
        enqueueFSM = FILL_BUF_0;
      } else {
        wait_drain_cnt--;
      }
      break;

#ifdef WRAPPER_TEST
    case FILL_BUF_0:
    //we distribute on the channels only, cutting in right bitsize in dequeue process
      if( (!siData.empty() || hangover_present) && !sTohls4ml_parallelBuffer_chan1.full() ) //&& !sTohls4ml_parallelBuffer_chan2.full() )
      {
        if(hangover_present)
        {
          tmp_read_0 = hangover_axis;
        } else {
          tmp_read_0 = siData.read();
          new_bytes_cnt = extractByteCnt(tmp_read_0);
        }
        hangover_present = false;
        if((current_frame_byte_cnt + new_bytes_cnt) >= HLS4ML_PARALLEL_INPUT_FRAME_BYTE_CNT)
        {
          uint32_t bytes_to_this_frame = HLS4ML_PARALLEL_INPUT_FRAME_BYTE_CNT - current_frame_byte_cnt;
          int32_t bytes_to_next_frame = new_bytes_cnt - bytes_to_this_frame;
          ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH> cur_input = tmp_read_0.getTData();
          ap_uint<(DOSA_WRAPPER_INPUT_IF_BITWIDTH+7)/8> cur_tkeep = tmp_read_0.getTKeep();
          ap_uint<(DOSA_WRAPPER_INPUT_IF_BITWIDTH+7)/8> this_tkeep = 0x0;
          ap_uint<(DOSA_WRAPPER_INPUT_IF_BITWIDTH+7)/8> next_tkeep = 0x0;
          for(uint32_t i = 0; i < WRAPPER_INPUT_IF_BYTES; i++)
          {
            ap_uint<1> cur_tkeep_bit = (ap_uint<1>) (cur_tkeep >> i);
            if(i < bytes_to_this_frame)
            {
              this_tkeep |= ((ap_uint<(DOSA_WRAPPER_INPUT_IF_BITWIDTH+7)/8>) cur_tkeep_bit) << i;
            } else {
              next_tkeep |= ((ap_uint<(DOSA_WRAPPER_INPUT_IF_BITWIDTH+7)/8>) cur_tkeep_bit) << (i - bytes_to_this_frame);
            }
          }
          Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> tmp_write_this =  Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH>(cur_input, this_tkeep, 0);
          Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> tmp_write_next =  Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH>(cur_input >> (bytes_to_this_frame*8), next_tkeep, 0);
          sTohls4ml_parallelBuffer_chan1.write(tmp_write_this);
          if(bytes_to_next_frame > 0)
          {
            //sTohls4ml_parallelBuffer_chan2.write(tmp_write_next);
            hangover_present = true;
            hangover_axis = tmp_write_next;
            new_bytes_cnt = bytes_to_next_frame;
          }
          //current_frame_byte_cnt = bytes_to_next_frame;
          current_frame_byte_cnt = 0x0;
          enqueueFSM = FILL_BUF_1;
        } else {
          current_frame_byte_cnt += new_bytes_cnt;
          tmp_read_0.setTLast(0); //necessary? would be ignored anyhow...
          sTohls4ml_parallelBuffer_chan1.write(tmp_read_0);
          //stay here
        }
      }
      break;
    case FILL_BUF_1:
      if( (!siData.empty() || hangover_present) && !sTohls4ml_parallelBuffer_chan2.full() ) //&& !sTohls4ml_parallelBuffer_chan3.full() )
      {
        if(hangover_present)
        {
          tmp_read_0 = hangover_axis;
        } else {
          tmp_read_0 = siData.read();
          new_bytes_cnt = extractByteCnt(tmp_read_0);
        }
        hangover_present = false;
        if((current_frame_byte_cnt + new_bytes_cnt) >= HLS4ML_PARALLEL_INPUT_FRAME_BYTE_CNT)
        {
          uint32_t bytes_to_this_frame = HLS4ML_PARALLEL_INPUT_FRAME_BYTE_CNT - current_frame_byte_cnt;
          int32_t bytes_to_next_frame = new_bytes_cnt - bytes_to_this_frame;
          ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH> cur_input = tmp_read_0.getTData();
          ap_uint<(DOSA_WRAPPER_INPUT_IF_BITWIDTH+7)/8> cur_tkeep = tmp_read_0.getTKeep();
          ap_uint<(DOSA_WRAPPER_INPUT_IF_BITWIDTH+7)/8> this_tkeep = 0x0;
          ap_uint<(DOSA_WRAPPER_INPUT_IF_BITWIDTH+7)/8> next_tkeep = 0x0;
          for(uint32_t i = 0; i < WRAPPER_INPUT_IF_BYTES; i++)
          {
            ap_uint<1> cur_tkeep_bit = (ap_uint<1>) (cur_tkeep >> i);
            if(i < bytes_to_this_frame)
            {
              this_tkeep |= ((ap_uint<(DOSA_WRAPPER_INPUT_IF_BITWIDTH+7)/8>) cur_tkeep_bit) << i;
            } else {
              next_tkeep |= ((ap_uint<(DOSA_WRAPPER_INPUT_IF_BITWIDTH+7)/8>) cur_tkeep_bit) << (i - bytes_to_this_frame);
            }
          }
          Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> tmp_write_this =  Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH>(cur_input, this_tkeep, 0);
          Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> tmp_write_next =  Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH>(cur_input >> (bytes_to_this_frame*8), next_tkeep, 0);
          sTohls4ml_parallelBuffer_chan2.write(tmp_write_this);
          if(bytes_to_next_frame > 0)
          {
            //sTohls4ml_parallelBuffer_chan3.write(tmp_write_next);
            hangover_present = true;
            hangover_axis = tmp_write_next;
            new_bytes_cnt = bytes_to_next_frame;
          }
          //current_frame_byte_cnt = bytes_to_next_frame;
          current_frame_byte_cnt = 0x0;
          enqueueFSM = FILL_BUF_2;
        } else {
          current_frame_byte_cnt += new_bytes_cnt;
          tmp_read_0.setTLast(0); //necessary? would be ignored anyhow...
          sTohls4ml_parallelBuffer_chan2.write(tmp_read_0);
          //stay here
        }
      }
      break;
    case FILL_BUF_2:
      if( (!siData.empty() || hangover_present) && !sTohls4ml_parallelBuffer_chan3.full() ) //&& !sTohls4ml_parallelBuffer_chan1.full() )
      {
        if(hangover_present)
        {
          tmp_read_0 = hangover_axis;
        } else {
          tmp_read_0 = siData.read();
          new_bytes_cnt = extractByteCnt(tmp_read_0);
        }
        hangover_present = false;
        if((current_frame_byte_cnt + new_bytes_cnt) >= HLS4ML_PARALLEL_INPUT_FRAME_BYTE_CNT)
        {
          uint32_t bytes_to_this_frame = HLS4ML_PARALLEL_INPUT_FRAME_BYTE_CNT - current_frame_byte_cnt;
          int32_t bytes_to_next_frame = new_bytes_cnt - bytes_to_this_frame;
          ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH> cur_input = tmp_read_0.getTData();
          ap_uint<(DOSA_WRAPPER_INPUT_IF_BITWIDTH+7)/8> cur_tkeep = tmp_read_0.getTKeep();
          ap_uint<(DOSA_WRAPPER_INPUT_IF_BITWIDTH+7)/8> this_tkeep = 0x0;
          ap_uint<(DOSA_WRAPPER_INPUT_IF_BITWIDTH+7)/8> next_tkeep = 0x0;
          for(uint32_t i = 0; i < WRAPPER_INPUT_IF_BYTES; i++)
          {
            ap_uint<1> cur_tkeep_bit = (ap_uint<1>) (cur_tkeep >> i);
            if(i < bytes_to_this_frame)
            {
              this_tkeep |= ((ap_uint<(DOSA_WRAPPER_INPUT_IF_BITWIDTH+7)/8>) cur_tkeep_bit) << i;
            } else {
              next_tkeep |= ((ap_uint<(DOSA_WRAPPER_INPUT_IF_BITWIDTH+7)/8>) cur_tkeep_bit) << (i - bytes_to_this_frame);
            }
          }
          Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> tmp_write_this =  Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH>(cur_input, this_tkeep, 0);
          Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> tmp_write_next =  Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH>(cur_input >> (bytes_to_this_frame*8), next_tkeep, 0);
          sTohls4ml_parallelBuffer_chan3.write(tmp_write_this);
          if(bytes_to_next_frame > 0)
          {
            //sTohls4ml_parallelBuffer_chan1.write(tmp_write_next);
            hangover_present = true;
            hangover_axis = tmp_write_next;
            new_bytes_cnt = bytes_to_next_frame;
          }
          //current_frame_byte_cnt = bytes_to_next_frame;
          current_frame_byte_cnt = 0x0;
          //last channel --> go to start
          enqueueFSM = FILL_BUF_0;
        } else {
          current_frame_byte_cnt += new_bytes_cnt;
          tmp_read_0.setTLast(0); //necessary? would be ignored anyhow...
          sTohls4ml_parallelBuffer_chan3.write(tmp_read_0);
          //stay here
        }
      }
      break;
#else
      //DOSA_ADD_demux_fsm
#endif
  }

  *debug = (uint8_t) enqueueFSM;
  *debug |= ((uint16_t) current_frame_byte_cnt) << 8;

}


#ifdef WRAPPER_TEST
void pTohls4ml_parallelNarrow_1(
  stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >    &sTohls4ml_parallelBuffer_chanX,
  stream<ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> >   &sTohls4ml_parallelPixelChain_chanX
  )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static threeStatesFSM narrowFSM = RESET3;
#pragma HLS reset variable=narrowFSM
  static ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH> cur_read;
#pragma HLS reset variable=cur_read
  static ap_uint<(DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+7)/8> cur_tkeep;
#pragma HLS reset variable=cur_tkeep
  //-- LOCAL VARIABLES ------------------------------------------------------
  bool not_empty = false;

  switch(narrowFSM)
  {
    default:
    case RESET3:
      cur_read = 0x0;
      cur_tkeep = 0x0;
      if(!sTohls4ml_parallelBuffer_chanX.empty())
      {
        sTohls4ml_parallelBuffer_chanX.read();
        not_empty = true;
      }
      if(!not_empty)
      {
        narrowFSM = FORWARD3;
      }
      break;

    case FORWARD3:
      if(!sTohls4ml_parallelBuffer_chanX.empty() && !sTohls4ml_parallelPixelChain_chanX.full())
      {
        Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> tmp_read_0 = sTohls4ml_parallelBuffer_chanX.read();
        cur_read = tmp_read_0.getTData();
        cur_tkeep = tmp_read_0.getTKeep();
        if(cur_tkeep > 0)
        {
          ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP_WIDTH> cur_tkeep_bit = (ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP_WIDTH>) cur_tkeep;
          ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> cur_pixel = (ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH>) cur_read;
          if(cur_tkeep_bit > 0)
          {
            //if not --> process in next state, additional delay not dramatic since all pixels are buffered in parallel
            sTohls4ml_parallelPixelChain_chanX.write(cur_pixel);
          }
          cur_read >>= DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH;
          cur_tkeep >>= DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP_WIDTH;
          if(cur_tkeep > 0)
          {
            narrowFSM = BACKLOG3;
          }
        }
      }
      break;

    case BACKLOG3:
      if(!sTohls4ml_parallelPixelChain_chanX.full())
      {
        ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP_WIDTH> cur_tkeep_bit = (ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP_WIDTH>) cur_tkeep;
        ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> cur_pixel = (ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH>) cur_read;
        if(cur_tkeep_bit > 0)
        {
          //if not --> process in next state, additional delay not dramatic since all pixels are buffered in parallel
          sTohls4ml_parallelPixelChain_chanX.write(cur_pixel);
        }
        cur_read >>= DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH;
        cur_tkeep >>= DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP_WIDTH;
        if(cur_tkeep == 0)
        {
          narrowFSM = FORWARD3;
        }
      }
      break;
  }
}

void pTohls4ml_parallelNarrow_2(
  stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >    &sTohls4ml_parallelBuffer_chanX,
  stream<ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> >   &sTohls4ml_parallelPixelChain_chanX
  )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static threeStatesFSM narrowFSM = RESET3;
#pragma HLS reset variable=narrowFSM
  static ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH> cur_read;
#pragma HLS reset variable=cur_read
  static ap_uint<(DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+7)/8> cur_tkeep;
#pragma HLS reset variable=cur_tkeep
  //-- LOCAL VARIABLES ------------------------------------------------------
  bool not_empty = false;

  switch(narrowFSM)
  {
    default:
    case RESET3:
      cur_read = 0x0;
      cur_tkeep = 0x0;
      if(!sTohls4ml_parallelBuffer_chanX.empty())
      {
        sTohls4ml_parallelBuffer_chanX.read();
        not_empty = true;
      }
      if(!not_empty)
      {
        narrowFSM = FORWARD3;
      }
      break;

    case FORWARD3:
      if(!sTohls4ml_parallelBuffer_chanX.empty() && !sTohls4ml_parallelPixelChain_chanX.full())
      {
        Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> tmp_read_0 = sTohls4ml_parallelBuffer_chanX.read();
        cur_read = tmp_read_0.getTData();
        cur_tkeep = tmp_read_0.getTKeep();
        if(cur_tkeep > 0)
        {
          ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP_WIDTH> cur_tkeep_bit = (ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP_WIDTH>) cur_tkeep;
          ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> cur_pixel = (ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH>) cur_read;
          if(cur_tkeep_bit > 0)
          {
            //if not --> process in next state, additional delay not dramatic since all pixels are buffered in parallel
            sTohls4ml_parallelPixelChain_chanX.write(cur_pixel);
          }
          cur_read >>= DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH;
          cur_tkeep >>= DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP_WIDTH;
          if(cur_tkeep > 0)
          {
            narrowFSM = BACKLOG3;
          }
        }
      }
      break;

    case BACKLOG3:
      if(!sTohls4ml_parallelPixelChain_chanX.full())
      {
        ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP_WIDTH> cur_tkeep_bit = (ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP_WIDTH>) cur_tkeep;
        ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> cur_pixel = (ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH>) cur_read;
        if(cur_tkeep_bit > 0)
        {
          //if not --> process in next state, additional delay not dramatic since all pixels are buffered in parallel
          sTohls4ml_parallelPixelChain_chanX.write(cur_pixel);
        }
        cur_read >>= DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH;
        cur_tkeep >>= DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP_WIDTH;
        if(cur_tkeep == 0)
        {
          narrowFSM = FORWARD3;
        }
      }
      break;
  }
}

void pTohls4ml_parallelNarrow_3(
  stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >    &sTohls4ml_parallelBuffer_chanX,
  stream<ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> >   &sTohls4ml_parallelPixelChain_chanX
  )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static threeStatesFSM narrowFSM = RESET3;
#pragma HLS reset variable=narrowFSM
  static ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH> cur_read;
#pragma HLS reset variable=cur_read
  static ap_uint<(DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+7)/8> cur_tkeep;
#pragma HLS reset variable=cur_tkeep
  //-- LOCAL VARIABLES ------------------------------------------------------
  bool not_empty = false;

  switch(narrowFSM)
  {
    default:
    case RESET3:
      cur_read = 0x0;
      cur_tkeep = 0x0;
      if(!sTohls4ml_parallelBuffer_chanX.empty())
      {
        sTohls4ml_parallelBuffer_chanX.read();
        not_empty = true;
      }
      if(!not_empty)
      {
        narrowFSM = FORWARD3;
      }
      break;

    case FORWARD3:
      if(!sTohls4ml_parallelBuffer_chanX.empty() && !sTohls4ml_parallelPixelChain_chanX.full())
      {
        Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> tmp_read_0 = sTohls4ml_parallelBuffer_chanX.read();
        cur_read = tmp_read_0.getTData();
        cur_tkeep = tmp_read_0.getTKeep();
        if(cur_tkeep > 0)
        {
          ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP_WIDTH> cur_tkeep_bit = (ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP_WIDTH>) cur_tkeep;
          ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> cur_pixel = (ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH>) cur_read;
          if(cur_tkeep_bit > 0)
          {
            //if not --> process in next state, additional delay not dramatic since all pixels are buffered in parallel
            sTohls4ml_parallelPixelChain_chanX.write(cur_pixel);
          }
          cur_read >>= DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH;
          cur_tkeep >>= DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP_WIDTH;
          if(cur_tkeep > 0)
          {
            narrowFSM = BACKLOG3;
          }
        }
      }
      break;

    case BACKLOG3:
      if(!sTohls4ml_parallelPixelChain_chanX.full())
      {
        ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP_WIDTH> cur_tkeep_bit = (ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP_WIDTH>) cur_tkeep;
        ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> cur_pixel = (ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH>) cur_read;
        if(cur_tkeep_bit > 0)
        {
          //if not --> process in next state, additional delay not dramatic since all pixels are buffered in parallel
          sTohls4ml_parallelPixelChain_chanX.write(cur_pixel);
        }
        cur_read >>= DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH;
        cur_tkeep >>= DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP_WIDTH;
        if(cur_tkeep == 0)
        {
          narrowFSM = FORWARD3;
        }
      }
      break;
  }
}
#else
  //DOSA_ADD_pTohls4ml_parallelNarrow_X_declaration
#endif


void pTohls4ml_parallelDeq(
#ifdef WRAPPER_TEST
  stream<ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> >    &sTohls4ml_parallelPixelChain_chan1,
  stream<ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> >    &sTohls4ml_parallelPixelChain_chan2,
  stream<ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> >    &sTohls4ml_parallelPixelChain_chan3,
#else
    //DOSA_ADD_tohls4ml_parallel_pixelChain_param_decl
#endif
    stream<ap_uint<DOSA_HLS4ML_PARALLEL_INPUT_BITDIWDTH> > &po_hls4ml_parallel_data,
    stream<bool>                              &shls4ml_parallelUnitProcessing,
    uint16_t *debug
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static twoStatesFSM dequeueFSM = RESET;
#pragma HLS reset variable=dequeueFSM
  //-- LOCAL VARIABLES ------------------------------------------------------
  bool one_not_empty = false;
  //for debugging
  bool did_forward = false;

  switch(dequeueFSM)
  {
    default:
    case RESET:
#ifdef WRAPPER_TEST
      if( !sTohls4ml_parallelPixelChain_chan1.empty() )
      {
        sTohls4ml_parallelPixelChain_chan1.read();
        one_not_empty = true;
      }
      if( !sTohls4ml_parallelPixelChain_chan2.empty() )
      {
        sTohls4ml_parallelPixelChain_chan2.read();
        one_not_empty = true;
      }
      if( !sTohls4ml_parallelPixelChain_chan3.empty() )
      {
        sTohls4ml_parallelPixelChain_chan3.read();
        one_not_empty = true;
      }
#else
      //DOSA_ADD_tohls4ml_parallel_deq_pixelChain_drain
#endif
      if( !one_not_empty )
      {
        dequeueFSM = FORWARD;
      }
      break;

    case FORWARD:
      if( !shls4ml_parallelUnitProcessing.full()
#ifdef WRAPPER_TEST
          && !sTohls4ml_parallelPixelChain_chan1.empty() && !sTohls4ml_parallelPixelChain_chan2.empty() && !sTohls4ml_parallelPixelChain_chan3.empty()
#else
          //DOSA_ADD_tohls4ml_parallel_deq_if_clause
#endif
          && !po_hls4ml_parallel_data.full()
        )
      {
        ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> pixel_array[DOSA_HLS4ML_PARALLEL_INPUT_CHAN_NUM];
#ifdef WRAPPER_TEST
        pixel_array[0] = sTohls4ml_parallelPixelChain_chan1.read();
        pixel_array[1] = sTohls4ml_parallelPixelChain_chan2.read();
        pixel_array[2] = sTohls4ml_parallelPixelChain_chan3.read();
#else
        //DOSA_ADD_deq_flatten
#endif
        ap_uint<DOSA_HLS4ML_PARALLEL_INPUT_BITDIWDTH> output_data = 0x0;
        for(int i = 0; i<DOSA_HLS4ML_PARALLEL_INPUT_CHAN_NUM; i++)
        {
          //DynLayerInput requires (chan2,chan1,chan0) vector layout
          output_data |= ((ap_uint<DOSA_HLS4ML_PARALLEL_INPUT_BITDIWDTH>) pixel_array[i]) << (i*DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH);
        }
        po_hls4ml_parallel_data.write(output_data);
        printf("pTohls4ml_parallelDeq: write 0x%6.6X\n", (uint32_t) output_data);
        shls4ml_parallelUnitProcessing.write(true);
        did_forward = true;
      }
      break;
  }

  *debug = (uint16_t) did_forward;
  *debug |= ((uint16_t) dequeueFSM) << 8;
}


void pFromhls4ml_parallelEnq(
    stream<ap_uint<DOSA_HLS4ML_PARALLEL_OUTPUT_BITDIWDTH> > &pi_hls4ml_parallel_data,
    stream<bool>                              &shls4ml_parallelUnitProcessing,
    stream<ap_uint<DOSA_HLS4ML_PARALLEL_OUTPUT_BITDIWDTH> >  &sFromhls4ml_parallelBuffer
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static Fromhls4ml_parallelEnqStates enqueueFSM = RESET2;
#pragma HLS reset variable=enqueueFSM
  static uint32_t invalid_pixel_cnt = DOSA_HLS4ML_PARALLEL_VALID_WAIT_CNT;
#pragma HLS reset variable=invalid_pixel_cnt
  //-- LOCAL VARIABLES ------------------------------------------------------
  ap_uint<DOSA_HLS4ML_PARALLEL_OUTPUT_BITDIWDTH> input_data = 0x0;

  switch(enqueueFSM)
  {
    default:
    case RESET2:
      invalid_pixel_cnt = DOSA_HLS4ML_PARALLEL_VALID_WAIT_CNT;
      if(!shls4ml_parallelUnitProcessing.empty())
      {
        shls4ml_parallelUnitProcessing.read();
      } else {
        if(DOSA_HLS4ML_PARALLEL_VALID_WAIT_CNT > 0)
        {
          enqueueFSM = CNT_UNTIL_VAILD;
        } else {
          enqueueFSM = FORWARD2;
        }
      }
      break;

    case CNT_UNTIL_VAILD:
      if(!shls4ml_parallelUnitProcessing.empty() && !pi_hls4ml_parallel_data.empty()
        )
      {
        ap_uint<DOSA_HLS4ML_PARALLEL_OUTPUT_BITDIWDTH> ignore_me = pi_hls4ml_parallel_data.read();
        bool ignore_me_too = shls4ml_parallelUnitProcessing.read();
        printf("pFromhls4ml_parallelEnq: ignoring 0x%6.6X\n", (uint32_t) ignore_me);

        invalid_pixel_cnt--;
        if(invalid_pixel_cnt == 0)
        {
          enqueueFSM = FORWARD2;
        }
      }
      else if ( !shls4ml_parallelUnitProcessing.empty()
          && pi_hls4ml_parallel_data.empty() //yes, empty!
          )
      { // consume one HUP token, so that the fifo doesn't sand
        bool ignore_me = shls4ml_parallelUnitProcessing.read();
        printf("pFromhls4ml_parallelEnq: read HUP token to avoid sanding...(initial phase)\n");
        invalid_pixel_cnt--;
        if(invalid_pixel_cnt == 0)
        {
          enqueueFSM = FORWARD2;
        }
      }
      break;

    case FORWARD2:
      //if ( !shls4ml_parallelUnitProcessing.empty() && !sFromhls4ml_parallelBuffer.full()
      //    && !pi_hls4ml_parallel_data.empty()
      //   )
      //{
      //  //ignore pi_hls4ml_parallel_frame_valid?
      //  //if( *pi_hls4ml_parallel_data_valid == 0b1 )
      //  //{
      //  //input_data = *pi_hls4ml_parallel_data_vector;
      //  input_data = pi_hls4ml_parallel_data.read();
      //  //read only if able to process
      //  bool ignore_me = shls4ml_parallelUnitProcessing.read();
      //  printf("pFromhls4ml_parallelEnq: read 0x%6.6X\n", (uint32_t) input_data);
      //  sFromhls4ml_parallelBuffer.write(input_data);
      //  //}
      //}
      //else if ( !shls4ml_parallelUnitProcessing.empty() && !sFromhls4ml_parallelBuffer.full()
      //    && pi_hls4ml_parallel_data.empty() //yes, empty!
      //    )
      //{ // consume one HUP token, so that the fifo doesn't sand
      //  // and, since sFromhls4ml_parallelBuffer isn't full, we can do this safely
      //  bool ignore_me = shls4ml_parallelUnitProcessing.read();
      //  printf("pFromhls4ml_parallelEnq: read HUP token to avoid sanding...\n");
      //}
      if ( !shls4ml_parallelUnitProcessing.empty() && !sFromhls4ml_parallelBuffer.full()
          && !pi_hls4ml_parallel_data.empty()
         )
      {
        //ignore pi_hls4ml_parallel_frame_valid?
        //if( *pi_hls4ml_parallel_data_valid == 0b1 )
        //{
        //input_data = *pi_hls4ml_parallel_data_vector;
        input_data = pi_hls4ml_parallel_data.read();
        //read only if able to process
        bool ignore_me = shls4ml_parallelUnitProcessing.read();
        printf("pFromhls4ml_parallelEnq: read 0x%6.6X\n", (uint32_t) input_data);
        sFromhls4ml_parallelBuffer.write(input_data);
        //}
      }
      //TODO: now, with correct valid signals, we can read all that's valid
      else if ( shls4ml_parallelUnitProcessing.empty() && !sFromhls4ml_parallelBuffer.full() //yes empty
          && !pi_hls4ml_parallel_data.empty()
         )
      {
        //ignore pi_hls4ml_parallel_frame_valid?
        //if( *pi_hls4ml_parallel_data_valid == 0b1 )
        //{
        //input_data = *pi_hls4ml_parallel_data_vector;
        input_data = pi_hls4ml_parallel_data.read();
        //read only if able to process
        printf("pFromhls4ml_parallelEnq: read 0x%6.6X\n", (uint32_t) input_data);
        sFromhls4ml_parallelBuffer.write(input_data);
        //}
      }
      else if ( !shls4ml_parallelUnitProcessing.empty() && !sFromhls4ml_parallelBuffer.full()
          && pi_hls4ml_parallel_data.empty() //yes, empty!
          )
      { // consume one HUP token, so that the fifo doesn't sand
        // and, since sFromhls4ml_parallelBuffer isn't full, we can do this safely
        bool ignore_me = shls4ml_parallelUnitProcessing.read();
        printf("pFromhls4ml_parallelEnq: read HUP token to avoid sanding...\n");
      }
      break;
  }

}


void pFromhls4ml_parallelFlatten(
#ifdef WRAPPER_TEST
  stream<ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> >    &sFromhls4ml_parallelBuffer_chan1,
  stream<ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> >    &sFromhls4ml_parallelBuffer_chan2,
  stream<ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> >    &sFromhls4ml_parallelBuffer_chan3,
  stream<ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> >    &sFromhls4ml_parallelBuffer_chan4,
#else
    //DOSA_ADD_from_hls4ml_parallel_stream_param_decl
#endif
    stream<ap_uint<DOSA_HLS4ML_PARALLEL_OUTPUT_BITDIWDTH> >  &sFromhls4ml_parallelBuffer,
    //stream<bool>                              &sFromhls4ml_parallelIfLineComplete,
    uint16_t *debug
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static twoStatesFSM flattenFSM = RESET;
#pragma HLS reset variable=flattenFSM
//  static ap_uint<64> current_array_write_pnt = 0x0;
//#pragma HLS reset variable=current_array_write_pnt
//  static ap_uint<2> current_array_slot_pnt = 0x0;
//#pragma HLS reset variable=current_array_slot_pnt
  //-- LOCAL VARIABLES ------------------------------------------------------
  ap_uint<DOSA_HLS4ML_PARALLEL_OUTPUT_BITDIWDTH> input_data = 0x0;
  bool not_empty = false;

  if(flattenFSM == RESET)
  {
    //current_array_write_pnt = 0x0;
    if(!sFromhls4ml_parallelBuffer.empty())
    {
      sFromhls4ml_parallelBuffer.read();
      not_empty = true;
    }
    if(!not_empty)
    {
      flattenFSM = FORWARD;
    }
  } else { //FORWARD
      if ( !sFromhls4ml_parallelBuffer.empty() //&& !sFromhls4ml_parallelIfLineComplete.full()
#ifdef WRAPPER_TEST
          && !sFromhls4ml_parallelBuffer_chan1.full() && !sFromhls4ml_parallelBuffer_chan2.full() && !sFromhls4ml_parallelBuffer_chan3.full() && !sFromhls4ml_parallelBuffer_chan4.full()
#else
          //DOSA_ADD_from_hls4ml_parallel_stream_full_check
#endif
         )
      {
        input_data = sFromhls4ml_parallelBuffer.read();
#ifdef WRAPPER_TEST
        sFromhls4ml_parallelBuffer_chan1.write((ap_uint<DOSA_HLS4ML_PARALLEL_FROMACCEL_BITWIDTH>) (input_data >> 0 * DOSA_HLS4ML_PARALLEL_FROMACCEL_BITWIDTH));
        sFromhls4ml_parallelBuffer_chan2.write((ap_uint<DOSA_HLS4ML_PARALLEL_FROMACCEL_BITWIDTH>) (input_data >> 1 * DOSA_HLS4ML_PARALLEL_FROMACCEL_BITWIDTH));
        sFromhls4ml_parallelBuffer_chan3.write((ap_uint<DOSA_HLS4ML_PARALLEL_FROMACCEL_BITWIDTH>) (input_data >> 2 * DOSA_HLS4ML_PARALLEL_FROMACCEL_BITWIDTH));
        sFromhls4ml_parallelBuffer_chan4.write((ap_uint<DOSA_HLS4ML_PARALLEL_FROMACCEL_BITWIDTH>) (input_data >> 3 * DOSA_HLS4ML_PARALLEL_FROMACCEL_BITWIDTH));
#else
        //DOSA_ADD_from_hls4ml_parallel_stream_write
#endif
        //printf("pFromhls4ml_parallelFlatten: sorted incoming %2.2x to position %d in slot %d\n", (uint32_t) input_data, (uint32_t) current_array_write_pnt, (uint8_t) current_array_slot_pnt);
        //current_array_write_pnt++;
        //if( current_array_write_pnt >= WRAPPER_OUTPUT_IF_HLS4ML_PARALLEL_WORDS_CNT_CEIL)
        //{
        //  sFromhls4ml_parallelIfLineComplete.write(true);
        //  current_array_write_pnt = 0;
        //}
      }
  }

  //*debug = ((uint16_t) current_array_write_pnt);
  *debug = ((uint16_t) input_data);

}


// THANKS to vivado HLS...pralellize it explicitly
#ifdef WRAPPER_TEST
void pFromhls4ml_parallelWiden_1(
    stream<ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> >    &sFromhls4ml_parallelBuffer_chanX,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >    &sOutBuffer_chanX
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static twoStatesFSM widenFSM = RESET;
#pragma HLS reset variable=widenFSM
  static uint32_t current_frame_bit_cnt;
#pragma HLS reset variable=current_frame_bit_cnt
  static uint32_t current_line_read_pnt;
#pragma HLS reset variable=current_line_read_pnt
  static ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> hangover_store;
#pragma HLS reset variable=hangover_store
  static ap_uint<32> hangover_store_valid_bits;
#pragma HLS reset variable=hangover_store_valid_bits
  static ap_uint<(DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+7)/8> tkeep;
#pragma HLS reset variable=tkeep
  //-- LOCAL VARIABLES ------------------------------------------------------
  bool not_empty = false;
  ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> combined_output = 0x0;

  switch(widenFSM)
  {
    default:
    case RESET:
      //    for(int i = 0; i < DOSA_HLS4ML_PARALLEL_OUTPUT_CHAN_NUM; i++)
      //    {
      //      current_frame_bit_cnt[i]= 0x0;
      //      current_line_read_pnt[i] = 0x0;
      //      hangover_store[i] = 0x0;
      //      hangover_store_valid_bits[i] = 0x0;
      //    }
      current_frame_bit_cnt = 0x0;
      current_line_read_pnt = 0x0;
      hangover_store = 0x0;
      hangover_store_valid_bits = 0x0;
      tkeep = 0x0;
      if(!sFromhls4ml_parallelBuffer_chanX.empty())
      {
        sFromhls4ml_parallelBuffer_chanX.read();
        not_empty = true;
      }
      if(!not_empty)
      {
        widenFSM = FORWARD;
      }
      break;

    case FORWARD:
      if(!sFromhls4ml_parallelBuffer_chanX.empty() && !sOutBuffer_chanX.full())
      {
        combined_output = hangover_store;
        //ap_uint<(DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+7)/8> tkeep = 0x0;

        ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> nv = sFromhls4ml_parallelBuffer_chanX.read();
        combined_output |= ((ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH>) nv) << (hangover_store_valid_bits);
        current_line_read_pnt++;
        tkeep <<= DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP_WIDTH;
        tkeep |= (ap_uint<(DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+7)/8>) DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP;

        //current_frame_bit_cnt += DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH - hangover_store_valid_bits;
        current_frame_bit_cnt += DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH;
        //tkeep = bitCntToTKeep(ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH/8> (DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH + hangover_store_valid_bits)); //do it HERE, to achieve II=1

        if(current_line_read_pnt >= WRAPPER_OUTPUT_IF_HLS4ML_PARALLEL_WORDS_CNT_CEIL || current_frame_bit_cnt >= HLS4ML_PARALLEL_OUTPUT_FRAME_BIT_CNT)
        {//write to stream
          current_line_read_pnt = 0x0;
          ap_uint<1> tlast = 0;
          if(current_frame_bit_cnt >= HLS4ML_PARALLEL_OUTPUT_FRAME_BIT_CNT)
          {
            //TODO: what if there is hangover data left?
            current_frame_bit_cnt = 0x0;
            tlast = 0b1;
          }
          Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> tmp_write_0 = Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH>((ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH>) combined_output, tkeep, tlast);
          sOutBuffer_chanX.write(tmp_write_0);
          tkeep = 0x0;
          printf("genericWiden: write Axis tdata: %16.16llx, tkeep: %2.2x, tlast: %x;\n", (uint64_t) tmp_write_0.getTData(), (uint8_t) tmp_write_0.getTKeep(), (uint8_t) tmp_write_0.getTLast());

          hangover_store = (ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH>) (combined_output >> DOSA_WRAPPER_OUTPUT_IF_BITWIDTH);
          if((hangover_store_valid_bits+DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH) > DOSA_WRAPPER_OUTPUT_IF_BITWIDTH)
          {
            hangover_store_valid_bits -= (DOSA_WRAPPER_OUTPUT_IF_BITWIDTH-DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH);
          } else {
            hangover_store_valid_bits = 0x0;
          }
        } else {
          //wait
          hangover_store = combined_output;
          hangover_store_valid_bits += DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH;
        }
        printf("genericWiden: combined %16.16llx, hangover_bits_valid_bits: %d, current_frame_bit_cnt: %d\n", (uint64_t) combined_output, (uint64_t) hangover_store_valid_bits, (uint32_t) current_frame_bit_cnt);
      }
      break;
  }

}
void pFromhls4ml_parallelWiden_2(
    stream<ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> >    &sFromhls4ml_parallelBuffer_chanX,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >    &sOutBuffer_chanX
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static twoStatesFSM widenFSM = RESET;
#pragma HLS reset variable=widenFSM
  static uint32_t current_frame_bit_cnt;
#pragma HLS reset variable=current_frame_bit_cnt
  static uint32_t current_line_read_pnt;
#pragma HLS reset variable=current_line_read_pnt
  static ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> hangover_store;
#pragma HLS reset variable=hangover_store
  static ap_uint<32> hangover_store_valid_bits;
#pragma HLS reset variable=hangover_store_valid_bits
  static ap_uint<(DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+7)/8> tkeep;
#pragma HLS reset variable=tkeep
  //-- LOCAL VARIABLES ------------------------------------------------------
  bool not_empty = false;
  ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> combined_output = 0x0;

  switch(widenFSM)
  {
    default:
    case RESET:
      //    for(int i = 0; i < DOSA_HLS4ML_PARALLEL_OUTPUT_CHAN_NUM; i++)
      //    {
      //      current_frame_bit_cnt[i]= 0x0;
      //      current_line_read_pnt[i] = 0x0;
      //      hangover_store[i] = 0x0;
      //      hangover_store_valid_bits[i] = 0x0;
      //    }
      current_frame_bit_cnt = 0x0;
      current_line_read_pnt = 0x0;
      hangover_store = 0x0;
      hangover_store_valid_bits = 0x0;
      tkeep = 0x0;
      if(!sFromhls4ml_parallelBuffer_chanX.empty())
      {
        sFromhls4ml_parallelBuffer_chanX.read();
        not_empty = true;
      }
      if(!not_empty)
      {
        widenFSM = FORWARD;
      }
      break;

    case FORWARD:
      if(!sFromhls4ml_parallelBuffer_chanX.empty() && !sOutBuffer_chanX.full())
      {
        combined_output = hangover_store;
        //ap_uint<(DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+7)/8> tkeep = 0x0;

        ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> nv = sFromhls4ml_parallelBuffer_chanX.read();
        combined_output |= ((ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH>) nv) << (hangover_store_valid_bits);
        current_line_read_pnt++;
        tkeep <<= DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP_WIDTH;
        tkeep |= (ap_uint<(DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+7)/8>) DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP;

        //current_frame_bit_cnt += DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH - hangover_store_valid_bits;
        current_frame_bit_cnt += DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH;
        //tkeep = bitCntToTKeep(ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH/8> (DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH + hangover_store_valid_bits)); //do it HERE, to achieve II=1

        if(current_line_read_pnt >= WRAPPER_OUTPUT_IF_HLS4ML_PARALLEL_WORDS_CNT_CEIL || current_frame_bit_cnt >= HLS4ML_PARALLEL_OUTPUT_FRAME_BIT_CNT)
        {//write to stream
          current_line_read_pnt = 0x0;
          ap_uint<1> tlast = 0;
          if(current_frame_bit_cnt >= HLS4ML_PARALLEL_OUTPUT_FRAME_BIT_CNT)
          {
            //TODO: what if there is hangover data left?
            current_frame_bit_cnt = 0x0;
            tlast = 0b1;
          }
          Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> tmp_write_0 = Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH>((ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH>) combined_output, tkeep, tlast);
          sOutBuffer_chanX.write(tmp_write_0);
          tkeep = 0x0;
          printf("genericWiden: write Axis tdata: %16.16llx, tkeep: %2.2x, tlast: %x;\n", (uint64_t) tmp_write_0.getTData(), (uint8_t) tmp_write_0.getTKeep(), (uint8_t) tmp_write_0.getTLast());

          hangover_store = (ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH>) (combined_output >> DOSA_WRAPPER_OUTPUT_IF_BITWIDTH);
          if((hangover_store_valid_bits+DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH) > DOSA_WRAPPER_OUTPUT_IF_BITWIDTH)
          {
            hangover_store_valid_bits -= (DOSA_WRAPPER_OUTPUT_IF_BITWIDTH-DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH);
          } else {
            hangover_store_valid_bits = 0x0;
          }
        } else {
          //wait
          hangover_store = combined_output;
          hangover_store_valid_bits += DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH;
        }
        printf("genericWiden: combined %16.16llx, hangover_bits_valid_bits: %d, current_frame_bit_cnt: %d\n", (uint64_t) combined_output, (uint64_t) hangover_store_valid_bits, (uint32_t) current_frame_bit_cnt);
      }
      break;
  }

}
void pFromhls4ml_parallelWiden_3(
    stream<ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> >    &sFromhls4ml_parallelBuffer_chanX,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >    &sOutBuffer_chanX
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static twoStatesFSM widenFSM = RESET;
#pragma HLS reset variable=widenFSM
  static uint32_t current_frame_bit_cnt;
#pragma HLS reset variable=current_frame_bit_cnt
  static uint32_t current_line_read_pnt;
#pragma HLS reset variable=current_line_read_pnt
  static ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> hangover_store;
#pragma HLS reset variable=hangover_store
  static ap_uint<32> hangover_store_valid_bits;
#pragma HLS reset variable=hangover_store_valid_bits
  static ap_uint<(DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+7)/8> tkeep;
#pragma HLS reset variable=tkeep
  //-- LOCAL VARIABLES ------------------------------------------------------
  bool not_empty = false;
  ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> combined_output = 0x0;

  switch(widenFSM)
  {
    default:
    case RESET:
      //    for(int i = 0; i < DOSA_HLS4ML_PARALLEL_OUTPUT_CHAN_NUM; i++)
      //    {
      //      current_frame_bit_cnt[i]= 0x0;
      //      current_line_read_pnt[i] = 0x0;
      //      hangover_store[i] = 0x0;
      //      hangover_store_valid_bits[i] = 0x0;
      //    }
      current_frame_bit_cnt = 0x0;
      current_line_read_pnt = 0x0;
      hangover_store = 0x0;
      hangover_store_valid_bits = 0x0;
      tkeep = 0x0;
      if(!sFromhls4ml_parallelBuffer_chanX.empty())
      {
        sFromhls4ml_parallelBuffer_chanX.read();
        not_empty = true;
      }
      if(!not_empty)
      {
        widenFSM = FORWARD;
      }
      break;

    case FORWARD:
      if(!sFromhls4ml_parallelBuffer_chanX.empty() && !sOutBuffer_chanX.full())
      {
        combined_output = hangover_store;
        //ap_uint<(DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+7)/8> tkeep = 0x0;

        ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> nv = sFromhls4ml_parallelBuffer_chanX.read();
        combined_output |= ((ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH>) nv) << (hangover_store_valid_bits);
        current_line_read_pnt++;
        tkeep <<= DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP_WIDTH;
        tkeep |= (ap_uint<(DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+7)/8>) DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP;

        //current_frame_bit_cnt += DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH - hangover_store_valid_bits;
        current_frame_bit_cnt += DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH;
        //tkeep = bitCntToTKeep(ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH/8> (DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH + hangover_store_valid_bits)); //do it HERE, to achieve II=1

        if(current_line_read_pnt >= WRAPPER_OUTPUT_IF_HLS4ML_PARALLEL_WORDS_CNT_CEIL || current_frame_bit_cnt >= HLS4ML_PARALLEL_OUTPUT_FRAME_BIT_CNT)
        {//write to stream
          current_line_read_pnt = 0x0;
          ap_uint<1> tlast = 0;
          if(current_frame_bit_cnt >= HLS4ML_PARALLEL_OUTPUT_FRAME_BIT_CNT)
          {
            //TODO: what if there is hangover data left?
            current_frame_bit_cnt = 0x0;
            tlast = 0b1;
          }
          Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> tmp_write_0 = Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH>((ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH>) combined_output, tkeep, tlast);
          sOutBuffer_chanX.write(tmp_write_0);
          tkeep = 0x0;
          printf("genericWiden: write Axis tdata: %16.16llx, tkeep: %2.2x, tlast: %x;\n", (uint64_t) tmp_write_0.getTData(), (uint8_t) tmp_write_0.getTKeep(), (uint8_t) tmp_write_0.getTLast());

          hangover_store = (ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH>) (combined_output >> DOSA_WRAPPER_OUTPUT_IF_BITWIDTH);
          if((hangover_store_valid_bits+DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH) > DOSA_WRAPPER_OUTPUT_IF_BITWIDTH)
          {
            hangover_store_valid_bits -= (DOSA_WRAPPER_OUTPUT_IF_BITWIDTH-DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH);
          } else {
            hangover_store_valid_bits = 0x0;
          }
        } else {
          //wait
          hangover_store = combined_output;
          hangover_store_valid_bits += DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH;
        }
        printf("genericWiden: combined %16.16llx, hangover_bits_valid_bits: %d, current_frame_bit_cnt: %d\n", (uint64_t) combined_output, (uint64_t) hangover_store_valid_bits, (uint32_t) current_frame_bit_cnt);
      }
      break;
  }

}
void pFromhls4ml_parallelWiden_4(
    stream<ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> >    &sFromhls4ml_parallelBuffer_chanX,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >    &sOutBuffer_chanX
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static twoStatesFSM widenFSM = RESET;
#pragma HLS reset variable=widenFSM
  static uint32_t current_frame_bit_cnt;
#pragma HLS reset variable=current_frame_bit_cnt
  static uint32_t current_line_read_pnt;
#pragma HLS reset variable=current_line_read_pnt
  static ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> hangover_store;
#pragma HLS reset variable=hangover_store
  static ap_uint<32> hangover_store_valid_bits;
#pragma HLS reset variable=hangover_store_valid_bits
  static ap_uint<(DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+7)/8> tkeep;
#pragma HLS reset variable=tkeep
  //-- LOCAL VARIABLES ------------------------------------------------------
  bool not_empty = false;
  ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> combined_output = 0x0;

  switch(widenFSM)
  {
    default:
    case RESET:
      //    for(int i = 0; i < DOSA_HLS4ML_PARALLEL_OUTPUT_CHAN_NUM; i++)
      //    {
      //      current_frame_bit_cnt[i]= 0x0;
      //      current_line_read_pnt[i] = 0x0;
      //      hangover_store[i] = 0x0;
      //      hangover_store_valid_bits[i] = 0x0;
      //    }
      current_frame_bit_cnt = 0x0;
      current_line_read_pnt = 0x0;
      hangover_store = 0x0;
      hangover_store_valid_bits = 0x0;
      tkeep = 0x0;
      if(!sFromhls4ml_parallelBuffer_chanX.empty())
      {
        sFromhls4ml_parallelBuffer_chanX.read();
        not_empty = true;
      }
      if(!not_empty)
      {
        widenFSM = FORWARD;
      }
      break;

    case FORWARD:
      if(!sFromhls4ml_parallelBuffer_chanX.empty() && !sOutBuffer_chanX.full())
      {
        combined_output = hangover_store;
        //ap_uint<(DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+7)/8> tkeep = 0x0;

        ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> nv = sFromhls4ml_parallelBuffer_chanX.read();
        combined_output |= ((ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH>) nv) << (hangover_store_valid_bits);
        current_line_read_pnt++;
        tkeep <<= DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP_WIDTH;
        tkeep |= (ap_uint<(DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+7)/8>) DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH_TKEEP;

        //current_frame_bit_cnt += DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH - hangover_store_valid_bits;
        current_frame_bit_cnt += DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH;
        //tkeep = bitCntToTKeep(ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH/8> (DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH + hangover_store_valid_bits)); //do it HERE, to achieve II=1

        if(current_line_read_pnt >= WRAPPER_OUTPUT_IF_HLS4ML_PARALLEL_WORDS_CNT_CEIL || current_frame_bit_cnt >= HLS4ML_PARALLEL_OUTPUT_FRAME_BIT_CNT)
        {//write to stream
          current_line_read_pnt = 0x0;
          ap_uint<1> tlast = 0;
          if(current_frame_bit_cnt >= HLS4ML_PARALLEL_OUTPUT_FRAME_BIT_CNT)
          {
            //TODO: what if there is hangover data left?
            current_frame_bit_cnt = 0x0;
            tlast = 0b1;
          }
          Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> tmp_write_0 = Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH>((ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH>) combined_output, tkeep, tlast);
          sOutBuffer_chanX.write(tmp_write_0);
          tkeep = 0x0;
          printf("genericWiden: write Axis tdata: %16.16llx, tkeep: %2.2x, tlast: %x;\n", (uint64_t) tmp_write_0.getTData(), (uint8_t) tmp_write_0.getTKeep(), (uint8_t) tmp_write_0.getTLast());

          hangover_store = (ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH>) (combined_output >> DOSA_WRAPPER_OUTPUT_IF_BITWIDTH);
          if((hangover_store_valid_bits+DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH) > DOSA_WRAPPER_OUTPUT_IF_BITWIDTH)
          {
            hangover_store_valid_bits -= (DOSA_WRAPPER_OUTPUT_IF_BITWIDTH-DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH);
          } else {
            hangover_store_valid_bits = 0x0;
          }
        } else {
          //wait
          hangover_store = combined_output;
          hangover_store_valid_bits += DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH;
        }
        printf("genericWiden: combined %16.16llx, hangover_bits_valid_bits: %d, current_frame_bit_cnt: %d\n", (uint64_t) combined_output, (uint64_t) hangover_store_valid_bits, (uint32_t) current_frame_bit_cnt);
      }
      break;
  }

}
#else
  //DOSA_ADD_pFromhls4ml_parallelWiden_X_declaration
#endif



void pFromhls4ml_parallelDeq(
#ifdef WRAPPER_TEST
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >    &sOutBuffer_chan1,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >    &sOutBuffer_chan2,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >    &sOutBuffer_chan3,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >    &sOutBuffer_chan4,
#else
    //DOSA_ADD_output_stream_param_decl
#endif
    //stream<bool>                              &sFromhls4ml_parallelIfLineComplete,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >  &soData,
    uint16_t *debug
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static Fromhls4ml_parallelDeqStates dequeueFSM = RESET1;
#pragma HLS reset variable=dequeueFSM
  static ap_uint<64> current_frame_bit_cnt = 0x0;
#pragma HLS reset variable=current_frame_bit_cnt
  //  static ap_uint<64> current_batch_bit_cnt = 0x0;
  //#pragma HLS reset variable=current_batch_bit_cnt
  //-- LOCAL VARIABLES ------------------------------------------------------
  bool not_empty = false;
  Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> tmp_read_0;

  switch (dequeueFSM)
  {
    default:
    case RESET1:
      current_frame_bit_cnt = 0x0;
      //current_batch_bit_cnt = 0x0;
#ifdef WRAPPER_TEST
      if(!sOutBuffer_chan1.empty())
      {
        sOutBuffer_chan1.read();
        not_empty = true;
      }
      if(!sOutBuffer_chan2.empty())
      {
        sOutBuffer_chan2.read();
        not_empty = true;
      }
      if(!sOutBuffer_chan3.empty())
      {
        sOutBuffer_chan3.read();
        not_empty = true;
      }
      if(!sOutBuffer_chan4.empty())
      {
        sOutBuffer_chan4.read();
        not_empty = true;
      }
#else
      //DOSA_ADD_out_stream_drain
#endif
      if(!not_empty)
      {
        dequeueFSM = READ_BUF_0;
      }
      break;

      //case WAIT_LINE:
      //  if( !sFromhls4ml_parallelIfLineComplete.empty() )
      //  {
      //    bool ignore_me = sFromhls4ml_parallelFrameComplete.read();
      //    current_frame_bit_cnt = 0x0;
      //    current_array_read_pnt = 0x0;
      //    current_batch_bit_cnt = 0x0;
      //    current_array_slot_pnt = 0x0;
      //    hangover_store = 0x0;
      //    hangover_store_valid_bits = 0x0;
      //    dequeueFSM = READ_FRAME;
      //  }
      //  break;
#ifdef WRAPPER_TEST
    case READ_BUF_0:
      if(!soData.full() && !sOutBuffer_chan1.empty())
      {
        tmp_read_0 = sOutBuffer_chan1.read();
        uint32_t bit_read = extractByteCnt(tmp_read_0) * 8;
        current_frame_bit_cnt += bit_read;
        //current_batch_bit_cnt += bit_read;
        if(current_frame_bit_cnt >= HLS4ML_PARALLEL_OUTPUT_FRAME_BIT_CNT || tmp_read_0.getTLast() == 1)
        {
          current_frame_bit_cnt = 0x0;
          dequeueFSM = READ_BUF_1;
          //if(current_batch_bit_cnt >= HLS4ML_PARALLEL_OUTPUT_FRAME_BIT_CNT*DOSA_HLS4ML_PARALLEL_OUTPUT_CHAN_NUM)
          //{
          //  current_batch_bit_cnt = 0x0;
          //  //in all cases
          //  tlast = 0b1;
          //}
          //check for tlast after each frame
          if(!DOSA_HLS4ML_PARALLEL_OUTPUT_BATCH_FLATTEN)
          {
            tmp_read_0.setTLast(0b1);
          } else {
            tmp_read_0.setTLast(0b0);
          }
        }
        soData.write(tmp_read_0);
      }
      break;

    case READ_BUF_1:
      if(!soData.full() && !sOutBuffer_chan2.empty())
      {
        tmp_read_0 = sOutBuffer_chan2.read();
        uint32_t bit_read = extractByteCnt(tmp_read_0) * 8;
        current_frame_bit_cnt += bit_read;
        if(current_frame_bit_cnt >= HLS4ML_PARALLEL_OUTPUT_FRAME_BIT_CNT || tmp_read_0.getTLast() == 1)
        {
          current_frame_bit_cnt = 0x0;
          dequeueFSM = READ_BUF_2;
          //check for tlast after each frame
          if(!DOSA_HLS4ML_PARALLEL_OUTPUT_BATCH_FLATTEN)
          {
            tmp_read_0.setTLast(0b1);
          } else {
            tmp_read_0.setTLast(0b0);
          }
        }
        soData.write(tmp_read_0);
      }
      break;

    case READ_BUF_2:
      if(!soData.full() && !sOutBuffer_chan3.empty())
      {
        tmp_read_0 = sOutBuffer_chan3.read();
        uint32_t bit_read = extractByteCnt(tmp_read_0) * 8;
        current_frame_bit_cnt += bit_read;
        if(current_frame_bit_cnt >= HLS4ML_PARALLEL_OUTPUT_FRAME_BIT_CNT || tmp_read_0.getTLast() == 1)
        {
          current_frame_bit_cnt = 0x0;
          dequeueFSM = READ_BUF_3;
          //check for tlast after each frame
          if(!DOSA_HLS4ML_PARALLEL_OUTPUT_BATCH_FLATTEN)
          {
            tmp_read_0.setTLast(0b1);
          } else {
            tmp_read_0.setTLast(0b0);
          }
        }
        soData.write(tmp_read_0);
      }
      break;

    case READ_BUF_3:
      if(!soData.full() && !sOutBuffer_chan4.empty())
      {
        tmp_read_0 = sOutBuffer_chan4.read();
        uint32_t bit_read = extractByteCnt(tmp_read_0) * 8;
        current_frame_bit_cnt += bit_read;
        if(current_frame_bit_cnt >= HLS4ML_PARALLEL_OUTPUT_FRAME_BIT_CNT || tmp_read_0.getTLast() == 1)
        {
          current_frame_bit_cnt = 0x0;
          dequeueFSM = READ_BUF_0; //back to start
          //check for tlast after each frame
          //if(!DOSA_HLS4ML_PARALLEL_OUTPUT_BATCH_FLATTEN)
          //{
          // tmp_read_0.setTLast(0b1);
          //} else {
          // tmp_read_0.setTLast(0b0);
          //}
          //in all cases
          tmp_read_0.setTLast(0b1);
        }
        soData.write(tmp_read_0);
      }
      break;
#else
      //DOSA_ADD_from_hls4ml_parallel_deq_buf_read
#endif

  }

  *debug = current_frame_bit_cnt;
}

void pMergeDebug(
  uint16_t *debug0,
  uint16_t *debug1,
  uint16_t *debug2,
  uint16_t *debug3,
    ap_uint<64> *debug_out
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------

  *debug_out = (uint64_t) *debug0;
  *debug_out |= ((uint64_t) *debug1) << 16;
  *debug_out |= ((uint64_t) *debug2) << 32;
  *debug_out |= ((uint64_t) *debug3) << 48;

}



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
    )
{
  //-- DIRECTIVES FOR THE BLOCK ---------------------------------------------
#pragma HLS INTERFACE ap_ctrl_none port=return

#pragma HLS INTERFACE ap_fifo port=siData
#pragma HLS INTERFACE ap_fifo port=soData

#pragma HLS INTERFACE ap_ovld register port=debug_out
//#pragma HLS INTERFACE axis register port=po_hls4ml_parallel_data
//#pragma HLS INTERFACE axis register port=pi_hls4ml_parallel_data
#pragma HLS INTERFACE ap_fifo port=po_hls4ml_parallel_data
#pragma HLS INTERFACE ap_fifo port=pi_hls4ml_parallel_data


  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS DATAFLOW

#ifndef __SYNTHESIS__
  assert(DOSA_WRAPPER_INPUT_IF_BITWIDTH >= DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH); //currently, the assumption is that one "pixel" is smaller than the interface bitwidth
  assert(HLS4ML_PARALLEL_INPUT_FRAME_BIT_CNT % 8 == 0); //currently, only byte-aligned FRAMES are supported
  //printf("cnn_input_frame_size: %d\n", cnn_input_frame_size);
  //printf("cnn_output_frame_size: %d\n", cnn_output_frame_size);
#endif

  //-- STATIC VARIABLES (with RESET) ------------------------------------------

  //-- STATIC DATAFLOW VARIABLES ------------------------------------------
#ifdef WRAPPER_TEST
  static stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> > sTohls4ml_parallelBuffer_chan1 ("sTohls4ml_parallelBuffer_chan1");
#pragma HLS STREAM variable=sTohls4ml_parallelBuffer_chan1   depth=cnn_input_frame_size
  static stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> > sTohls4ml_parallelBuffer_chan2 ("sTohls4ml_parallelBuffer_chan2");
#pragma HLS STREAM variable=sTohls4ml_parallelBuffer_chan2   depth=cnn_input_frame_size
  static stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> > sTohls4ml_parallelBuffer_chan3 ("sTohls4ml_parallelBuffer_chan3");
#pragma HLS STREAM variable=sTohls4ml_parallelBuffer_chan3   depth=cnn_input_frame_size
  static stream<ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> > sTohls4ml_parallelPixelChain_chan1 ("sTohls4ml_parallelPixelChain_chan1");
#pragma HLS STREAM variable=sTohls4ml_parallelPixelChain_chan1   depth=2*cnn_input_frame_size
  static stream<ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> > sTohls4ml_parallelPixelChain_chan2 ("sTohls4ml_parallelPixelChain_chan2");
#pragma HLS STREAM variable=sTohls4ml_parallelPixelChain_chan2   depth=2*cnn_input_frame_size
  static stream<ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> > sTohls4ml_parallelPixelChain_chan3 ("sTohls4ml_parallelPixelChain_chan3");
#pragma HLS STREAM variable=sTohls4ml_parallelPixelChain_chan3   depth=2*cnn_input_frame_size
  static stream<ap_uint<DOSA_HLS4ML_PARALLEL_FROMACCEL_BITWIDTH> > sFromhls4ml_parallelBuffer_chan1 ("sFromhls4ml_parallelBuffer_chan1");
#pragma HLS STREAM variable=sFromhls4ml_parallelBuffer_chan1 depth=cnn_output_frame_size
  static stream<ap_uint<DOSA_HLS4ML_PARALLEL_FROMACCEL_BITWIDTH> > sFromhls4ml_parallelBuffer_chan2 ("sFromhls4ml_parallelBuffer_chan2");
#pragma HLS STREAM variable=sFromhls4ml_parallelBuffer_chan2 depth=cnn_output_frame_size
  static stream<ap_uint<DOSA_HLS4ML_PARALLEL_FROMACCEL_BITWIDTH> > sFromhls4ml_parallelBuffer_chan3 ("sFromhls4ml_parallelBuffer_chan3");
#pragma HLS STREAM variable=sFromhls4ml_parallelBuffer_chan3 depth=cnn_output_frame_size
  static stream<ap_uint<DOSA_HLS4ML_PARALLEL_FROMACCEL_BITWIDTH> > sFromhls4ml_parallelBuffer_chan4 ("sFromhls4ml_parallelBuffer_chan4");
#pragma HLS STREAM variable=sFromhls4ml_parallelBuffer_chan4 depth=cnn_output_frame_size
  static stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> > sOutBuffer_chan1 ("sOutBuffer_chan1");
#pragma HLS STREAM variable=sOutBuffer_chan1 depth=cnn_output_frame_size
  static stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> > sOutBuffer_chan2 ("sOutBuffer_chan2");
#pragma HLS STREAM variable=sOutBuffer_chan2 depth=cnn_output_frame_size
  static stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> > sOutBuffer_chan3 ("sOutBuffer_chan3");
#pragma HLS STREAM variable=sOutBuffer_chan3 depth=cnn_output_frame_size
  static stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> > sOutBuffer_chan4 ("sOutBuffer_chan4");
#pragma HLS STREAM variable=sOutBuffer_chan4 depth=cnn_output_frame_size
#else
  //DOSA_ADD_hls4ml_parallel_buffer_instantiation
#endif
  static stream<bool> shls4ml_parallelUnitProcessing ("shls4ml_parallelUnitProcessing");
  const int hls4ml_parallel_processing_fifo_depth = (HLS4ML_PARALLEL_AVG_LAYER_LATENCY + 1) * DOSA_HLS4ML_PARALLEL_LAYER_CNT;
  #pragma HLS STREAM variable=shls4ml_parallelUnitProcessing depth=hls4ml_parallel_processing_fifo_depth
  static stream<ap_uint<DOSA_HLS4ML_PARALLEL_OUTPUT_BITDIWDTH> > sFromhls4ml_parallelBuffer ("sFromhls4ml_parallelBuffer");
  #pragma HLS STREAM variable=sFromhls4ml_parallelBuffer depth=2*hls4ml_parallel_processing_fifo_depth  //so that we can receive, what we send out...

  //-- LOCAL VARIABLES ------------------------------------------------------------
  uint16_t debug0 = 0;
  uint16_t debug1 = 0;
  uint16_t debug2 = 0;
  uint16_t debug3 = 0;
  //-- PROCESS INSTANTIATION ------------------------------------------------------

  // with the current strategy [INPUT] -> [CHANEL, with same bitwidth] -> [BATCH_PIXEL], we can maintain
  // ~ (DOSA_HLS4ML_PARALLEL_INPUT_CHAN_NUM * DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH)/DOSA_WRAPPER_INPUT_IF_BITWIDTH of the bandwidth
//  pTohls4ml_parallelEnq(
//#ifdef WRAPPER_TEST
//      sTohls4ml_parallelBuffer_chan1, sTohls4ml_parallelBuffer_chan2, sTohls4ml_parallelBuffer_chan3,
//#else
//      //DOSA_ ADD_tohls4ml_parallel_buffer_list
//#endif
//      siData, &debug0);


  pTohls4ml_parallelDemux(
#ifdef WRAPPER_TEST
      sTohls4ml_parallelBuffer_chan1, sTohls4ml_parallelBuffer_chan2, sTohls4ml_parallelBuffer_chan3,
#else
      //DOSA_ADD_tohls4ml_parallel_buffer_list
#endif
      siData, &debug0);


#ifdef WRAPPER_TEST
  pTohls4ml_parallelNarrow_1(sTohls4ml_parallelBuffer_chan1, sTohls4ml_parallelPixelChain_chan1);
  pTohls4ml_parallelNarrow_2(sTohls4ml_parallelBuffer_chan2, sTohls4ml_parallelPixelChain_chan2);
  pTohls4ml_parallelNarrow_3(sTohls4ml_parallelBuffer_chan3, sTohls4ml_parallelPixelChain_chan3);
#else
  //DOSA_ADD_pTohls4ml_parallelNarrow_X_instantiate
#endif

  pTohls4ml_parallelDeq(
#ifdef WRAPPER_TEST
      sTohls4ml_parallelPixelChain_chan1, sTohls4ml_parallelPixelChain_chan2, sTohls4ml_parallelPixelChain_chan3,
#else
      //DOSA_ADD_tohls4ml_parallel_pixelChain_list
#endif
      po_hls4ml_parallel_data,
      shls4ml_parallelUnitProcessing, &debug1);

  pFromhls4ml_parallelEnq(
      pi_hls4ml_parallel_data,
      shls4ml_parallelUnitProcessing, sFromhls4ml_parallelBuffer);

  pFromhls4ml_parallelFlatten(
#ifdef WRAPPER_TEST
      sFromhls4ml_parallelBuffer_chan1, sFromhls4ml_parallelBuffer_chan2, sFromhls4ml_parallelBuffer_chan3, sFromhls4ml_parallelBuffer_chan4,
#else
      //DOSA_ADD_from_hls4ml_parallel_stream_list
#endif
      sFromhls4ml_parallelBuffer, &debug2);

#ifdef WRAPPER_TEST
  pFromhls4ml_parallelWiden_1(sFromhls4ml_parallelBuffer_chan1, sOutBuffer_chan1);
  pFromhls4ml_parallelWiden_2(sFromhls4ml_parallelBuffer_chan2, sOutBuffer_chan2);
  pFromhls4ml_parallelWiden_3(sFromhls4ml_parallelBuffer_chan3, sOutBuffer_chan3);
  pFromhls4ml_parallelWiden_4(sFromhls4ml_parallelBuffer_chan4, sOutBuffer_chan4);
#else
  //DOSA_ADD_pFromhls4ml_parallelWiden_X_instantiate
#endif

  pFromhls4ml_parallelDeq(
#ifdef WRAPPER_TEST
      sOutBuffer_chan1, sOutBuffer_chan2, sOutBuffer_chan3, sOutBuffer_chan4,
#else
      //DOSA_ADD_from_hls4ml_parallel_out_list
#endif
      soData, &debug3);

  pMergeDebug(&debug0, &debug1, &debug2, &debug3, debug_out);

}



