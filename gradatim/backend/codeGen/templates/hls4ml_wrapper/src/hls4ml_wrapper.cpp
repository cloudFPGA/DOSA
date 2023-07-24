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
//  *     Created: Mar 2022
//  *     Authors: NGL
//  *
//  *     Description:
//  *        Template for a hls4ml wrapper
//  *

#include <stdio.h>
#include <stdint.h>
#include "ap_int.h"
#include "ap_utils.h"
#include <hls_stream.h>
#include <cassert>

#include "hls4ml_wrapper.hpp"

using namespace hls;



//ap_uint<64> flattenAxisBuffer(
//    Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> &in_read,
//    ap_uint<2*DOSA_WRAPPER_INPUT_IF_BITWIDTH>  &combined_input,
//    ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH>  &hangover_bits,
//    ap_uint<64>                            &hangover_bits_valid_bits
//    )
//{
//#pragma HLS INLINE
//  ap_uint<64> cur_line_bit_cnt = 0;
//  // consider hangover
//  if(hangover_bits_valid_bits > 0)
//  {
//    combined_input = hangover_bits;
//    hangover_bits = 0x0;
//    cur_line_bit_cnt = hangover_bits_valid_bits;
//    hangover_bits_valid_bits = 0;
//  }
//
//  for(int i = 0; i < WRAPPER_INPUT_IF_BYTES; i++)
//  {
//#pragma HLS unroll
//    //TODO: should not make a difference, since we count them?
//    //if((in_read.getTKeep() >> i) == 0)
//    //{
//    //  printf("flatten buffer: skipped due to tkeep\n");
//    //  continue;
//    //}
//    //TODO: what if input is not byte aligned?
//    ap_uint<8> current_byte = (ap_uint<8>) (((ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH>) in_read.getTData()) >> (i*8));
//    combined_input |= ((ap_uint<2*DOSA_WRAPPER_INPUT_IF_BITWIDTH>) current_byte) << cur_line_bit_cnt;
//    //printf("flatten buffer: read 0x%16llx %16llx, bit cnt: %d\n", (uint64_t) (combined_input >> 64), (uint64_t) combined_input, (uint32_t) cur_line_bit_cnt);
//    cur_line_bit_cnt += 8;
//  }
//
//  printf("flatten buffer: read 0x%16llx %16llx, ret bit cnt: %d\n", (uint64_t) (combined_input >> 64), (uint64_t) combined_input, (uint32_t) cur_line_bit_cnt);
//  return cur_line_bit_cnt;
//}


ap_uint<64> flattenAxisBuffer(
    Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> &in_read,
    ap_uint<2*DOSA_WRAPPER_INPUT_IF_BITWIDTH>  &combined_input,
    ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH>  &hangover_bits,
    ap_uint<64>                            &hangover_bits_valid_bits
    )
{
#pragma HLS INLINE
  //ap_uint<64> cur_line_bit_cnt = 0;
  // consider hangover
  if(hangover_bits_valid_bits > 0)
  {
    combined_input = hangover_bits;
    hangover_bits = 0x0;
    //printf("flatten: used hangover %d bits: 0x%16llx\n", (uint64_t) hangover_bits_valid_bits, (uint64_t) combined_input);
    //cur_line_bit_cnt = hangover_bits_valid_bits;
    //hangover_bits_valid_bits = 0;
  }

  ap_uint<2*DOSA_WRAPPER_INPUT_IF_BITWIDTH> tmp_input = 0x0;
  for(int i = 0; i < WRAPPER_INPUT_IF_BYTES; i++)
  {
#pragma HLS unroll
    //TODO: should not make a difference, since we count them?
    //if((in_read.getTKeep() >> i) == 0)
    //{
    //  printf("flatten buffer: skipped due to tkeep\n");
    //  continue;
    //}
    //TODO: what if input is not byte aligned?
    ap_uint<8> current_byte = (ap_uint<8>) (((ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH>) in_read.getTData()) >> (i*8));
    //combined_input |= ((ap_uint<2*DOSA_WRAPPER_INPUT_IF_BITWIDTH>) current_byte) << cur_line_bit_cnt;
    tmp_input |= ((ap_uint<2*DOSA_WRAPPER_INPUT_IF_BITWIDTH>) current_byte) << (i*8);
    //printf("flatten buffer: read 0x%16llx %16llx, bit cnt: %d\n", (uint64_t) (combined_input >> 64), (uint64_t) combined_input, (uint32_t) cur_line_bit_cnt);
    //cur_line_bit_cnt += 8;
  }
  //printf("flatten buffer: tmp input 0x%16llx 0x%2.2X\n", (uint64_t) tmp_input, (uint32_t) in_read.getTKeep());
  combined_input |= (ap_uint<2*DOSA_WRAPPER_INPUT_IF_BITWIDTH>) (tmp_input << hangover_bits_valid_bits);
  ap_uint<64> cur_line_bit_cnt = extractByteCnt(in_read)*8 + hangover_bits_valid_bits;
  hangover_bits_valid_bits = 0;
  printf("flatten buffer: read 0x%16llx %16llx, bit cnt: %d\n", (uint64_t) (combined_input >> 64), (uint64_t) combined_input, (uint32_t) cur_line_bit_cnt);

  //ap_uint<2*DOSA_WRAPPER_INPUT_IF_BITWIDTH> tmp_input = 0x0;
  //for(int i = 0; i < DOSA_WRAPPER_INPUT_IF_BITWIDTH; i++)
  //{
//  #pragma HLS unroll
  //  if(i > in_read.getTKeep())
  //  {
  //    printf("flatten buffer: skipped due to tkeep\n");
  //    continue;
  //  }
  //  //TODO: what if input is not byte aligned?
  //  ap_uint<1> current_byte = (ap_uint<1>) (((ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH>) in_read.getTData()) >> i);
  //  //combined_input |= ((ap_uint<2*DOSA_WRAPPER_INPUT_IF_BITWIDTH>) current_byte) << cur_line_bit_cnt;
  //  tmp_input |= ((ap_uint<2*DOSA_WRAPPER_INPUT_IF_BITWIDTH>) current_byte) << i;
  //  //printf("flatten buffer: read 0x%16llx %16llx, bit cnt: %d\n", (uint64_t) (combined_input >> 64), (uint64_t) combined_input, (uint32_t) cur_line_bit_cnt);
  //  //cur_line_bit_cnt += 8;
  //}
  //combined_input |= tmp_input << hangover_bits_valid_bits;
  //ap_uint<64> cur_line_bit_cnt = in_read.getTKeep() + hangover_bits_valid_bits;

  //printf("flatten buffer: read 0x%16llx %16llx, ret bit cnt: %d\n", (uint64_t) (combined_input >> 64), (uint64_t) combined_input, (uint32_t) cur_line_bit_cnt);
  return cur_line_bit_cnt;
}


void pToAcc(
    stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >   &siData,
    stream<ap_uint<DOSA_HLS4ML_INPUT_BITWIDTH> >    &soToHls4mlData,
    ap_uint<32>                                     *debug_out
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static twoStatesFSM toAccFsm = RESET;
#pragma HLS reset variable=toAccFsm
  static ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH> hangover_bits;
#pragma HLS reset variable=hangover_bits
  static ap_uint<64> hangover_bits_valid_bits;
#pragma HLS reset variable=hangover_bits_valid_bits
  static bool only_hangover_processing = false;
#pragma HLS reset variable=only_hangover_processing
  //-- LOCAL VARIABLES ------------------------------------------------------

  ap_uint<2*DOSA_WRAPPER_INPUT_IF_BITWIDTH> combined_input;
  ap_uint<64> cur_line_bit_cnt;
  Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> tmp_read_0;

  //if-else better than 2-state switch?
  if(toAccFsm == RESET)
  {
    hangover_bits = 0x0;
    hangover_bits_valid_bits = 0x0;
    only_hangover_processing = false;
    bool one_not_empty = false;
    if( !siData.empty() )
    {
      siData.read();
      one_not_empty = true;
    }
    if( !one_not_empty )
    {
      toAccFsm = FORWARD;
    }
  } else {
    if( (!siData.empty() || only_hangover_processing) && !soToHls4mlData.full() )
    {
      combined_input = 0x0;
      cur_line_bit_cnt = 0x0;

      if( !only_hangover_processing && !siData.empty())
      {
        tmp_read_0 = siData.read();
        cur_line_bit_cnt = flattenAxisBuffer(tmp_read_0, combined_input, hangover_bits, hangover_bits_valid_bits);
      } else {
        //need to reduce backlog
        combined_input = hangover_bits;
        cur_line_bit_cnt = hangover_bits_valid_bits;
        hangover_bits = 0x0;
        hangover_bits_valid_bits = 0x0;
      }

      only_hangover_processing = false;
      ap_uint<DOSA_HLS4ML_INPUT_BITWIDTH> output_data = (ap_uint<DOSA_HLS4ML_INPUT_BITWIDTH>) combined_input;
      hangover_bits_valid_bits = cur_line_bit_cnt - DOSA_HLS4ML_INPUT_BITWIDTH;
      hangover_bits = (ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH>) (combined_input >> DOSA_HLS4ML_INPUT_BITWIDTH);
      printf("pToAcc: combined %16.16llx, hangover_bits_valid_bits: %d, cur_line_bit_cnt: %d\n", (uint64_t) combined_input, (uint64_t) hangover_bits_valid_bits, (uint32_t) cur_line_bit_cnt);
      if(hangover_bits_valid_bits >= DOSA_HLS4ML_INPUT_BITWIDTH)
      {
        only_hangover_processing = true;
      }

      printf("pToAcc: write 0x%2.2X\n", (uint8_t) output_data);
      soToHls4mlData.write(output_data);
    }
  }

  //debugging?
  *debug_out = 0x0;

}


void pFromAcc(
    stream<ap_uint<DOSA_HLS4ML_OUTPUT_BITWIDTH> >   &siFromHls4mlData,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >  &soData
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static twoStatesFSM fromAccFSM = RESET;
#pragma HLS reset variable=fromAccFSM
  static ap_uint<64> current_frame_bit_cnt = 0x0;
#pragma HLS reset variable=current_frame_bit_cnt
  static ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> hangover_store = 0x0;
#pragma HLS reset variable=hangover_store
  static ap_uint<64> hangover_store_valid_bits = 0x0;
#pragma HLS reset variable=hangover_store_valid_bits
  //-- LOCAL VARIABLES ------------------------------------------------------
  ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+DOSA_HLS4ML_OUTPUT_BITWIDTH> combined_output = 0x0;

  //if-else better than 2-state switch?
  if(fromAccFSM == RESET)
  {
    current_frame_bit_cnt = 0x0;
    hangover_store = 0x0;
    hangover_store_valid_bits = 0x0;
    bool one_not_empty = false;
    if( !siFromHls4mlData.empty() )
    {
      siFromHls4mlData.read();
      one_not_empty = true;
    }
    if( !one_not_empty )
    {
      fromAccFSM = FORWARD;
    }
  } else {
    if( !soData.full() && (!siFromHls4mlData.empty() || hangover_store_valid_bits > 0) )
    {
      combined_output = hangover_store;
      //uint8_t tkeep_offset_hangover = 0x0;
      //uint32_t cur_line_bit_cnt = 0x0;
      //for(int k = 0; k < (DOSA_HLS4ML_OUTPUT_BITWIDTH+7)/8; k++)
      //{
      //  if(hangover_store_valid_bits > k*8)
      //  {
      //    tkeep |= ((ap_uint<(DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+7)/8>) 0b1) << k;
      //    tkeep_offset_hangover++;
      //  } else {
      //    break;
      //  }
      //}
      //uint32_t bits_read = 0x0;

      ap_uint<DOSA_HLS4ML_OUTPUT_BITWIDTH> nv = 0x0;
      //if(cur_line_bit_cnt < DOSA_WRAPPER_OUTPUT_IF_BITWIDTH && !siFromHls4mlData.empty())
      //{
      nv = siFromHls4mlData.read();
      //bits_read += DOSA_HLS4ML_OUTPUT_BITWIDTH;
      //}
      combined_output |= ((ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+DOSA_HLS4ML_OUTPUT_BITWIDTH>) nv) << (hangover_store_valid_bits);

      current_frame_bit_cnt += DOSA_HLS4ML_OUTPUT_BITWIDTH;
      if( (DOSA_HLS4ML_OUTPUT_BITWIDTH < DOSA_WRAPPER_OUTPUT_IF_BITWIDTH)
          && ((hangover_store_valid_bits + DOSA_HLS4ML_OUTPUT_BITWIDTH) < DOSA_WRAPPER_OUTPUT_IF_BITWIDTH)
          && (current_frame_bit_cnt < HLS4ML_OUTPUT_FRAME_BIT_CNT)
        )
      {//not a full line yet, and not end of frame
        hangover_store = combined_output;
        hangover_store_valid_bits += DOSA_HLS4ML_OUTPUT_BITWIDTH;
        //current_frame_bit_cnt += DOSA_HLS4ML_OUTPUT_BITWIDTH;
      } else {
        //full line
        ap_uint<(DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+7)/8> tkeep = 0x0;
        uint32_t total_bit_cnt = hangover_store_valid_bits + DOSA_HLS4ML_OUTPUT_BITWIDTH;
        //printf("total_bit_cnt: %d\n", total_bit_cnt);
        //for(int k = 0; k < DOSA_WRAPPER_OUTPUT_IF_BITWIDTH/8; k++)
        //{
        //  if(k*8 < total_bit_cnt)
        //  {
        //    tkeep |= ((ap_uint<(DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+7)/8>) 0b1) << k;
        //  }
        //}
        //tkeep = (ap_uint<(DOSA_HLS4ML_OUTPUT_BITWIDTH+7)/8>) byteCntToTKeep((uint8_t) (bits_read + hangover_store_valid_bits));
        //TODO: make data type dynamic
        tkeep = byteCntToTKeep((uint8_t) (total_bit_cnt/8));

        hangover_store = (ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH>) (combined_output >> DOSA_WRAPPER_OUTPUT_IF_BITWIDTH);
        if(total_bit_cnt > DOSA_WRAPPER_OUTPUT_IF_BITWIDTH)
        {
          hangover_store_valid_bits -= DOSA_HLS4ML_OUTPUT_BITWIDTH;
        } else {
          hangover_store_valid_bits = 0x0;
        }
        //current_frame_bit_cnt += total_bit_cnt - hangover_store_valid_bits;
        printf("\tpFromAcc: combined %16.16llx, hangover_bits_valid_bits: %d, current_frame_bit_cnt: %d\n", (uint64_t) combined_output, (uint64_t) hangover_store_valid_bits, (uint32_t) current_frame_bit_cnt);

        ap_uint<1> tlast = 0;
        if(current_frame_bit_cnt >= HLS4ML_OUTPUT_FRAME_BIT_CNT)
        {
          current_frame_bit_cnt -= HLS4ML_OUTPUT_FRAME_BIT_CNT;
          tlast = 0b1; //TODO: does anybody care?
        }
        Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> tmp_write_0 = Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH>((ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH>) combined_output, tkeep, tlast);
        soData.write(tmp_write_0);
        printf("\tpFromAcc: write Axis tdata: %16.16llx, tkeep: %2.2x, tlast: %x;\n", (uint64_t) tmp_write_0.getTData(), (uint8_t) tmp_write_0.getTKeep(), (uint8_t) tmp_write_0.getTLast());
      }
    }
  }

}



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
    )
{
  //-- DIRECTIVES FOR THE BLOCK ---------------------------------------------
#pragma HLS INTERFACE ap_ctrl_none port=return

#pragma HLS INTERFACE ap_fifo port=siData
#pragma HLS INTERFACE ap_fifo port=soData

#pragma HLS INTERFACE axis port=soToHls4mlData
#pragma HLS INTERFACE axis port=siFromHls4mlData
#pragma HLS INTERFACE ap_ovld register port=debug_out


  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS DATAFLOW

#ifndef __SYNTHESIS__
  assert(HLS4ML_INPUT_FRAME_BIT_CNT % 8 == 0); //currently, only byte-aligned FRAMES are supported
  assert(HLS4ML_OUTPUT_FRAME_BIT_CNT % 8 == 0); //currently, only byte-aligned FRAMES are supported
  assert(DOSA_WRAPPER_OUTPUT_IF_BITWIDTH % 8 == 0);
  assert(DOSA_WRAPPER_INPUT_IF_BITWIDTH % 8 == 0);
  //printf("cnn_input_frame_size: %d\n", cnn_input_frame_size);
  //printf("cnn_output_frame_size: %d\n", cnn_output_frame_size);
#endif

  //-- STATIC VARIABLES (with RESET) ------------------------------------------

  //-- STATIC DATAFLOW VARIABLES ------------------------------------------

  //-- DATAFLOW PROCESS ---------------------------------------------------

  pToAcc(siData, soToHls4mlData, debug_out);

  pFromAcc(siFromHls4mlData, soData);

}



