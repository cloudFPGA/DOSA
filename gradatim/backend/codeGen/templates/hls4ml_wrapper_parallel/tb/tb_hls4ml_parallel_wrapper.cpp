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
//  *     Created: Jan 2022
//  *     Authors: NGL
//  *
//  *     Description:
//  *        Template for a hls4ml_parallel2 wrapper
//  *

#include <stdio.h>
#include <hls_stream.h>
#include <cassert>

#include "../src/hls4ml_parallel_wrapper.hpp"

using namespace std;


#define OK          true
#define KO          false
#define VALID       true
#define UNVALID     false
#define DEBUG_TRACE true

#define ENABLED     (ap_uint<1>)1
#define DISABLED    (ap_uint<1>)0

//------------------------------------------------------
//-- DUT INTERFACES AS GLOBAL VARIABLES
//------------------------------------------------------
// ----- Wrapper Interface -----
stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >   siData("siData");
stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >  soData("soData");
// ----- hls4ml_parallel Interface -----
//ap_uint<1>                                *po_hls4ml_parallel_data_valid,
//ap_uint<DOSA_HLS4ML_PARALLEL_INPUT_BITDIWDTH>      *po_hls4ml_parallel_data_vector,
ap_uint<1>                                po_hls4ml_parallel_frame_valid = 0;
stream<ap_uint<DOSA_HLS4ML_PARALLEL_INPUT_BITDIWDTH> > po_hls4ml_parallel_data ("po_hls4ml_parallel_data");
//ap_uint<1>                                *pi_hls4ml_parallel_data_valid,
//ap_uint<DOSA_HLS4ML_PARALLEL_OUTPUT_BITDIWDTH>     *pi_hls4ml_parallel_data_vector,
ap_uint<1>                                pi_hls4ml_parallel_frame_valid = 0;
stream<ap_uint<DOSA_HLS4ML_PARALLEL_OUTPUT_BITDIWDTH> > pi_hls4ml_parallel_data ("pi_hls4ml_parallel_data");
//// ----- hls4ml_parallel Interface -----
//ap_uint<1>                                po_hls4ml_parallel_data_valid = 0;
//ap_uint<1>                                po_hls4ml_parallel_frame_valid = 0;
//ap_uint<DOSA_HLS4ML_PARALLEL_INPUT_BITDIWDTH>      po_hls4ml_parallel_data_vector = 0;
//ap_uint<1>                                pi_hls4ml_parallel_data_valid = 0;
//ap_uint<1>                                pi_hls4ml_parallel_frame_valid = 0;
//ap_uint<DOSA_HLS4ML_PARALLEL_OUTPUT_BITDIWDTH>     pi_hls4ml_parallel_data_vector = 0;
// ----- DEBUG IO ------
ap_uint<64> debug_out = 0;


//------------------------------------------------------
//-- TESTBENCH GLOBAL VARIABLES
//------------------------------------------------------
int         simCnt;
stream<ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH*DOSA_HLS4ML_PARALLEL_INPUT_CHAN_NUM> > shls4ml_parallelProcessing("shls4ml_parallelProcessing");
int hls4ml_parallel_wait_processing = 0;
//int ignore_hls4ml_parallel_value_cnt = 0;
#define TB_HLS4ML_PARALLEL_STORE_VALUE_PROP (((DOSA_HLS4ML_PARALLEL_INPUT_FRAME_WIDTH*DOSA_HLS4ML_PARALLEL_INPUT_FRAME_WIDTH)/(DOSA_HLS4ML_PARALLEL_OUTPUT_FRAME_WIDTH*DOSA_HLS4ML_PARALLEL_OUTPUT_FRAME_WIDTH))-2)
//#define TB_HLS4ML_PARALLEL_LATENCY (2*TB_HLS4ML_PARALLEL_STORE_VALUE_PROP)
#define TB_HLS4ML_PARALLEL_LATENCY 23

#define TB_PER_FRAME_FILTER_CNT 25
int forwarded_this_frame = 0;
int frame_pixel_cnt = 0;

/*****************************************************************************
 * @brief Run a single iteration of the DUT model.
 * @ingroup udp_app_flash
 * @return Nothing.
 ******************************************************************************/
void stepDut() {
  hls4ml_parallel_wrapper_test(
      siData, soData,
      //&po_hls4ml_parallel_data_valid, &po_hls4ml_parallel_frame_valid, &po_hls4ml_parallel_data_vector,
      &po_hls4ml_parallel_frame_valid, po_hls4ml_parallel_data,
      //&pi_hls4ml_parallel_data_valid, &pi_hls4ml_parallel_frame_valid, &pi_hls4ml_parallel_data_vector,
      &pi_hls4ml_parallel_frame_valid, pi_hls4ml_parallel_data,
      &debug_out);
    simCnt++;
    printf("[%4.4d] STEP DUT \n", simCnt);
}

void phls4ml_parallel() {
  //pi_hls4ml_parallel_data_valid = 0;
  pi_hls4ml_parallel_frame_valid = 0;
  //if(po_hls4ml_parallel_data_valid == 1 && po_hls4ml_parallel_frame_valid == 1)
  if(!po_hls4ml_parallel_data.empty() && po_hls4ml_parallel_frame_valid == 1)
  {
    ap_uint<DOSA_HLS4ML_PARALLEL_INPUT_BITDIWDTH>      po_hls4ml_parallel_data_vector = po_hls4ml_parallel_data.read();
    //if(ignore_hls4ml_parallel_value_cnt >= TB_HLS4ML_PARALLEL_STORE_VALUE_PROP)
    if(forwarded_this_frame < TB_PER_FRAME_FILTER_CNT)
    {
      printf("\t\t\t\t\t\t\t\t\t\t\t\tphls4ml_parallel: input %8.8X is forwareded.\n", (uint32_t) po_hls4ml_parallel_data_vector);
      shls4ml_parallelProcessing.write(po_hls4ml_parallel_data_vector);
      forwarded_this_frame++;
      //ignore_hls4ml_parallel_value_cnt = 0;
    } //else {
      //ignore_hls4ml_parallel_value_cnt++;
    //}
    frame_pixel_cnt++;
    if(frame_pixel_cnt > (DOSA_HLS4ML_PARALLEL_INPUT_FRAME_WIDTH*DOSA_HLS4ML_PARALLEL_INPUT_FRAME_WIDTH))
    {
      frame_pixel_cnt = 0;
      forwarded_this_frame = 0;
    }

    if(hls4ml_parallel_wait_processing <= TB_HLS4ML_PARALLEL_LATENCY)
    {
      hls4ml_parallel_wait_processing++;
    }
  }
  if(hls4ml_parallel_wait_processing == TB_HLS4ML_PARALLEL_LATENCY)
  {
    //adding ignore value
    for(int i = 0; i < (DOSA_HLS4ML_PARALLEL_VALID_WAIT_CNT-TB_HLS4ML_PARALLEL_LATENCY); i++)
    {
      pi_hls4ml_parallel_data.write(0xBEEF);
    }
    hls4ml_parallel_wait_processing++;
  }
  if(hls4ml_parallel_wait_processing >= TB_HLS4ML_PARALLEL_LATENCY)
  {
    if(!shls4ml_parallelProcessing.empty())
    {
      ap_uint<DOSA_HLS4ML_PARALLEL_OUTPUT_BITDIWDTH> tmp_out = shls4ml_parallelProcessing.read();
      tmp_out |= ((ap_uint<DOSA_HLS4ML_PARALLEL_OUTPUT_BITDIWDTH>) 0xBB) << DOSA_HLS4ML_PARALLEL_INPUT_BITDIWDTH;
      //pi_hls4ml_parallel_data_valid = 1;
      pi_hls4ml_parallel_frame_valid = 1;
      //pi_hls4ml_parallel_data_vector = tmp_out;
      pi_hls4ml_parallel_data.write(tmp_out);
    }
  }
}


int main() {

#ifndef WRAPPER_TEST
  printf("ERROR: This testbench works only with flag 'WRAPPER_TEST'. STOP.\n");
  exit(-1);
#endif

  //------------------------------------------------------
  //-- TESTBENCH LOCAL VARIABLES
  //------------------------------------------------------
  int         nrErr = 0;

  printf("#####################################################\n");
  printf("## TESTBENCH STARTS HERE                           ##\n");
  printf("#####################################################\n");

  //assert(DOSA_HLS4ML_PARALLEL_INPUT_CHAN_NUM == DOSA_HLS4ML_PARALLEL_OUTPUT_CHAN_NUM);
  //assert(DOSA_HLS4ML_PARALLEL_INPUT_BITDIWDTH == DOSA_HLS4ML_PARALLEL_OUTPUT_BITDIWDTH);
  assert(TB_HLS4ML_PARALLEL_STORE_VALUE_PROP > 1);

    // 01010101...
    // 020202....
    // ....
    // 0a0a0a0a0a
    // 111111......
    // 1212....
    // ....
    // ...
    // 212121...
    // ...
    // 2a2a...
    //siData.write(Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH>(0x010101010101010101,0xFF,0b0));
    //siData.write(Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH>(0x010102020202020202,0xFF,0b0));

    stream<ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH>> sTmpIn("sTmpIn");
    stream<ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH>> sGoldenOut("sGoldenOut");
    //int filter_out_value_cnt = 0;
    int golden_bytes_written = 0;
    int bytes_written_this_frame = 0;
    for(int c = 0; c < DOSA_HLS4ML_PARALLEL_INPUT_CHAN_NUM; c++)
    {
      bytes_written_this_frame = 0;
      for(int f = 0; f < DOSA_HLS4ML_PARALLEL_INPUT_FRAME_WIDTH; f++)
      {
        for(int l = 0; l < DOSA_HLS4ML_PARALLEL_INPUT_FRAME_WIDTH; l++)
        {
          ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> cur_word = ((ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH>) c) << (DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH-4);
          cur_word |= (ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH>) f;
          sTmpIn.write(cur_word);
          //printf("write: %2.2X\n", (uint32_t) cur_word);
          //if(filter_out_value_cnt >= TB_HLS4ML_PARALLEL_STORE_VALUE_PROP)
          if(bytes_written_this_frame < TB_PER_FRAME_FILTER_CNT)
          {
            sGoldenOut.write(cur_word);
            //printf("write golden: %2.2X\n", (uint32_t) cur_word);
            golden_bytes_written++;
            bytes_written_this_frame++;
            //filter_out_value_cnt = 0;
          } //else {
          //  filter_out_value_cnt++;
          //}
        }
      }
    }
    //adding additional out channel
    for(int c = 0; c < (DOSA_HLS4ML_PARALLEL_OUTPUT_CHAN_NUM - DOSA_HLS4ML_PARALLEL_INPUT_CHAN_NUM); c++)
    {
      bytes_written_this_frame = 0;
      for(int f = 0; f < DOSA_HLS4ML_PARALLEL_INPUT_FRAME_WIDTH; f++)
      {
        for(int l = 0; l < DOSA_HLS4ML_PARALLEL_INPUT_FRAME_WIDTH; l++)
        {
          if(bytes_written_this_frame < TB_PER_FRAME_FILTER_CNT)
          {
            sGoldenOut.write(0xBB);
            //printf("write golden: %2.2X\n", (uint32_t) 0xBB);
            golden_bytes_written++;
            bytes_written_this_frame++;
          }
        }
      }
    }


    assert((DOSA_WRAPPER_INPUT_IF_BITWIDTH % DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH) == 0);
    while(!sTmpIn.empty())
    {
      ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH> cur_inp = 0x0;
      for(int i = 0; i < (DOSA_WRAPPER_INPUT_IF_BITWIDTH/DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH); i++)
      {
        ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH> nw = (ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH>) sTmpIn.read();
        cur_inp |= (ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH>) (nw << i*DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH);
      }
      ap_uint<1> tlast = int(sTmpIn.empty()); //in a testbench, that is ok...
      siData.write(Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH>(cur_inp ,0xFFFF,tlast));
      printf("[TB] filling input FIFO: 0x%16.16llx\n", (uint64_t) cur_inp);
    }
    printf("[TB] (4 empty reads expected)\n");

    simCnt = 0;
    nrErr  = 0;

    //------------------------------------------------------
    //-- STEP-2 : MAIN TRAFFIC LOOP
    //------------------------------------------------------
    while (!nrErr) {

      if (simCnt < 160)
      {
        stepDut();
        phls4ml_parallel();

      } else {
        printf("## End of simulation at cycle=%3d. \n", simCnt);
        break;
      }

    }  // End: while()

    //------------------------------------------------------
    //-- STEP-3 : COMPARE INPUT AND OUTPUT STREAMS
    //------------------------------------------------------
    assert(DOSA_WRAPPER_INPUT_IF_BITWIDTH == DOSA_WRAPPER_OUTPUT_IF_BITWIDTH);
    int received_words = 0;
    while(!soData.empty())
    {
      Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> cur_read = soData.read();
      ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> cur_out = cur_read.getTData();
      for(int i = 0; i<(DOSA_WRAPPER_OUTPUT_IF_BITWIDTH/DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH); i++)
      {
        if((cur_read.getTKeep() >> i) == 0)
        {
          continue;
        }
        ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> cur_word = (ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH>) (cur_out >> (i*DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH));
        ap_uint<DOSA_HLS4ML_PARALLEL_GENERAL_BITWIDTH> expected_word = sGoldenOut.read();
        if(cur_word != expected_word)
        {
          nrErr++;
          printf("ERROR: Expected %2.2x, but got %2.2x, at position %d.\n", (uint8_t) expected_word, (uint8_t) cur_word, received_words);
        }
        received_words++;
      }
    }
    if(received_words != golden_bytes_written)
    {
      nrErr++;
      printf("ERROR: Received %d words, but expected %d.\n", received_words, golden_bytes_written);
    }

    printf("#####################################################\n");
    if (nrErr)
        printf("## ERROR - TESTBENCH FAILED (RC=%d) !!!             ##\n", nrErr);
    else
        printf("## SUCCESSFULL END OF TESTBENCH (RC=0)             ##\n");

    printf("#####################################################\n");

    return(nrErr);
}
