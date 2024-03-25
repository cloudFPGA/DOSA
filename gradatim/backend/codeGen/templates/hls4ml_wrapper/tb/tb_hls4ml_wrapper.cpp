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
//  *        Template for a Haddoc2 wrapper
//  *

#include <stdio.h>
#include <hls_stream.h>
#include <cassert>

#include "../src/hls4ml_wrapper.hpp"

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
// ----- hls4ml interface -----
stream<ap_uint<DOSA_HLS4ML_INPUT_BITWIDTH> >    soToHls4mlData("soToHls4mlData");
stream<ap_uint<DOSA_HLS4ML_OUTPUT_BITWIDTH> >   siFromHls4mlData("siFromHls4mlData");
// ----- DEBUG IO ------
ap_uint<32> debug_out = 0;

#define DOSA_HLS4ML_INPUT_CHAN_NUM 3
#define DOSA_HLS4ML_OUTPUT_CHAN_NUM 4
#define DOSA_HLS4ML_INPUT_FRAME_WIDTH 10

//------------------------------------------------------
//-- TESTBENCH GLOBAL VARIABLES
//------------------------------------------------------
int         simCnt;
stream<ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> > sHls4mlProcessing("sHls4mlProcessing");

#define TB_PER_FRAME_FILTER_CNT 25
int forwarded_this_frame = 0;
int frame_pixel_cnt = 0;
int frame_num_forwarded = 0;

/*****************************************************************************
 * @brief Run a single iteration of the DUT model.
 * @ingroup udp_app_flash
 * @return Nothing.
 ******************************************************************************/
void stepDut() {
  printf("----- [%4.4d] STEP DUT start ----\n", simCnt);
  hls4ml_wrapper_test(
      siData, soData,
      soToHls4mlData, siFromHls4mlData,
      &debug_out);
  printf("----- [%4.4d] STEP DUT done -----\n", simCnt);
  simCnt++;
}

void pHls4ml() {
  if(!soToHls4mlData.empty() && !siFromHls4mlData.full() )
  {
    ap_uint<DOSA_HLS4ML_OUTPUT_BITWIDTH> cur_data = soToHls4mlData.read();
    if(forwarded_this_frame < TB_PER_FRAME_FILTER_CNT)
    {
      printf("\t\t\t\t\t\t\t\t\t\t\t\tpHls4ml: input %2.2X is forwareded.\n", (uint8_t) cur_data);
      siFromHls4mlData.write(cur_data);
      forwarded_this_frame++;
    }
    frame_pixel_cnt++;
    if(frame_pixel_cnt >= CNN_INPUT_FRAME_SIZE)
    {
      frame_pixel_cnt = 0;
      forwarded_this_frame = 0;
      frame_num_forwarded++;
    }
  }
  //add additional out channel
  if(frame_num_forwarded == 3)
  {
    int bytes_written_this_frame = 0;
    for(int c = 0; c < (DOSA_HLS4ML_OUTPUT_CHAN_NUM - DOSA_HLS4ML_INPUT_CHAN_NUM); c++)
    {
      bytes_written_this_frame = 0;
      for(int f = 0; f < DOSA_HLS4ML_INPUT_FRAME_WIDTH; f++)
      {
        for(int l = 0; l < DOSA_HLS4ML_INPUT_FRAME_WIDTH; l++)
        {
          if(bytes_written_this_frame < TB_PER_FRAME_FILTER_CNT)
          {
            siFromHls4mlData.write(0xBB);
            bytes_written_this_frame++;
            printf("\t\t\t\t\t\t\t\t\t\t\t\tpHls4ml: input %2.2X is added.\n", (uint8_t) 0xBB);
          }
        }
      }
    }
    frame_num_forwarded++;
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

  assert(DOSA_WRAPPER_INPUT_IF_BITWIDTH == DOSA_WRAPPER_OUTPUT_IF_BITWIDTH);
  assert(DOSA_HLS4ML_OUTPUT_BITWIDTH == DOSA_HLS4ML_INPUT_BITWIDTH);
  //------------------------------------------------------
  //-- Process Reset
  //------------------------------------------------------

  simCnt = 0;
  nrErr  = 0;
  stepDut();

  //------------------------------------------------------
  //-- Fill Streams
  //------------------------------------------------------

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

    stream<ap_uint<DOSA_HLS4ML_OUTPUT_BITWIDTH>> sTmpIn("sTmpIn");
    stream<ap_uint<DOSA_HLS4ML_OUTPUT_BITWIDTH>> sGoldenOut("sGoldenOut");
    //int filter_out_value_cnt = 0;
    int golden_bytes_written = 0;
    int bytes_written_this_frame = 0;
    for(int c = 0; c < DOSA_HLS4ML_INPUT_CHAN_NUM; c++)
    {
      bytes_written_this_frame = 0;
      for(int f = 0; f < DOSA_HLS4ML_INPUT_FRAME_WIDTH; f++)
      {
        for(int l = 0; l < DOSA_HLS4ML_INPUT_FRAME_WIDTH; l++)
        {
          ap_uint<DOSA_HLS4ML_OUTPUT_BITWIDTH> cur_word = ((ap_uint<DOSA_HLS4ML_OUTPUT_BITWIDTH>) c) << (DOSA_HLS4ML_OUTPUT_BITWIDTH-4);
          cur_word |= (ap_uint<DOSA_HLS4ML_OUTPUT_BITWIDTH>) f;
          sTmpIn.write(cur_word);
          //printf("write: %2.2X\n", (uint32_t) cur_word);
          //if(filter_out_value_cnt >= TB_HADDOC_STORE_VALUE_PROP)
          if(bytes_written_this_frame < TB_PER_FRAME_FILTER_CNT)
          {
            sGoldenOut.write(cur_word);
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
    for(int c = 0; c < (DOSA_HLS4ML_OUTPUT_CHAN_NUM - DOSA_HLS4ML_INPUT_CHAN_NUM); c++)
    {
      bytes_written_this_frame = 0;
      for(int f = 0; f < DOSA_HLS4ML_INPUT_FRAME_WIDTH; f++)
      {
        for(int l = 0; l < DOSA_HLS4ML_INPUT_FRAME_WIDTH; l++)
        {
          if(bytes_written_this_frame < TB_PER_FRAME_FILTER_CNT)
          {
            sGoldenOut.write(0xBB);
            golden_bytes_written++;
            bytes_written_this_frame++;
          }
        }
      }
    }


    assert((DOSA_WRAPPER_INPUT_IF_BITWIDTH % DOSA_HLS4ML_OUTPUT_BITWIDTH) == 0);
    while(!sTmpIn.empty())
    {
      ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH> cur_inp = 0x0;
      ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH/8> tkeep = 0x0;
      for(int i = 0; i < (DOSA_WRAPPER_INPUT_IF_BITWIDTH/DOSA_HLS4ML_OUTPUT_BITWIDTH); i++)
      {
        if(sTmpIn.empty())
        {
          continue;
        }
        ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH> nw = (ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH>) sTmpIn.read();
        cur_inp |= (ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH>) (nw << i*DOSA_HLS4ML_OUTPUT_BITWIDTH);
        tkeep |= ((ap_uint<8>) 0b1) << i;
      }
      ap_uint<1> tlast = int(sTmpIn.empty()); //in a testbench, that is ok...
      siData.write(Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH>(cur_inp , tkeep, tlast));
      printf("[TB] filling input FIFO: 0x%16.16llx (tkeep: %2.2x, tlast: %x)\n", (uint64_t) cur_inp, (uint32_t) tkeep, (uint8_t) tlast);
    }
    //printf("[TB] (4 empty reads expected)\n");

    //------------------------------------------------------
    //-- STEP-2 : MAIN TRAFFIC LOOP
    //------------------------------------------------------
    while (!nrErr) {

      if (simCnt < 340)
      {
        stepDut();
        pHls4ml();

      } else {
        printf("## End of simulation at cycle=%3d. \n", simCnt);
        break;
      }

    }  // End: while()

    //------------------------------------------------------
    //-- STEP-3 : COMPARE INPUT AND OUTPUT STREAMS
    //------------------------------------------------------
    int received_words = 0;
    while(!soData.empty())
    {
      Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> cur_read = soData.read();
      ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> cur_out = cur_read.getTData();
      for(int i = 0; i<(DOSA_WRAPPER_OUTPUT_IF_BITWIDTH/DOSA_HLS4ML_OUTPUT_BITWIDTH); i++)
      {
        if((cur_read.getTKeep() >> i) == 0)
        {
          continue;
        }
        ap_uint<DOSA_HLS4ML_OUTPUT_BITWIDTH> cur_word = (ap_uint<DOSA_HLS4ML_OUTPUT_BITWIDTH>) (cur_out >> (i*DOSA_HLS4ML_OUTPUT_BITWIDTH));
        ap_uint<DOSA_HLS4ML_OUTPUT_BITWIDTH> expected_word = sGoldenOut.read();
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
