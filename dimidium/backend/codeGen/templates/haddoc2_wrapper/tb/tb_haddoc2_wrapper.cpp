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
#include <hls_stream.h>
#include <cassert>

#include "../src/haddoc_wrapper.hpp"

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
// ----- Haddoc Interface -----
ap_uint<1>                                po_haddoc_data_valid = 0;
ap_uint<1>                                po_haddoc_frame_valid = 0;
ap_uint<DOSA_HADDOC_INPUT_BITDIWDTH>      po_haddoc_data_vector = 0;
ap_uint<1>                                pi_haddoc_data_valid = 0;
ap_uint<1>                                pi_haddoc_frame_valid = 0;
ap_uint<DOSA_HADDOC_OUTPUT_BITDIWDTH>     pi_haddoc_data_vector = 0;
// ----- DEBUG IO ------
ap_uint<32> debug_out = 0;


//------------------------------------------------------
//-- TESTBENCH GLOBAL VARIABLES
//------------------------------------------------------
int         simCnt;
stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH*DOSA_HADDOC_INPUT_CHAN_NUM> > sHaddocProcessing("sHaddocProcessing");
int haddoc_wait_processing = 0;
int ignore_haddoc_value_cnt = 0;
#define TB_HADDOC_STORE_VALUE_PROP (((DOSA_HADDOC_INPUT_FRAME_WIDTH*DOSA_HADDOC_INPUT_FRAME_WIDTH)/(DOSA_HADDOC_OUTPUT_FRAME_WIDTH*DOSA_HADDOC_OUTPUT_FRAME_WIDTH))-1)
#define TB_HADDOC_LATENCY (2*TB_HADDOC_STORE_VALUE_PROP)

/*****************************************************************************
 * @brief Run a single iteration of the DUT model.
 * @ingroup udp_app_flash
 * @return Nothing.
 ******************************************************************************/
void stepDut() {
  haddoc_wrapper(
      siData, soData,
      &po_haddoc_data_valid, &po_haddoc_frame_valid, &po_haddoc_data_vector,
      &pi_haddoc_data_valid, &pi_haddoc_frame_valid, &pi_haddoc_data_vector,
      &debug_out);
    simCnt++;
    printf("[%4.4d] STEP DUT \n", simCnt);
}

void pHaddoc() {
  pi_haddoc_data_valid = 0;
  pi_haddoc_frame_valid = 0;
  if(po_haddoc_data_valid == 1 && po_haddoc_frame_valid == 1)
  {
    if(ignore_haddoc_value_cnt >= TB_HADDOC_STORE_VALUE_PROP)
    {
      printf("pHaddoc: input %d is forwareded.\n", (uint32_t) po_haddoc_data_vector);
      sHaddocProcessing.write(po_haddoc_data_vector);
      ignore_haddoc_value_cnt = 0;
    } else {
      ignore_haddoc_value_cnt++;
    }
    if(haddoc_wait_processing <= TB_HADDOC_LATENCY)
    {
      haddoc_wait_processing++;
    }
  }
  if(haddoc_wait_processing >= TB_HADDOC_LATENCY)
  {
    if(!sHaddocProcessing.empty())
    {
      ap_uint<DOSA_HADDOC_OUTPUT_BITDIWDTH> tmp_out = sHaddocProcessing.read();
      pi_haddoc_data_valid = 1;
      pi_haddoc_frame_valid = 1;
      pi_haddoc_data_vector = tmp_out;
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

  assert(DOSA_HADDOC_INPUT_CHAN_NUM == DOSA_HADDOC_OUTPUT_CHAN_NUM);
  assert(DOSA_HADDOC_INPUT_BITDIWDTH == DOSA_HADDOC_OUTPUT_BITDIWDTH);
  assert(TB_HADDOC_STORE_VALUE_PROP > 1);

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
    
    stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH>> sTmpIn("sTmpIn");
    stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH>> sGoldenOut("sGoldenOut");
    int filter_out_value_cnt = 0;
    int golden_bytes_written = 0;
    for(int c = 0; c < DOSA_HADDOC_INPUT_CHAN_NUM; c++)
    {
      ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> cur_word = ((ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH>) c) << (DOSA_HADDOC_GENERAL_BITWIDTH-4);
      for(int f = 0; f < DOSA_HADDOC_INPUT_FRAME_WIDTH; f++)
      {
        for(int l = 0; l < DOSA_HADDOC_INPUT_FRAME_WIDTH; l++)
        {
          cur_word |= (ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH>) l;
          sTmpIn.write(cur_word);
          if(filter_out_value_cnt >= TB_HADDOC_STORE_VALUE_PROP)
          {
            sGoldenOut.write(cur_word);
            golden_bytes_written++;
            filter_out_value_cnt = 0;
          } else {
            filter_out_value_cnt++;
          }
        }
      }
    }
    assert((DOSA_WRAPPER_INPUT_IF_BITWIDTH % DOSA_HADDOC_GENERAL_BITWIDTH) == 0);
    while(!sTmpIn.empty())
    {
      ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH> cur_inp = 0x0;
      for(int i = 0; i < (DOSA_WRAPPER_INPUT_IF_BITWIDTH/DOSA_HADDOC_GENERAL_BITWIDTH); i++)
      {
        ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH> nw = (ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH>) sTmpIn.read();
        cur_inp |= (ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH>) (nw << i*DOSA_HADDOC_GENERAL_BITWIDTH);
      }
      ap_uint<1> tlast = int(sTmpIn.empty()); //in a testbench, that is ok...
      siData.write(Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH>(cur_inp ,0xFFFF,tlast));
    }
    printf("(4 empty reads expected)\n");

    simCnt = 0;
    nrErr  = 0;

    ignore_haddoc_value_cnt = 0;
    haddoc_wait_processing = 0;
    //------------------------------------------------------
    //-- STEP-2 : MAIN TRAFFIC LOOP
    //------------------------------------------------------
    while (!nrErr) {

      if (simCnt < 300)
      {
        stepDut();
        pHaddoc();

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
      ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> cur_out = soData.read().getTData();
      for(int i = 0; i<(DOSA_WRAPPER_OUTPUT_IF_BITWIDTH/DOSA_HADDOC_GENERAL_BITWIDTH); i++)
      {
        ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> cur_word = (ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH>) cur_out >> (i*DOSA_HADDOC_GENERAL_BITWIDTH);
        ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> expected_word = sGoldenOut.read();
        if(cur_word != expected_word)
        {
          nrErr++;
          printf("ERROR: Expected %d, but got %d, at position %d.\n", expected_word, cur_word, received_words);
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
