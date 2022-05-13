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

#include "../src/tips.hpp"

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
// ----- DEBUG IO ------
ap_uint<80> debug_out = 0;

//------------------------------------------------------
//-- TESTBENCH GLOBAL VARIABLES
//------------------------------------------------------
int         simCnt;


/*****************************************************************************
 * @brief Run a single iteration of the DUT model.
 ******************************************************************************/
void stepDut() {
  printf("----- [%4.4d] STEP DUT start ----\n", simCnt);
  tips_test(
      siData,
      soData,
      &debug_out
      );
  printf("        debug_out: %4.4llX%16.16llX\n", (uint16_t) (debug_out >> 64), (uint64_t) debug_out);
  printf("----- [%4.4d] STEP DUT done -----\n", simCnt);
  simCnt++;
}


int main() {

#ifndef TIPS_TEST
  printf("ERROR: This testbench works only with flag 'TIPS_TEST'. STOP.\n");
  exit(-1);
#endif

  //------------------------------------------------------
  //-- TESTBENCH LOCAL VARIABLES
  //------------------------------------------------------
  int         nrErr = 0;

  printf("#####################################################\n");
  printf("## TESTBENCH STARTS HERE                           ##\n");
  printf("#####################################################\n");

  //------------------------------------------------------
  //-- Process Reset
  //------------------------------------------------------

  simCnt = 0;
  nrErr  = 0;
  stepDut();

  //------------------------------------------------------
  //-- Fill Streams
  //------------------------------------------------------

  stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> > sGoldenOut("sGoldenOut");
  //first instruction DENSE_BIAS
  // with uint16_t
  //siData.write(Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH>(0x0004000300020001,0xFF,0b1));
  //sGoldenOut.write(Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH>(0x0000014b00e70079, 0x3F, 1));
  // with ap_fixed<16,14>
  siData.write(Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH>(0x0010000c00080004,0xFF,0b1));
  sGoldenOut.write(Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH>(0x0000052c039c01e4, 0x3F, 1));
  //second instruction TANH
  // with uint16_t
  siData.write(Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH>(0x0000014b00e70079, 0x3F, 1));
  sGoldenOut.write(Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH>(0x0000000100010001, 0x3F, 1));

    //TODO
    //------------------------------------------------------
    //-- STEP-2 : MAIN TRAFFIC LOOP
    //------------------------------------------------------
    while (!nrErr) {

      if (simCnt < DOSA_TIPS_PROGRAM_LENGTH*7)
      {
        stepDut();

      } else {
        printf("## End of simulation at cycle=%3d. \n", simCnt);
        break;
      }

    }  // End: while()

    //------------------------------------------------------
    //-- STEP-3 : COMPARE INPUT AND OUTPUT STREAMS
    //------------------------------------------------------
    while(!soData.empty())
    {
      if(sGoldenOut.empty())
      {
        printf("ERROR: DUT produced to much output.\n");
      }
      Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> cur_read = soData.read();
      Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> expected_read = sGoldenOut.read();
      if(cur_read.getTData() != expected_read.getTData() || cur_read.getTKeep() != expected_read.getTKeep()
          || cur_read.getTLast() != expected_read.getTLast() )
      {
        printf("ERROR: Expected (0x%16.16llX, %2.2X, %X), but got (0x%16.16llX, %2.2X, %X).\n", (unsigned long long) expected_read.getTData(), (uint32_t) expected_read.getTKeep(), (uint8_t) expected_read.getTLast(),
            (unsigned long long) cur_read.getTData(), (uint32_t) cur_read.getTKeep(), (uint8_t) cur_read.getTLast());
        nrErr++;
      }
    }
    if(!sGoldenOut.empty())
    {
        printf("ERROR: DUT produced to few output.\n");
        nrErr++;
    }
    printf("#####################################################\n");
    if (nrErr)
        printf("## ERROR - TESTBENCH FAILED (RC=%d) !!!             ##\n", nrErr);
    else
        printf("## SUCCESSFULL END OF TESTBENCH (RC=0)             ##\n");

    printf("#####################################################\n");

    return(nrErr);
}
