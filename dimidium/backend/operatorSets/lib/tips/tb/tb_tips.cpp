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

  siData.write(Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH>(0x0004000300020001,0xFF,0b0));
  siData.write(Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH>(0x0008000600060005,0xFF,0b0));
  siData.write(Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH>(0x0000000000000009,0x02,0b1));

  stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> > sGoldenOut("sGoldenOut");
  sGoldenOut.write(Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH>(0x00a4009e00980092, 0xFF, 0));
  sGoldenOut.write(Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH>(0x0179016a015b014c, 0xFF, 0));
  sGoldenOut.write(Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH>(0x024e0236021e0206, 0xFF, 1));

    //TODO
    //------------------------------------------------------
    //-- STEP-2 : MAIN TRAFFIC LOOP
    //------------------------------------------------------
    while (!nrErr) {

      if (simCnt < 32)
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
