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
//  *     Created: Jan 2022
//  *     Authors: NGL
//  *
//  *     Description:
//  *        Testbench for the ZRLMPI wrapper
//  *

#include <stdio.h>
#include <hls_stream.h>
#include <cassert>

#include "../src/zrlmpi_wrapper.hpp"

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
// ----- FROM FMC -----
ap_uint<32> role_rank_arg = 0x1;
ap_uint<32> cluster_size_arg = 0x5;
// ----- Wrapper Interface -----
stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >   siData ("siData");
stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >  soData ("soData");
// ----- MPI_Interface -----
stream<MPI_Interface>  soMPIif ("soMPIif");
stream<MPI_Feedback>  siMPIFeB ("siMPIFeB");
stream<Axis<64> >   soMPI_data ("soMPI_data");
stream<Axis<64> >   siMPI_data ("siMPI_data");
// ----- DEBUG IO ------
ap_uint<80> debug_out = 0;
ap_uint<32> debug_out_ignore = 0;


//------------------------------------------------------
//-- TESTBENCH GLOBAL VARIABLES
//------------------------------------------------------
int         simCnt;
stream<Axis<64> >  Loopback_data ("Loopback_data");
stream<Axis<64> >  recv_data ("recv_data");
stream<Axis<64> >  send_data ("send_data");

/*****************************************************************************
 * @brief Run a single iteration of the DUT model.
 * @ingroup udp_app_flash
 * @return Nothing.
 ******************************************************************************/
void stepDut() {
    zrlmpi_wrapper(&role_rank_arg, &cluster_size_arg,
        siData, soData, soMPIif, siMPIFeB,
        soMPI_data, siMPI_data,
        &debug_out, &debug_out_ignore);
    simCnt++;
    printf("[%4.4d] STEP DUT\n", simCnt);
    printf("        debug_out: %4.4llX%16.16llX\n", (uint16_t) (debug_out >> 64), (uint64_t) debug_out);
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
  int feb_2_cnt = -1;
  //int fw_cnt = 0;

  printf("#####################################################\n");
  printf("## TESTBENCH STARTS HERE                           ##\n");
  printf("#####################################################\n");

  assert(DOSA_WRAPPER_INPUT_IF_BITWIDTH == 64);
  assert(DOSA_WRAPPER_OUTPUT_IF_BITWIDTH == 64);
  assert(DOSA_WRAPPER_INPUT_IF_BITWIDTH == DOSA_WRAPPER_OUTPUT_IF_BITWIDTH);

  //------------------------------------------------------
  //-- STEP-0 : PROCESS RESET STATES
  //------------------------------------------------------
  stepDut();
  printf("[TB] (Reset processed.)\n");

  //------------------------------------------------------
  //-- STEP-1 : INITIALIZE STREAMS
  //------------------------------------------------------

  //simple data test
  siMPI_data.write(Axis<64>((uint64_t) 0x0101010101010101,0xFF,0b0));
  siMPI_data.write(Axis<64>((uint64_t) 0x0202020202020202,0xFF,0b0));
  siMPI_data.write(Axis<64>((uint64_t) 0x0A0B030303030303,0x3F,0b1));
  Loopback_data.write(Axis<64>((uint64_t) 0x0101010101010101,0xFF,0b0));
  Loopback_data.write(Axis<64>((uint64_t) 0x0202020202020202,0xFF,0b0));
  Loopback_data.write(Axis<64>((uint64_t) 0x0A0B030303030303,0x3F,0b1));
  send_data.write(Axis<64>((uint64_t) 0x0101010101010101,0xFF,0b0));
  send_data.write(Axis<64>((uint64_t) 0x0202020202020202,0x0F,0b0));
  send_data.write(Axis<64>((uint64_t) 0x0303030302020202,0xFF,0b0));
  send_data.write(Axis<64>((uint64_t) 0x0A0B030303030303,0x03,0b1));

    simCnt = 0;
    nrErr  = 0;
    feb_2_cnt = simCnt + 6;
    //int half_tkeep_cnt = 0;
    bool start_send = false;

    //------------------------------------------------------
    //-- STEP-2 : MAIN TRAFFIC LOOP
    //------------------------------------------------------
    while (!nrErr) {

      if (simCnt < 32)
      {
        stepDut();

        // Loopback test
        if( !soData.empty() )
        {
          Axis<64> tmp_fw = soData.read();
          recv_data.write(tmp_fw);
          printf("[TB] Recving (0x%16.16llX, %2.2X, %X).\n", (unsigned long long) tmp_fw.getTData(), (uint32_t) tmp_fw.getTKeep(), (uint8_t) tmp_fw.getTLast());
          //if(half_tkeep_cnt == 1)
          //{
          //  Axis<64> tmp_fw_1 = Axis<64>(tmp_fw.getTData(), 0x0F, 0);
          //  Axis<64> tmp_fw_2 = Axis<64>(tmp_fw.getTData() >> 32, 0x0F, 0);
          //  printf("[TB] Forwarding (0x%16.16llX, %2.2X, %X).\n", (unsigned long long) tmp_fw_1.getTData(), (uint32_t) tmp_fw_1.getTKeep(), (uint8_t) tmp_fw_1.getTLast());
          //  printf("[TB] Forwarding (0x%16.16llX, %2.2X, %X).\n", (unsigned long long) tmp_fw_2.getTData(), (uint32_t) tmp_fw_2.getTKeep(), (uint8_t) tmp_fw_2.getTLast());
          //  siData.write(tmp_fw_1);
          //  siData.write(tmp_fw_2);
          //} else {
          //  printf("[TB] Forwarding (0x%16.16llX, %2.2X, %X).\n", (unsigned long long) tmp_fw.getTData(), (uint32_t) tmp_fw.getTKeep(), (uint8_t) tmp_fw.getTLast());
          //  siData.write(tmp_fw);
          //}
          //fw_cnt++;
          feb_2_cnt = simCnt + 6;
          //half_tkeep_cnt++;
          start_send = true;
        }

        if(start_send && !send_data.empty())
        {
          Axis<64> tmp_fw = send_data.read();
          printf("[TB] Writing (0x%16.16llX, %2.2X, %X).\n", (unsigned long long) tmp_fw.getTData(), (uint32_t) tmp_fw.getTKeep(), (uint8_t) tmp_fw.getTLast());
          siData.write(tmp_fw);
        }

        //if( fw_cnt == 1 )
        //{
        //  siMPIFeB.write(ZRLMPI_FEEDBACK_OK);
        //}
        if( feb_2_cnt == simCnt )
        {
          siMPIFeB.write(ZRLMPI_FEEDBACK_OK);
        }

      } else {
        printf("## End of simulation at cycle=%3d. \n", simCnt);
        break;
      }

    }  // End: while()

    //------------------------------------------------------
    //-- STEP-3 : COMPARE INPUT AND OUTPUT STREAMS
    //------------------------------------------------------
    MPI_Interface info_out = soMPIif.read();
    if( !(info_out.rank == 0 && info_out.count == 6 && info_out.mpi_call == MPI_RECV_INT) )
    {
      nrErr++;
      printf("ERROR: First MPI_Interface command is wrong.\n");
    }
    info_out = soMPIif.read();
    if( !(info_out.rank == 0 && info_out.count == 6 && info_out.mpi_call == MPI_SEND_INT) )
    {
      nrErr++;
      printf("ERROR: Second MPI_Interface command is wrong.\n");
    }
    if( !soMPIif.empty() )
    {
      info_out = soMPIif.read();
      //in case the wrapper issued the next recv already, this would be ok
      if( !(info_out.rank == 0 && info_out.count == 6 && info_out.mpi_call == MPI_RECV_INT)
          || !soMPIif.empty()
        )
      {
        nrErr++;
        printf("ERROR: MPI_Interface fifo contains to many (or wrong) commands.\n");
      }
    }
    if( !siMPIFeB.empty() )
    {
      nrErr++;
      printf("ERROR: MPI_Feedback fifo contains to many entries.\n");
    }

    while( !soMPI_data.empty() )
    {
      Axis<64> out_data = soMPI_data.read();
      printf("[TB] Reading (0x%16.16llX, %2.2X, %X).\n", (unsigned long long) out_data.getTData(), (uint32_t) out_data.getTKeep(), (uint8_t) out_data.getTLast());
      Axis<64> golden_data = Loopback_data.read();
      Axis<64> recv_word = recv_data.read();
      if( (out_data.getTData() & golden_data.getTKeep()) != (golden_data.getTData() & golden_data.getTKeep()) ) //yes, two times golden tkeep
      {
        nrErr++;
        printf("ERROR: out tdata: Expected 0x%16.16llX but got 0x%16.16llX\n", (unsigned long long) golden_data.getTData(), (unsigned long long) out_data.getTData());
      }
      if( out_data.getTKeep() != golden_data.getTKeep() )
      {
        nrErr++;
        printf("ERROR: out tkeep: Expected %2.2X but got %2.2X\n", (uint32_t) golden_data.getTKeep(), (uint32_t) out_data.getTKeep());
      }
      if( out_data.getTLast() != golden_data.getTLast() )
      {
        nrErr++;
        printf("ERROR: out tlast: Expected %X but got %X\n", (uint32_t) golden_data.getTLast(), (uint32_t) out_data.getTLast());
      }
      if( recv_word.getTData() != golden_data.getTData() )
      {
        nrErr++;
        printf("ERROR: recv word tdata: Expected 0x%16.16llX but got 0x%16.16llX\n", (unsigned long long) golden_data.getTData(), (unsigned long long) recv_word.getTData());
      }
      if( recv_word.getTKeep() != golden_data.getTKeep() )
      {
        nrErr++;
        printf("ERROR: recv word tkeep: Expected %2.2X but got %2.2X\n", (uint32_t) golden_data.getTKeep(), (uint32_t) recv_word.getTKeep());
      }
      if( recv_word.getTLast() != golden_data.getTLast() )
      {
        nrErr++;
        printf("ERROR: recv word tlast: Expected %X but got %X\n", (uint32_t) golden_data.getTLast(), (uint32_t) recv_word.getTLast());
      }
    }
    if( !Loopback_data.empty() )
    {
      nrErr++;
      printf("ERROR: DUT produced to few output data.\n");
    }
    if( !recv_data.empty() )
    {
      nrErr++;
      printf("ERROR: DUT prduced to much recv data.\n");
    }


    printf("#####################################################\n");
    if (nrErr)
        printf("## ERROR - TESTBENCH FAILED (RC=%d) !!!             ##\n", nrErr);
    else
        printf("## SUCCESSFULL END OF TESTBENCH (RC=0)             ##\n");

    printf("#####################################################\n");

    return(nrErr);
}
