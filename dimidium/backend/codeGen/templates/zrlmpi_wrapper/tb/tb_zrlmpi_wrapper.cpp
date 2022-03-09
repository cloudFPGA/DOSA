//  *
//  *                       cloudFPGA
//  *     Copyright IBM Research, All Rights Reserved
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
ap_uint<32> debug_out = 0;


//------------------------------------------------------
//-- TESTBENCH GLOBAL VARIABLES
//------------------------------------------------------
int         simCnt;
stream<Axis<64> >  Loopback_data ("Loopback_data");

/*****************************************************************************
 * @brief Run a single iteration of the DUT model.
 * @ingroup udp_app_flash
 * @return Nothing.
 ******************************************************************************/
void stepDut() {
    zrlmpi_wrapper(&role_rank_arg, &cluster_size_arg,
        siData, soData, soMPIif, siMPIFeB,
        soMPI_data, siMPI_data,
        &debug_out);
    simCnt++;
    printf("[%4.4d] STEP DUT \n", simCnt);
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
  int fw_cnt = 0;

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

    simCnt = 0;
    nrErr  = 0;

    //------------------------------------------------------
    //-- STEP-2 : MAIN TRAFFIC LOOP
    //------------------------------------------------------
    while (!nrErr) {

      if (simCnt < 22)
      {
        stepDut();

        // Loopback test
        if( !soData.empty() )
        {
          Axis<64> tmp_fw = soData.read();
          printf("[TB] Forwarding (0x%16.16llX, %2.2X, %X).\n", (unsigned long long) tmp_fw.getTData(), (uint32_t) tmp_fw.getTKeep(), (uint8_t) tmp_fw.getTLast());
          siData.write(tmp_fw);
          fw_cnt++;
          feb_2_cnt = simCnt + 6;
        }

        if( fw_cnt == 1 )
        {
          siMPIFeB.write(ZRLMPI_FEEDBACK_OK);
        }
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
      Axis<64> golden_data = Loopback_data.read();
      if( out_data.getTData() != golden_data.getTData() )
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
    }
    if( !Loopback_data.empty() )
    {
      nrErr++;
      printf("ERROR: DUT produced to few output data.\n");
    }


    printf("#####################################################\n");
    if (nrErr)
        printf("## ERROR - TESTBENCH FAILED (RC=%d) !!!             ##\n", nrErr);
    else
        printf("## SUCCESSFULL END OF TESTBENCH (RC=0)             ##\n");

    printf("#####################################################\n");

    return(nrErr);
}
