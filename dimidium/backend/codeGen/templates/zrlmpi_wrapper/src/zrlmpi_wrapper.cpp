//  *
//  *                       cloudFPGA
//  *     Copyright IBM Research, All Rights Reserved
//  *    =============================================
//  *     Created: Jan 2022
//  *     Authors: NGL
//  *
//  *     Description:
//  *        ZRLMPI wrapper to be used by DOSA
//  *

#include <stdio.h>
#include <stdint.h>
#include "ap_int.h"
#include "ap_utils.h"
#include <hls_stream.h>
#include <cassert>

#include "zrlmpi_wrapper.hpp"

using namespace hls;



void pStateControl(
    ap_uint<32>       *role_rank_arg,
    ap_uint<32>       *cluster_size_arg,
    stream<MPI_Interface>   &soMPIif,
    stream<MPI_Feedback>    &siMPIFeB,
    stream<uint32_t>        &sReceiveLength,
    stream<uint32_t>        &sSendLength,
    stream<bool>            &sDataArrived,
    stream<bool>            &sReceiveReset,
    stream<bool>            &sSendReset,
    stream<bool>            &sReceiveDone,
    stream<bool>            &sSendDone,
    uint16_t* debug,
    ap_uint<32> *debug_out_ignore
  )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static controlFsmStates controlFSM = RESET;
#pragma HLS reset variable=controlFSM
  static uint8_t nextCommandPtr = 0x0;
#pragma HLS reset variable=nextCommandPtr
  static uint8_t curIterationCnt = 0x0;
#pragma HLS reset variable=curIterationCnt
  static uint8_t curCmnd = MPI_INSTR_NOP;
#pragma HLS reset variable=curCmnd
  static uint8_t curRank = MPI_NO_RANK;
#pragma HLS reset variable=curRank
  static uint32_t curCount = 0;
#pragma HLS reset variable=curCount
  static uint8_t curRep = 0;
#pragma HLS reset variable=curRep
  static bool reuse_prev_data = false;
#pragma HLS reset variable=reuse_prev_data
  static bool save_cur_data = false;
#pragma HLS reset variable=save_cur_data
  //-- STATIC VARIABLES ------------------------------------------------------
  static uint8_t mpiCommands[DOSA_WRAPPER_PROG_LENGTH];
  #pragma HLS ARRAY_PARTITION variable=mpiCommands complete
  static uint8_t mpiRanks[DOSA_WRAPPER_PROG_LENGTH];
  #pragma HLS ARRAY_PARTITION variable=mpiRanks complete
  static uint32_t mpiCounts[DOSA_WRAPPER_PROG_LENGTH];
  #pragma HLS ARRAY_PARTITION variable=mpiCounts complete
  static uint8_t commandRepetitions[DOSA_WRAPPER_PROG_LENGTH];
  #pragma HLS ARRAY_PARTITION variable=commandRepetitions complete
  static bool saveCurData[DOSA_WRAPPER_PROG_LENGTH];
  #pragma HLS ARRAY_PARTITION variable=saveCurData complete

  //-- LOCAL VARIABLES ------------------------------------------------------
  bool not_empty = false;
  MPI_Feedback fedb;
  bool ignore_me;

  switch(controlFSM)
  {
    default:
    case RESET:
      nextCommandPtr = 0x0;
      curIterationCnt = 0x0;
      for(int i = 0; i < DOSA_WRAPPER_PROG_LENGTH; i++)
      {
#pragma HLS unroll
        mpiCommands[i] = MPI_INSTR_NOP;
        mpiRanks[i] = MPI_NO_RANK;
        mpiCounts[i] = 0;
        commandRepetitions[i] = 0;
        saveCurData[i] = false;
      }
      if( !siMPIFeB.empty() )
      {
        siMPIFeB.read();
        not_empty = true;
      }
      if( !sReceiveDone.empty() )
      {
        sReceiveDone.read();
        not_empty = true;
      }
      if( !sSendDone.empty() )
      {
        sSendDone.read();
        not_empty = true;
      }
      if( !sDataArrived.empty() )
      {
        sDataArrived.read();
        not_empty = true;
      }
      curCmnd = MPI_INSTR_NOP;
      curRep = 0;
      curRank = MPI_NO_RANK;
      reuse_prev_data = false;
      save_cur_data = false;
      if( !not_empty && *cluster_size_arg != 0)
      {
        controlFSM = WRITE_PROGRAM;
      }
      break;
    case WRITE_PROGRAM:
      //extra state to avoid fatal HLS optimizations (?)
#ifdef WRAPPER_TEST
      if(*role_rank_arg == 1)
      {
        mpiCommands[0]          = MPI_INSTR_RECV;
        mpiRanks[0]             = 0;
        mpiCounts[0]            = 6; // 22/4;  //MUST be wordsize!
        commandRepetitions[0]   = 1;
        saveCurData[0]          = false;
        mpiCommands[1]          = MPI_INSTR_SEND;
        mpiRanks[1]             = 0;
        mpiCounts[1]            = 6; // 22/4; //MUST be wordsize!
        commandRepetitions[1]   = 1;
        saveCurData[1]          = false;
      }
#else
      //DOSA_ADD_mpi_commands
#endif
      controlFSM = LOAD_COMMAND;
      break;

    case LOAD_COMMAND:
      if(curIterationCnt >= curRep)
      {//read new command
        curCmnd = mpiCommands[nextCommandPtr];
        curRank = mpiRanks[nextCommandPtr];
        curCount = mpiCounts[nextCommandPtr];
        curRep = commandRepetitions[nextCommandPtr];
        //relevant for send only
        save_cur_data = saveCurData[nextCommandPtr];
        nextCommandPtr++;
        if(nextCommandPtr >= DOSA_WRAPPER_PROG_LENGTH)
        {
          nextCommandPtr = 0x0;
        }
        curIterationCnt = 1;
      } else { //issue same again
        curIterationCnt++;
      }
      controlFSM = LOAD_COMMAND2;
      break;

    case LOAD_COMMAND2:
      // just here to get II=1 working
      if(curCmnd != MPI_INSTR_NOP && curRep > 0 && curCount > 0)
      {
        controlFSM = ISSUE_COMMAND;
      } else {
        //else, read next
        controlFSM = LOAD_COMMAND;
      }
      break;

    case ISSUE_COMMAND:
      if( !soMPIif.full() && !sReceiveLength.full() && !sSendLength.full()
          && !sReceiveReset.full()
          && !sSendReset.full()
        )
      {
          MPI_Interface info = MPI_Interface();
          //curCount is WORD length
          uint32_t byte_length = (curCount*4) - 3; //since it would be friction WITHIN a line -> should work
          if(curCmnd == MPI_INSTR_SEND)
          {
            if(!reuse_prev_data)
            {
              sSendLength.write(byte_length);
              controlFSM = WAIT_DATA;
            } else {
              MPI_Interface info = MPI_Interface();
              info.mpi_call = MPI_SEND_INT;
              info.rank = (uint32_t) curRank;
              info.count = (uint32_t) curCount;
              soMPIif.write(info);
              printf("[pStateControl] Issuing %d command, rank %d, count %d (as repeat).\n", (uint8_t) info.mpi_call, (uint8_t) info.rank, (uint32_t) info.count);
              controlFSM = PROC_SEND;
            }
          } else {
            info.mpi_call = MPI_RECV_INT;
            info.rank = (uint32_t) curRank;
            info.count = (uint32_t) curCount;
            soMPIif.write(info);
            printf("[pStateControl] Issuing %d command, rank %d, count %d.\n", (uint8_t) info.mpi_call, (uint8_t) info.rank, (uint32_t) info.count);
            sReceiveLength.write(byte_length);
            controlFSM = PROC_RECEIVE;
          }
      }
      break;

    case WAIT_DATA:
      if( !sDataArrived.empty() && !soMPIif.full() )
      {
        ignore_me = sDataArrived.read();
        MPI_Interface info = MPI_Interface();
        info.mpi_call = MPI_SEND_INT;
        info.rank = (uint32_t) curRank;
        info.count = (uint32_t) curCount;
        soMPIif.write(info);
        printf("[pStateControl] Issuing %d command, rank %d, count %d.\n", (uint8_t) info.mpi_call, (uint8_t) info.rank, (uint32_t) info.count);
        controlFSM = PROC_SEND;
      }
      break;

    case PROC_SEND:
      if( !siMPIFeB.empty()
          && !sSendReset.full()
        )
      {
        fedb = siMPIFeB.read();
        if( fedb == ZRLMPI_FEEDBACK_OK )
        {
          if(save_cur_data)
          {
            printf("[pStateControl] Send succesfull, reuse data.\n");
            reuse_prev_data = true;
            controlFSM = LOAD_COMMAND;
            //reuse reset as fake
            sSendReset.write(true);
          } else {
            reuse_prev_data = false;
            controlFSM = WAIT_SEND;
            sSendReset.write(false);
            printf("[pStateControl] Issue send ok.\n");
          }
        } else {
          //Timeout occured
          sSendReset.write(true);
          printf("[pStateControl] Issue send reset.\n");
          //stay here
        }
      }
      break;

    case WAIT_SEND:
      if( !sSendDone.empty() )
      {
        ignore_me = sSendDone.read();
        printf("[pStateControl] Send done.\n");
        controlFSM = LOAD_COMMAND;
      }
      break;

    case PROC_RECEIVE:
      if( !siMPIFeB.empty()
          && !sReceiveReset.full()
        )
      {
        fedb = siMPIFeB.read();
        if( fedb == ZRLMPI_FEEDBACK_OK )
        {
          //after complete transfer received --> Deq process knows already
          controlFSM = WAIT_RECEIVE;
          sReceiveReset.write(false);
          printf("[pStateControl] Receive successfull.\n");
        } else {
          //Timeout occured
          sReceiveReset.write(true);
          printf("[pStateControl] Issue receive reset.\n");
          //stay here
        }
      }
      break;

    case WAIT_RECEIVE:
      if( !sReceiveDone.empty() )
      {
        ignore_me = sReceiveDone.read();
        printf("[pStateControl] Receive done.\n");
        reuse_prev_data = false; //just to be sure
        controlFSM = LOAD_COMMAND;
      }
      break;
  }

  //to always "use" piFMC_to_ROLE_rank
  *debug_out_ignore = *role_rank_arg;
  *debug = controlFSM;
}


void pRecvEnq(
    stream<Axis<64> >       &siMPI_data,
    stream<uint32_t>        &sReceiveLength,
    stream<bool>            &sReceiveReset,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >   &sRecvBuff_0,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >   &sRecvBuff_1,
    stream<deqBufferCmd>    &sRecvBufferCmds,
    stream<bool>            &sReceiveDone,
    uint16_t* debug
  )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static recvEnqStates recvEnqFsm = RESET0;
#pragma HLS reset variable=recvEnqFsm
  static uint32_t curLength = 0x0;
#pragma HLS reset variable=curLength
  static uint32_t curCnt = 0x0;
#pragma HLS reset variable=curCnt
  static uint8_t nextBuffer = 0;
#pragma HLS reset variable=nextBuffer
  //-- STATIC VARIABLES ------------------------------------------------------
  //-- LOCAL VARIABLES ------------------------------------------------------
  bool not_empty = false;
  Axis<64> tmp_read;
  //uint32_t this_cnt = 0;
  //bool ignore_me;


  switch(recvEnqFsm)
  {
    default:
    case RESET0:
      curLength = 0;
      curCnt = 0;
      nextBuffer = 0;
      if( !sReceiveLength.empty() )
      {
        sReceiveLength.read();
        not_empty = true;
      }
      if( !sReceiveReset.empty() )
      {
        sReceiveReset.read();
        not_empty = true;
      }
      if( !siMPI_data.empty() )
      {
        siMPI_data.read();
        not_empty = true;
      }
      if( !not_empty )
      {
        recvEnqFsm = RECV_WAIT;
      }
      break;

    case RECV_WAIT:
      if( !sReceiveLength.empty() )
      {
        curLength = sReceiveLength.read();
        curCnt = 0;
        if(nextBuffer == 0)
        {
          recvEnqFsm = RECV_BUF_0;
          nextBuffer = 1;
        } else {
          recvEnqFsm = RECV_BUF_1;
          nextBuffer = 0;
        }
      }
      break;

    case RECV_BUF_0:
      //if( !sReceiveReset.empty() && !sRecvBuff_0.full() && !sRecvBufferCmds.full() )
      //{
      //  ignore_me = sReceiveReset.read();
      //  sRecvBufferCmds.write(DRAIN_0);
      //  //"poison pill"
      //  sRecvBuff_0.write(Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH>(0,0,1));
      //  curCnt = 0;
      //  recvEnqFsm = RECV_BUF_1;
      //  nextBuffer = 0;
      //}
      //else
      if( !siMPI_data.empty() && !sRecvBuff_0.full()
          //&& !sRecvBufferCmds.full()
        )
      {
        tmp_read = siMPI_data.read();
        curCnt += extractByteCnt(tmp_read);
        //TODO: remove couning? MPE does always set tlast?
        //if(curCnt >= curLength)
        //{
        //  tmp_read.setTLast(1);
        //}

        // MPE sets the tlast --> we can trust it
        if(tmp_read.getTLast() == 1)
        {
          //also fail will set TLAST
          //recvEnqFsm = RECV_WAIT;
          recvEnqFsm = WAIT_CONFIRMATION;
          nextBuffer = 1;
          //sRecvBufferCmds.write(FORWARD_0);
          //sReceiveDone.write(true);
        }
        sRecvBuff_0.write(tmp_read);
      }
      break;

    case RECV_BUF_1:
      //if( !sReceiveReset.empty() && !sRecvBuff_1.full() && !sRecvBufferCmds.full() )
      //{
      //  ignore_me = sReceiveReset.read();
      //  sRecvBufferCmds.write(DRAIN_1);
      //  //"poison pill"
      //  sRecvBuff_1.write(Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH>(0,0,1));
      //  curCnt = 0;
      //  recvEnqFsm = RECV_BUF_0;
      //  nextBuffer = 1;
      //}
      //else
      if( !siMPI_data.empty() && !sRecvBuff_1.full()
          //&& !sRecvBufferCmds.full()
        )
      {
        tmp_read = siMPI_data.read();
        curCnt += extractByteCnt(tmp_read);
        //TODO: remove couning? MPE does always set tlast?
        //if(curCnt >= curLength)
        //{
        //  tmp_read.setTLast(1);
        //}

        // MPE sets the tlast --> we can trust it
        if(tmp_read.getTLast() == 1)
        {
          //also fail will set TLAST
          //recvEnqFsm = RECV_WAIT;
          recvEnqFsm = WAIT_CONFIRMATION;
          nextBuffer = 0;
          //sRecvBufferCmds.write(FORWARD_1);
          //sReceiveDone.write(true);
        }
        sRecvBuff_1.write(tmp_read);
      }
      break;

    case WAIT_CONFIRMATION:
      if( !sReceiveReset.empty() && !sRecvBufferCmds.full()
          && !sReceiveDone.full()
        )
      {
        bool result = sReceiveReset.read();
        if(result)
        {//failed
          //"poison pill" is already in the fifo
          if(nextBuffer == 0)
          {
            sRecvBufferCmds.write(DRAIN_1);
      //  recvEnqFsm = RECV_BUF_0;
          } else {
            sRecvBufferCmds.write(DRAIN_0);
          }
        } else {
          //success
          sReceiveDone.write(true);
          if(nextBuffer == 0)
          {
            sRecvBufferCmds.write(FORWARD_1);
          } else {
            sRecvBufferCmds.write(FORWARD_0);
          }
          recvEnqFsm = RECV_WAIT;
        }
      }
      break;
  }

  *debug = recvEnqFsm;
}


void pRecvDeq(
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >   &sRecvBuff_0,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >   &sRecvBuff_1,
    stream<deqBufferCmd>    &sRecvBufferCmds,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >   &soData,
    uint16_t* debug
  )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static DeqStates recvDeqFsm = RESET1;
#pragma HLS reset variable=recvDeqFsm
  //-- STATIC VARIABLES ------------------------------------------------------
  //-- LOCAL VARIABLES ------------------------------------------------------
  bool not_empty = false;
  Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> tmp_read;
  deqBufferCmd tmp_cmd;

  switch(recvDeqFsm)
  {
    default:
    case RESET1:
      if( !sRecvBuff_0.empty() )
      {
        sRecvBuff_0.read();
        not_empty = true;
      }
      if( !sRecvBuff_1.empty() )
      {
        sRecvBuff_1.read();
        not_empty = true;
      }
      if( !sRecvBufferCmds.empty() )
      {
        sRecvBufferCmds.read();
        not_empty = true;
      }
      if( !not_empty )
      {
        recvDeqFsm = WAIT_CMD;
      }
      break;

    case WAIT_CMD:
      if( !sRecvBufferCmds.empty() )
      {
        tmp_cmd = sRecvBufferCmds.read();
        switch(tmp_cmd)
        {
          case FORWARD_0:
            recvDeqFsm = DEQ_FW_0;
            break;
          case FORWARD_1:
            recvDeqFsm = DEQ_FW_1;
            break;
          case DRAIN_0:
            recvDeqFsm = DEQ_DRAIN_0;
            break;
          case DRAIN_1:
            recvDeqFsm = DEQ_DRAIN_1;
            break;
        }
      }
      break;

    case DEQ_FW_0:
      if( !sRecvBuff_0.empty() && !soData.full() )
      {
        tmp_read = sRecvBuff_0.read();
        soData.write(tmp_read);
        if( tmp_read.getTLast() == 1 )
        {
          recvDeqFsm = WAIT_CMD;
        }
      }
      break;

    case DEQ_FW_1:
      if( !sRecvBuff_1.empty() && !soData.full() )
      {
        tmp_read = sRecvBuff_1.read();
        soData.write(tmp_read);
        if( tmp_read.getTLast() == 1 )
        {
          recvDeqFsm = WAIT_CMD;
        }
      }
      break;

    case DEQ_DRAIN_0:
      if( !sRecvBuff_0.empty() )
      {
        tmp_read = sRecvBuff_0.read();
        if( tmp_read.getTLast() == 1 )
        {
          recvDeqFsm = WAIT_CMD;
        }
      }
      break;

    case DEQ_DRAIN_1:
      if( !sRecvBuff_1.empty() )
      {
        tmp_read = sRecvBuff_1.read();
        if( tmp_read.getTLast() == 1 )
        {
          recvDeqFsm = WAIT_CMD;
        }
      }
      break;
  }

  *debug = recvDeqFsm;
}



void pSendEnq(
    stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >   &siData,
    stream<uint32_t>        &sSendLength,
    stream<bool>            &sDataArrived,
    stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >   &sSendBuff_0,
    stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >   &sSendBuff_1,
    stream<sendBufferCmd>    &sSendBufferCmds,
    uint16_t* debug
  )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static sendEnqStates sendEnqFsm = RESET2;
#pragma HLS reset variable=sendEnqFsm
  static uint32_t curLength = 0x0;
#pragma HLS reset variable=curLength
  static uint32_t curCnt = 0x0;
#pragma HLS reset variable=curCnt
  static uint8_t nextBuffer = 0;
#pragma HLS reset variable=nextBuffer
  //-- STATIC VARIABLES ------------------------------------------------------
  //-- LOCAL VARIABLES ------------------------------------------------------
  bool not_empty = false;
  Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> tmp_read;
  //uint32_t this_cnt = 0;
  bool ignore_me;


  switch(sendEnqFsm)
  {
    default:
    case RESET2:
      curLength = 0;
      curCnt = 0;
      nextBuffer = 0;
      if( !sSendLength.empty() )
      {
        sSendLength.read();
        not_empty = true;
      }
      if( !siData.empty() )
      {
        siData.read();
        not_empty = true;
      }
      if( !not_empty )
      {
        sendEnqFsm = SEND_WAIT;
      }
      break;

    case SEND_WAIT:
      if( !sSendLength.empty() && !sSendBufferCmds.full() )
      {
        curLength = sSendLength.read();
        curCnt = 0;
        if(nextBuffer == 0)
        {
          sendEnqFsm = SEND_BUF_0_INIT;
          nextBuffer = 1;
          sSendBufferCmds.write(SEND_0);
        } else {
          sendEnqFsm = SEND_BUF_1_INIT;
          sSendBufferCmds.write(SEND_1);
          nextBuffer = 0;
        }
      }
      break;

    case SEND_BUF_0_INIT:
      if( !siData.empty() && !sSendBuff_0.full() && !sDataArrived.full() )
      {
        tmp_read = siData.read();
        curCnt += extractByteCnt(tmp_read);
        //we know the length, so we don't trust the incoming tlast
        if(curCnt >= curLength)
        {
          tmp_read.setTLast(1);
          sendEnqFsm = SEND_WAIT;
          nextBuffer = 1;
        } else {
          tmp_read.setTLast(0);
        }
        sSendBuff_0.write(tmp_read);
        sDataArrived.write(true);
        sendEnqFsm = SEND_BUF_0;
      }
      break;

    case SEND_BUF_0:
      if( !siData.empty() && !sSendBuff_0.full() )
      {
        tmp_read = siData.read();
        curCnt += extractByteCnt(tmp_read);
        //printf("curCnt: %d\n", curCnt);
        //we know the length, so we don't trust the incoming tlast
        if(curCnt >= curLength)
        {
          tmp_read.setTLast(1);
          sendEnqFsm = SEND_WAIT;
          nextBuffer = 1;
        } else {
          tmp_read.setTLast(0);
        }
        sSendBuff_0.write(tmp_read);
      }
      break;

    case SEND_BUF_1_INIT:
      if( !siData.empty() && !sSendBuff_1.full() && !sDataArrived.full() )
      {
        tmp_read = siData.read();
        curCnt += extractByteCnt(tmp_read);
        //we know the length, so we don't trust the incoming tlast
        if(curCnt >= curLength)
        {
          tmp_read.setTLast(1);
          sendEnqFsm = SEND_WAIT;
          nextBuffer = 0;
        } else {
          tmp_read.setTLast(0);
        }
        sSendBuff_1.write(tmp_read);
        sDataArrived.write(true);
        sendEnqFsm = SEND_BUF_1;
      }
      break;

    case SEND_BUF_1:
      if( !siData.empty() && !sSendBuff_1.full() )
      {
        tmp_read = siData.read();
        curCnt += extractByteCnt(tmp_read);
        //we know the length, so we don't trust the incoming tlast
        if(curCnt >= curLength)
        {
          tmp_read.setTLast(1);
          sendEnqFsm = SEND_WAIT;
          nextBuffer = 0;
        } else {
          tmp_read.setTLast(0);
        }
        sSendBuff_1.write(tmp_read);
      }
      break;
  }

  *debug = sendEnqFsm;
}


void pSendDeq(
    stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >   &sSendBuff_0,
    stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >   &sSendBuff_1,
    stream<sendBufferCmd>    &sSendBufferCmds,
    stream<bool>             &sSendReset,
    stream<bool>             &sSendDone,
    stream<Axis<64> >        &soMPI_data,
    uint16_t* debug
  )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static SendDeqStates sendDeqFsm = RESET3;
#pragma HLS reset variable=recvDeqFsm
  static uint8_t lastBuffer = 0;
#pragma HLS reset variable=lastBuffer
  static uint8_t nextCC = 0;
#pragma HLS reset variable=nextCC
//  static bool drain_cc_0 = false;
//#pragma HLS reset variable=drain_cc_0
//  static bool drain_cc_1 = false;
//#pragma HLS reset variable=drain_cc_1
//  static bool back_to_other_cc = false;
//#pragma HLS reset variable=back_to_other_cc
  //-- STATIC VARIABLES ------------------------------------------------------
  static stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> > sCopyContainer_0 ("sCopyContainer_0");
  #pragma HLS STREAM variable=sCopyContainer_0 depth=buffer_fifo_depth
  static stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> > sCopyContainer_1 ("sCopyContainer_1");
  #pragma HLS STREAM variable=sCopyContainer_1 depth=buffer_fifo_depth
  //-- LOCAL VARIABLES ------------------------------------------------------
  bool not_empty = false;
  Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> tmp_read;
  sendBufferCmd tmp_cmd;
  bool tmp_reset;

  switch(sendDeqFsm)
  {
    default:
    case RESET3:
      lastBuffer = 0;
      //drain_cc_0 = false;
      //drain_cc_1 = false;
      nextCC = 0;
      //back_to_other_cc = false;
      if( !sSendBuff_0.empty() )
      {
        sSendBuff_0.read();
        not_empty = true;
      }
      if( !sSendBuff_1.empty() )
      {
        sSendBuff_1.read();
        not_empty = true;
      }
      if( !sSendBufferCmds.empty() )
      {
        sSendBufferCmds.read();
        not_empty = true;
      }
      if( !sSendReset.empty() )
      {
        sSendReset.read();
        not_empty = true;
      }
      if( !sCopyContainer_0.empty() )
      {
        sCopyContainer_0.read();
        not_empty = true;
      }
      if( !sCopyContainer_1.empty() )
      {
        sCopyContainer_1.read();
        not_empty = true;
      }
      if( !not_empty )
      {
        sendDeqFsm = WAIT_START;
      }
      break;

    case WAIT_START:
      if( !sSendBufferCmds.empty() )
      {
        tmp_cmd = sSendBufferCmds.read();
        if( tmp_cmd == SEND_0 )
        {
          sendDeqFsm = SEND_DEQ_0;
          lastBuffer = 0;
        } else {
          sendDeqFsm = SEND_DEQ_1;
          lastBuffer = 1;
        }
      }
      break;

    case SEND_DEQ_0:
      //if( !sSendReset.empty() )
      //{
      //  tmp_reset = sSendReset.read();
      //  if( tmp_reset == true )
      //  {
      //    lastBuffer = 0;
      //    if( nextCC == 0 )
      //    {
      //      sendDeqFsm = SEND_CC_0;
      //    } else {
      //      sendDeqFsm = SEND_CC_1;
      //    }
      //  }
      //  //else, better ignore?
      //}
      //else
      if( !sSendBuff_0.empty() && !soMPI_data.full()
              && nextCC == 0 && !sCopyContainer_0.full()
          )
      {
        tmp_read = sSendBuff_0.read();
        soMPI_data.write(tmp_read);
        sCopyContainer_0.write(tmp_read);
        if( tmp_read.getTLast() == 1 )
        {
          lastBuffer = 0;
          sendDeqFsm = WAIT_OK;
        }
      }
      else if( !sSendBuff_0.empty() && !soMPI_data.full()
              && nextCC == 1 && !sCopyContainer_1.full()
          )
      {
        tmp_read = sSendBuff_0.read();
        soMPI_data.write(tmp_read);
        sCopyContainer_1.write(tmp_read);
        if( tmp_read.getTLast() == 1 )
        {
          lastBuffer = 0;
          sendDeqFsm = WAIT_OK;
        }
      }
      break;

    case SEND_DEQ_1:
      //if( !sSendReset.empty() )
      //{
      //  tmp_reset = sSendReset.read();
      //  if( tmp_reset == true )
      //  {
      //    lastBuffer = 1;
      //    if( nextCC == 0 )
      //    {
      //      sendDeqFsm = SEND_CC_0;
      //    } else {
      //      sendDeqFsm = SEND_CC_1;
      //    }
      //  }
      //  //else, better ignore?
      //}
      //else
      if( !sSendBuff_1.empty() && !soMPI_data.full()
              && nextCC == 0 && !sCopyContainer_0.full()
          )
      {
        tmp_read = sSendBuff_1.read();
        soMPI_data.write(tmp_read);
        sCopyContainer_0.write(tmp_read);
        if( tmp_read.getTLast() == 1 )
        {
          lastBuffer = 1;
          sendDeqFsm = WAIT_OK;
        }
      }
      else if( !sSendBuff_1.empty() && !soMPI_data.full()
              && nextCC == 1 && !sCopyContainer_1.full()
          )
      {
        tmp_read = sSendBuff_1.read();
        soMPI_data.write(tmp_read);
        sCopyContainer_1.write(tmp_read);
        if( tmp_read.getTLast() == 1 )
        {
          lastBuffer = 1;
          sendDeqFsm = WAIT_OK;
        }
      }
      break;

    case SEND_CC_0:
      //if( !sSendReset.empty() )
      //{
      //  tmp_reset = sSendReset.read();
      //  if( tmp_reset == true )
      //  {
      //    //TODO: implement true call stack?
      //    back_to_other_cc = true;
      //    sendDeqFsm = SEND_CC_1;
      //  }
      //  //else, better ignore?
      //}
      //else
      if( !sCopyContainer_0.empty() && !soMPI_data.full()
                && !sCopyContainer_1.full()
          )
      {
        tmp_read = sCopyContainer_0.read();
        soMPI_data.write(tmp_read);
        sCopyContainer_1.write(tmp_read);
        if( tmp_read.getTLast() == 1 )
        {
          nextCC = 0;
          //back_to_other_cc = false;
          sendDeqFsm = WAIT_OK;
        }
      }
      //else if( sCopyContainer_0.empty() && !sCopyContainer_1.full() )
      //{ // continue were stopped
      //  nextCC = 0;
      //  if( back_to_other_cc )
      //  {
      //    //TODO: implement true call stack?
      //    back_to_other_cc = false;
      //    sendDeqFsm = SEND_CC_1;
      //  } else {
      //    if( lastBuffer == 0 )
      //    {
      //      sendDeqFsm = SEND_DEQ_0;
      //    } else {
      //      sendDeqFsm = SEND_DEQ_1;
      //    }
      //    drain_cc_1 = true;
      //    //"poison pill"
      //    sCopyContainer_1.write(Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH>(0,0,1));
      //  }
      //}
      break;

    case SEND_CC_1:
      //if( !sSendReset.empty() )
      //{
      //  tmp_reset = sSendReset.read();
      //  if( tmp_reset == true )
      //  {
      //    //TODO: implement true call stack?
      //    back_to_other_cc = true;
      //    sendDeqFsm = SEND_CC_0;
      //  }
      //  //else, better ignore?
      //}
      //else
      if( !sCopyContainer_1.empty() && !soMPI_data.full()
                && !sCopyContainer_0.full()
          )
      {
        tmp_read = sCopyContainer_1.read();
        soMPI_data.write(tmp_read);
        sCopyContainer_0.write(tmp_read);
        if( tmp_read.getTLast() == 1 )
        {
          nextCC = 1;
          //back_to_other_cc = false;
          sendDeqFsm = WAIT_OK;
        }
      }
      //else if( sCopyContainer_1.empty() && !sCopyContainer_0.full() )
      //{ // continue were stopped
      //  nextCC = 1;
      //  if( back_to_other_cc )
      //  {
      //    //TODO: implement true call stack?
      //    back_to_other_cc = false;
      //    sendDeqFsm = SEND_CC_0;
      //  } else {
      //    if( lastBuffer == 0 )
      //    {
      //      sendDeqFsm = SEND_DEQ_0;
      //    } else {
      //      sendDeqFsm = SEND_DEQ_1;
      //    }
      //    drain_cc_0 = true;
      //    //"poison pill"
      //    sCopyContainer_0.write(Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH>(0,0,1));
      //  }
      //}
      break;

    case WAIT_OK:
      if( !sSendReset.empty() &&
          !sSendDone.full() )
      {
        tmp_reset = sSendReset.read();
        printf("[pSendDeq] received reset: %d\n", (uint8_t) tmp_reset);
        if( tmp_reset == true )
        {
          //will go to right buffer one cycle later, if necessary
          if( nextCC == 0 )
          {
            sendDeqFsm = SEND_CC_0;
          } else {
            sendDeqFsm = SEND_CC_1;
          }
        } else {
          //success
          sSendDone.write(true);
          //sendDeqFsm = WAIT_DRAIN;
          if( nextCC == 0 )
          {
            //drain_cc_0 = true;
            sendDeqFsm = DRAIN_CC_0;
            nextCC = 1;
          } else {
            //drain_cc_1 = true;
            sendDeqFsm = DRAIN_CC_1;
            nextCC = 0;
          }
        }
      }
      break;

    case DRAIN_CC_0:
      if( !sCopyContainer_0.empty() )
      {
        tmp_read = sCopyContainer_0.read();
        if( tmp_read.getTLast() == 1)
        {
          sendDeqFsm = WAIT_START;
        }
      } else {
          sendDeqFsm = WAIT_START;
      }
      break;

    case DRAIN_CC_1:
      if( !sCopyContainer_1.empty() )
      {
        tmp_read = sCopyContainer_1.read();
        if( tmp_read.getTLast() == 1)
        {
          sendDeqFsm = WAIT_START;
        }
      } else {
          sendDeqFsm = WAIT_START;
      }
      break;

    //case WAIT_DRAIN:
    //  if(drain_cc_0)
    //  {
    //    if( !sCopyContainer_0.empty() )
    //    {
    //      tmp_read = sCopyContainer_0.read();
    //      if( tmp_read.getTLast() == 1)
    //      {
    //        drain_cc_0 = false;
    //      }
    //    } else {
    //      drain_cc_0 = false;
    //    }
    //  }
    //  if(drain_cc_1)
    //  {
    //    if( !sCopyContainer_1.empty() )
    //    {
    //      tmp_read = sCopyContainer_1.read();
    //      if( tmp_read.getTLast() == 1)
    //      {
    //        drain_cc_1 = false;
    //      }
    //    } else {
    //      drain_cc_1 = false;
    //    }
    //  }
    //  if( !drain_cc_0 && !drain_cc_1)
    //  {
    //    sendDeqFsm = WAIT_START;
    //  }
    //  break;
  }

  *debug = sendDeqFsm;
}


void pMergeDebug(
  uint16_t *debug0,
  uint16_t *debug1,
  uint16_t *debug2,
  uint16_t *debug3,
  uint16_t *debug4,
    ap_uint<80> *debug_out
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------

  *debug_out =   (ap_uint<80>) *debug0;
  *debug_out |= ((ap_uint<80>) *debug1) << 16;
  *debug_out |= ((ap_uint<80>) *debug2) << 32;
  *debug_out |= ((ap_uint<80>) *debug3) << 48;
  *debug_out |= ((ap_uint<80>) *debug4) << 64;

}



void zrlmpi_wrapper(
    // ----- FROM FMC ----- TODO: necessary?
    ap_uint<32>       *role_rank_arg,
    ap_uint<32>       *cluster_size_arg,
    // ----- Wrapper Interface -----
    stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >   &siData,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >  &soData,
    // ----- MPI_Interface -----
    stream<MPI_Interface>   &soMPIif,
    stream<MPI_Feedback>    &siMPIFeB,
    stream<Axis<64> >       &soMPI_data,
    stream<Axis<64> >       &siMPI_data,
    // ----- DEBUG out ------
    ap_uint<80> *debug_out,
    ap_uint<32> *debug_out_ignore
    )
{
  //-- DIRECTIVES FOR THE BLOCK ---------------------------------------------
#pragma HLS INTERFACE ap_ctrl_none port=return

//TODO: necessary?
#pragma HLS INTERFACE ap_vld register port=role_rank_arg name=piFMC_to_ROLE_rank
#pragma HLS INTERFACE ap_vld register port=cluster_size_arg name=piFMC_to_ROLE_size

#pragma HLS INTERFACE ap_fifo port=siData
#pragma HLS INTERFACE ap_fifo port=soData

#pragma HLS INTERFACE ap_ovld register port=debug_out
#pragma HLS INTERFACE ap_ovld register port=debug_out_ignore

#pragma HLS INTERFACE ap_fifo port=soMPIif
#pragma HLS DATA_PACK     variable=soMPIif
#pragma HLS INTERFACE ap_fifo port=siMPIFeB
  //#pragma HLS DATA_PACK     variable=siMPIFeB
#pragma HLS INTERFACE ap_fifo port=soMPI_data
#pragma HLS DATA_PACK     variable=soMPI_data
#pragma HLS INTERFACE ap_fifo port=siMPI_data
#pragma HLS DATA_PACK     variable=siMPI_data


  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS DATAFLOW

#ifndef __SYNTHESIS__
  //TODO: make dynamic
  assert(DOSA_WRAPPER_INPUT_IF_BITWIDTH == 64);
  assert(DOSA_WRAPPER_OUTPUT_IF_BITWIDTH == 64);
  //TODO: add bitwidth conversation
  assert(DOSA_WRAPPER_INPUT_IF_BITWIDTH == DOSA_WRAPPER_OUTPUT_IF_BITWIDTH);
#endif

  //-- STATIC VARIABLES (with RESET) ------------------------------------------

  //-- STATIC DATAFLOW VARIABLES ------------------------------------------

  static stream<uint32_t> sReceiveLength ("sReceiveLength");
  #pragma HLS STREAM variable=sReceiveLength depth=2  //TODO?
  static stream<bool> sReceiveDone ("sReceiveDone");
  #pragma HLS STREAM variable=sReceiveDone depth=2
  static stream<bool> sReceiveReset ("sReceiveReset");
  #pragma HLS STREAM variable=sReceiveReset depth=2
  static stream<uint32_t> sSendLength ("sSendLength");
  #pragma HLS STREAM variable=sSendLength depth=2
  static stream<bool> sSendDone ("sSendDone");
  #pragma HLS STREAM variable=sSendDone depth=2
  static stream<bool> sSendReset ("sSendReset");
  #pragma HLS STREAM variable=sSendReset depth=2
  static stream<bool> sDataArrived ("sDataArrived");
  #pragma HLS STREAM variable=sDataArrived depth=2

  static stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> > sRecvBuff_0 ("sRecvBuff_0");
  #pragma HLS STREAM variable=sRecvBuff_0 depth=buffer_fifo_depth
  static stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> > sRecvBuff_1 ("sRecvBuff_1");
  #pragma HLS STREAM variable=sRecvBuff_1 depth=buffer_fifo_depth
  static stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> > sSendBuff_0 ("sSendBuff_0");
  #pragma HLS STREAM variable=sSendBuff_0 depth=buffer_fifo_depth
  static stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> > sSendBuff_1 ("sSendBuff_1");
  #pragma HLS STREAM variable=sSendBuff_1 depth=buffer_fifo_depth

  static stream<sendBufferCmd> sSendBufferCmds ("sSendBufferCmds");
  #pragma HLS STREAM variable=sSendBufferCmds depth=2
  static stream<deqBufferCmd> sRecvBufferCmds ("sRecvBufferCmds");
  #pragma HLS STREAM variable=sRecvBufferCmds depth=2

  //-- LOCAL VARIABLES ------------------------------------------------------------
  uint16_t debug0 = 0;
  uint16_t debug1 = 0;
  uint16_t debug2 = 0;
  uint16_t debug3 = 0;
  uint16_t debug4 = 0;


  //-- PROCESS INSTANTIATION ------------------------------------------------------

  pStateControl(role_rank_arg, cluster_size_arg, soMPIif, siMPIFeB, sReceiveLength, sSendLength,
      sDataArrived, sReceiveReset, sSendReset,
                sReceiveDone, sSendDone, &debug0, debug_out_ignore);

  pRecvEnq(siMPI_data, sReceiveLength,
      sReceiveReset,
      sRecvBuff_0, sRecvBuff_1, sRecvBufferCmds, sReceiveDone, &debug1);

  pRecvDeq(sRecvBuff_0, sRecvBuff_1, sRecvBufferCmds, soData, &debug2);

  pSendEnq(siData, sSendLength, sDataArrived, sSendBuff_0, sSendBuff_1, sSendBufferCmds, &debug3);

  pSendDeq(sSendBuff_0, sSendBuff_1, sSendBufferCmds,
      sSendReset,
      sSendDone, soMPI_data, &debug4);

  pMergeDebug(&debug0, &debug1, &debug2, &debug3, &debug4, debug_out);

}



