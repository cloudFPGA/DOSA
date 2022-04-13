//  *
//  *                       cloudFPGA
//  *     Copyright IBM Research, All Rights Reserved
//  *    =============================================
//  *     Created: Apr 2022
//  *     Authors: NGL
//  *
//  *     Description:
//  *        Template for a Tensor processor without
//  *          Interlocked Pipeline Stages (TIPS) engine
//  *

#include <stdio.h>
#include <stdint.h>
#include "ap_int.h"
#include "ap_utils.h"
#include <hls_stream.h>
#include <hls_math.h>
#include <cassert>

#include "tips.hpp"

using namespace hls;


void pLoadNetwork(
    stream<TipsNetworkInstr>  &sNetworkLoadCmd,
    stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >   &siData,
    stream<ap_uint<DOSA_TIPS_LONGEST_INPUT> >  &sNetworkInput,
    uint16_t *debug
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static LoadNetworkStates loadNetFSM = RESET1;
#pragma HLS reset variable=loadNetFSM
  static TipsLength cur_length = 0;
#pragma HLS reset variable=cur_length
  static TipsLength req_input_length = 0;
#pragma HLS reset variable=req_input_length
  static TipsLength hangover_valid_bytes = 0x0;
#pragma HLS reset variable=hangover_valid_bytes
  //-- STATIC VARIABLES -----------------------------------------------------
  static ap_uint<DOSA_TIPS_LONGEST_INPUT> cur_input;
  static ap_uint<DOSA_TIPS_LONGEST_INPUT> hangover_store;
  static ap_uint<DOSA_TIPS_LONGEST_INPUT> mask;
  //-- LOCAL VARIABLES ------------------------------------------------------

  switch(loadNetFSM)
  {
    default:
    case RESET1:
      //necessary?
      cur_length = 0;
      hangover_valid_bytes = 0;
      cur_input = 0;
      hangover_store = 0;
      req_input_length = 0;
      loadNetFSM = READ_INSTR;
      break;

    case READ_INSTR:
      if( !sNetworkLoadCmd.empty() && !sNetworkInput.full() )
      {
        req_input_length = sNetworkLoadCmd.read().length;
        if(req_input_length == 0)
        {
          sNetworkInput.write(0x0);
          //stay here
        } else {
          mask = exp2(req_input_length*8)-1;

          if(hangover_valid_bytes >= req_input_length)
          {
            ap_uint<DOSA_TIPS_LONGEST_INPUT> out_vector = hangover_store & mask;
            sNetworkInput.write(out_vector);
            hangover_valid_bytes -= req_input_length;
            hangover_store =>> req_input_length * 8;
            //stay here
          } else {
            cur_length = hangover_valid_bytes;
            cur_input = hangover_store;
            hangover_store = 0;
            hangover_valid_bytes = 0;
            loadNetFSM = READ_NETWORK;
          }
        }
      }
      break;

    case READ_NETWORK:
      if( !siData.empty() && !sNetworkInput.full() )
      {
        NetworkWord in_word = siData.read();
        TipsLength byte_read = extractByteCnt(in_word);
        cur_input |= ((ap_uint<DOSA_TIPS_LONGEST_INPUT>) in_word.getTData()) << cur_length*8;
        cur_length += byte_read;
        //ignore tlast?
        if(cur_length >= req_input_length)
        {
            //ap_uint<DOSA_TIPS_LONGEST_INPUT> mask = exp2(req_input_length*8)-1;
            ap_uint<DOSA_TIPS_LONGEST_INPUT> out_vector = cur_input & mask;
            sNetworkInput.write(out_vector);
            hangover_valid_bytes = cur_length - req_input_length;
            hangover_store = (cur_input >> req_input_length * 8);
            cur_input = 0;
            cur_length = 0;
            loadNetFSM = READ_INSTR;
        }
      }
      break;
  }

  *debug = (uint8_t) loadNetFSM;
  *debug |= ((uint16_t) cur_length) << 8;

}


void pSendNetwork(
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  //-- STATIC VARIABLES -----------------------------------------------------
  //-- LOCAL VARIABLES ------------------------------------------------------

}


void pTipsControl(
    stream<TipsNetworkInstr>  &sNetworkLoadCmd,
    stream<TipsLoadInstr>     &sOpLoadCmd,
    stream<TipsAluInstr>      &sAluInstr,
    uint16_t *debug
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static uint16_t next_command_pointer = 0;
#pragma HLS reset varialbe=next_command_pointer
  //-- STATIC VARIABLES -----------------------------------------------------
  const TipsOp program[] = {
    [0] = { .opcode = DENSE_BIAS, .in_addr = NETWORK_ALIAS_ADDRESS, .in_length = 9,
            .op0_addr = 0, .op0_length = 12, .op1_addr = 12, .op1_length = 12,
            .out_addr = ACCUM_ALIAS_ADDRESS, .out_length = 12
          },
    [1] = { .opcode = TANH, .in_addr = ACCUM_ALIAS_ADDRESS, .in_length = 12,
            .op0_addr = NO_ADDRESS_ALIAS, .op0_length = 0,
            .op1_addr = NO_ADDRESS_ALIAS, .op1_length = 0,
            .out_addr = ACCUM_ALIAS_ADDRESS, .out_length = 12
          }
  };
//#pragma HLS reset varialbe=program

  //-- LOCAL VARIABLES ------------------------------------------------------
  TipsOp cur_op = program[next_command_pointer];


  if( !sNetworkLoadCmd.full() && !sOpLoadCmd.full() && !sAluInstr.full() )
  {
    if(cur_op.opcode == TIPS_NOP)
    {
      if(cur_op.in_addr == NETWORK_ALIAS_ADDRESS)
      {
        sNetworkLoadCmd.write({ .length = cur_op.in_length });
      } else {
        sNetworkLoadCmd.write({ .length = 0x0 });
      }
      TipsLoadInstr new_op_load = { .addr_0 = cur_op.op0_addr, .length_0 = cur_op.op0_length,
        .addr_1 = cur_op.op1_addr, .length_1 = cur_op.op1_length
      };
      sOpLoadCmd.write(new_op_load);

      TipsAluInstr new_alu_cmd = { .operation = cur_op.opcode, .in_addr = cur_op.in_addr,
        .in_length = cur_op.in_length, .op0_length = cur_op.op0_length,
        .op1_length = cur_op.op1_length, .out_addr = cur_op.out_addr,
        .out_length = cur_op.out_length
      };
      sAluInstr.write(new_alu_cmd);
    }

    next_command_pointer++;
    if(next_command_pointer >= DOSA_TIPS_PROGRAM_LENGTH)
    {
      next_command_pointer = 0x0;
    }
  }

  *debug = (uint8_t) next_command_pointer;
  *debug |= ((uint16_t) cur_op.opcode) << 8;
}


void pLoadOpDual(
    stream<TipsLoadInstr>     &sOpLoadCmd,
    stream<TipsAluInstr>      &sAluInstr_in,
    stream<TipsAluInstr>      &sAluInstr_out,
    stream<ap_uint<DOSA_TIPS_LONGEST_OP0> >  &sOp0,
    stream<ap_uint<DOSA_TIPS_LONGEST_OP1> >  &sOp1,
    uint16_t *debug
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  //-- STATIC VARIABLES -----------------------------------------------------
  const usedDtype opStore[DOSA_TIPS_ADDR_SPACE_LENGTH] = {
    10,10,10,11,11,11,12,12,12,13,13,13, //op0
    1,1,1,1,1,1,1,1,1,1,1,1, //op1
    0,0,0,0,0,0 //fill-to-end
  };
#pragma HLS ARRAY_PARTITION variable=opStore complete //TODO: to in-efficient?
  //for debugging
  static TipsLoadInstr last_instr;
  //-- LOCAL VARIABLES ------------------------------------------------------

  //use reset as init, no explicit init
  //also forward alu instr --> for feedback to control unit

  if( !sOpLoadCmd.empty() && !sAluInstr_in.empty()
      && !sAluInstr_out.full() && !sOp0.full() && !sOp1.full()
    )
  {
    TipsLoadInstr cur_instr = sOpLoadCmd.read();
    TipsAluInstr alu_instr_fw = sAluInstr_in.read();
    ap_uint<DOSA_TIPS_LONGEST_OP0> new_op0 = 0x0;
    //if(cur_instr.addr_0 < TIPS_MAX_ADDRESS)
    if(cur_instr.addr_0 < DOSA_TIPS_ADDR_SPACE_LENGTH)
    {
      for(int i = 0; i < DOSA_TIPS_LONGEST_OP0; i++)
      {
        if(i >= cur_instr.length_0)
        {
          continue;
        }
        //new_op0 |= ((ap_uint<DOSA_TIPS_LONGEST_OP0>) opStore[cur_instr.addr_0 + i]) << (i*DOSA_TIPS_USED_BITWIDTH);
        //0 at end?
        new_op0 |= ((ap_uint<DOSA_TIPS_LONGEST_OP0>) opStore[cur_instr.addr_0 + i]) << (DOSA_TIPS_LONGEST_OP0 - 1 - (i*DOSA_TIPS_USED_BITWIDTH));
      }
    }
    ap_uint<DOSA_TIPS_LONGEST_OP1> new_op1 = 0x0;
    //if(cur_instr.addr_1 < TIPS_MAX_ADDRESS)
    if(cur_instr.addr_1 < DOSA_TIPS_ADDR_SPACE_LENGTH)
    {
      for(int i = 0; i < DOSA_TIPS_LONGEST_OP1; i++)
      {
        if(i >= cur_instr.length_1)
        {
          continue;
        }
        //new_op0 |= ((ap_uint<DOSA_TIPS_LONGEST_OP0>) opStore[cur_instr.addr_0 + i]) << (i*DOSA_TIPS_USED_BITWIDTH);
        //0 at end?
        new_op1 |= ((ap_uint<DOSA_TIPS_LONGEST_OP1>) opStore[cur_instr.addr_1 + i]) << (DOSA_TIPS_LONGEST_OP1 - 1 - (i*DOSA_TIPS_USED_BITWIDTH));
      }
    }

    sAluInstr_out.write(alu_instr_fw);
    sOp0.write(new_op0);
    sOp0.write(new_op1);
    last_instr = cur_instr;
  }

  *debug = (uint16_t) last_instr;

}


void pALU(
    stream<TipsAluInstr>      &sAluInstr,
    stream<ap_uint<DOSA_TIPS_LONGEST_INPUT> >  &sNetworkInput,
    stream<ap_uint<DOSA_TIPS_LONGEST_OP0> >  &sOp0,
    stream<ap_uint<DOSA_TIPS_LONGEST_OP1> >  &sOp1,
    stream<TipsNetworkInstr>  &sNetworkStoreCmnd,
    stream<ap_uint<DOSA_TIPS_LONGEST_OUTPUT> >  &sNetworkOutput,
    uint16_t *debug
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static aluFSM aluState = INIT;
#pragma HLS reset variable=aluState
  static ap_uint<TIPS_ACCUM_LENGTH> accum = 0x0;
#pragma HLS reset variable=accum
  //-- STATIC VARIABLES -----------------------------------------------------
  //-- LOCAL VARIABLES ------------------------------------------------------

  //with internal accum
  //superscalar architecture: can schedule different alu operations in paralell --> later

  if(aluFSM == INIT)
  {

  } else {
  }

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

  *debug_out = (uint64_t) *debug0;
  *debug_out |= ((uint64_t) *debug1) << 16;
  *debug_out |= ((uint64_t) *debug2) << 32;
  *debug_out |= ((uint64_t) *debug3) << 48;

}


//DOSA_ADD_ip_name_BELOW
void tips_test(
    // ----- DOSA Interface -----
    stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >   &siData,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >  &soData,
    // ----- add potential DRAM Interface -----
    // ----- DEBUG out ------
    ap_uint<64> *debug_out
    )
{
  //-- DIRECTIVES FOR THE BLOCK ---------------------------------------------
#pragma HLS INTERFACE ap_ctrl_none port=return

#pragma HLS INTERFACE ap_fifo port=siData
#pragma HLS INTERFACE ap_fifo port=soData

#pragma HLS INTERFACE ap_ovld register port=debug_out


  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS DATAFLOW

#ifndef __SYNTHESIS__
  assert(DOSA_TIPS_ADDR_SPACE_LENGTH < 0xFFFF);
  assert(DOSA_TIPS_USED_BITWIDTH % 8 == 0); //only byte aligned for now
#endif

  //-- STATIC VARIABLES (with RESET) ------------------------------------------

  //-- STATIC DATAFLOW VARIABLES ------------------------------------------
  //streams with depth 1
  static stream<TipsNetworkInstr> sNetworkLoadCmd ("sNetworkLoadCmd");
  static stream<TipsLoadInstr> sOpLoadCmd ("sOpLoadCmd");
  static stream<TipsAluInstr> sAluInstr ("sAluInstr");
  static stream<ap_uint<DOSA_TIPS_LONGEST_INPUT> > sNetworkInput ("sNetworkInput");
  static stream<TipsAluInstr> sAluInstr_from_op_load ("sAluInstr_from_op_load");
  static stream<ap_uint<DOSA_TIPS_LONGEST_OP0> >  sOp0 ("sOp0");
  static stream<ap_uint<DOSA_TIPS_LONGEST_OP1> >  sOp1 ("sOp1");

  //-- LOCAL VARIABLES ------------------------------------------------------------
  uint16_t debug0 = 0;
  uint16_t debug1 = 0;
  uint16_t debug2 = 0;
  uint16_t debug3 = 0;

  //-- DATAFLOW PROCESS ---------------------------------------------------

  pTipsControl(sNetworkLoadCmd, sOpLoadCmd, sAluInstr, &debug0);

  pLoadNetwork(sNetworkLoadCmd, siData, sNetworkInput, &debug1);

  pLoadOpDual(sOpLoadCmd, sAluInstr, sAluInstr_from_op_load, sOp0, sOp1, &debug2);

  pMergeDebug(&debug0, &debug1, &debug2, &debug3, debug_out);
}



