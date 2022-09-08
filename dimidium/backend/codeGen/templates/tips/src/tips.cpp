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
//#include <vector>
#include <array>
#define AP_INT_MAX_W 4096
#include "ap_int.h"
#include "ap_utils.h"
#include <hls_stream.h>
//#include <hls_math.h>
#include <cassert>

#include "tips.hpp"
#include "alu.hpp"

using namespace hls;


void pLoadNetwork(
    stream<TipsNetworkInstr>  &sNetworkLoadCmd,
    stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >   &siData,
    stream<ap_uint<DOSA_TIPS_LONGEST_INPUT*DOSA_TIPS_USED_BITWIDTH> >  &sNetworkInput,
    //stream<std:array<usedDtype, DOSA_TIPS_LONGEST_INPUT> >  &sNetworkInput, TODO: later
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
  static ap_uint<DOSA_TIPS_LONGEST_INPUT*DOSA_TIPS_USED_BITWIDTH> cur_input;
  static ap_uint<DOSA_TIPS_LONGEST_INPUT*DOSA_TIPS_USED_BITWIDTH> hangover_store;
  static ap_uint<DOSA_TIPS_LONGEST_INPUT*DOSA_TIPS_USED_BITWIDTH> mask;
  //-- LOCAL VARIABLES ------------------------------------------------------

  //TODO: update for non int8 types (i.e. different length)
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
      if( !sNetworkLoadCmd.empty() && !sNetworkInput.full() )  //no siData
      {
        req_input_length = sNetworkLoadCmd.read().length*(DOSA_TIPS_USED_BITWIDTH/8);
        printf("[pLoadNetwork] new req_input_length: %d Bytes;\n", (uint32_t) req_input_length);
        if(req_input_length == 0)
        {
          sNetworkInput.write(0x0);
          //stay here
        } else {
          //mask = ((ap_uint<DOSA_TIPS_LONGEST_INPUT*DOSA_TIPS_USED_BITWIDTH>) exp2(req_input_length*8))-1;
          mask = 0x0;
          for(uint32_t i = 0; i < DOSA_TIPS_LONGEST_INPUT; i++)
          {
            if(i < req_input_length)
            {
              mask |= ((ap_uint<DOSA_TIPS_LONGEST_INPUT*DOSA_TIPS_USED_BITWIDTH>) DOSA_TIPS_USED_BITWIDTH_PARTIAL_MASK) << i*DOSA_TIPS_USED_BITWIDTH;
            }
          }
          printf("[pLoadNetwork] using mask %llX\n", (uint64_t) mask);

          if(hangover_valid_bytes >= req_input_length)
          {
            ap_uint<DOSA_TIPS_LONGEST_INPUT*DOSA_TIPS_USED_BITWIDTH> out_vector = hangover_store & mask;
            sNetworkInput.write(out_vector);
            hangover_valid_bytes -= req_input_length;
            hangover_store >>= req_input_length * 8;
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
        //printf("cur_length: %d\n", cur_length);
        Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> in_word = siData.read();
        TipsLength byte_read = extractByteCnt(in_word);
        cur_input |= ((ap_uint<DOSA_TIPS_LONGEST_INPUT*DOSA_TIPS_USED_BITWIDTH>) in_word.getTData()) << cur_length*8;
        cur_length += byte_read;
        //ignore tlast?
        if(cur_length >= req_input_length)
        {
            //ap_uint<DOSA_TIPS_LONGEST_INPUT*DOSA_TIPS_USED_BITWIDTH> mask = exp2(req_input_length*8)-1;
            ap_uint<DOSA_TIPS_LONGEST_INPUT*DOSA_TIPS_USED_BITWIDTH> out_vector = cur_input & mask;
            sNetworkInput.write(out_vector);
            printf("[pLoadNetwork] forwarding (last 64 bits): %16.16llX\n", (uint64_t) out_vector);
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
    stream<TipsNetworkInstr>  &sNetworkStoreCmnd,
    stream<ap_uint<DOSA_TIPS_LONGEST_OUTPUT*DOSA_TIPS_USED_BITWIDTH> >  &sNetworkOutput,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >  &soData,
    uint16_t *debug
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static StoreNetworkStates storeNetFSM = RESET2;
#pragma HLS reset variable=storeNetFSM
  static TipsLength cur_length = 0;
#pragma HLS reset variable=cur_length
  static TipsLength req_output_length = 0;
#pragma HLS reset variable=req_output_length
  //-- STATIC VARIABLES -----------------------------------------------------
  static ap_uint<DOSA_TIPS_LONGEST_OUTPUT*DOSA_TIPS_USED_BITWIDTH> cur_output;
  //-- LOCAL VARIABLES ------------------------------------------------------

  switch(storeNetFSM)
  {
    default:
    case RESET2:
      //necessary?
      cur_length = 0;
      req_output_length = 0;
      cur_output = 0;
      storeNetFSM = READ_INSTR1;
      break;

    case READ_INSTR1:
      if( !sNetworkStoreCmnd.empty() && !sNetworkOutput.empty()
          && !soData.full() )
      {
        req_output_length = sNetworkStoreCmnd.read().length*(DOSA_TIPS_USED_BITWIDTH/8);
        printf("[pSendNetwork] new req_output_length: %d Bytes;\n", (uint32_t) req_output_length);
        cur_output = sNetworkOutput.read();
        printf("[pSendNetwork] forwarding (last 64 bits): %16.16llX\n", (uint64_t) cur_output);
        cur_length = 0;
        if(req_output_length > 0)
        {
          storeNetFSM = WRITE_NETWORK;
        }
      }
      break;

    case WRITE_NETWORK:
      if( !soData.full() )
      {
        ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> new_tdata = (ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH>) (cur_output >> cur_length);
        Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> out_word = Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH>(new_tdata);
        //TODO: make type dynamic
        uint8_t tmp_cnt = DOSA_WRAPPER_OUTPUT_BYTES_PER_LINE;
        out_word.setTLast(0);
        if( (req_output_length - cur_length) <= DOSA_WRAPPER_OUTPUT_BYTES_PER_LINE)
        {
          tmp_cnt = req_output_length - cur_length;
          out_word.setTLast(1);
          storeNetFSM = READ_INSTR1;
        }
        ap_uint<(DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+7)/8> new_tkeep = byteCntToTKeep(tmp_cnt);
        out_word.setTKeep(new_tkeep);
        soData.write(out_word);
        cur_length += tmp_cnt;
      }
      break;
  }

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
#pragma HLS reset variable=next_command_pointer
  //-- STATIC VARIABLES -----------------------------------------------------
#ifdef TIPS_TEST
  const TipsOp program[] = {
    [0] = { .opcode = DENSE_BIAS, .op_param = 4,
            .in_addr = NETWORK_ALIAS_ADDRESS, .in_length = 4,
            .op0_addr = 0, .op0_length = 12, .op1_addr = 12, .op1_length = 3,
            .out_addr = NETWORK_ALIAS_ADDRESS, .out_length = 3
          },
    [1] = { .opcode = TANH, .op_param = 0,
            .in_addr = NETWORK_ALIAS_ADDRESS, .in_length = 3,
            .op0_addr = NO_ADDRESS_ALIAS, .op0_length = 0,
            .op1_addr = NO_ADDRESS_ALIAS, .op1_length = 0,
            .out_addr = NETWORK_ALIAS_ADDRESS, .out_length = 3
          }
  };
#else
  //DOSA_ADD_program
#endif
#pragma HLS RESOURCE variable=program core=ROM_2P_BRAM

  //-- LOCAL VARIABLES ------------------------------------------------------
  TipsOp cur_op = program[next_command_pointer];


  if( !sNetworkLoadCmd.full() && !sOpLoadCmd.full() && !sAluInstr.full() )
  {
#ifndef __SYNTHESIS__
    if(next_command_pointer < DOSA_TIPS_PROGRAM_LENGTH)
    {
      printf("[pTipsControl] processing operation at postion %d: opcode %d, op_param %d, in_addr %d, in_length %d, op0_addr %d, op0_length %d, op1_addr %d, op1_length %d, out_addr %d, out_length %d;\n", next_command_pointer, cur_op.opcode, cur_op.op_param, cur_op.in_addr, cur_op.in_length, cur_op.op0_addr, cur_op.op0_length, cur_op.op1_addr, cur_op.op1_length, cur_op.out_addr, cur_op.out_length);
#endif
    if(cur_op.opcode != TIPS_NOP)
    {
      if(cur_op.in_addr == NETWORK_ALIAS_ADDRESS)
      {
        TipsNetworkInstr ni = { .length = cur_op.in_length };
        sNetworkLoadCmd.write(ni);
      } else {
        TipsNetworkInstr ni = { .length = 0x0 };
        sNetworkLoadCmd.write(ni);
      }
      TipsLoadInstr new_op_load = { .addr_0 = cur_op.op0_addr, .length_0 = cur_op.op0_length,
        .addr_1 = cur_op.op1_addr, .length_1 = cur_op.op1_length};
      printf("new_op_load: %d, %d, %d, %d\n", new_op_load.addr_0, new_op_load.length_0, new_op_load.addr_1, new_op_load.length_1);
      sOpLoadCmd.write(new_op_load);

      TipsAluInstr new_alu_cmd = { .operation = cur_op.opcode, .op_param = cur_op.op_param,
        .in_addr = cur_op.in_addr,
        .in_length = cur_op.in_length, .op0_length = cur_op.op0_length,
        .op1_length = cur_op.op1_length, .out_addr = cur_op.out_addr,
        .out_length = cur_op.out_length
      };
      sAluInstr.write(new_alu_cmd);
      printf("[pTipsControl] Issuing %d ALU instr at position %d.\n", (uint8_t) cur_op.opcode, (uint8_t) next_command_pointer);
    }

    next_command_pointer++;
#ifdef __SYNTHESIS__
    if(next_command_pointer >= DOSA_TIPS_PROGRAM_LENGTH)
    {
      next_command_pointer = 0;
#endif
    }
  }

  *debug = (uint8_t) next_command_pointer;
  *debug |= ((uint16_t) cur_op.opcode) << 8;
}


void pLoadOpDual(
    stream<TipsLoadInstr>     &sOpLoadCmd,
    stream<TipsAluInstr>      &sAluInstr_in,
    stream<TipsAluInstr>      &sAluInstr_out,
    //stream<ap_uint<DOSA_TIPS_LONGEST_OP0*DOSA_TIPS_USED_BITWIDTH> >  &sOp0,
    stream<std::array<usedDtype, DOSA_TIPS_LONGEST_OP0> >  &sOp0,
    //stream<ap_uint<DOSA_TIPS_LONGEST_OP1*DOSA_TIPS_USED_BITWIDTH> >  &sOp1,
    stream<std::array<usedDtype, DOSA_TIPS_LONGEST_OP1> >  &sOp1,
    uint16_t *debug
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
//#pragma HLS pipeline //II=1
#pragma HLS pipeline II=alu_op_pipeline_ii
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  //-- STATIC VARIABLES -----------------------------------------------------
#ifdef TIPS_TEST
  const usedDtype opStore[DOSA_TIPS_ADDR_SPACE_LENGTH] = {
    //uin16_t 10,11,12,13,21,22,23,24,31,32,33,34, //op0
    //fixed ['0x28', '0x2c', '0x30', '0x34', '0x54', '0x58', '0x5c', '0x60', '0x7c', '0x80', '0x84', '0x88']
    0x28, 0x2c, 0x30, 0x34, 0x54, 0x58, 0x5c, 0x60, 0x7c, 0x80, 0x84, 0x88,
    4,4,4, //op1
    0,0,0,0,0 //fill-to-end
  };
#else
  //DOSA_ADD_op_store
#endif
//#pragma HLS ARRAY_PARTITION variable=opStore complete //TODO: too in-efficient?
#pragma HLS RESOURCE variable=opStore core=ROM_2P_BRAM
  //for debugging
  static TipsLoadInstr last_instr;
  //-- LOCAL VARIABLES ------------------------------------------------------
  //std::vector<usedDtype, DOSA_TIPS_LONGEST_OP0> new_op0(DOSA_TIPS_LONGEST_OP0, 0);
  std::array<usedDtype, DOSA_TIPS_LONGEST_OP0> new_op0;
  std::array<usedDtype, DOSA_TIPS_LONGEST_OP1> new_op1;
  //use reset as init, no explicit init
  //also forward alu instr --> for feedback to control unit

  if( !sOpLoadCmd.empty() && !sAluInstr_in.empty()
      && !sAluInstr_out.full() && !sOp0.full() && !sOp1.full()
    )
  {
    TipsLoadInstr cur_instr = sOpLoadCmd.read();
    //printf("cur_instr: %d, %d, %d, %d\n", cur_instr.addr_0, cur_instr.length_0, cur_instr.addr_1, cur_instr.length_1);
    TipsAluInstr alu_instr_fw = sAluInstr_in.read();
    //ap_uint<DOSA_TIPS_LONGEST_OP0*DOSA_TIPS_USED_BITWIDTH> new_op0 = 0x0;
    //if(cur_instr.addr_0 < TIPS_MAX_ADDRESS)
    if(cur_instr.addr_0 < DOSA_TIPS_ADDR_SPACE_LENGTH)
    {
      for(int i = 0; i < DOSA_TIPS_LONGEST_OP0; i++)
      {
#pragma HLS unroll factor=512
        if(i >= cur_instr.length_0)
        {
          new_op0[i] = 0x0;
        }
        else {
          new_op0[i] = opStore[cur_instr.addr_0 + i];
        }
        //new_op0 |= ((ap_uint<DOSA_TIPS_LONGEST_OP0*DOSA_TIPS_USED_BITWIDTH>) opStore[cur_instr.addr_0 + i]) << (i*DOSA_TIPS_USED_BITWIDTH);
        //0 at end?
        //new_op0 |= ((ap_uint<DOSA_TIPS_LONGEST_OP0*DOSA_TIPS_USED_BITWIDTH>) opStore[cur_instr.addr_0 + i]) << (DOSA_TIPS_LONGEST_OP0 - 1 - (i*DOSA_TIPS_USED_BITWIDTH));
      }
    }
    //ap_uint<DOSA_TIPS_LONGEST_OP1*DOSA_TIPS_USED_BITWIDTH> new_op1 = 0x0;
    //if(cur_instr.addr_1 < TIPS_MAX_ADDRESS)
    if(cur_instr.addr_1 < DOSA_TIPS_ADDR_SPACE_LENGTH)
    {
      for(int i = 0; i < DOSA_TIPS_LONGEST_OP1; i++)
      {
#pragma HLS unroll factor=512
        if(i >= cur_instr.length_1)
        {
          new_op1[i] = 0x0;
          //continue;
        } else {
          new_op1[i] = opStore[cur_instr.addr_1 + 1];
        }
        //new_op1 |= ((ap_uint<DOSA_TIPS_LONGEST_OP1*DOSA_TIPS_USED_BITWIDTH>) opStore[cur_instr.addr_1 + i]) << (i*DOSA_TIPS_USED_BITWIDTH);
        //0 at end?
        //new_op1 |= ((ap_uint<DOSA_TIPS_LONGEST_OP1*DOSA_TIPS_USED_BITWIDTH>) opStore[cur_instr.addr_1 + i]) << (DOSA_TIPS_LONGEST_OP1 - 1 - (i*DOSA_TIPS_USED_BITWIDTH));
      }
    }

    sAluInstr_out.write(alu_instr_fw);
    sOp0.write(new_op0);
    sOp1.write(new_op1);
    printf("[pLoadOpDual] forwarding Op0 from position %d (last 64 bits): %16.16llX\n", (uint32_t) cur_instr.addr_0, (uint64_t) new_op0[0]); //TODO
    printf("[pLoadOpDual] forwarding Op1 from position %d (last 64 bits): %16.16llX\n", (uint32_t) cur_instr.addr_1, (uint64_t) new_op1[0]); //TODO
    last_instr = cur_instr;
  }

  *debug = (uint8_t) last_instr.addr_0;
  *debug |= (uint16_t) (last_instr.addr_1 << 8);

}


void pALU(
    stream<TipsAluInstr>      &sAluInstr,
    stream<ap_uint<DOSA_TIPS_LONGEST_INPUT*DOSA_TIPS_USED_BITWIDTH> >  &sNetworkInput,
    //stream<ap_uint<DOSA_TIPS_LONGEST_OP0*DOSA_TIPS_USED_BITWIDTH> >  &sOp0,
    //stream<usedDtype[DOSA_TIPS_LONGEST_OP0] >  &sOp0,
    stream<std::array<usedDtype, DOSA_TIPS_LONGEST_OP0> >  &sOp0,
    //stream<ap_uint<DOSA_TIPS_LONGEST_OP1*DOSA_TIPS_USED_BITWIDTH> >  &sOp1,
    stream<std::array<usedDtype, DOSA_TIPS_LONGEST_OP1> >  &sOp1,
    stream<TipsNetworkInstr>  &sNetworkStoreCmnd,
    stream<ap_uint<DOSA_TIPS_LONGEST_OUTPUT*DOSA_TIPS_USED_BITWIDTH> >  &sNetworkOutput,
    uint16_t *debug
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
//#pragma HLS pipeline II=512
#pragma HLS pipeline II=alu_op_pipeline_ii
//#pragma HLS DATAFLOW
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static aluFSM aluState = INIT;
#pragma HLS reset variable=aluState
  //-- STATIC VARIABLES -----------------------------------------------------
  static ap_uint<TIPS_ACCUM_LENGTH*DOSA_TIPS_USED_BITWIDTH> accum;
  static aluAccumDtype tanh_table[N_TABLE];
//  static usedDtype accum_scratchpad[TIPS_ACCUM_LENGTH];
//#pragma HLS ARRAY_PARTITION variable=accum_scratchpad complete
  //-- LOCAL VARIABLES ------------------------------------------------------
  TipsAluInstr cur_instr;
  TipsNetworkInstr ni;
  ap_uint<DOSA_TIPS_LONGEST_INPUT*DOSA_TIPS_USED_BITWIDTH> cur_network_in;
  //ap_uint<DOSA_TIPS_LONGEST_OP0*DOSA_TIPS_USED_BITWIDTH> cur_op0;
  //ap_uint<DOSA_TIPS_LONGEST_OP1*DOSA_TIPS_USED_BITWIDTH> cur_op1;
  ap_uint<DOSA_TIPS_LONGEST_INPUT*DOSA_TIPS_USED_BITWIDTH> cur_input;
  ap_uint<DOSA_TIPS_LONGEST_OUTPUT*DOSA_TIPS_USED_BITWIDTH> cur_output;
  quantDtype input_scratchpad[DOSA_TIPS_LONGEST_INPUT];
//#pragma HLS ARRAY_PARTITION variable=input_scratchpad complete
  quantDtype op0_scratchpad[DOSA_TIPS_LONGEST_OP0];
//#pragma HLS ARRAY_PARTITION variable=op0_scratchpad complete
//#pragma HLS ARRAY_PARTITION variable=op0_scratchpad cyclic factor=512
  quantDtype op1_scratchpad[DOSA_TIPS_LONGEST_OP1];
//#pragma HLS ARRAY_PARTITION variable=op1_scratchpad complete
//#pragma HLS ARRAY_PARTITION variable=op1_scratchpad cyclic factor=512
  quantDtype output_scratchpad[DOSA_TIPS_LONGEST_OUTPUT];
//#pragma HLS ARRAY_PARTITION variable=output_scratchpad complete
  //usedDtype cur_op0[DOSA_TIPS_LONGEST_OP0];
  std::array<usedDtype, DOSA_TIPS_LONGEST_OP0> cur_op0;
  std::array<usedDtype, DOSA_TIPS_LONGEST_OP1> cur_op1;

//TODO: customize ALU for each Brick/Block?

  if(aluState == INIT)
  {
    init_tanh_table(tanh_table);
    accum = 0x0;
    aluState = ALU;
  } else {
    if(!sAluInstr.empty() && !sNetworkInput.empty() && !sOp0.empty() && !sOp1.empty()
        && !sNetworkStoreCmnd.full() && !sNetworkOutput.full()
      )
    {
      //read all inputs to clear streams
      cur_instr = sAluInstr.read();
      cur_network_in = sNetworkInput.read();
      //cur_op0 = sOp0.read();
      sOp0.read(cur_op0);
      //cur_op1 = sOp1.read();
      sOp1.read(cur_op1);
      cur_input = 0x0;
      if(cur_instr.in_addr == ACCUM_ALIAS_ADDRESS)
      {
        cur_input = (ap_uint<DOSA_TIPS_LONGEST_INPUT*DOSA_TIPS_USED_BITWIDTH>) accum;
      } else if(cur_instr.in_addr == NETWORK_ALIAS_ADDRESS)
      {
        cur_input = cur_network_in;
      }
      for(int i = 0; i < DOSA_TIPS_LONGEST_INPUT; i++)
      {
//#pragma HLS unroll
        if( i >= cur_instr.in_length )
        {
          input_scratchpad[i] = 0x0;
        }
        input_scratchpad[i] = (quantDtype) ((usedDtype) (cur_input >> (i*DOSA_TIPS_USED_BITWIDTH)));
        //printf("input_scratchpad[%d]: %d\n", i, (uint16_t) (input_scratchpad[i] >> DEBUG_FRACTIONAL_BITS));
      }
      for(int i = 0; i < DOSA_TIPS_LONGEST_OUTPUT; i++)
      {
//#pragma HLS unroll
        output_scratchpad[i] = 0x0;
      }
      //"type cast" op vectors
      for(int i = 0; i < DOSA_TIPS_LONGEST_OP0; i++)
      {
//#pragma HLS unroll factor=512
        if( i >= cur_instr.op0_length )
        {
          op0_scratchpad[i] = 0x0;
        }
        //op0_scratchpad[i] = (quantDtype) ((usedDtype) (cur_op0 >> (i*DOSA_TIPS_USED_BITWIDTH)));
        op0_scratchpad[i] = (quantDtype) cur_op0[i];
      }
      for(int i = 0; i < DOSA_TIPS_LONGEST_OP1; i++)
      {
//#pragma HLS unroll factor=512
        if( i >= cur_instr.op1_length )
        {
          op1_scratchpad[i] = 0x0;
        }
        //op1_scratchpad[i] = (quantDtype) ((usedDtype) (cur_op1 >> (i*DOSA_TIPS_USED_BITWIDTH)));
        op1_scratchpad[i] = (quantDtype) cur_op1[i];
      }

      //process AluOp
      printf("[pALU] Executing %d AluOp with param %d.\n", (uint8_t) cur_instr.operation, cur_instr.op_param);
      switch(cur_instr.operation)
      {
        default:
        case TIPS_NOP:
          accum = 0x0;
          ni = (TipsNetworkInstr) { .length = 0x0 };
          sNetworkStoreCmnd.write(ni);
          sNetworkOutput.write(0x0);
          break;

        case DENSE:
          //is the same, since bias is then already at 0
        case DENSE_BIAS:
#ifndef __SYNTHESIS__
          assert(cur_instr.op_param == cur_instr.in_length);
#endif
          //void dense(usedDtype data[DOSA_TIPS_LONGEST_INPUT], usedDtype res[DOSA_TIPS_LONGEST_OUTPUT], usedDtype weights[DOSA_TIPS_LONGEST_OP0], usedDtype biases[DOSA_TIPS_LONGEST_OP1], int m);
          //dense(input_scratchpad, output_scratchpad, op0_scratchpad, op1_scratchpad, (int) cur_instr.op_param);
          dense(input_scratchpad, output_scratchpad, op0_scratchpad, op1_scratchpad, 4);
          break;

        case RELU:
          //void relu(usedDtype data[DOSA_TIPS_LONGEST_OUTPUT], usedDtype res[DOSA_TIPS_LONGEST_OUTPUT]);
          relu(input_scratchpad, output_scratchpad);
          break;

        case TANH:
          //void tanh(usedDtype data[DOSA_TIPS_LONGEST_OUTPUT], usedDtype res[DOSA_TIPS_LONGEST_OUTPUT], usedDtype tanh_table[N_TABLE]);
          tanh(input_scratchpad, output_scratchpad, tanh_table);
          break;
      }

      //write output
      cur_output = 0x0;
      for(int i = 0; i < DOSA_TIPS_LONGEST_OUTPUT; i++)
      {
//#pragma HLS unroll
        if( i >= cur_instr.out_length)
        {
          continue;
        }
        cur_output |= ((ap_uint<DOSA_TIPS_LONGEST_OUTPUT*DOSA_TIPS_USED_BITWIDTH>) ((usedDtype) output_scratchpad[i])) << (i*DOSA_TIPS_USED_BITWIDTH);
      }
      if(cur_instr.out_addr == NETWORK_ALIAS_ADDRESS)
      {
        ni = (TipsNetworkInstr) { .length = cur_instr.out_length };
        sNetworkStoreCmnd.write(ni);
        sNetworkOutput.write(cur_output);
        printf("[pALU] forwarding to network store (last 64 bits): %16.16llX\n", (uint64_t) cur_output);
      } else if(cur_instr.out_addr == ACCUM_ALIAS_ADDRESS)
      {
        accum = (ap_uint<TIPS_ACCUM_LENGTH>) cur_output;
      }
    }
  }

  *debug = (uint16_t) cur_instr.operation;
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

  *debug_out =   (ap_uint<80>) *debug0;
  *debug_out |= ((ap_uint<80>) *debug1) << 16;
  *debug_out |= ((ap_uint<80>) *debug2) << 32;
  *debug_out |= ((ap_uint<80>) *debug3) << 48;
  *debug_out |= ((ap_uint<80>) *debug4) << 64;

}


//DOSA_ADD_ip_name_BELOW
void tips_test(
    // ----- DOSA Interface -----
    stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >   &siData,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >  &soData,
    // ----- add potential DRAM Interface -----
    // ----- DEBUG out ------
    ap_uint<80> *debug_out
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
  assert(DOSA_WRAPPER_INPUT_IF_BITWIDTH == 64);
  assert(DOSA_WRAPPER_OUTPUT_IF_BITWIDTH == 64);
#endif

  //-- STATIC VARIABLES (with RESET) ------------------------------------------

  //-- STATIC DATAFLOW VARIABLES ------------------------------------------
  //streams with depth 1
  static stream<TipsNetworkInstr> sNetworkLoadCmd ("sNetworkLoadCmd");
#pragma HLS STREAM variable=sNetworkLoadCmd depth=1
  static stream<TipsLoadInstr> sOpLoadCmd ("sOpLoadCmd");
#pragma HLS STREAM variable=sOpLoadCm depth=1
  static stream<TipsAluInstr> sAluInstr ("sAluInstr");
#pragma HLS STREAM variable=sAluInstr depth=1
  static stream<ap_uint<DOSA_TIPS_LONGEST_INPUT*DOSA_TIPS_USED_BITWIDTH> > sNetworkInput ("sNetworkInput");
#pragma HLS STREAM variable=sNetworkInput depth=1
  static stream<TipsAluInstr> sAluInstr_from_op_load ("sAluInstr_from_op_load");
#pragma HLS STREAM variable=sAluInstr_from_op_loa depth=1
  //static stream<ap_uint<DOSA_TIPS_LONGEST_OP0*DOSA_TIPS_USED_BITWIDTH> >  sOp0 ("sOp0");
  //static stream<usedDtype[DOSA_TIPS_LONGEST_OP0] >  sOp0 ("sOp0");
  static stream<std::array<usedDtype, DOSA_TIPS_LONGEST_OP0> >  sOp0 ("sOp0");
#pragma HLS STREAM variable=sOp0 depth=1
  //static stream<ap_uint<DOSA_TIPS_LONGEST_OP1*DOSA_TIPS_USED_BITWIDTH> >  sOp1 ("sOp1");
  static stream<std::array<usedDtype, DOSA_TIPS_LONGEST_OP1> >  sOp1 ("sOp1");
#pragma HLS STREAM variable=sOp1 depth=1
  static stream<TipsNetworkInstr>  sNetworkStoreCmnd ("sNetworkStoreCmnd");
#pragma HLS STREAM variable=sNetworkStoreCmn depth=1
  static stream<ap_uint<DOSA_TIPS_LONGEST_OUTPUT*DOSA_TIPS_USED_BITWIDTH> >  sNetworkOutput ("sNetworkOutput");
#pragma HLS STREAM variable=sNetworkOutput depth=1

  //-- LOCAL VARIABLES ------------------------------------------------------------
  uint16_t debug0 = 0;
  uint16_t debug1 = 0;
  uint16_t debug2 = 0;
  uint16_t debug3 = 0;
  uint16_t debug4 = 0;

  //-- DATAFLOW PROCESS ---------------------------------------------------

  pTipsControl(sNetworkLoadCmd, sOpLoadCmd, sAluInstr, &debug0);

  pLoadNetwork(sNetworkLoadCmd, siData, sNetworkInput, &debug1);

  pSendNetwork(sNetworkStoreCmnd, sNetworkOutput, soData, &debug4);

  //slow processes last

  pLoadOpDual(sOpLoadCmd, sAluInstr, sAluInstr_from_op_load, sOp0, sOp1, &debug2);

  pALU(sAluInstr_from_op_load, sNetworkInput, sOp0, sOp1, sNetworkStoreCmnd, sNetworkOutput, &debug3);

  pMergeDebug(&debug0, &debug1, &debug2, &debug3, &debug4, debug_out);
}



