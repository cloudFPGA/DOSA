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

#ifndef _TIPS_ENGINE_H_
#define _TIPS_ENGINE_H_

#include <stdio.h>
#include <stdint.h>
#include "ap_int.h"
#include "ap_utils.h"
#include <hls_stream.h>

#include "../../lib/axi_utils.hpp"
//#include "../../lib/interface_utils.hpp"
#include "alu.hpp"


using namespace hls;

//generated defines
//DOSA_REMOVE_START
#ifdef TIPS_TEST
#define DOSA_WRAPPER_INPUT_IF_BITWIDTH 64
#define DOSA_WRAPPER_OUTPUT_IF_BITWIDTH 64
#define DOSA_TIPS_LONGEST_INPUT 9  // 3x3
#define DOSA_TIPS_LONGEST_OP0 12   // 3x4
#define DOSA_TIPS_LONGEST_OP1 12
#define DOSA_TIPS_LONGEST_OUTPUT 12
#define DOSA_TIPS_PROGRAM_LENGTH 2
#define DOSA_TIPS_ADDR_SPACE_LENGTH 30
typedef int8_t usedDtype;
#define DOSA_TIPS_USED_BITWIDTH 8
typedef int32_t aluAccumDtype;
#define DOSA_TIPS_ALU_ACCUM_BITWIDTH 32
//#define DOSA_TIPS_ALU_BACK_CAST_BIT_SHIFT 5
#define DOSA_TIPS_USED_BITWIDTH_TKEEP 1
#define DOSA_TIPS_USED_BITWIDTH_TKEEP_WIDTH 1
#endif

//DOSA_REMOVE_STOP
//DOSA_ADD_INTERFACE_DEFINES

//TODO: define ALU ops dynamically based on what is needed?
//#define DOSA_TIPS_ALU_PARALLEL_SLOT 3  //but only one per operation

//#define MIN(a,b) (((a)<(b))?(a):(b))
//#define MAX(a,b) (((a)>(b))?(a):(b))

//derived defines
//#define TIPS_ACCUM_LENGTH (3 * DOSA_TIPS_LONGEST_OUTPUT)
#define TIPS_ACCUM_LENGTH DOSA_TIPS_LONGEST_OUTPUT
//#define TIPS_OP_LOOP_MAX MAX(DOSA_TIPS_LONGEST_OP0, DOSA_TIPS_LONGEST_OP1)

//as constants for HLS pragmas
//const uint32_t cnn_input_frame_size = (CNN_INPUT_FRAME_SIZE);


//independent defines
enum twoStatesFSM {RESET = 0, FORWARD};
enum aluFSM {INIT = 0, ALU};
enum LoadNetworkStates {RESET1 = 0, READ_INSTR, READ_NETWORK};

typedef uint8_t TipsOpcode;
typedef uint8_t AluOpcode;
typedef uint32_t TipsAddr;
#define TIPS_MAX_ADDRESS       0x0FFFF
typedef uint16_t TipsLength;
//typedef uint8_t TipsExecId;
//pseudo addresses
#define NETWORK_ALIAS_ADDRESS  0x10001  //network in/out
#define ACCUM_ALIAS_ADDRESS    0x10002  //to store in ALU accum?
#define ZERO_ALIAS_ADDRESS     0x10003
#define NO_ADDRESS_ALIAS       0x10004

//define TipsOpcode
#define TIPS_NOP    0
#define DENSE_BIAS  1
#define DENSE       2
#define RELU        3
#define TANH        4


struct TipsOp {
  TipsOpcode opcode;
  TipsAddr in_addr;
  TipsLength in_length;
  TipsAddr op0_addr;
  TipsLength op0_length;
  TipsAddr op1_addr;
  TipsLength op1_length;
  TipsAddr out_addr;
  TipsLength out_length;
  //bool out_forward; //re-use within execute?
};

struct TipsNetworkInstr {
  TipsLength length;
};

struct TipsLoadInstr {
  TipsAddr addr_0;
  TipsLength length_0;
  TipsAddr addr_1;
  TipsLength length_1;
};

struct TipsAluInstr {
  //TipsExecId id;
  TipsAluInstr operation;
  TipsAddr in_addr;
  TipsLength in_length;
  TipsLength op0_length;
  TipsLength op1_length;
  TipsAddr out_addr;
  TipsLength out_length;
}



//DOSA_ADD_ip_name_BELOW
void tips_test(
    // ----- DOSA Interface -----
    stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >   &siData,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >  &soData,
    // ----- add potential DRAM Interface -----
    // ----- DEBUG out ------
    ap_uint<64> *debug_out
);

#endif

