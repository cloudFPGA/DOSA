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

#ifndef _ZRLMPI_WRAPPER_H_
#define _ZRLMPI_WRAPPER_H_

#include <stdio.h>
#include <stdint.h>
#include "ap_int.h"
#include "ap_utils.h"
#include <hls_stream.h>

#include "../../lib/interface_utils.hpp"
#include "../../lib/axi_utils.hpp"
#include "zrlmpi_common.hpp"

using namespace hls;

#define MPI_INSTR_NOP 0
#define MPI_INSTR_SEND 1
#define MPI_INSTR_RECV 2
#define MPI_NO_RANK 0xFE


//generated defines
//DOSA_REMOVE_START
#ifdef WRAPPER_TEST
#define DOSA_WRAPPER_INPUT_IF_BITWIDTH 64
#define DOSA_WRAPPER_OUTPUT_IF_BITWIDTH 64
#define DOSA_WRAPPER_DEFAULT_TKEEP 0xFF
#define DOSA_WRAPPER_BUFFER_FIFO_DEPTH_LINES 1500
#define DOSA_WRAPPER_PROG_LENGTH 2
#define DOSA_COMM_PLAN_AFTER_FILL_JUMP 0
#endif
//LESSON LEARNED: ifdef/else for constants that affect INTERFACES does not work with vivado HLS...it uses the first occurence, apparently...

//DOSA_REMOVE_STOP
//DOSA_ADD_INTERFACE_DEFINES

//derived defines
const uint32_t buffer_fifo_depth = DOSA_WRAPPER_BUFFER_FIFO_DEPTH_LINES;
#define DOSA_WRAPPER_INPUT_IF_BYTES_PER_LINE ((DOSA_WRAPPER_INPUT_IF_BITWIDTH+7)/8)

enum controlFsmStates {RESET = 0, WRITE_PROGRAM, LOAD_COMMAND, LOAD_COMMAND2, ISSUE_COMMAND, WAIT_DATA, PROC_SEND, WAIT_SEND, PROC_RECEIVE, WAIT_RECEIVE};
//                          0        1                2             3               4           5          6          7           8            9
enum deqBufferCmd {FORWARD_0 = 0, DRAIN_0, FORWARD_1, DRAIN_1};
enum sendBufferCmd {SEND_0 = 0, SEND_1};
enum recvEnqStates {RESET0 = 0, RECV_WAIT, RECV_BUF_0, RECV_BUF_1, WAIT_CONFIRMATION};
enum DeqStates {RESET1 = 0, WAIT_CMD, DEQ_FW_0, DEQ_FW_1, DEQ_DRAIN_0, DEQ_DRAIN_1};
enum sendEnqStates {RESET2 = 0, SEND_WAIT, SEND_BUF_0_INIT, SEND_BUF_0, SEND_BUF_0_HANGOVER, SEND_BUF_1_INIT, SEND_BUF_1, SEND_BUF_1_HANGOVER};
//enum SendDeqStates {RESET3 = 0, WAIT_START, SEND_DEQ_0, SEND_CC_0, SEND_CC_1, SEND_DEQ_1, WAIT_OK, WAIT_DRAIN};
enum SendDeqStates {RESET3 = 0, WAIT_START, SEND_DEQ_0, SEND_CC_0, SEND_CC_1, SEND_DEQ_1, WAIT_OK, DRAIN_CC_0, DRAIN_CC_1};


void zrlmpi_wrapper(
    // ----- FROM FMC -----
    ap_uint<32>   *role_rank_arg,
    ap_uint<32>   *cluster_size_arg,
    // ----- Wrapper Interface -----
    stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >   &siData,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >  &soData,
    // ----- MPI_Interface -----
    stream<MPI_Interface>     &soMPIif,
    stream<MPI_Feedback>      &siMPIFeB,
    stream<Axis<64> >         &soMPI_data,
    stream<Axis<64> >         &siMPI_data,
    // ----- DEBUG out ------
    ap_uint<80> *debug_out,
    ap_uint<32> *debug_out_ignore
);

#endif

