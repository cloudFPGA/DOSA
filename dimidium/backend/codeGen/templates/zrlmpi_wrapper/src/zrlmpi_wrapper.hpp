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

#include "../../lib/axi_utils.hpp"
#include "../../lib/interface_utils.hpp"
#include "zrlmpi_common.hpp"

using namespace hls;

#define MPI_INSTR_NOP 0
#define MPI_INSTR_SEND 1
#define MPI_INSTR_RECV 2
#define MPI_NO_RANK 0xFE


//generated defines
#ifdef WRAPPER_TEST
#define DOSA_WRAPPER_INPUT_IF_BITWIDTH 64
#define DOSA_WRAPPER_OUTPUT_IF_BITWIDTH 64
#define DOSA_WRAPPER_BUFFER_FIFO_DEPTH_LINES 1500
#define DOSA_WRAPPER_PROG_LENGTH 2
#else
//DOSA_ADD_INTERFACE_DEFINES
#endif

const uint32_t buffer_fifo_depth = DOSA_WRAPPER_BUFFER_FIFO_DEPTH_LINES;
enum controlFsmStates {RESET = 0, ISSUE_COMMAND, PROC_SEND, WAIT_SEND, PROC_RECEIVE, WAIT_RECEIVE};
enum deqBufferCmd {FORWARD_0 = 0, DRAIN_0, FORWARD_1, DRAIN_1};
enum sendBufferCmd {SEND_0 = 0, SEND_1};
enum recvEnqStates {RESET0 = 0, RECV_WAIT, RECV_BUF_0, RECV_BUF_1};
enum DeqStates {RESET1 = 0, WAIT_CMD, DEQ_FW_0, DEQ_FW_1, DEQ_DRAIN_0, DEQ_DRAIN_1};
enum sendEnqStates {RESET2 = 0, SEND_WAIT, SEND_BUF_0, SEND_BUF_1};
enum SendDeqStates {RESET3 = 0, WAIT_START, SEND_DEQ_0, SEND_CC_0, SEND_CC_1, SEND_DEQ_1, WAIT_OK, WAIT_DRAIN};


void zrlmpi_wrapper(
    // ----- FROM FMC -----
    ap_uint<32> role_rank_arg,
    ap_uint<32> cluster_size_arg,
    // ----- Wrapper Interface -----
    stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >   &siData,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >  &soData,
    // ----- MPI_Interface -----
    stream<MPI_Interface> *soMPIif,
    stream<MPI_Feedback> *siMPIFeB,
    stream<Axis<64> > *soMPI_data,
    stream<Axis<64> > *siMPI_data,
    // ----- DEBUG IO ------
    ap_uint<32> *debug_out
);

#endif

