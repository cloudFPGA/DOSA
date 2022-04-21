//  *
//  *                       cloudFPGA
//  *     Copyright IBM Research, All Rights Reserved
//  *    =============================================
//  *     Created: Feb 2022
//  *     Authors: NGL
//  *
//  *     Description:
//  *        C++ module to wrap ZRLMPI communication library
//  *
//  *


#ifndef _DOSA_INFER_H_
#define _DOSA_INFER_H_

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "ZRLMPI.hpp"


#define MPI_INSTR_NOP 0
#define MPI_INSTR_SEND 1
#define MPI_INSTR_RECV 2
#define MPI_NO_RANK 0xFE

//generated defines
#ifdef WRAPPER_TEST
#define DOSA_WRAPPER_PROG_LENGTH 2
//#define DOSA_MINIMAL_PROG_LENGTH 2
#define DOSA_PIPELINE_STORE_DETPH 1
#define DOSA_MINIMAL_INPUT_NUM 1
#define DOSA_MINIMAL_OUTPUT_NUM 1
#else
//DOSA_ADD_APP_NODE_DEFINES
#endif

#define DEBUG
//#define DEBUG2

//DOCSTRING...
extern "C" void init(int argc, char **argv);

//DOCSTRING...
extern "C" void reset_state(void);

//DOCSTRING...
extern "C" uint32_t get_pipeline_store_depth(void);

//DOCSTRING...
extern "C" uint32_t get_batch_input_size(void);

//DOCSTRING...
extern "C" uint32_t get_batch_output_size(void);

//DOCSTRING...
//extern "C" int infer(int *input, uint32_t input_length, int *output, uint32_t output_length);
//TODO: also allow single frame inference?
extern "C" int infer_batch(int *input, uint32_t input_num, int *output, uint32_t output_num);


#endif

