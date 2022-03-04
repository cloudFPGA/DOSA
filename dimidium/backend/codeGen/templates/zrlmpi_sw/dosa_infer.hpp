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
#define DOSA_MINIMAL_PROG_LENGTH 2
#else
//DOSA_ADD_APP_NODE_DEFINES
#endif

#define DEBUG
//#define DEBUG2

//DOCSTRING...
extern "C" void init(int argc, char **argv);

//DOCSTRING...
extern "C" void reset_state();

//DOCSTRING...
extern "C" int infer(int *input, uint32_t input_length, int *output, uint32_t output_length);


#endif

