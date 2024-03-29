/*******************************************************************************
 * Copyright 2019 -- 2024 IBM Corporation
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
#define DOSA_WRAPPER_MAX_PROG_LENGTH 2
//#define DOSA_MINIMAL_PROG_LENGTH 2
//#define DOSA_PIPELINE_STORE_DETPH 1
//#define DOSA_MINIMAL_INPUT_NUM 1
//#define DOSA_MINIMAL_OUTPUT_NUM 1
//#define DOSA_COMM_PLAN_AFTER_FILL_JUMP 0
#define DOSA_PIPELINE_FULL_BATCH_SIZE 1
#define DOSA_MAX_PARALLEL_RANKS 1
#else
//DOSA_ADD_APP_NODE_DEFINES
#endif

//#define DEBUG
//#define DEBUG2

//DOCSTRING...
extern "C" void init(int argc, char **argv);

//DOCSTRING...
extern "C" void cleanup(void);

//DOCSTRING...
extern "C" void reset_state(void);

//DOCSTRING...
extern "C" uint32_t get_pipeline_store_depth(void);

//DOCSTRING...
extern "C" uint32_t get_batch_input_size(void);

//DOCSTRING...
extern "C" uint32_t get_batch_output_size(void);

//DOCSTRING...
extern "C" uint32_t get_pipeline_full_batch_size(void);

//DOCSTRING...
extern "C" bool are_processing_pipelines_filled(void);

//DOCSTRING...
//extern "C" int infer(int *input, uint32_t input_length, int *output, uint32_t output_length);
//TODO: also allow single frame inference?
extern "C" int infer_batch(char *input, uint32_t input_num, char *output, uint32_t output_num);


#endif

