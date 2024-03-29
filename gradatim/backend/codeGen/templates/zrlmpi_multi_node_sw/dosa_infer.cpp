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
//  *     Created: Mar 2023
//  *     Authors: NGL
//  *
//  *     Description:
//  *        C++ module to wrap ZRLMPI communication library
//  *        Multi-cpu-node version
//  *
//  *


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include "ZRLMPI.hpp"
#include "dosa_infer.hpp"
#include <sys/time.h>
#include <time.h>
#include <assert.h>


//internal state
//to match comm plan state of network
static uint8_t mpiCommands[DOSA_WRAPPER_MAX_PROG_LENGTH];
static uint8_t mpiRanks[DOSA_WRAPPER_MAX_PROG_LENGTH];
static uint32_t mpiCounts[DOSA_WRAPPER_MAX_PROG_LENGTH];
static uint8_t commandRepetitions[DOSA_WRAPPER_MAX_PROG_LENGTH];
static bool saveCurData[DOSA_WRAPPER_MAX_PROG_LENGTH];
static uint32_t byteCounts[DOSA_WRAPPER_MAX_PROG_LENGTH];

static int my_argc;
static char **my_argv;
static bool is_init = false;
static int rank;
static int size;
static uint32_t my_prog_length;
static uint32_t my_jump_addr;

static uint32_t nextCommandPtr = 0x0;
static uint8_t curIterationCnt = 0x0;
static uint8_t curCmnd = MPI_INSTR_NOP;
static uint8_t curRank = MPI_NO_RANK;
static uint32_t curCount = 0;
static uint32_t curBytes = 0;
static uint8_t curRep = 0;
static bool save_cur_data = false;
static bool processing_pipeline_filled = false;

static uint32_t pipeline_depth[DOSA_MAX_PARALLEL_RANKS];
static uint32_t batch_input_size[DOSA_MAX_PARALLEL_RANKS];
static uint32_t batch_output_size[DOSA_MAX_PARALLEL_RANKS];


typedef unsigned long long timestamp_t;

static timestamp_t get_timestamp()
{
  struct timeval now;
  gettimeofday(&now, NULL);
  return now.tv_usec + (timestamp_t)now.tv_sec*1000000;
}


void init(int argc, char **argv)
{
  //for(int i =0; i<argc; i++)
  //{
  //  printf("%d: %s\n",i,argv[i]);
  //}
  my_argc = argc;
  my_argv = argv;
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

#ifdef WRAPPER_TEST
  if(rank == 0)
  {
    mpiCommands[0]          = MPI_INSTR_SEND;
    mpiRanks[0]             = 1;
    mpiCounts[0]            = 6;  // 22/4;  //MUST be wordsize!
    byteCounts[0]           = 22;
    commandRepetitions[0]   = 1;
    saveCurData[0]          = false;
    mpiCommands[1]          = MPI_INSTR_RECV;
    mpiRanks[1]             = 5;
    mpiCounts[1]            = 6;  // 22/4;  //MUST be wordsize!
    byteCounts[0]           = 22;
    commandRepetitions[1]   = 1;
    saveCurData[1]          = false;
    my_prog_length = 2;
    my_jump_addr = 0;
  }

  pipeline_depth[0] = 1;
  batch_input_size[0] = 1;
  batch_output_size[0] = 1;

#else
  //DOSA_ADD_mpi_config
#endif

  is_init = true;
}

void cleanup(void)
{
  ZRLMPI_cleanup();
}


void reset_state(void)
{
  //ZRLMPI_cleanup();
  nextCommandPtr = 0x0;
  curIterationCnt = 0x0;
  curCmnd = MPI_INSTR_NOP;
  curRep = 0;
  curCount = 0;
  curBytes = 0;
  curRank = MPI_NO_RANK;
  save_cur_data = false;
  processing_pipeline_filled = false;
  //MPI_Init(&my_argc, &my_argv);
}


uint32_t get_pipeline_store_depth(void)
{
  assert(is_init);
  return pipeline_depth[rank];
  //return DOSA_PIPELINE_STORE_DETPH;
}


uint32_t get_batch_input_size(void)
{
  assert(is_init);
  return batch_input_size[rank];
  //return DOSA_MINIMAL_INPUT_NUM;
}


uint32_t get_batch_output_size(void)
{
  assert(is_init);
  return batch_output_size[rank];
  //return DOSA_MINIMAL_OUTPUT_NUM;
}


uint32_t get_pipeline_full_batch_size(void)
{
  return DOSA_PIPELINE_FULL_BATCH_SIZE;
}


bool are_processing_pipelines_filled(void)
{
  return processing_pipeline_filled;
}


//TODO: also allow single frame inference?
//int infer(int *input, uint32_t input_length, int *output, uint32_t output_length)
int infer_batch(char *input, uint32_t input_num, char *output, uint32_t output_num)
{
  if(!processing_pipeline_filled)
  {
    if(input_num < batch_input_size[rank])
    {
      fprintf(stderr, "[DOSA:ERROR]  processing pipeline not yet filled: Invalid input_num. It must be a at least %d (c.f. get_batch_input_size).\n", batch_input_size[rank]);
      return 1;
    }
    if(output_num < batch_output_size[rank])
    {
      fprintf(stderr, "[DOSA:ERROR]  processing pipeline not yet filled: Invalid output_num. It must be at least %d (c.f. get_batch_output_size).\n", batch_output_size[rank]);
      return 1;
    }
    if(input_num - output_num != pipeline_depth[rank])
    {
      //printf("%d %d  ", input_num, output_num);
      fprintf(stderr, "[DOSA:ERROR] processing pipeline not yet filled: input_num must be exactly %d larger than output_num.\n", pipeline_depth[rank]);
      return 1;
    }
  } else {
    if(input_num % DOSA_PIPELINE_FULL_BATCH_SIZE != 0)
    {
      fprintf(stderr, "[DOSA:ERROR] Invalid input_num. With filled processing pipelines, it must be a multiple of %d (c.f. get_pipeline_full_batch_size).\n", DOSA_PIPELINE_FULL_BATCH_SIZE);
      return 1;
    }
    if(output_num % DOSA_PIPELINE_FULL_BATCH_SIZE != 0)
    {
      fprintf(stderr, "[DOSA:ERROR] Invalid output_num. With filled processing pipelines, it must be a multiple of %d (c.f. get_pipeline_full_batch_size).\n", DOSA_PIPELINE_FULL_BATCH_SIZE);
      return 1;
    }
    if(input_num != output_num)
    {
      fprintf(stderr, "[DOSA:ERROR] input_num and output_num must be euqal, since processing pipelines are filled (c.f. are_processing_pipelines_filled).\n");
      return 1;
    }
  }


  uint8_t status;
#ifdef DEBUG
  printf("[DOSA:INFO] Here is rank %d, size is %d.\n", rank, size);
#endif

  //DOSA_ADD_MPI_barrier

  uint32_t total_input_processed = 0;
  uint32_t total_output_processed = 0;
  char *cur_send_pointer = input;
  char *cur_recv_pointer = output;
  bool last_instruction_was_recv = true;

  printf("[DOSA:INFO] Performing %d inference(s)...\n", input_num);
#ifdef PROGRESS_PRINT
  fflush(stdout);
#endif
  timestamp_t t0 = get_timestamp();
  //output is same multiple, checked above
  while( (total_input_processed < input_num) || (total_output_processed < output_num) )
  {
    //for(uint32_t i = 0; i < DOSA_MINIMAL_PROG_LENGTH; i++)
    //{
    if(curIterationCnt >= curRep)
    {//read new command
      curCmnd = mpiCommands[nextCommandPtr];
      curRank = mpiRanks[nextCommandPtr];
      curCount = mpiCounts[nextCommandPtr];
      curBytes = byteCounts[nextCommandPtr];
      curRep = commandRepetitions[nextCommandPtr];
      save_cur_data = saveCurData[nextCommandPtr];
      nextCommandPtr++;
      if(nextCommandPtr >= my_prog_length)
      {
        nextCommandPtr = my_jump_addr;
      }
      if(nextCommandPtr >= my_jump_addr)
      {
        processing_pipeline_filled = true;
      }
      curIterationCnt = 1;
#ifdef DEBUG
      if(curCmnd == MPI_INSTR_SEND && last_instruction_was_recv)
      {
        printf("[DOSA:DEBUG] Performing %d inference(s), each with length %d (words)...\n", curRep, curCount);
        fflush(stdout);
      }
#endif
    } else { //issue same again
      curIterationCnt++;
    }

    if(curCmnd != MPI_INSTR_NOP && curRep > 0 && curCount > 0)
    {
      if(curCmnd == MPI_INSTR_SEND)
      {
        //send data to infer
        //MPI_Send(input + (total_input_processed*curCount*(4/sizeof(int))), curCount, MPI_INTEGER, curRank, 0, MPI_COMM_WORLD);
        MPI_Send((int*)cur_send_pointer, curCount, MPI_INTEGER, curRank, rank, MPI_COMM_WORLD);
        if(!save_cur_data)
        {
          //cur_send_pointer += curCount*(4/sizeof(int));
          //cur_send_pointer += (curBytes+ sizeof(int)-1)/sizeof(int);
          cur_send_pointer += curBytes;
          total_input_processed++;
        }
        last_instruction_was_recv = false;
      } else {
        //receive result
        //MPI_Recv(output + (total_output_processed*curCount*(4/sizeof(int))), curCount, MPI_INTEGER, curRank, 0, MPI_COMM_WORLD, &status);
        MPI_Recv((int*)cur_recv_pointer, curCount, MPI_INTEGER, curRank, rank, MPI_COMM_WORLD, &status);
        total_output_processed++;
        //cur_recv_pointer += curCount*(4/sizeof(int));
        cur_recv_pointer += curBytes;
        last_instruction_was_recv = true;
#ifdef PROGRESS_PRINT
        printf("#");
        fflush(stdout);
#endif
      }
    }

    //}
  }

  timestamp_t t1 = get_timestamp();
  double elapsed_time_secs = (double)(t1 - t0) / 1000000.0L;
#ifdef PROGRESS_PRINT
  printf("\n");
#endif
  printf("[DOSA:INFO] ...done with %d inferences, %d results stored.\n    >>>>>>> Total clib-execution time: %lfs\n", total_input_processed, total_output_processed, elapsed_time_secs);

#ifdef MEASURE_PROTOCOL_WAIT
  ZRLMPI_print_stats();
#endif

  //return success
  return 0;
}


//function dummy to get ZRLMPI to compile
int app_main(int argc, char **argv)
{
  return 0;
}


