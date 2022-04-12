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


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include "ZRLMPI.hpp"
#include "dosa_infer.hpp"
#include <sys/time.h>
#include <time.h>


//internal state
//to match comm plan state of network
static uint8_t mpiCommands[DOSA_WRAPPER_PROG_LENGTH];
static uint8_t mpiRanks[DOSA_WRAPPER_PROG_LENGTH];
static uint32_t mpiCounts[DOSA_WRAPPER_PROG_LENGTH];
static uint8_t commandRepetitions[DOSA_WRAPPER_PROG_LENGTH];
static bool saveCurData[DOSA_WRAPPER_PROG_LENGTH];

static uint32_t nextCommandPtr = 0x0;
static uint8_t curIterationCnt = 0x0;
static uint8_t curCmnd = MPI_INSTR_NOP;
static uint8_t curRank = MPI_NO_RANK;
static uint32_t curCount = 0;
static uint8_t curRep = 0;
static bool save_cur_data = false;


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
  MPI_Init(&argc, &argv);
}


void reset_state(void)
{
  nextCommandPtr = 0x0;
  curIterationCnt = 0x0;
  curCmnd = MPI_INSTR_NOP;
  curRep = 0;
  curRank = MPI_NO_RANK;
  save_cur_data = false;
}


uint32_t get_pipeline_store_depth(void)
{
  return DOSA_PIPELINE_STORE_DETPH;
}


uint32_t get_batch_input_size(void)
{
  return DOSA_MINIMAL_INPUT_NUM;
}


uint32_t get_batch_output_size(void)
{
  return DOSA_MINIMAL_OUTPUT_NUM;
}


//TODO: also allow single frame inference?
//int infer(int *input, uint32_t input_length, int *output, uint32_t output_length)
int infer_batch(int *input, uint32_t input_num, int *output, uint32_t output_num)
{
  if(input_num % DOSA_MINIMAL_INPUT_NUM != 0)
  {
    fprintf(stderr, "[DOSA:ERROR] Invalid input_num. It must be a multiple of %d (c.f. get_batch_input_size).\n", DOSA_MINIMAL_INPUT_NUM);
    return 1;
  }
  if(output_num % DOSA_MINIMAL_OUTPUT_NUM != 0)
  {
    fprintf(stderr, "[DOSA:ERROR] Invalid output_num. It must be a multiple of %d (c.f. get_batch_output_size).\n", DOSA_MINIMAL_OUTPUT_NUM);
    return 1;
  }
  if(input_num/DOSA_MINIMAL_INPUT_NUM != output_num/DOSA_MINIMAL_OUTPUT_NUM)
  {
    fprintf(stderr, "[DOSA:ERROR] input_num and output_num must be the same multiple of each minmum size (for this cluster: %d for input and %d for output).\n", DOSA_MINIMAL_INPUT_NUM, DOSA_MINIMAL_OUTPUT_NUM);
    return 1;
  }
  //if(input_num%4 != 0)
  //{
  //  fprintf(stderr, "[DOSA:ERROR] input_num must have a 'word-aligned' (4 bytes) length.\n");
  //  return 1;
  //}
  //if(output_num%4 != 0)
  //{
  //  fprintf(stderr, "[DOSA:ERROR] output_nu must have a 'word-aligned' (4 bytes) length.\n");
  //  return 1;
  //}


  int rank;
  int size;
  uint8_t status;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  printf("[DOSA:INFO] Here is rank %d, size is %d.\n", rank, size);

#ifdef WRAPPER_TEST
  mpiCommands[0]          = MPI_INSTR_SEND;
  mpiRanks[0]             = 1;
  mpiCounts[0]            = 6;  // 22/4;  //MUST be wordsize!
  commandRepetitions[0]   = 1;
  saveCurData[0]          = false;
  mpiCommands[1]          = MPI_INSTR_RECV;
  mpiRanks[1]             = 5;
  mpiCounts[1]            = 6;  // 22/4;  //MUST be wordsize!
  commandRepetitions[1]   = 1;
  saveCurData[1]          = false;
#else
  //DOSA_ADD_mpi_config
#endif

  uint32_t total_input_processed = 0;
  uint32_t total_output_processed = 0;
  timestamp_t t0 = get_timestamp();

  //output is same multiple, checked above
  while(total_input_processed < input_num)
  {
    for(uint32_t i = 0; i < DOSA_MINIMAL_PROG_LENGTH; i++)
    {
      if(curIterationCnt >= curRep)
      {//read new command
        curCmnd = mpiCommands[nextCommandPtr];
        curRank = mpiRanks[nextCommandPtr];
        curCount = mpiCounts[nextCommandPtr];
        curRep = commandRepetitions[nextCommandPtr];
        save_cur_data = saveCurData[nextCommandPtr];
        nextCommandPtr++;
        if(nextCommandPtr >= DOSA_WRAPPER_PROG_LENGTH)
        {
          nextCommandPtr = 0x0;
        }
        curIterationCnt = 1;
        printf("[DOSA:INFO] Performing %d inference(s), each with length %d (words)...\n", curRep, curCount);
      } else { //issue same again
        curIterationCnt++;
      }

      if(curCmnd != MPI_INSTR_NOP && curRep > 0 && curCount > 0)
      {
        if(curCmnd == MPI_INSTR_SEND)
        {
          //send data to infer
          MPI_Send(input + (total_input_processed*curCount*(4/sizeof(int))), curCount, MPI_INTEGER, curRank, 0, MPI_COMM_WORLD);
          if(!save_cur_data)
          {
            total_input_processed++;
          }
        } else {
          //receive result
          MPI_Recv(output + (total_output_processed*curCount*(4/sizeof(int))), curCount, MPI_INTEGER, curRank, 0, MPI_COMM_WORLD, &status);
          total_output_processed++;
        }
      }

    }
  }

  timestamp_t t1 = get_timestamp();
  double elapsed_time_secs = (double)(t1 - t0) / 1000000.0L;
  printf("[DOSA:INFO] Done with %d inferences, %d results stored.\n>>>>>>>>>>>> Total execution time: %lfs\n", total_input_processed, total_output_processed, elapsed_time_secs);

  //return success
  //MPI_Finalize();
  return 0;
}


//function dummy to get ZRLMPI to compile
int app_main(int argc, char **argv)
{
  return 0;
}


