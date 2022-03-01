
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include "ZRLMPI.hpp"
#include "dosa_infer.hpp"

//internal state
//to match comm plan state of network
static uint8_t mpiCommands[DOSA_WRAPPER_PROG_LENGTH];
static uint8_t mpiRanks[DOSA_WRAPPER_PROG_LENGTH];
static uint32_t mpiCounts[DOSA_WRAPPER_PROG_LENGTH];
static uint8_t commandRepetitions[DOSA_WRAPPER_PROG_LENGTH];
static uint32_t nextCommandPtr = 0x0;
static uint8_t curIterationCnt = 0x0;
static uint8_t curCmnd = MPI_INSTR_NOP;
static uint8_t curRank = MPI_NO_RANK;
static uint8_t curCount = 0;
static uint8_t curRep = 0;


void reset_state()
{
  nextCommandPtr = 0x0;
  curIterationCnt = 0x0;
  curCmnd = MPI_INSTR_NOP;
  curRep = 0;
  curRank = MPI_NO_RANK;
}


int infer(int *input, int input_length, int *output, int output_length)
{
  int rank;
  int size;
  uint8_t status;
  MPI_Init(); //TODO: ZRLMPI init?, make extra method?
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  printf("[DOSA:INFO] Here is rank %d, size is %d.\n", rank, size);
  printf("[DOSA:INFO] Performing inference on input with length %d...\n", input_length);


#ifdef WRAPPER_TEST
  mpiCommands[0]          = MPI_INSTR_SEND;
  mpiRanks[0]             = 1;
  mpiCounts[0]            = 22;
  commandRepetitions[0]   = 1;
  mpiCommands[1]          = MPI_INSTR_RECV;
  mpiRanks[1]             = 5;
  mpiCounts[1]            = 22;
  commandRepetitions[1]   = 1;
#else
  //DOSA_ADD_mpi_config
#endif
  //TODO: add repeat factor? with batch input/output

  for(uint32_t i = 0; i < DOSA_MINIMAL_PROG_LENGTH; i++)
  {
    if(curIterationCnt >= curRep)
    {//read new command
      curCmnd = mpiCommands[nextCommandPtr];
      curRank = mpiRanks[nextCommandPtr];
      curCount = mpiCounts[nextCommandPtr];
      curRep = commandRepetitions[nextCommandPtr];
      nextCommandPtr++;
      if(nextCommandPtr >= DOSA_WRAPPER_PROG_LENGTH)
      {
        nextCommandPtr = 0x0;
      }
      curIterationCnt = 1;
    } else { //issue same again
      curIterationCnt++;
    }

    //TODO: add possible repeat with additional buffers
    if(curCmnd == MPI_INSTR_SEND && curCount != input_length)
    {
      fprintf(stderr, "[DOSA:ERROR] input_length %d is different from expected size %d. ABORTING.\n", input_length, curCount);
      return 1;
    }
    if(curCmnd == MPI_INSTR_RECV && curCount != output_length)
    {
      fprintf(stderr, "[DOSA:ERROR] output_length %d is different from expected size %d. ABORTING.\n", output_length, curCount);
      return 1;
    }

    if(curCmnd != MPI_INSTR_NOP && curRep > 0 && curCount > 0)
    {
      if(curCmnd == MPI_INSTR_SEND)
      {
        //send data to infer
        MPI_Send(input, curCount, MPI_INTEGER, curRank, 0, MPI_COMM_WORLD);
      } else {
        //receive result
        MPI_Recv(output, curCount, MPI_INTEGER, curRank, 0, MPI_COMM_WORLD, &status);
      }
    }
  }

  //return success
  MPI_Finalize();
  return 0;
}


