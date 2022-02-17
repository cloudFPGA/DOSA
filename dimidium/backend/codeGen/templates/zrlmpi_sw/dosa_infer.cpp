
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include "ZRLMPI.hpp"
#include "dosa_infer.hpp"


int infer(int *input, int input_length, int *output, int output_length)
{
  int rank;
  int size;
  uint8_t status;
  MPI_Init();
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  printf("[DOSA:INFO] Here is rank %d, size is %d.\n", rank, size);
  printf("[DOSA:INFO] Performing inference on input with length %d...\n", input_length);


#ifdef WRAPPER_TEST
  int first_node = 1;
  int send_mesage_length = 22;
  int last_node = 5;
  int recv_mesage_length = 22;
#else
  //DOSA_ADD_mpi_config
#endif

  if(send_mesage_length != input_length)
  {
    fprintf(stderr, "[DOSA:ERROR] input_length %d is different from expected size %d. ABORTING.\n", input_length, send_mesage_length);
    return 1;
  }
  if(recv_mesage_length != output_length)
  {
    fprintf(stderr, "[DOSA:ERROR] output_length %d is different from expected size %d. ABORTING.\n", output_length, recv_mesage_length);
    return 1;
  }
  //send data to infer
  MPI_Send(input, send_mesage_length, MPI_INTEGER, first_node, 0, MPI_COMM_WORLD);
  //receive result
  MPI_Recv(output, recv_mesage_length, MPI_INTEGER, last_node, 0, MPI_COMM_WORLD, &status);
  //return success
  MPI_Finalize();
  return 0;
}


