#ifndef _ZRLMPI_COMMON_H_
#define _ZRLMPI_COMMON_H_

#include <stdint.h>

#include "zrlmpi_int.hpp"

#define MPI_SEND_INT 0
#define MPI_RECV_INT 1
#define MPI_SEND_FLOAT 2
#define MPI_RECV_FLOAT 3
#define MPI_BARRIER 4
#define mpiCall uint8_t

#define MPI_STATUS_IGNORE 0

#define SEND_REQUEST 1
#define CLEAR_TO_SEND 2
#define DATA 3
#define ACK 4
#define ERROR 5
#define packetType uint8_t

#define ZRLMPI_DEFAULT_PORT 2718

#define ZRLMPI_MAX_CLUSTER_SIZE 64


//#define ZRLMPI_PROTOCOL_TIMEOUT_MS 10
//#define ZRLMPI_PROTOCOL_TIMEOUT_CYCLES 1562500  //10ms, with cycle=6.4ns, 20bit
#define ZRLMPI_PROTOCOL_TIMEOUT_MS 100
#define ZRLMPI_PROTOCOL_TIMEOUT_CYCLES 15625000  //100ms, with cycle=6.4ns, 23bit

#define ZRLMPI_PROTOCOL_TIMEOUT_INC_FACTOR 10
#define ZRLMPI_PROTOCOL_MAX_INC 3
#define PROTOCOL_ACK_DELAY_FACTOR 2

//to disable debug printfs
#define DEBUG0
//#define DEBUG
//#define DEBUG2
//#define DEBUG3

//UDP Header + MPIF Header = 74
//main constraint is ZYC2 VXLAN setup
#define ZRLMPI_MAX_MESSAGE_SIZE_BYTES 1416  //Bytes inclusive header!
#define ZRLMPI_MAX_MESSAGE_SIZE_WORDS 354   //int or float inclusive header!
#define ZRLMPI_MAX_MESSAGE_SIZE_LINES 177   //line of 8 byte (inclusive header)


/*
 * ZRLMPI Interface
 */
struct MPI_Interface {
  UINT8     mpi_call;
  UINT32    count;
  UINT32    rank;
  MPI_Interface() {}
};

/*
 * ZRLMPI Feedback link (for HW only)
 */
typedef uint8_t MPI_Feedback;

#define ZRLMPI_FEEDBACK_OK 1
#define ZRLMPI_FEEDBACK_FAIL 2


/*
 * ZRLMPI Header
 */
struct MPI_Header {
  UINT32 dst_rank;
  UINT32 src_rank;
  UINT32 size;
  mpiCall call;
  packetType type;
  MPI_Header() {}
};
#define MPIF_HEADER_LENGTH 32 //Bytes

/*
 * MPI_Op operations
 */

void MPI_SUM_INTEGER(int32_t *accum, int32_t *source, uint32_t length);
void MPI_SUM_FLOAT(float *accum, float *source, uint32_t length);

/*
 * Utility functions
 */
void my_memcpy(int *dst, int *src, uint32_t length);

//UINT32 littleEndianToInteger(UINT8 *buffer, int lsb);
UINT32 bigEndianToInteger(UINT8 *buffer, int lsb);
void integerToBigEndian(UINT32 n, UINT8 *bytes);
int bytesToHeader(UINT8 bytes[MPIF_HEADER_LENGTH], MPI_Header &header);
void headerToBytes(MPI_Header header, UINT8 bytes[MPIF_HEADER_LENGTH]);



#endif
