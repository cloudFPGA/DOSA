
//  *
//  *                       cloudFPGA
//  *    =============================================
//  *     Created: Jan 2022
//  *     Authors: NGL
//  *
//  *     Description:
//  *        This file contains helper functions for DOSA interface wrappers
//  *

#ifndef _DOSA_INTERFACE_UTILS_
#define _DOSA_INTERFACE_UTILS_

#include "axi_utils.hpp"


//TODO: add 512 and 2048 versions
inline uint8_t extractByteCnt(Axis<64> &currWord)
{
#pragma HLS INLINE

  uint8_t ret = 0;

  switch (currWord.getTKeep()) {
    //case 0b11111111:
    case 0xFF:
      ret = 8;
      break;
    //case 0b01111111:
    case 0x7F:
      ret = 7;
      break;
    //case 0b00111111:
    case 0x3F:
      ret = 6;
      break;
    //case 0b00011111:
    case 0x1F:
      ret = 5;
      break;
    //case 0b00001111:
    case 0x0F:
      ret = 4;
      break;
    //case 0b00000111:
    case 0x07:
      ret = 3;
      break;
    //case 0b00000011:
    case 0x03:
      ret = 2;
      break;
    default:
    //case 0b00000001:
    case 0x01:
      ret = 1;
      break;
  }
  return ret;
}

inline uint8_t byteCntToTKeep(uint8_t byte_cnt)
{
#pragma HLS INLINE

  uint8_t ret = 0;
//  const uint8_t byte2keep[9] = {
//  [0] = 0b00000000,
//  [1] = 0b00000001,
//  [2] = 0b00000011,
//  [3] = 0b00000111,
//  [4] = 0b00001111,
//  [5] = 0b00011111,
//  [6] = 0b00111111,
//  [7] = 0b01111111,
//  [8] = 0b11111111};

//LESSON LEARNED: switch-case with default better than if with array lookup

  switch (byte_cnt) {
    case 1:
      ret = 0b00000001;
      break;
    case 2:
      ret = 0b00000011;
      break;
    case 3:
      ret = 0b00000111;
      break;
    case 4:
      ret = 0b00001111;
      break;
    case 5:
      ret = 0b00011111;
      break;
    case 6:
      ret = 0b00111111;
      break;
    case 7:
      ret = 0b01111111;
      break;
    default:
    case 8:
      ret = 0b11111111;
      break;
  }
  //if(byte_cnt == 0 || byte_cnt > 8)
  //{
  //  //TODO
  //  return 0xFF;
  //}

  //uint8_t byte2keep[9];
  //byte2keep[8] =  0b11111111;
  //byte2keep[7] =  0b01111111;
  //byte2keep[6] =  0b00111111;
  //byte2keep[5] =  0b00011111;
  //byte2keep[4] =  0b00001111;
  //byte2keep[3] =  0b00000111;
  //byte2keep[2] =  0b00000011;
  //byte2keep[1] =  0b00000001;
  //byte2keep[0] =  0b00000000;

  //ret = byte2keep[byte_cnt];
  //printf("\tgetting tkeep for %d bytes is %X\n", byte_cnt, ret);
  return ret;
}

inline ap_uint<8> bitCntToTKeep(ap_uint<8> bit_cnt)
{
#pragma HLS INLINE
  uint8_t byte_cnt = (((uint16_t) bit_cnt) + 7)/8;
  return (ap_uint<8>) byteCntToTKeep(byte_cnt);
}


inline ap_uint<64> bitCntToTKeep(ap_uint<64> bit_cnt)
{
#pragma HLS INLINE
  ap_uint<64> ret = 0;
  for(int i = 0; i < 8; i++)
  {
#pragma HLS unroll
    uint8_t cur_bit_cnt = (uint8_t) (bit_cnt >> (i*8));
    uint8_t byte_cnt = (((uint16_t) cur_bit_cnt) + 7)/8;
    ap_uint<8> tmp_tkeep = byteCntToTKeep(byte_cnt);
    ret |= ((ap_uint<64>) tmp_tkeep) << i*8;
  }
  return ret;
}


inline ap_uint<256> bitCntToTKeep(ap_uint<256> bit_cnt)
{
#pragma HLS INLINE
  ap_uint<256> ret = 0;
  for(int i = 0; i < 4; i++)
  {
#pragma HLS unroll
    ap_uint<64> cur_bit_cnt = (ap_uint<64>) (bit_cnt >> (i*64));
    ap_uint<64> tmp_tkeep = bitCntToTKeep(cur_bit_cnt);
    ret |= ((ap_uint<256>) tmp_tkeep) << i*64;
  }
  return ret;
}


#endif

