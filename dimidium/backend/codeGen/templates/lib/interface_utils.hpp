
//  *
//  *                       cloudFPGA
//  *     Copyright IBM Research, All Rights Reserved
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


inline uint8_t extractByteCnt(Axis<64> &currWord)
{
#pragma HLS INLINE

  uint8_t ret = 0;

  switch (currWord.getTKeep()) {
    case 0b11111111:
      ret = 8;
      break;
    case 0b01111111:
      ret = 7;
      break;
    case 0b00111111:
      ret = 6;
      break;
    case 0b00011111:
      ret = 5;
      break;
    case 0b00001111:
      ret = 4;
      break;
    case 0b00000111:
      ret = 3;
      break;
    case 0b00000011:
      ret = 2;
      break;
    default:
    case 0b00000001:
      ret = 1;
      break;
  }
  return ret;
}

inline uint8_t byteCntToTKeep(uint8_t byte_cnt)
{
#pragma HLS INLINE

  uint8_t ret = 0;

  switch (byte_cnt) {
    case 8:
      ret = 0b11111111;
      break;
    case 7:
      ret = 0b01111111;
      break;
    case 6:
      ret = 0b00111111;
      break;
    case 5:
      ret = 0b00011111;
      break;
    case 4:
      ret = 0b00001111;
      break;
    case 3:
      ret = 0b00000111;
      break;
    case 2:
      ret = 0b00000011;
      break;
    default:
    case 1:
      ret = 0b00000001;
      break;
  }
  return ret;
}

//TODO: add 512 and 2048 version?


#endif

