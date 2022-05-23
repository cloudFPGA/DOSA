//  *
//  *                       cloudFPGA
//  *     Copyright IBM Research, All Rights Reserved
//  *    =============================================
//  *     Created: Apr 2022
//  *     Authors: NGL
//  *
//  *     Description:
//  *        Module containing the ALU functions.
//  *          Partially copied from hls4ml.
//  *

#include <cmath>
#include <cassert>
#include "ap_fixed.h"
#include "tips.hpp"
#include "alu.hpp"


// *************************************************
//       TanH Activation
// *************************************************
void init_tanh_table(aluAccumDtype table_out[N_TABLE])
{
#pragma HLS inline
  // Implement tanh lookup
  for (int ii = 0; ii < N_TABLE; ii++) {
    // First, convert from table index to X-value (signed 8-bit, range -4 to +4)
    float in_val = 2*4.0*(ii-float(N_TABLE)/2.0)/float(N_TABLE);
    // Next, compute lookup table function
    aluAccumDtype real_val = (aluAccumDtype) tanh(in_val);
#ifdef ALU_DEBUG
    std::cout << "Tanh:  Lookup table Index: " <<  ii << " In Value: " << in_val << " Result: " << real_val << std::endl;
#endif
    table_out[ii] = real_val;
  }
}


void tanh(quantDtype data[DOSA_TIPS_LONGEST_OUTPUT], quantDtype res[DOSA_TIPS_LONGEST_OUTPUT], aluAccumDtype tanh_table[N_TABLE])
{
//#pragma HLS inline
#pragma HLS INLINE off
  // Index into the lookup table based on data
  int data_round;
  int index;
//#ifdef ALU_DEBUG
//  printf("TANH for input:\n\t");
//  for(int i = 0; i < DOSA_TIPS_LONGEST_OUTPUT; i++)
//  {
//    printf(" %d", data[i]);
//  }
//  printf("\nwith tanh_table:\n\t");
//  for(int i = 0; i < N_TABLE; i++)
//  {
//    printf(" %d", tanh_table[i]);
//  }
//  printf("\n");
//#endif
  for (int ii=0; ii<DOSA_TIPS_LONGEST_OUTPUT; ii++) {
#pragma HLS PIPELINE
    data_round = data[ii]*N_TABLE/8;
    index = data_round + 4*N_TABLE/8;
#ifdef ALU_DEBUG
    std::cout << "Input: "  << data[ii] << " Round: " << data_round << " Index: " << index << std::endl;
#endif
    if (index < 0)   index = 0;
    if (index > N_TABLE-1) index = N_TABLE-1;
    res[ii] = (quantDtype) tanh_table[index];
  }
}


// *************************************************
//       RELU Activation
// *************************************************
void relu(quantDtype data[DOSA_TIPS_LONGEST_OUTPUT], quantDtype res[DOSA_TIPS_LONGEST_OUTPUT])
{
//#pragma HLS inline
#pragma HLS INLINE off
  quantDtype datareg;
  for (int ii=0; ii<DOSA_TIPS_LONGEST_OUTPUT; ii++) {
#pragma HLS PIPELINE
    datareg = data[ii];
    if (datareg > 0) res[ii] = datareg;
    else res[ii] = 0;
  }
}


// *************************************************
//       Dense
//       aka. Matrix-Vector multiplication
//          (i.e. data is 1D)
// *************************************************
void dense(
    quantDtype data[DOSA_TIPS_LONGEST_INPUT],
    quantDtype res[DOSA_TIPS_LONGEST_OUTPUT],
    quantDtype weights[DOSA_TIPS_LONGEST_OP0],
    quantDtype biases[DOSA_TIPS_LONGEST_OP1],
    int m //colum number of weights
  )
{
#ifndef __SYNTHESIS__
  //assert(m*DOSA_TIPS_LONGEST_INPUT <= DOSA_TIPS_LONGEST_OP0);
  assert(m <= DOSA_TIPS_LONGEST_INPUT);
  assert(DOSA_TIPS_LONGEST_OUTPUT <= DOSA_TIPS_LONGEST_OP1);
#endif
//#pragma HLS inline
#pragma HLS INLINE off
//#pragma HLS function_instantiate variable=m
#pragma HLS PIPELINE
  quantDtype cache;
  aluAccumDtype mult[DOSA_TIPS_LONGEST_OP0];
  aluAccumDtype acc[DOSA_TIPS_LONGEST_OUTPUT];
  //#pragma HLS ARRAY_PARTITION variable=biases complete
  //#pragma HLS ARRAY_PARTITION variable=mult complete
  //#pragma HLS ARRAY_PARTITION variable=acc complete
  //TODO
  //#pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation

#ifdef ALU_DEBUG
  printf("Dense operation with:\ninput:\n\t");
  for(int i = 0; i < DOSA_TIPS_LONGEST_INPUT; i++)
  {
    printf(" %d", (uint16_t) (data[i] >> DEBUG_FRACTIONAL_BITS));
  }
  printf("\nweights:\n\t");
  for(int i = 0; i < DOSA_TIPS_LONGEST_OP0; i++)
  {
    printf(" %d", (uint16_t) (weights[i] >> DEBUG_FRACTIONAL_BITS));
    if(i > 0 && i%m == (m-1))
    {
      printf("\n\t");
    }
  }
  printf("\nbias:\n\t");
  for(int i = 0; i < DOSA_TIPS_LONGEST_OP1; i++)
  {
    printf(" %d", (uint16_t) (biases[i] >> DEBUG_FRACTIONAL_BITS));
  }
  printf("\n");
#endif

  // Do the matrix-vector multiply, column-wise
  // data*weights^T, with m =number of colums of weights
//Product1: for(int ii = 0; ii < DOSA_TIPS_LONGEST_INPUT; ii++) {
//#pragma HLS PIPELINE
//            cache = data[ii];
//Product2: for(int jj = 0; jj < m; jj++) {
//            int index = jj*m+ii;
//            //mult[index] = (((aluAccumDtype) cache) * ((aluAccumDtype) weights[index])) / QUANT_SCALE_BACK_VALUE;
//            if(index < DOSA_TIPS_LONGEST_OP0)
//            {
//              mult[index] = (((aluAccumDtype) cache) * ((aluAccumDtype) weights[index]));
//            }
//          }
//          }
Product2: for(int jj = 0; jj < m; jj++) {
Product1: for(int ii = 0; ii < DOSA_TIPS_LONGEST_INPUT; ii++) {
#pragma HLS PIPELINE
            cache = data[ii];
            int index = jj*m+ii;
            //mult[index] = (((aluAccumDtype) cache) * ((aluAccumDtype) weights[index])) / QUANT_SCALE_BACK_VALUE;
            if(index < DOSA_TIPS_LONGEST_OP0)
            {
              mult[index] = (((aluAccumDtype) cache) * ((aluAccumDtype) weights[index]));
            }
          }
          }
#ifdef ALU_DEBUG
  printf("mult intermediate:\n\t");
  for(int i = 0; i < DOSA_TIPS_LONGEST_OP0; i++)
  {
    printf(" %d", (uint16_t) (mult[i] >> DEBUG_FRACTIONAL_BITS));
    //printf(" %d", (uint16_t) (mult[i] >> DEBUG_ALU_ACCUM_FRACTIONAL_BITS));
    //printf(" %d", (uint16_t) (mult[i] >> 2*DEBUG_ALU_ACCUM_FRACTIONAL_BITS));
    if(i > 0 && i%m == (m-1))
    {
      printf("\n\t");
    }
  }
  printf("\n");
#endif

          // Initialize accumulator with input biases
ResetAccum: for(int iacc = 0; iacc < DOSA_TIPS_LONGEST_OUTPUT; iacc++) {
              //acc[iacc] = (aluAccumDtype) biases[iacc];
              acc[iacc] = (aluAccumDtype) biases[iacc]*QUANT_SCALE_BACK_VALUE;
            }

#ifdef ALU_DEBUG
  printf("accum intermediate:\n\t");
  for(int i = 0; i < DOSA_TIPS_LONGEST_OUTPUT; i++)
  {
    //printf(" %d", (uint16_t) (acc[i] >> DEBUG_ALU_ACCUM_FRACTIONAL_BITS));
    printf(" %d", (uint16_t) (acc[i] >> DEBUG_FRACTIONAL_BITS));
  }
  printf("\n");
#endif
            // Accumulate multiplication result
Accum1: for(int ii = 0; ii < DOSA_TIPS_LONGEST_OUTPUT; ii++) {
Accum2: for(int jj = 0; jj < m; jj++) {
          int index = ii*m+jj;
          acc[ii] += mult[index];
        }
        }

        // Cast to result type
Result: for(int ires = 0; ires < DOSA_TIPS_LONGEST_OUTPUT; ires++){
          //res[ires] = (quantDtype) (acc[ires]);
          //the library can't now how much to scale back...we have to do
          //res[ires] = (quantDtype) (acc[ires] / QUANT_SCALE_BACK_VALUE);
          res[ires] = (quantDtype) ((acc[ires] + QUANT_SCALE_BACK_VALUE/2) / QUANT_SCALE_BACK_VALUE);
          //res[ires] = (quantDtype) (acc[ires] / ((aluAccumDtype) QUANT_SCALE_BACK_VALUE*4));
          //res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
        }
#ifdef ALU_DEBUG
  printf("Dense operation result:\n\t");
  for(int i = 0; i < DOSA_TIPS_LONGEST_OUTPUT; i++)
  {
    printf(" %d", (uint16_t) (res[i] >> DEBUG_FRACTIONAL_BITS));
  }
  printf("\n");
#endif
}



