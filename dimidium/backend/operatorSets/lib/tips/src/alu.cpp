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
#include "ap_fixed.h"
#include "tips.hpp"
#include "alu.hpp"


// *************************************************
//       TanH Activation
// *************************************************
void init_tanh_table(usedDtype table_out[N_TABLE])
{
#pragma HLS inline
  // Implement tanh lookup
  for (int ii = 0; ii < N_TABLE; ii++) {
    // First, convert from table index to X-value (signed 8-bit, range -4 to +4)
    float in_val = 2*4.0*(ii-float(N_TABLE)/2.0)/float(N_TABLE);
    // Next, compute lookup table function
    usedDtype real_val = (usedDtype) tanh(in_val);
    //std::cout << "Tanh:  Lookup table Index: " <<  ii<< " In Value: " << in_val << " Result: " << real_val << std::endl;
    table_out[ii] = real_val;
  }
}


void tanh(usedDtype data[DOSA_TIPS_LONGEST_OUTPUT], usedDtype res[DOSA_TIPS_LONGEST_OUTPUT], usedDtype tanh_table[N_TABLE])
{
#pragma HLS inline
  // Index into the lookup table based on data
  int data_round;
  int index;
  for (int ii=0; ii<DOSA_TIPS_LONGEST_OUTPUT; ii++) {
#pragma HLS PIPELINE
    data_round = data[ii]*N_TABLE/8;
    index = data_round + 4*N_TABLE/8;
    //std::cout << "Input: "  << data[ii] << " Round: " << data_round << " Index: " << index << std::endl;
    if (index < 0)   index = 0;
    if (index > N_TABLE-1) index = N_TABLE-1;
    res[ii] = (usedDtype) tanh_table[index];
  }
}


// *************************************************
//       RELU Activation
// *************************************************
void relu(usedDtype data[DOSA_TIPS_LONGEST_OUTPUT], usedDtype res[DOSA_TIPS_LONGEST_OUTPUT])
{
#pragma HLS inline
  usedDtype datareg;
  for (int ii=0; ii<DOSA_TIPS_LONGEST_OUTPUT; ii++) {
#pragma HLS PIPELINE
    datareg = data[ii];
    if (datareg > 0) res[ii] = datareg;
    else res[ii] = 0;
  }
}


// *************************************************
//       Dense
// *************************************************
void dense(
    usedDtype data[DOSA_TIPS_LONGEST_INPUT],
    usedDtype res[DOSA_TIPS_LONGEST_OUTPUT],
    usedDtype weights[DOSA_TIPS_LONGEST_OP0],
    usedDtype biases[DOSA_TIPS_LONGEST_OP1])
{
#pragma HLS inline
  usedDtype cache;
  aluAccumDtype mult[DOSA_TIPS_LONGEST_OP0];
  aluAccumDtype acc[DOSA_TIPS_LONGEST_OUTPUT];
  //#pragma HLS ARRAY_PARTITION variable=biases complete
  //#pragma HLS ARRAY_PARTITION variable=mult complete
  //#pragma HLS ARRAY_PARTITION variable=acc complete
  //TODO
  //#pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation

  // Do the matrix-multiply
Product1: for(int ii = 0; ii < DOSA_TIPS_LONGEST_INPUT; ii++) {
#pragma HLS PIPELINE
            cache = data[ii];
Product2: for(int jj = 0; jj < DOSA_TIPS_LONGEST_OUTPUT; jj++) {
            int index = ii*DOSA_TIPS_LONGEST_OUTPUT+jj;
            mult[index] = cache * weights[index];
          }
          }

          // Initialize accumulator with input biases
ResetAccum: for(int iacc = 0; iacc < DOSA_TIPS_LONGEST_OUTPUT; iacc++) {
              acc[iacc] = (aluAccumDtype) biases[iacc];
            }

            // Accumulate multiplication result
Accum1: for(int ii = 0; ii < DOSA_TIPS_LONGEST_INPUT; ii++) {
Accum2: for(int jj = 0; jj < DOSA_TIPS_LONGEST_OUTPUT; jj++) {
          int index = ii*DOSA_TIPS_LONGEST_OUTPUT+jj;
          acc[jj] += mult[index];
        }
        }

        // Cast to result type
Result: for(int ires = 0; ires < DOSA_TIPS_LONGEST_OUTPUT; ires++){
          res[ires] = (usedDtype) (acc[ires]); //TODO: take care of quantization?
          //res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
        }
}



