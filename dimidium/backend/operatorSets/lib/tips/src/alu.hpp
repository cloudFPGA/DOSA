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

#ifndef _TIPS_ALU_MOD_H_
#define _TIPS_ALU_MOD_H_

#include <cmath>
#include "ap_fixed.h"

#define N_TABLE 1024

void init_tanh_table(usedDtype table_out[N_TABLE]);
void tanh(usedDtype data[DOSA_TIPS_LONGEST_OUTPUT], usedDtype res[DOSA_TIPS_LONGEST_OUTPUT], usedDtype tanh_table[N_TABLE]);

void relu(usedDtype data[DOSA_TIPS_LONGEST_OUTPUT], usedDtype res[DOSA_TIPS_LONGEST_OUTPUT]);

void dense(usedDtype data[DOSA_TIPS_LONGEST_INPUT], usedDtype res[DOSA_TIPS_LONGEST_OUTPUT], usedDtype weights[DOSA_TIPS_LONGEST_OP0], usedDtype biases[DOSA_TIPS_LONGEST_OP1]);


#endif

