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
#include "tips.hpp"

#ifndef __SYNTHESIS__
#define ALU_DEBUG
#endif

#define N_TABLE 1024

void init_tanh_table(aluAccumDtype table_out[N_TABLE]);
void tanh(quantDtype data[DOSA_TIPS_LONGEST_OUTPUT], quantDtype res[DOSA_TIPS_LONGEST_OUTPUT], aluAccumDtype tanh_table[N_TABLE]);

void relu(quantDtype data[DOSA_TIPS_LONGEST_OUTPUT], quantDtype res[DOSA_TIPS_LONGEST_OUTPUT]);

void dense(quantDtype data[DOSA_TIPS_LONGEST_INPUT], quantDtype res[DOSA_TIPS_LONGEST_OUTPUT], quantDtype weights[DOSA_TIPS_LONGEST_OP0], quantDtype biases[DOSA_TIPS_LONGEST_OP1], int m);


#endif

