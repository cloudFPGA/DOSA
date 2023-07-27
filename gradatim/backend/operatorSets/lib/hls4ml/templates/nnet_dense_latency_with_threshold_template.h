/*******************************************************************************
 * Copyright 2023 IBM Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*******************************************************************************/

//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

//  *
//  *                       cloudFPGA
//  *    =============================================
//  *     Created: July 2023
//  *     Authors: NGL
//  *
//  *     Description:
//  *        Template for dense layer with thresholding operation
//  *        Based on the nnet_dense_latency.h from hls4ml
//  *

//DOSA_infdef_define

#include "nnet_common.h"
#include "nnet_mult.h"
#include "nnet_helpers.h"
#include "hls_stream.h"
#include <math.h>

namespace nnet {

template<class data_T, class res_T, typename CONFIG_T>
//DOSA_insert_function_name
void dense_latency(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
{
    data_T cache;
    typename CONFIG_T::accum_t mult[CONFIG_T::n_in*CONFIG_T::n_out];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=weights,biases

    if (CONFIG_T::io_type == io_parallel || CONFIG_T::io_type == io_stream){
      // For parallel inputs:
      //   - completely partition arrays -- target fabric
      //   - if we have an unroll factor, limit number of multipliers
      #pragma HLS PIPELINE II=CONFIG_T::reuse_factor enable_flush
      //if (CONFIG_T::reuse_factor == 1)
      //{
      //  #pragma HLS LATENCY max=1
      //}
      //THIS block is IGNORED...I don't know why (some constant folding?), so here in different form
      //#pragma HLS LATENCY max=CONFIG_T::reuse_factor

      // #pragma HLS ARRAY_PARTITION variable=weights complete // remove this line for now, it breaks compression sometimes
      #pragma HLS ARRAY_PARTITION variable=biases complete
      #pragma HLS ARRAY_PARTITION variable=mult complete
      #pragma HLS ARRAY_PARTITION variable=acc complete

      int multiplier_limit  = ceil(float(CONFIG_T::n_in*CONFIG_T::n_out) / float(CONFIG_T::reuse_factor)) - floor(float(CONFIG_T::n_zeros) / float(CONFIG_T::reuse_factor));
      CONFIG_T::template product<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::accum_t>::limit(multiplier_limit);

    } else if (CONFIG_T::io_type == io_serial){
        // Only reduce cycle_factor if n_out is evenly divisible by reuse_factor
        // Otherwise, HLS wont be happy
        int cycle_factor = CONFIG_T::n_out / CONFIG_T::reuse_factor;
        int reused_cycle = DIV_ROUNDUP(CONFIG_T::n_out, CONFIG_T::reuse_factor);
        if (cycle_factor != reused_cycle) {
            cycle_factor = CONFIG_T::n_out;
        }
        /*int cycle_factor = CONFIG_T::n_out;
        float reused_cycle = CONFIG_T::n_out / CONFIG_T::reuse_factor;
        if (reused_cycle == ceil(reused_cycle)){
            // Dont use "ceil" here; as of 2018.2, HLS crashes mysteriously
            cycle_factor = cycle_factor / CONFIG_T::reuse_factor;
        }*/
        #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=cycle_factor
        #pragma HLS ARRAY_PARTITION variable=mult cyclic factor=cycle_factor
        #pragma HLS ARRAY_PARTITION variable=acc complete
        #pragma HLS DATAFLOW
        #pragma HLS STREAM variable=mult depth=1
        #pragma HLS STREAM variable=acc depth=1
        if (CONFIG_T::store_weights_in_bram){
            #pragma HLS RESOURCE variable=weights core=ROM_2P_BRAM
        }
    }

    // Do the matrix-multiply
    Product1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial || CONFIG_T::n_in > 128){
            //#pragma HLS PIPELINE
            #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        }
        cache = data[ii];
        Product2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
            if (CONFIG_T::io_type == io_serial) {
                int multiplier_limit  = ceil(float(CONFIG_T::n_out) / float(CONFIG_T::reuse_factor));
                CONFIG_T::template product<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::accum_t>::limit(multiplier_limit);
            }
        int index = ii*CONFIG_T::n_out+jj;
        mult[index] = CONFIG_T::template product<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::accum_t>::product(cache, weights[index]);
        }
    }

    // Initialize accumulator with input biases
    ResetAccum: for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        if (CONFIG_T::io_type == io_serial){
          if (CONFIG_T::reuse_factor > 1)
          {
            #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
          } else {
            #pragma HLS UNROLL
          }
        }
        acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
    }

    // Accumulate multiplication result
    Accum1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
            //#pragma HLS PIPELINE
        }
        Accum2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
        int index = ii*CONFIG_T::n_out+jj;
        acc[jj] += mult[index];
        }
    }

    //DOSA_insert_thresholding

    //// Cast to "res_t" type
    //Result: for(int ires = 0; ires < CONFIG_T::n_out; ires++){
    //    #pragma HLS UNROLL
    //    res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    //}
}

}

#endif

