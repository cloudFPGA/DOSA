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
//  *     Created: Aug 2023
//  *     Authors: NGL
//  *
//  *     Description:
//  *        Template for a standalone thresholding operation
//  *        Based on the nnet_activation.h from hls4ml
//  *

//DOSA_infdef_define

#include <cmath>
#include "ap_fixed.h"
#include "nnet_common.h"

namespace nnet {


template<class data_T, class res_T, int MAX_INT, typename CONFIG_T>
//DOSA_insert_function_name
void  multi_thresholding(
    data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }

    //data_T datareg;
    //for (int ii=0; ii<CONFIG_T::n_in; ii++) {
    //    if (CONFIG_T::io_type == io_serial){
    //        #pragma HLS PIPELINE
    //    }
    //    datareg = data[ii];
    //    if (datareg < 0) res[ii] = 0;
    //    else if (datareg > MAX_INT) res[ii] = MAX_INT;
    //    else res[ii] = datareg;
    //}

    //DOSA_insert_thresholding

}

}

#endif

