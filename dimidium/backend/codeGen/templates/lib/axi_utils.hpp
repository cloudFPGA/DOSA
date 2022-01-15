/*******************************************************************************
 * Copyright 2016 -- 2020 IBM Corporation
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

//  *
//  *                       cloudFPGA
//  *    =============================================
//  *     Created: Apr 2019
//  *     Authors: FAB, WEI, NGL
//  *
//  *     Description:
//  *        This file contains AXI types and functions that are shared between SHELL and ROLE.
//  *

#ifndef _CF_AXI_USER_UTILS_
#define _CF_AXI_USER_UTILS_


#include <stdint.h>
#include <stdio.h>

#include "ap_int.h"
#include "ap_utils.h"
#include <hls_stream.h>


using namespace hls;

/*
 * A generic unsigned AXI4-Stream interface used all over the cloudFPGA place.
 * It has NO defined byte order. The user can use this for BE and LE (and must ensure the encoding)!.
 */
template<int D>
struct Axis {
  protected:
    ap_uint<D>       tdata;
    ap_uint<(D+7)/8> tkeep;
    ap_uint<1>       tlast;
  public:
    Axis() {}
    Axis(ap_uint<D> single_data) : tdata((ap_uint<D>)single_data), tkeep(~(((ap_uint<D>) single_data) & 0)), tlast(1) {}
    Axis(ap_uint<D> new_data, ap_uint<(D+7/8)> new_keep, ap_uint<1> new_last) : tdata(new_data), tkeep(new_keep), tlast(new_last) {}
    ap_uint<D> getTData() {
      return tdata;
    }
    ap_uint<(D+7)/8> getTKeep() {
      return tkeep;
    }
    ap_uint<1> getTLast() {
      return tlast;
    }
    void setTData(ap_uint<D> new_data) {
      tdata = new_data;
    }
    void setTKeep(ap_uint<(D+7)/8> new_keep) {
      tkeep = new_keep;
    }
    void setTLast(ap_uint<1> new_last) {
      tlast = new_last;
    }
    //Axis<64>& operator= (const NetworkWord& nw) {
    //  this->tdata = nw.tdata;
    //  this->tkeep = nw.tkeep;
    //  this->tlast = nw.tlast;
    //  return *this;
    //}
    Axis<D>& operator= (const Axis<D>& ot) {
      this->tdata = ot.tdata;
      this->tkeep = ot.tkeep;
      this->tlast = ot.tlast;
      return *this;
    }
    //operator NetworkWord() {
    //  return NetworkWord(this->tdata, this->tkeep, this->tlast);
    //}
    //Axis(NetworkWord nw) : tdata((ap_uint<D>) nw.tdata), tkeep((ap_uint<(D+7/8)>) nw.tkeep), tlast(nw.tlast) {}
};



#endif
