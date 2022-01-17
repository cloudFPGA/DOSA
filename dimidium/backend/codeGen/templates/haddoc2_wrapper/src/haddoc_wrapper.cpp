//  *
//  *                       cloudFPGA
//  *     Copyright IBM Research, All Rights Reserved
//  *    =============================================
//  *     Created: Jan 2022
//  *     Authors: NGL
//  *
//  *     Description:
//  *        Template for a Haddoc2 wrapper
//  *

#include <stdio.h>
#include <stdint.h>
#include "ap_int.h"
#include "ap_utils.h"
#include <hls_stream.h>
#include <cassert>

#include "haddoc_wrapper.hpp"

using namespace hls;


void processLargeInput(
    Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> &in_read,
    ap_uint<2*DOSA_WRAPPER_INPUT_IF_BITWIDTH>  &combined_input,
    bool *comp_inp_valid,
    ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH> &cur_read_offset
    )
{
#pragma HLS INLINE
  for(int i = 0; i < WRAPPER_INPUT_IF_BYTES; i++)
  {
#pragma HLS unroll
    if((in_read.tkeep >> i) == 0)
    {
      continue;
    }
    ap_uint<8> current_byte = (ap_uint<8>) (in_read.tdata >> i*8);
    //TODO: what if input is not byte aligned?
    combined_input |= ((ap_uint<2*DOSA_WRAPPER_INPUT_IF_BITWIDTH>) current_byte) << cur_read_offset;
    for(int j = 0; j<8; j++)
    {
#pragma HLS unroll
      comb_inp_valid[j + cur_read_offset] = true;
    }
    cur_read_offset += 8;
  }
}


void flattenAxisInput(
    Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> &in_read,
    ap_uint<2*DOSA_WRAPPER_INPUT_IF_BITWIDTH>  &combined_input,
    ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH> &cur_line_bit_cnt
    )
{
#pragma HLS INLINE
  for(int i = 0; i < WRAPPER_INPUT_IF_BYTES; i++)
  {
#pragma HLS unroll
    if((in_read.tkeep >> i) == 0)
    {
      continue;
    }
    ap_uint<8> current_byte = (ap_uint<8>) (in_read.tdata >> i*8);
    //TODO: what if input is not byte aligned?
    combined_input |= ((ap_uint<2*DOSA_WRAPPER_INPUT_IF_BITWIDTH>) current_byte) << cur_line_bit_cnt;
    cur_line_bit_cnt += 8;
  }
}


ap_uint<WRAPPER_INPUT_IF_BYTES> createTReady(valid_bit_cnt)
{
#pragma HLS INLINE
  ap_uint<WRAPPER_INPUT_IF_BYTES> tready = 0x0;
  for(int i = 0; i < WRAPPER_INPUT_IF_BYTES; i++)
  {
#pragma HLS unroll
    if(i < valid_bit_cnt)
    {
      tready |= (ap_uint<WRAPPER_INPUT_IF_BYTES> 0x1) << i;
    }
    //TODO: not byte aligned?
  }
}

bool genericEnqState(
  stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >   &siData,
  stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >    &cur_buffer,
  ap_uint<64> &current_frame_bit_cnt,
  ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH> &hangover_store,
  ap_uint<64> &hangover_store_valid_bits
    )
{
#pragma HLS INLINE
  ap_uint<2*DOSA_WRAPPER_INPUT_IF_BITWIDTH> combined_input = 0x0;
  ap_uint<64> cur_line_bit_cnt = 0;

  bool go_to_next_state = false;

  // consider hangover
  if(hangover_store_valid_bits > 0)
  {
    combined_input = hangover_store;
    hangover_store = 0x0;
    cur_line_bit_cnt = hangover_store_valid_bits;
    hangover_store_valid_bits = 0;
  }

  //read axis, determine byte count, fill in buffer
  Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> tmp_read_0 = siData.read();
  flattenAxisInput(tmp_read_0, combined_input, cur_line_bit_cnt);
  //TODO: what about siData.tlast?
  

  ap_uint<64> to_axis_cnt = cur_line_bit_cnt;
  if( to_axis_cnt > DOSA_WRAPPER_INPUT_IF_BITWIDTH )
  {
    to_axis_cnt = DOSA_WRAPPER_INPUT_IF_BITWIDTH;
    //write to hangover
    hangover_store_valid_bits = cur_line_bit_cnt - DOSA_WRAPPER_INPUT_IF_BITWIDTH;
    hangover_store = (ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH>) (combined_input >> DOSA_WRAPPER_INPUT_IF_BITWIDTH);
  }
  ap_uint<1> to_axis_tlast = 0;
  current_frame_bit_cnt += to_axis_cnt;
  if( current_frame_bit_cnt >= HADDOC_INPUT_FRAME_BIT_CNT )
  {
    to_axis_tlast = 1;
    //enqueueFSM = FILL_BUF_1;
    current_frame_bit_cnt = 0;
    go_to_next_state = true;
  } else {
    //enqueueFSM = FILL_BUF_0;
    go_to_next_state = false;
  }
  Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> tmp_write_0 = Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH>(ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH> combined_input,
      createTReady(to_axis_cnt), to_axis_tlast);
  cur_buffer.write(tmp_write_0);
  
  return go_to_next_state;
}


ap_uint<64> flattenAxisBuffer(
    Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> &in_read,
    ap_uint<2*DOSA_WRAPPER_INPUT_IF_BITWIDTH>  &combined_input,
    ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH>  &hangover_bits,
    ap_uint<32>                            &hangover_bits_valid_bits,
    )
{
#pragma HLS INLINE
  ap_uint<64> cur_line_bit_cnt = 0;
  // consider hangover
  if(hangover_bits_valid_bits > 0)
  {
    combined_input = hangover_bits;
    hangover_bits = 0x0;
    cur_line_bit_cnt = hangover_bits_valid_bits;
    hangover_bits_valid_bits = 0;
  }

  for(int i = 0; i < WRAPPER_INPUT_IF_BYTES; i++)
  {
#pragma HLS unroll
    if((in_read.tkeep >> i) == 0)
    {
      continue;
    }
    ap_uint<8> current_byte = (ap_uint<8>) (in_read.tdata >> i*8);
    //TODO: what if input is not byte aligned?
    combined_input |= ((ap_uint<2*DOSA_WRAPPER_INPUT_IF_BITWIDTH>) current_byte) << cur_line_bit_cnt;
    cur_line_bit_cnt += 8;
  }

  return cur_line_bit_cnt;
}


void pToHaddocEnq(
#ifdef WRAPPER_TEST
  stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >    &sToHaddocBuffer_chan1,
  stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >    &sToHaddocBuffer_chan2,
  stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >    &sToHaddocBuffer_chan3,
#else
    //DOSA_ADD_toHaddoc_buffer_param_decl
#endif
    stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >   &siData
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static ToHaddocEnqStates enqueueFSM = RESET;
#pragma HLS reset variable=enqueueFSM
  static ap_uint<64> current_frame_bit_cnt = 0x0;
#pragma HLS reset variable=current_frame_bit_cnt
  //TODO: ensure 2^64 is enough... ;)
  static ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH> hangover_store = 0x0;
#pragma HLS reset variable=hangover_store
  static ap_uint<64> hangover_store_valid_bits = 0x0;
#pragma HLS reset variable=hangover_store_valid_bits
  static ap_uint<32> wait_drain_cnt = input_fifo_depth;
#pragma HLS reset variable=wait_drain_cnt

  //-- LOCAL VARIABLES ------------------------------------------------------

  switch(enqueueFSM)
  {
    default:
    case RESET:
      //TODO: necessary?
      current_frame_bit_cnt = 0x0;
      hangover_store = 0x0;
      hangover_store_valid_bits = 0;
      wait_drain_cnt = input_fifo_depth;
      enqueueFSM = WAIT_DRAIN;
      break;

    case WAIT_DRAIN:
      if(wait_drain_cnt == 0)
      {
        enqueueFSM = FILL_BUF_0;
      } else {
        wait_drain_cnt--;
      }
      break;

    case FILL_BUF_0:
#ifdef WRAPPER_TEST
    //we distribute on the channels only, cutting in right bitsize in dequeue process
      if( !siData.empty() && !sToHaddocBuffer_chan1.full() )
      {
        if(genericEnqState(siData, sToHaddocBuffer_chan1, current_frame_bit_cnt, hangover_store, hangover_store_valid_bits))
        {
          enqueueFSM = FILL_BUF_1;
        }
      }
      break;
    case FILL_BUF_1:
      if( !siData.empty() && !sToHaddocBuffer_chan2.full() )
      {
        if(genericEnqState(siData, sToHaddocBuffer_chan2, current_frame_bit_cnt, hangover_store, hangover_store_valid_bits))
        {
          enqueueFSM = FILL_BUF_2;
        }
      }
      break;
    case FILL_BUF_2:
      if( !siData.empty() && !sToHaddocBuffer_chan3.full() )
      {
        if(genericEnqState(siData, sToHaddocBuffer_chan3, current_frame_bit_cnt, hangover_store, hangover_store_valid_bits))
        {
          //last channel -> go to start
          enqueueFSM = FILL_BUF_1;
        }
      }
#else
      //DOSA_ADD_enq_fsm
#endif
      break;
  }

}


void pToHaddocDeq(
#ifdef WRAPPER_TEST
  stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >    &sToHaddocBuffer_chan1,
  stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >    &sToHaddocBuffer_chan2,
  stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >    &sToHaddocBuffer_chan3,
#else
    //DOSA_ADD_toHaddoc_buffer_param_decl
#endif
    ap_uint<1>                                *po_haddoc_data_valid,
    ap_uint<1>                                *po_haddoc_frame_valid,
    ap_uint<DOSA_HADDOC_INPUT_BITDIWDTH> >    *po_haddoc_data_vector,
    stream<bool>                              &sHaddocUnitProcessing
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static ToHaddocDeqStates dequeueFSM = RESET;
#pragma HLS reset variable=dequeueFSM
  static ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> hangover_bits[DOSA_HADDOC_INPUT_CHAN_NUM];
#pragma HLS ARRAY_PARTITION variable=hangover_bits complete
  static ap_uint<32> hangover_bits_valid_bits[DOSA_HADDOC_INPUT_CHAN_NUM];
#pragma HLS ARRAY_PARTITION variable=hangover_bits_valid_bits complete
  static bool only_hangover_processing = false;
#pragma HLS reset variable=only_hangover_processing
  //-- LOCAL VARIABLES ------------------------------------------------------

  ap_uint<2*DOSA_WRAPPER_INPUT_IF_BITWIDTH> combined_input[DOSA_HADDOC_INPUT_CHAN_NUM];
#pragma HLS ARRAY_PARTITION variable=combined_input complete
  ap_uint<32> cur_line_bit_cnt[DOSA_HADDOC_INPUT_CHAN_NUM];
#pragma HLS ARRAY_PARTITION variable=cur_line_bit_cnt complete


  //if-else better than 2-state switch?
  if(dequeueFSM == RESET)
  {
    for(int i = 0; i<DOSA_HADDOC_INPUT_CHAN_NUM; i++)
    {
      hangover_bits[i] = 0x0;
      hangover_bits_valid_bits[i] = 0x0;
    }
    bool one_not_empty = false;
#ifdef WRAPPER_TEST
    if( !sToHaddocBuffer_chan1.empty() )
    {
      sToHaddocBuffer_chan1.read();
      one_not_empty = true;
    }
    if( !sToHaddocBuffer_chan2.empty() )
    {
      sToHaddocBuffer_chan2.read();
      one_not_empty = true;
    }
    if( !sToHaddocBuffer_chan3.empty() )
    {
      sToHaddocBuffer_chan3.read();
      one_not_empty = true;
    }
#else
    //DOSA_ADD_toHaddoc_deq_buffer_drain
#endif
    *po_haddoc_data_valid = 0x0;
    *po_haddoc_frame_valid = 0x0;
    *po_haddoc_data_vector = 0x0;
    only_hangover_processing = false;
    if( !one_not_empty )
    {
      dequeueFSM = FORWARD;
    }
  } else {
    if( !sHaddocUnitProcessing.full() &&
#ifdef WRAPPER_TEST
        && !sToHaddocBuffer_chan1.empty() && !sToHaddocBuffer_chan2.empty() && !sToHaddocBuffer_chan3.empty()
#else
        //DOSA_ADD_toHaddoc_deq_if_clause
#endif
      )
    {
      for(int i = 0; i<DOSA_HADDOC_INPUT_CHAN_NUM; i++)
      {
        combined_input[i] = 0x0;
        cur_line_bit_cnt[i] = 0x0;
      }
      
      if( !only_hangover_processing )
      {
#ifdef WRAPPER_TEST
        cur_line_bit_cnt[0] = flattenAxisBuffer(sToHaddocBuffer_chan1.read(), combined_input[0], hangover_bits[0], hangover_store_valid_bits[0]);
        cur_line_bit_cnt[1] = flattenAxisBuffer(sToHaddocBuffer_chan2.read(), combined_input[1], hangover_bits[1], hangover_store_valid_bits[1]);
        cur_line_bit_cnt[2] = flattenAxisBuffer(sToHaddocBuffer_chan3.read(), combined_input[2], hangover_bits[2], hangover_store_valid_bits[2]);
#else
        //DOSA_ADD_deq_flatten
#endif
      } else {
        //need to reduce backlog
        for(int i = 0; i<DOSA_HADDOC_INPUT_CHAN_NUM; i++)
        {
          combined_input[i] = hangover_bits[i];
          cur_line_bit_cnt[i] = hangover_bits_valid_bits[i];
          hangover_bits[i] = 0x0;
          hangover_bits_valid_bits[i] = 0x0;
        }
        only_hangover_processing = false;
      }
      ap_uint<DOSA_HADDOC_INPUT_BITDIWDTH> output_data = 0x0;
      for(int i = 0; i<DOSA_HADDOC_INPUT_CHAN_NUM; i++)
      {
        ap_uint<2*DOSA_WRAPPER_INPUT_IF_BITWIDTH> tmp_bit_cnt = cur_line_bit_cnt[i];
        //actually, it is either > or ==..., since frames are transmitted aligned, can't be less than that
        if(tmp_bit_cnt > DOSA_HADDOC_GENERAL_BITWIDTH)
        {
          tmp_bit_cnt = DOSA_HADDOC_GENERAL_BITWIDTH;
          hangover_bits_valid_bits[i] = cur_line_bit_cnt[i] - DOSA_HADDOC_GENERAL_BITWIDTH;
          hangover_bits[i] = (ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH>) (combined_input[i] >> DOSA_HADDOC_GENERAL_BITWIDTH);
          if(hangover_bits_valid_bits[i] >= DOSA_HADDOC_GENERAL_BITWIDTH)
          {
            only_hangover_processing = true;
          }
        }
        //DynLayerInput requires (chan2,chan1,chan0) vector layout
        output_data |= (ap_uint<DOSA_HADDOC_INPUT_BITDIWDTH>) (((ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH>) combined_input) << i*DOSA_HADDOC_GENERAL_BITWIDTH);
      }
    
      *po_haddoc_data_valid = 0x1;
      *po_haddoc_data_vector = output_data;
      sHaddocUnitProcessing.write(true);
    } else {
      *po_haddoc_data_valid = 0x0;
    }
    *po_haddoc_frame_valid = 0x1;
  }

}

void pFromHaddocEnq(
    ap_uint<1>                                *pi_haddoc_data_valid,
    ap_uint<1>                                *pi_haddoc_frame_valid,
    ap_uint<DOSA_HADDOC_INPUT_BITDIWDTH> >    *pi_haddoc_data_vector,
    stream<bool>                              &sHaddocUnitProcessing,
    stream<ap_uint<DOSA_HADDOC_OUTPUT_BITDIWDTH> >  &sFromHaddocBuffer
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static FromHaddocEnqStates enqueueFSM = RESET;
#pragma HLS reset variable=enqueueFSM
  //-- LOCAL VARIABLES ------------------------------------------------------
  ap_uint<DOSA_HADDOC_OUTPUT_BITDIWDTH> input_data = 0x0;

  if(enqueueFSM == RESET)
  {
    if(!sHaddocUnitProcessing.empty())
    {
      sHaddocUnitProcessing.read();
    } else {
      enqueueFSM = FORWARD;
    }
  } else {
    if ( !sHaddocUnitProcessing.empty() && !sFromHaddocBuffer.full() )
    {
      //ignore pi_haddoc_frame_valid?
      if( *pi_haddoc_data_valid == 0b1 )
      {
        input_data = *pi_haddoc_data_vector;
        //read only if able to process
        bool ignore_me = sHaddocUnitProcessing.read();
        sFromHaddocBuffer.write(input_data);
      }
    }
  }

}
    


void pFromHaddocFlatten(
#ifdef WRAPPER_TEST
  //stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> >    &sFromHaddocBuffer_chan1,
  //stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> >    &sFromHaddocBuffer_chan2,
  //stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> >    &sFromHaddocBuffer_chan3,
  ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> *arrFromHaddocBuffer_chan1,
  ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> *arrFromHaddocBuffer_chan2,
  ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> *arrFromHaddocBuffer_chan3,
#else
    //DOSA_ADD_toHaddoc_buffer_param_decl
#endif
    stream<ap_uint<DOSA_HADDOC_OUTPUT_BITDIWDTH> >  &sFromHaddocBuffer
    stream<bool>                              &sFromHaddocFrameComplete
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static FromHaddocFlattenStates flattenFSM = RESET;
#pragma HLS reset variable=flattenFSM
  static ap_uint<64> current_frame_bit_cnt = 0x0;
#pragma HLS reset variable=current_frame_bit_cnt
  static ap_uint<64> current_array_write_pnt = 0x0;
#pragma HLS reset variable=current_array_write_pnt
  static ap_uint<2> current_array_slot_pnt = 0x0;
#pragma HLS reset variable=current_array_slot_pnt
  //-- LOCAL VARIABLES ------------------------------------------------------
  ap_uint<DOSA_HADDOC_OUTPUT_BITDIWDTH> input_data = 0x0;

  //if-else better than 2-state switch?
  if(flattenFSM == RESET)
  {
    current_frame_bit_cnt = 0x0;
    current_array_slot_pnt = 0x0;
    current_array_write_pnt = 0x0;
    //we have to write the array...
    for(int i = 0; i < 2*output_fifo_depth; i++)
    {
#ifdef WRAPPER_TEST
      arrFromHaddocBuffer_chan1[i] = 0;
      arrFromHaddocBuffer_chan2[i] = 0;
      arrFromHaddocBuffer_chan3[i] = 0;
#else
      //DOSA_ADD_fromHaddoc_enq_buffer_reset
#endif
    }
    flattenFSM = FORWARD;
  } else {
    if( !sFromHaddocBuffer.empty() && !sFromHaddocFrameComplete.full()
//#ifdef WRAPPER_TEST
//        && !sFromHaddocBuffer_chan1.full() && !sFromHaddocBuffer_chan2.full() && !sFromHaddocBuffer_chan3.full()
//#else
//        //DOSA_ADD_fromHaddoc_enq_if_clause
//#endif
      )
    {
      input_data = sFromHaddocBuffer.read();
#ifdef WRAPPER_TEST
      //sFromHaddocBuffer_chan1.write((ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH>) (input_data >> 0 * DOSA_HADDOC_GENERAL_BITWIDTH));
      //sFromHaddocBuffer_chan2.write((ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH>) (input_data >> 1 * DOSA_HADDOC_GENERAL_BITWIDTH));
      //sFromHaddocBuffer_chan3.write((ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH>) (input_data >> 2 * DOSA_HADDOC_GENERAL_BITWIDTH));
      arrFromHaddocBuffer_chan1[current_array_slot_pnt*output_fifo_depth + current_array_write_pnt] = (ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH>) (input_data >> 0 * DOSA_HADDOC_GENERAL_BITWIDTH)
      arrFromHaddocBuffer_chan2[current_array_slot_pnt*output_fifo_depth + current_array_write_pnt] = (ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH>) (input_data >> 1 * DOSA_HADDOC_GENERAL_BITWIDTH)
      arrFromHaddocBuffer_chan3[current_array_slot_pnt*output_fifo_depth + current_array_write_pnt] = (ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH>) (input_data >> 2 * DOSA_HADDOC_GENERAL_BITWIDTH)
#else
        //DOSA_ADD_fromHaddoc_enq_chan_write
#endif

        current_array_write_pnt++;
        current_frame_bit_cnt += DOSA_HADDOC_GENERAL_BITWIDTH;
        if( current_frame_bit_cnt >= HADDOC_OUTPUT_FRAME_BIT_CNT )
        {
          sFromHaddocFrameComplete.write(true);
          current_frame_bit_cnt = 0;
          current_array_write_pnt = 0;
          current_array_slot_pnt++;
          if(current_array_slot_pnt > 1)
          {//double buffering
            current_array_slot_pnt = 0;
          }
        }
    }
  }

}


void pFromHaddocDeq(
#ifdef WRAPPER_TEST
  //stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> >    &sFromHaddocBuffer_chan1,
  //stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> >    &sFromHaddocBuffer_chan2,
  //stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> >    &sFromHaddocBuffer_chan3,
  ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> *arrFromHaddocBuffer_chan1,
  ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> *arrFromHaddocBuffer_chan2,
  ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> *arrFromHaddocBuffer_chan3,
#else
    //DOSA_ADD_toHaddoc_buffer_param_decl
#endif
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >  &soData,
    stream<bool>                              &sFromHaddocFrameComplete
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static FromHaddocDeqStates dequeueFSM = RESET;
#pragma HLS reset variable=dequeueFSM
  static ap_uint<64> current_frame_bit_cnt = 0x0;
#pragma HLS reset variable=current_frame_bit_cnt
  static ap_uint<64> current_batch_bit_cnt = 0x0;
#pragma HLS reset variable=current_batch_bit_cnt
  //-- LOCAL VARIABLES ------------------------------------------------------
  ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+DOSA_HADDOC_GENERAL_BITWIDTH> combined_output = 0x0;

  //TODO: do we need a FSM?
  // we can read WRAPPER_OUTPUT_IF_HADDOC_WORDS_CNT_CEIL array entries at max, then maybe some bits are in hangover...
  // maybe don't have three arrays, have one array? so that the loop here is rather simple?
  // back pressure via FrameComplete...

  switch(dequeueFSM)
  {
    default:
    case RESET:
      current_frame_bit_cnt = 0;
      current_batch_bit_cnt = 0;
//      bool one_not_empty = false;
//#ifdef WRAPPER_TEST
//      if( !sFromHaddocBuffer_chan1.empty() )
//      {
//        sFromHaddocBuffer_chan1.read();
//        one_not_empty = true;
//      }
//      if( !sFromHaddocBuffer_chan2.empty() )
//      {
//        sFromHaddocBuffer_chan2.read();
//        one_not_empty = true;
//      }
//      if( !sFromHaddocBuffer_chan3.empty() )
//      {
//        sFromHaddocBuffer_chan3.read();
//        one_not_empty = true;
//      }
//#else
//      //DOSA_ADD_fromHaddoc_deq_buffer_drain
//#endif
//      if( !one_not_empty )
//      {
//        dequeueFSM = WAIT_FRAME;
//      }
      dequeueFSM = WAIT_FRAME;
      break;
    case WAIT_FRAME:
      if( !sFromHaddocFrameComplete.empty() )
      {
        bool ignore_me = sFromHaddocFrameComplete.read();
        dequeueFSM = READ_BUF_0;
      }
      break;

    case READ_BUF_0:
#ifdef WRAPPER_TEST
#else
      //DOSA_ADD_fromHaddoc_deq_fsm
#endif
      break;
  }

}

void haddoc_wrapper(
    // ----- Wrapper Interface -----
    stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >   &siData,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >  &soData,
    // ----- Haddoc Interface -----
    ap_uint<1>                                *po_haddoc_data_valid,
    ap_uint<1>                                *po_haddoc_frame_valid,
    ap_uint<DOSA_HADDOC_INPUT_BITDIWDTH> >    *po_haddoc_data_vector,
    ap_uint<1>                                *pi_haddoc_data_valid,
    ap_uint<1>                                *pi_haddoc_frame_valid,
    ap_uint<DOSA_HADDOC_OUTPUT_BITDIWDTH> >   *pi_haddoc_data_vector,
    // ----- DEBUG IO ------
    ap_uint<32> *debug_out
    )
{
  //-- DIRECTIVES FOR THE BLOCK ---------------------------------------------
#pragma HLS INTERFACE ap_ctrl_none port=return

#pragma HLS INTERFACE ap_fifo port=siData
#pragma HLS INTERFACE ap_fifo port=soData

#pragma HLS INTERFACE ap_ovld register port=po_haddoc_data_valid
#pragma HLS INTERFACE ap_ovld register port=po_haddoc_frame_valid
#pragma HLS INTERFACE ap_ovld register port=po_haddoc_data_vector
#pragma HLS INTERFACE ap_vld register port=pi_haddoc_data_valid
#pragma HLS INTERFACE ap_vld register port=pi_haddoc_frame_valid
#pragma HLS INTERFACE ap_vld register port=pi_haddoc_data_vector
#pragma HLS INTERFACE ap_ovld register port=debug_out


  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS DATAFLOW

#ifndef __SYNTHESIS__
  assert(DOSA_WRAPPER_INPUT_IF_BITWIDTH >= DOSA_HADDOC_GENERAL_BITWIDTH); //currently, the assumption is that one "pixel" is smaller than the interface bitwidth
  assert(HADDOC_INPUT_FRAME_BIT_CNT % 8 == 0); //currently, only byte-aligned FRAMES are supported
#endif

  //-- STATIC VARIABLES (with RESET) ------------------------------------------

  //-- STATIC DATAFLOW VARIABLES ------------------------------------------
#ifdef WRAPPER_TEST
  static stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> > sToHaddocBuffer_chan1 ("sToHaddocBuffer_chan1");
  #pragma HLS STREAM variable=sToHaddocBuffer_chan1   depth=input_fifo_depth
  static stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> > sToHaddocBuffer_chan2 ("sToHaddocBuffer_chan2");
  #pragma HLS STREAM variable=sToHaddocBuffer_chan2   depth=input_fifo_depth
  static stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> > sToHaddocBuffer_chan3 ("sToHaddocBuffer_chan3");
  #pragma HLS STREAM variable=sToHaddocBuffer_chan3   depth=input_fifo_depth
  
  //static stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> > sFromHaddocBuffer_chan1 ("sFromHaddocBuffer_chan1");
  //#pragma HLS STREAM variable=sFromHaddocBuffer_chan1   depth=output_fifo_depth
  //static stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> > sFromHaddocBuffer_chan2 ("sFromHaddocBuffer_chan2");
  //#pragma HLS STREAM variable=sFromHaddocBuffer_chan2   depth=output_fifo_depth
  //static stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> > sFromHaddocBuffer_chan3 ("sFromHaddocBuffer_chan3");
  //#pragma HLS STREAM variable=sFromHaddocBuffer_chan3   depth=output_fifo_depth
  static ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> arrFromHaddocBuffer_chan1[2*output_fifo_depth];
  #pragma HLS ARRAY_PARTITION variable=arrFromHaddocBuffer_chan1 complete
  static ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> arrFromHaddocBuffer_chan2[2*output_fifo_depth];
  #pragma HLS ARRAY_PARTITION variable=arrFromHaddocBuffer_chan2 complete
  static ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> arrFromHaddocBuffer_chan3[2*output_fifo_depth];
  #pragma HLS ARRAY_PARTITION variable=arrFromHaddocBuffer_chan3 complete

#else
  //DOSA_ADD_haddoc_buffer_instantiation
#endif
  static stream<bool> sHaddocUnitProcessing ("sHaddocUnitProcessing");
  const int haddoc_processing_fifo_depth = HADDOC_AVG_LAYER_LATENCY * DOSA_HADDOC_LAYER_CNT;
  #pragma HLS STREAM variable=sHaddocUnitProcessing depth=haddoc_processing_fifo_depth
  static stream<ap_uint<DOSA_HADDOC_OUTPUT_BITDIWDTH> > sFromHaddocBuffer ("sFromHaddocBuffer");
  #pragma HLS STREAM variable=sFromHaddocBuffer depth=haddoc_processing_fifo_depth+3  //so that we can receive, what we send out...
  static stream<bool> sFromHaddocFrameComplete ("sFromHaddocFrameComplete");
  #pragma HLS STREAM variable=sFromHaddocFrameComplete depth=2  //2! since arrays are double-buffers
  
  //-- PROCESS INSTANTIATION ------------------------------------------------------

  // with the current strategy [INPUT] -> [CHANEL, with same bitwidth] -> [BATCH_PIXEL], we can maintain
  // ~ (DOSA_HADDOC_INPUT_CHAN_NUM * DOSA_HADDOC_GENERAL_BITWIDTH)/DOSA_WRAPPER_INPUT_IF_BITWIDTH of the bandwidth
  pToHaddocEnq(
#ifdef WRAPPER_TEST
      sToHaddocBuffer_chan1, sToHaddocBuffer_chan2, sToHaddocBuffer_chan3,
#else
      //DOSA_ADD_toHaddoc_buffer_list
#endif
      siData);

  pToHaddocDeq(
#ifdef WRAPPER_TEST
      sToHaddocBuffer_chan1, sToHaddocBuffer_chan2, sToHaddocBuffer_chan3,
#else
      //DOSA_ADD_toHaddoc_buffer_list
#endif
      po_haddoc_data_valid, po_haddoc_frame_valid, po_haddoc_data_vector, sHaddocUnitProcessing);


}



