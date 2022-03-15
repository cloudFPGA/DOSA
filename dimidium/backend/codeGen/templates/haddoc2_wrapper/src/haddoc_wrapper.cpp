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
    if((in_read.getTKeep() >> i) == 0)
    {
      printf("skipped due to tkeep\n");
      continue;
    }
    ap_uint<8> current_byte = (ap_uint<8>) (((ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH>) in_read.getTData()) >> (i*8));
    //TODO: what if input is not byte aligned?
    combined_input |= ((ap_uint<2*DOSA_WRAPPER_INPUT_IF_BITWIDTH>) current_byte) << cur_read_offset;
    for(int j = 0; j<8; j++)
    {
#pragma HLS unroll
      comp_inp_valid[j + cur_read_offset] = true;
    }
    cur_read_offset += 8;
  }
}


void flattenAxisInput(
    Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> &in_read,
    ap_uint<(2*DOSA_WRAPPER_INPUT_IF_BITWIDTH)>  &combined_input,
    ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH> &cur_line_bit_cnt
    )
{
#pragma HLS INLINE
  for(int i = 0; i < WRAPPER_INPUT_IF_BYTES; i++)
  {
#pragma HLS unroll
    if((in_read.getTKeep() >> i) == 0)
    {
      printf("flatten: skipped due to tkeep\n");
      continue;
    }
    //TODO: what if input is not byte aligned?
    ap_uint<8> current_byte = (ap_uint<8>) (((ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH>) in_read.getTData()) >> (i*8));
    //printf("flatten: cur_byte %x\n", (uint32_t) current_byte);
    //ap_uint<128> tmp_delme = ((ap_uint<128>) current_byte) << cur_line_bit_cnt;
    //printf("flatten: delm 0x%16llx %16llx, bitwidth %d\n", (uint64_t) (tmp_delme >> 64), (uint64_t) tmp_delme, 2*DOSA_WRAPPER_INPUT_IF_BITWIDTH);
    combined_input |= (ap_uint<(2*DOSA_WRAPPER_INPUT_IF_BITWIDTH)>) (((ap_uint<(2*DOSA_WRAPPER_INPUT_IF_BITWIDTH)>) current_byte) << cur_line_bit_cnt);
    //printf("flatten: read 0x%16llx %16llx, bit cnt: %d\n", (uint64_t) (combined_input >> 64), (uint64_t) combined_input, (uint32_t) cur_line_bit_cnt);
    cur_line_bit_cnt += 8;
  }
}


ap_uint<WRAPPER_INPUT_IF_BYTES> createTReady(ap_uint<64> valid_bit_cnt)
{
#pragma HLS INLINE
  ap_uint<WRAPPER_INPUT_IF_BYTES> tready = 0x0;
  for(int i = 0; i < WRAPPER_INPUT_IF_BYTES; i++)
  {
#pragma HLS unroll
    if(i < valid_bit_cnt)
    {
      tready |= ((ap_uint<WRAPPER_INPUT_IF_BYTES>) 0x1) << i;
    }
    //TODO: not byte aligned?
  }
  return tready;
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
  ap_uint<(2*DOSA_WRAPPER_INPUT_IF_BITWIDTH)> combined_input = 0x0;
  ap_uint<64> cur_line_bit_cnt = 0;

  bool go_to_next_state = false;

  // consider hangover
  if(hangover_store_valid_bits > 0)
  {
    //printf("considered hangover\n");
    combined_input = hangover_store;
    hangover_store = 0x0;
    cur_line_bit_cnt = hangover_store_valid_bits;
    hangover_store_valid_bits = 0;
  }

  if(cur_line_bit_cnt < DOSA_WRAPPER_INPUT_IF_BITWIDTH)
  {
    //read axis, determine byte count, fill in buffer
    Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> tmp_read_0 = siData.read();
    flattenAxisInput(tmp_read_0, combined_input, cur_line_bit_cnt);
    //TODO: what about siData.tlast?
  }
  //printf("enq: read 0x%16.16llx %16.16llx, bit cnt: %d\n", (uint64_t) (combined_input >> 64), (uint64_t) combined_input, (uint32_t) cur_line_bit_cnt);

  ap_uint<64> to_axis_cnt = cur_line_bit_cnt;
  if( to_axis_cnt > DOSA_WRAPPER_INPUT_IF_BITWIDTH )
  {
    //printf("hangover necessary\n");
    to_axis_cnt = DOSA_WRAPPER_INPUT_IF_BITWIDTH;
    //write to hangover
    hangover_store_valid_bits = cur_line_bit_cnt - DOSA_WRAPPER_INPUT_IF_BITWIDTH;
    hangover_store = (ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH>) (combined_input >> DOSA_WRAPPER_INPUT_IF_BITWIDTH);
  } else {
    hangover_store_valid_bits = 0x0;
  }
  ap_uint<1> to_axis_tlast = 0;
  current_frame_bit_cnt += to_axis_cnt;
  if( current_frame_bit_cnt >= HADDOC_INPUT_FRAME_BIT_CNT )
  {
    hangover_store <<= hangover_store_valid_bits;
    hangover_store_valid_bits += current_frame_bit_cnt - HADDOC_INPUT_FRAME_BIT_CNT;
    to_axis_cnt = HADDOC_INPUT_FRAME_BIT_CNT - (current_frame_bit_cnt - to_axis_cnt);
    hangover_store |= (ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH>) (combined_input >> to_axis_cnt);
    to_axis_tlast = 1;
    //printf("enque: go to next state, cur_frame_cnt: %d, hangover_valid_bits: %d\n", (uint32_t) current_frame_bit_cnt, (uint32_t) hangover_store_valid_bits);
    //enqueueFSM = FILL_BUF_1;
    current_frame_bit_cnt = 0;
    go_to_next_state = true;
  } else {
    //enqueueFSM = FILL_BUF_0;
    go_to_next_state = false;
  }
  //printf("enq: sorted %16.16llx, to_axis_cnt: %d\n", (uint64_t) combined_input, (uint32_t) to_axis_cnt);
  Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> tmp_write_0 = Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH>((ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH>) combined_input,
      createTReady(to_axis_cnt), to_axis_tlast);
  cur_buffer.write(tmp_write_0);

  return go_to_next_state;
}


ap_uint<64> flattenAxisBuffer(
    Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> &in_read,
    ap_uint<2*DOSA_WRAPPER_INPUT_IF_BITWIDTH>  &combined_input,
    ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH>  &hangover_bits,
    ap_uint<64>                            &hangover_bits_valid_bits
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
    if((in_read.getTKeep() >> i) == 0)
    {
      printf("flatten buffer: skipped due to tkeep\n");
      continue;
    }
    //TODO: what if input is not byte aligned?
    ap_uint<8> current_byte = (ap_uint<8>) (((ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH>) in_read.getTData()) >> (i*8));
    combined_input |= ((ap_uint<2*DOSA_WRAPPER_INPUT_IF_BITWIDTH>) current_byte) << cur_line_bit_cnt;
    //printf("flatten buffer: read 0x%16llx %16llx, bit cnt: %d\n", (uint64_t) (combined_input >> 64), (uint64_t) combined_input, (uint32_t) cur_line_bit_cnt);
    cur_line_bit_cnt += 8;
  }

  //printf("flatten buffer: read 0x%16llx %16llx, ret bit cnt: %d\n", (uint64_t) (combined_input >> 64), (uint64_t) combined_input, (uint32_t) cur_line_bit_cnt);
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
    stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >   &siData,
    uint16_t *debug
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static ToHaddocEnqStates enqueueFSM = RESET0;
#pragma HLS reset variable=enqueueFSM
  static ap_uint<64> current_frame_bit_cnt = 0x0;
#pragma HLS reset variable=current_frame_bit_cnt
  //TODO: ensure 2^64 is enough... ;)
  static ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH> hangover_store = 0x0;
#pragma HLS reset variable=hangover_store
  static ap_uint<64> hangover_store_valid_bits = 0x0;
#pragma HLS reset variable=hangover_store_valid_bits
  static uint32_t wait_drain_cnt = cnn_input_frame_size;
#pragma HLS reset variable=wait_drain_cnt

  //-- LOCAL VARIABLES ------------------------------------------------------

  switch(enqueueFSM)
  {
    default:
    case RESET0:
      //TODO: necessary?
      current_frame_bit_cnt = 0x0;
      hangover_store = 0x0;
      hangover_store_valid_bits = 0;
      wait_drain_cnt = cnn_input_frame_size;
#ifndef __SYNTHESIS_
      wait_drain_cnt = 10;
#endif
      enqueueFSM = WAIT_DRAIN;
      break;

    case WAIT_DRAIN:
      printf("wait_drain_cnt = %d\n", wait_drain_cnt);
      if(wait_drain_cnt == 0)
      {
        enqueueFSM = FILL_BUF_0;
      } else {
        wait_drain_cnt--;
      }
      break;

#ifdef WRAPPER_TEST
    case FILL_BUF_0:
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
          enqueueFSM = FILL_BUF_0;
        }
      }
      break;
#else
      //DOSA_ADD_enq_fsm
#endif
  }

  *debug = (uint8_t) enqueueFSM;
  *debug = ((uint16_t) hangover_store_valid_bits) << 8;

}


void pToHaddocDeq(
#ifdef WRAPPER_TEST
  stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >    &sToHaddocBuffer_chan1,
  stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >    &sToHaddocBuffer_chan2,
  stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >    &sToHaddocBuffer_chan3,
#else
    //DOSA_ADD_toHaddoc_buffer_param_decl
#endif
    //ap_uint<1>                                *po_haddoc_data_valid,
    //ap_uint<1>                                *po_haddoc_frame_valid,
    //ap_uint<DOSA_HADDOC_INPUT_BITDIWDTH>      *po_haddoc_data_vector,
    //ap_uint<DOSA_HADDOC_INPUT_BITDIWDTH+1>      *output_vector,
    stream<ap_uint<DOSA_HADDOC_INPUT_BITDIWDTH> > &po_haddoc_data,
    stream<bool>                              &sHaddocUnitProcessing,
    uint16_t *debug
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static twoStatesFSM dequeueFSM = RESET;
#pragma HLS reset variable=dequeueFSM
  static ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH> hangover_bits[DOSA_HADDOC_INPUT_CHAN_NUM];
#pragma HLS ARRAY_PARTITION variable=hangover_bits complete
#pragma HLS reset variable=hangover_bits
  static ap_uint<64> hangover_bits_valid_bits[DOSA_HADDOC_INPUT_CHAN_NUM];
#pragma HLS ARRAY_PARTITION variable=hangover_bits_valid_bits complete
#pragma HLS reset variable=hangover_bits_valid_bits
  static bool only_hangover_processing = false;
#pragma HLS reset variable=only_hangover_processing
  //-- LOCAL VARIABLES ------------------------------------------------------

  ap_uint<2*DOSA_WRAPPER_INPUT_IF_BITWIDTH> combined_input[DOSA_HADDOC_INPUT_CHAN_NUM];
#pragma HLS ARRAY_PARTITION variable=combined_input complete
  ap_uint<64> cur_line_bit_cnt[DOSA_HADDOC_INPUT_CHAN_NUM];
#pragma HLS ARRAY_PARTITION variable=cur_line_bit_cnt complete
  Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> tmp_read_0;

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
    //*po_haddoc_data_valid = 0x0;
    //*po_haddoc_frame_valid = 0x0;
    //*po_haddoc_data_vector = 0x0;
    //*output_vector = 0x0;
    only_hangover_processing = false;
    if( !one_not_empty )
    {
      dequeueFSM = FORWARD;
    }
  } else {
    if( !sHaddocUnitProcessing.full()
#ifdef WRAPPER_TEST
        && !sToHaddocBuffer_chan1.empty() && !sToHaddocBuffer_chan2.empty() && !sToHaddocBuffer_chan3.empty()
#else
        //DOSA_ADD_toHaddoc_deq_if_clause
#endif
        && !po_haddoc_data.full()
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
        tmp_read_0 = sToHaddocBuffer_chan1.read();
        cur_line_bit_cnt[0] = flattenAxisBuffer(tmp_read_0, combined_input[0], hangover_bits[0], hangover_bits_valid_bits[0]);
        tmp_read_0 = sToHaddocBuffer_chan2.read();
        cur_line_bit_cnt[1] = flattenAxisBuffer(tmp_read_0, combined_input[1], hangover_bits[1], hangover_bits_valid_bits[1]);
        tmp_read_0 = sToHaddocBuffer_chan3.read();
        cur_line_bit_cnt[2] = flattenAxisBuffer(tmp_read_0, combined_input[2], hangover_bits[2], hangover_bits_valid_bits[2]);
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
      }

      only_hangover_processing = false;
      ap_uint<DOSA_HADDOC_INPUT_BITDIWDTH> output_data = 0x0;
      for(int i = 0; i<DOSA_HADDOC_INPUT_CHAN_NUM; i++)
      {
        ap_uint<64> tmp_bit_cnt = cur_line_bit_cnt[i];
        //actually, it is either > or ==..., since frames are transmitted aligned, can't be less than that
        if(tmp_bit_cnt > DOSA_HADDOC_GENERAL_BITWIDTH)
        {
          tmp_bit_cnt = DOSA_HADDOC_GENERAL_BITWIDTH;
          hangover_bits_valid_bits[i] = cur_line_bit_cnt[i] - DOSA_HADDOC_GENERAL_BITWIDTH;
          hangover_bits[i] = (ap_uint<DOSA_WRAPPER_INPUT_IF_BITWIDTH>) (combined_input[i] >> DOSA_HADDOC_GENERAL_BITWIDTH);
          if(hangover_bits_valid_bits[i] >= DOSA_HADDOC_GENERAL_BITWIDTH)
          {
            only_hangover_processing = true;
          }
          //printf("deq: hangover %16.16llx, hangover_cnt: %d, only_hangover_processing: %d\n", (uint64_t) hangover_bits[i], (uint32_t) hangover_bits_valid_bits[i], only_hangover_processing);
        }
        //DynLayerInput requires (chan2,chan1,chan0) vector layout
        output_data |= ((ap_uint<DOSA_HADDOC_INPUT_BITDIWDTH>) ((ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH>) combined_input[i])) << (i*DOSA_HADDOC_GENERAL_BITWIDTH);
      }

      //*po_haddoc_data_valid = 0x1;
      //*po_haddoc_data_vector = output_data;
      //*output_vector = output_data;
      //*output_vector |= ((ap_uint<DOSA_HADDOC_INPUT_BITDIWDTH+1>) 0x1) << DOSA_HADDOC_INPUT_BITDIWDTH;
      po_haddoc_data.write(output_data);
      printf("pToHaddocDeq: write 0x%6.6X\n", (uint32_t) output_data);
      sHaddocUnitProcessing.write(true);
    } //else {
      //*po_haddoc_data_vector = 0x0;
      //*po_haddoc_data_valid = 0x0;
      //*output_vector = 0x0;
    //}
    //*po_haddoc_frame_valid = 0x1;
  }

  //debugging?
  //*debug_out = 0x0;
  *debug = (uint8_t) hangover_bits_valid_bits[0];
  *debug |= ((uint16_t) only_hangover_processing) << 15;

}

void pFromHaddocEnq(
    //ap_uint<1>                                *pi_haddoc_data_valid,
    //ap_uint<1>                                *pi_haddoc_frame_valid,
    //ap_uint<DOSA_HADDOC_OUTPUT_BITDIWDTH>     *pi_haddoc_data_vector,
    stream<ap_uint<DOSA_HADDOC_OUTPUT_BITDIWDTH> > &pi_haddoc_data,
    stream<bool>                              &sHaddocUnitProcessing,
    stream<ap_uint<DOSA_HADDOC_OUTPUT_BITDIWDTH> >  &sFromHaddocBuffer
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static twoStatesFSM enqueueFSM = RESET;
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
    if ( !sHaddocUnitProcessing.empty() && !sFromHaddocBuffer.full()
        && !pi_haddoc_data.empty()
        )
    {
      //ignore pi_haddoc_frame_valid?
      //if( *pi_haddoc_data_valid == 0b1 )
      //{
        //input_data = *pi_haddoc_data_vector;
        input_data = pi_haddoc_data.read();
        //read only if able to process
        bool ignore_me = sHaddocUnitProcessing.read();
        printf("pFromHaddocEnq: read 0x%6.6X\n", (uint32_t) input_data);
        sFromHaddocBuffer.write(input_data);
      //}
    }
  }

}


void pFromHaddocFlatten(
#ifdef WRAPPER_TEST
  stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> >    &sFromHaddocBuffer_chan1,
  stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> >    &sFromHaddocBuffer_chan2,
  stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> >    &sFromHaddocBuffer_chan3,
  stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> >    &sFromHaddocBuffer_chan4,
#else
    //DOSA_ADD_from_haddoc_stream_param_decl
#endif
    stream<ap_uint<DOSA_HADDOC_OUTPUT_BITDIWDTH> >  &sFromHaddocBuffer,
    //stream<bool>                              &sFromHaddocIfLineComplete,
    uint16_t *debug
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static twoStatesFSM flattenFSM = RESET;
#pragma HLS reset variable=flattenFSM
//  static ap_uint<64> current_array_write_pnt = 0x0;
//#pragma HLS reset variable=current_array_write_pnt
//  static ap_uint<2> current_array_slot_pnt = 0x0;
//#pragma HLS reset variable=current_array_slot_pnt
  //-- LOCAL VARIABLES ------------------------------------------------------
  ap_uint<DOSA_HADDOC_OUTPUT_BITDIWDTH> input_data = 0x0;
  bool not_empty = false;

  if(flattenFSM == RESET)
  {
    //current_array_write_pnt = 0x0;
    if(!sFromHaddocBuffer.empty())
    {
      sFromHaddocBuffer.read();
      not_empty = true;
    }
    if(!not_empty)
    {
      flattenFSM = FORWARD;
    }
  } else { //FORWARD
      if ( !sFromHaddocBuffer.empty() //&& !sFromHaddocIfLineComplete.full()
#ifdef WRAPPER_TEST
          && !sFromHaddocBuffer_chan1.full() && !sFromHaddocBuffer_chan2.full() && !sFromHaddocBuffer_chan3.full() && !sFromHaddocBuffer_chan4.full()
#else
          //DOSA_ADD_from_haddoc_stream_full_check
#endif
         )
      {
        input_data = sFromHaddocBuffer.read();
#ifdef WRAPPER_TEST
        sFromHaddocBuffer_chan1.write((ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH>) (input_data >> 0 * DOSA_HADDOC_GENERAL_BITWIDTH));
        sFromHaddocBuffer_chan2.write((ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH>) (input_data >> 1 * DOSA_HADDOC_GENERAL_BITWIDTH));
        sFromHaddocBuffer_chan3.write((ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH>) (input_data >> 2 * DOSA_HADDOC_GENERAL_BITWIDTH));
        sFromHaddocBuffer_chan4.write((ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH>) (input_data >> 3 * DOSA_HADDOC_GENERAL_BITWIDTH));
#else
        //DOSA_ADD_from_haddoc_stream_write
#endif
        //printf("pFromHaddocFlatten: sorted incoming %2.2x to position %d in slot %d\n", (uint32_t) input_data, (uint32_t) current_array_write_pnt, (uint8_t) current_array_slot_pnt);
        //current_array_write_pnt++;
        //if( current_array_write_pnt >= WRAPPER_OUTPUT_IF_HADDOC_WORDS_CNT_CEIL)
        //{
        //  sFromHaddocIfLineComplete.write(true);
        //  current_array_write_pnt = 0;
        //}
      }
  }

  //*debug = ((uint16_t) current_array_write_pnt);
  *debug = ((uint16_t) input_data);

}

void genericWiden(
  stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> >    &small_input,
  stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >    &wide_output,
  uint32_t &current_frame_bit_cnt,
  uint32_t &current_line_read_pnt,
  ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> &hangover_store,
  ap_uint<32> &hangover_store_valid_bits
    )
{
#pragma HLS INLINE
#pragma HLS latency max=1
  ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+DOSA_HADDOC_GENERAL_BITWIDTH> combined_output = 0x0;

  //if(!small_input.empty() && !wide_output.full())
  //{
    combined_output = hangover_store;
    ap_uint<(DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+7)/8> tkeep = 0x0;

    ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> nv = small_input.read();
    combined_output |= ((ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH+DOSA_HADDOC_GENERAL_BITWIDTH>) nv) << (hangover_store_valid_bits);
    current_line_read_pnt++;

    //current_frame_bit_cnt += DOSA_HADDOC_GENERAL_BITWIDTH - hangover_store_valid_bits;
    current_frame_bit_cnt += DOSA_HADDOC_GENERAL_BITWIDTH;

    if(current_line_read_pnt >= WRAPPER_OUTPUT_IF_HADDOC_WORDS_CNT_CEIL || current_frame_bit_cnt >= HADDOC_OUTPUT_FRAME_BIT_CNT)
    {//write to stream
      current_line_read_pnt = 0x0;
      tkeep = bitCntToTKeep(ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH/8> (DOSA_HADDOC_GENERAL_BITWIDTH + hangover_store_valid_bits));
      ap_uint<1> tlast = 0;
      if(current_frame_bit_cnt >= HADDOC_OUTPUT_FRAME_BIT_CNT)
      {
        //TODO: what if there is hangover data left?
        current_frame_bit_cnt = 0x0;
        tlast = 0b1;
      }
      hangover_store = (ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH>) (combined_output >> DOSA_WRAPPER_OUTPUT_IF_BITWIDTH);
      if((hangover_store_valid_bits+DOSA_HADDOC_GENERAL_BITWIDTH) > DOSA_WRAPPER_OUTPUT_IF_BITWIDTH)
      {
        hangover_store_valid_bits -= (DOSA_WRAPPER_OUTPUT_IF_BITWIDTH-DOSA_HADDOC_GENERAL_BITWIDTH);
      } else {
        hangover_store_valid_bits = 0x0;
      }
      Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> tmp_write_0 = Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH>((ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH>) combined_output, tkeep, tlast);
      wide_output.write(tmp_write_0);
      printf("genericWiden: write Axis tdata: %16.16llx, tkeep: %2.2x, tlast: %x;\n", (uint64_t) tmp_write_0.getTData(), (uint8_t) tmp_write_0.getTKeep(), (uint8_t) tmp_write_0.getTLast());
    } else {
      //wait
      hangover_store = combined_output;
      hangover_store_valid_bits += DOSA_HADDOC_GENERAL_BITWIDTH;
    }
    printf("genericWiden: combined %16.16llx, hangover_bits_valid_bits: %d, current_frame_bit_cnt: %d\n", (uint64_t) combined_output, (uint64_t) hangover_store_valid_bits, (uint32_t) current_frame_bit_cnt);
  //}
}



void pFromHaddocWiden(
#ifdef WRAPPER_TEST
    stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> >    &sFromHaddocBuffer_chan1,
    stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> >    &sFromHaddocBuffer_chan2,
    stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> >    &sFromHaddocBuffer_chan3,
    stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> >    &sFromHaddocBuffer_chan4,
#else
    //DOSA_ADD_from_haddoc_stream_param_decl
#endif
#ifdef WRAPPER_TEST
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >    &sOutBuffer_chan1,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >    &sOutBuffer_chan2,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >    &sOutBuffer_chan3,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >    &sOutBuffer_chan4,
#else
    //DOSA_ADD_output_stream_param_decl
#endif
    uint16_t param_void
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static twoStatesFSM widenFSM = RESET;
#pragma HLS reset variable=widenFSM
  static uint32_t current_frame_bit_cnt[DOSA_HADDOC_OUTPUT_CHAN_NUM];
#pragma HLS ARRAY_PARTITION variable=current_frame_bit_cnt complete
#pragma HLS reset variable=current_frame_bit_cnt
  static uint32_t current_line_read_pnt[DOSA_HADDOC_OUTPUT_CHAN_NUM];
#pragma HLS ARRAY_PARTITION variable=current_line_read_pnt complete
#pragma HLS reset variable=current_line_read_pnt
  static ap_uint<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> hangover_store[DOSA_HADDOC_OUTPUT_CHAN_NUM];
#pragma HLS ARRAY_PARTITION variable=hangover_store
#pragma HLS reset variable=hangover_store
  static ap_uint<32> hangover_store_valid_bits[DOSA_HADDOC_OUTPUT_CHAN_NUM];
#pragma HLS ARRAY_PARTITION variable=hangover_store_valid_bits
#pragma HLS reset variable=hangover_store_valid_bits
  //-- LOCAL VARIABLES ------------------------------------------------------
  bool not_empty = false;

  switch(widenFSM)
  {
    default:
    case RESET:
//    for(int i = 0; i < DOSA_HADDOC_OUTPUT_CHAN_NUM; i++)
//    {
//      current_frame_bit_cnt[i]= 0x0;
//      current_line_read_pnt[i] = 0x0;
//      hangover_store[i] = 0x0;
//      hangover_store_valid_bits[i] = 0x0;
//    }
#ifdef WRAPPER_TEST
    if(!sFromHaddocBuffer_chan1.empty())
    {
      sFromHaddocBuffer_chan1.read();
      not_empty = true;
    }
    if(!sFromHaddocBuffer_chan2.empty())
    {
      sFromHaddocBuffer_chan2.read();
        not_empty = true;
    }
    if(!sFromHaddocBuffer_chan3.empty())
    {
      sFromHaddocBuffer_chan3.read();
      not_empty = true;
    }
    if(!sFromHaddocBuffer_chan4.empty())
    {
      sFromHaddocBuffer_chan4.read();
      not_empty = true;
    }
#else
    //DOSA_ADD_from_haddoc_stream_drain
#endif
    if(!not_empty)
    {
      widenFSM = FORWARD;
    }
    break;
    case FORWARD:
#ifdef WRAPPER_TEST
    if(!sFromHaddocBuffer_chan1.empty() && !sOutBuffer_chan1.full())
    {
      genericWiden(sFromHaddocBuffer_chan1, sOutBuffer_chan1, current_frame_bit_cnt[0], current_line_read_pnt[0], hangover_store[0], hangover_store_valid_bits[0]);
    }
    if(!sFromHaddocBuffer_chan2.empty() && !sOutBuffer_chan2.full())
    {
      genericWiden(sFromHaddocBuffer_chan2, sOutBuffer_chan2, current_frame_bit_cnt[1], current_line_read_pnt[1], hangover_store[1], hangover_store_valid_bits[1]);
    }
    if(!sFromHaddocBuffer_chan3.empty() && !sOutBuffer_chan3.full())
    {
      genericWiden(sFromHaddocBuffer_chan3, sOutBuffer_chan3, current_frame_bit_cnt[2], current_line_read_pnt[2], hangover_store[2], hangover_store_valid_bits[2]);
    }
    if(!sFromHaddocBuffer_chan4.empty() && !sOutBuffer_chan4.full())
    {
      genericWiden(sFromHaddocBuffer_chan4, sOutBuffer_chan4, current_frame_bit_cnt[3], current_line_read_pnt[3], hangover_store[3], hangover_store_valid_bits[3]);
    }
#else
    //DOSA_ADD_widen
#endif
    break;
  }

}


void pFromHaddocDeq(
#ifdef WRAPPER_TEST
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >    &sOutBuffer_chan1,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >    &sOutBuffer_chan2,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >    &sOutBuffer_chan3,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >    &sOutBuffer_chan4,
#else
    //DOSA_ADD_output_stream_param_decl
#endif
    //stream<bool>                              &sFromHaddocIfLineComplete,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >  &soData,
    uint16_t *debug
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static FromHaddocDeqStates dequeueFSM = RESET1;
#pragma HLS reset variable=dequeueFSM
  static ap_uint<64> current_frame_bit_cnt = 0x0;
#pragma HLS reset variable=current_frame_bit_cnt
  //  static ap_uint<64> current_batch_bit_cnt = 0x0;
  //#pragma HLS reset variable=current_batch_bit_cnt
  //-- LOCAL VARIABLES ------------------------------------------------------
  bool not_empty = false;
  Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> tmp_read_0;

  switch (dequeueFSM)
  {
    default:
    case RESET1:
      current_frame_bit_cnt = 0x0;
      //current_batch_bit_cnt = 0x0;
#ifdef WRAPPER_TEST
      if(!sOutBuffer_chan1.empty())
      {
        sOutBuffer_chan1.read();
        not_empty = true;
      }
      if(!sOutBuffer_chan2.empty())
      {
        sOutBuffer_chan2.read();
        not_empty = true;
      }
      if(!sOutBuffer_chan3.empty())
      {
        sOutBuffer_chan3.read();
        not_empty = true;
      }
      if(!sOutBuffer_chan4.empty())
      {
        sOutBuffer_chan4.read();
        not_empty = true;
      }
#else
      //DOSA_ADD_out_stream_drain
#endif
      if(!not_empty)
      {
        dequeueFSM = READ_BUF_0;
      }
      break;

      //case WAIT_LINE:
      //  if( !sFromHaddocIfLineComplete.empty() )
      //  {
      //    bool ignore_me = sFromHaddocFrameComplete.read();
      //    current_frame_bit_cnt = 0x0;
      //    current_array_read_pnt = 0x0;
      //    current_batch_bit_cnt = 0x0;
      //    current_array_slot_pnt = 0x0;
      //    hangover_store = 0x0;
      //    hangover_store_valid_bits = 0x0;
      //    dequeueFSM = READ_FRAME;
      //  }
      //  break;
#ifdef WRAPPER_TEST
    case READ_BUF_0:
      if(!soData.full() && !sOutBuffer_chan1.empty())
      {
        tmp_read_0 = sOutBuffer_chan1.read();
        uint32_t bit_read = extractByteCnt(tmp_read_0) * 8;
        current_frame_bit_cnt += bit_read;
        //current_batch_bit_cnt += bit_read;
        if(current_frame_bit_cnt >= HADDOC_OUTPUT_FRAME_BIT_CNT || tmp_read_0.getTLast() == 1)
        {
          current_frame_bit_cnt = 0x0;
          dequeueFSM = READ_BUF_1;
          //if(current_batch_bit_cnt >= HADDOC_OUTPUT_FRAME_BIT_CNT*DOSA_HADDOC_OUTPUT_CHAN_NUM)
          //{
          //  current_batch_bit_cnt = 0x0;
          //  //in all cases
          //  tlast = 0b1;
          //}
          //check for tlast after each frame
          if(!DOSA_HADDOC_OUTPUT_BATCH_FLATTEN)
          {
            tmp_read_0.setTLast(0b1);
          } else {
            tmp_read_0.setTLast(0b0);
          }
        }
        soData.write(tmp_read_0);
      }
      break;

    case READ_BUF_1:
      if(!soData.full() && !sOutBuffer_chan2.empty())
      {
        tmp_read_0 = sOutBuffer_chan2.read();
        uint32_t bit_read = extractByteCnt(tmp_read_0) * 8;
        current_frame_bit_cnt += bit_read;
        if(current_frame_bit_cnt >= HADDOC_OUTPUT_FRAME_BIT_CNT || tmp_read_0.getTLast() == 1)
        {
          current_frame_bit_cnt = 0x0;
          dequeueFSM = READ_BUF_2;
          //check for tlast after each frame
          if(!DOSA_HADDOC_OUTPUT_BATCH_FLATTEN)
          {
            tmp_read_0.setTLast(0b1);
          } else {
            tmp_read_0.setTLast(0b0);
          }
        }
        soData.write(tmp_read_0);
      }
      break;

    case READ_BUF_2:
      if(!soData.full() && !sOutBuffer_chan3.empty())
      {
        tmp_read_0 = sOutBuffer_chan3.read();
        uint32_t bit_read = extractByteCnt(tmp_read_0) * 8;
        current_frame_bit_cnt += bit_read;
        if(current_frame_bit_cnt >= HADDOC_OUTPUT_FRAME_BIT_CNT || tmp_read_0.getTLast() == 1)
        {
          current_frame_bit_cnt = 0x0;
          dequeueFSM = READ_BUF_3;
          //check for tlast after each frame
          if(!DOSA_HADDOC_OUTPUT_BATCH_FLATTEN)
          {
            tmp_read_0.setTLast(0b1);
          } else {
            tmp_read_0.setTLast(0b0);
          }
        }
        soData.write(tmp_read_0);
      }
      break;

    case READ_BUF_3:
      if(!soData.full() && !sOutBuffer_chan4.empty())
      {
        tmp_read_0 = sOutBuffer_chan4.read();
        uint32_t bit_read = extractByteCnt(tmp_read_0) * 8;
        current_frame_bit_cnt += bit_read;
        if(current_frame_bit_cnt >= HADDOC_OUTPUT_FRAME_BIT_CNT || tmp_read_0.getTLast() == 1)
        {
          current_frame_bit_cnt = 0x0;
          dequeueFSM = READ_BUF_0; //back to start
          //check for tlast after each frame
          //if(!DOSA_HADDOC_OUTPUT_BATCH_FLATTEN)
          //{
          // tmp_read_0.setTLast(0b1);
          //} else {
          // tmp_read_0.setTLast(0b0);
          //}
          //in all cases
          tmp_read_0.setTLast(0b1);
        }
        soData.write(tmp_read_0);
      }
      break;
#else
      //DOSA_ADD_from_haddoc_deq_buf_read
#endif

  }

  *debug = current_frame_bit_cnt;

}



//DOSA_ADD_ip_name_BELOW
void haddoc_wrapper_test(
    // ----- Wrapper Interface -----
    stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> >   &siData,
    stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> >  &soData,
    // ----- Haddoc Interface -----
    //ap_uint<1>                                *po_haddoc_data_valid,
    //ap_uint<DOSA_HADDOC_INPUT_BITDIWDTH>      *po_haddoc_data_vector,
    ap_uint<1>                                *po_haddoc_frame_valid,
    stream<ap_uint<DOSA_HADDOC_INPUT_BITDIWDTH> > &po_haddoc_data,
    //ap_uint<1>                                *pi_haddoc_data_valid,
    //ap_uint<DOSA_HADDOC_OUTPUT_BITDIWDTH>     *pi_haddoc_data_vector,
    ap_uint<1>                                *pi_haddoc_frame_valid,
    stream<ap_uint<DOSA_HADDOC_OUTPUT_BITDIWDTH> > &pi_haddoc_data,
    // ----- DEBUG IO ------
    ap_uint<64> *debug_out
    )
{
  //-- DIRECTIVES FOR THE BLOCK ---------------------------------------------
#pragma HLS INTERFACE ap_ctrl_none port=return

#pragma HLS INTERFACE ap_fifo port=siData
#pragma HLS INTERFACE ap_fifo port=soData

//#pragma HLS INTERFACE ap_ovld register port=po_haddoc_data_valid
//#pragma HLS INTERFACE ap_ovld register port=po_haddoc_data_vector
#pragma HLS INTERFACE ap_ovld register port=po_haddoc_frame_valid
//#pragma HLS INTERFACE ap_vld register port=pi_haddoc_data_valid
//#pragma HLS INTERFACE ap_vld register port=pi_haddoc_data_vector
#pragma HLS INTERFACE ap_vld register port=pi_haddoc_frame_valid
#pragma HLS INTERFACE ap_ovld register port=debug_out
#pragma HLS INTERFACE axis register port=po_haddoc_data
#pragma HLS INTERFACE axis register port=pi_haddoc_data


  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS DATAFLOW

#ifndef __SYNTHESIS__
  assert(DOSA_WRAPPER_INPUT_IF_BITWIDTH >= DOSA_HADDOC_GENERAL_BITWIDTH); //currently, the assumption is that one "pixel" is smaller than the interface bitwidth
  assert(HADDOC_INPUT_FRAME_BIT_CNT % 8 == 0); //currently, only byte-aligned FRAMES are supported
  //printf("cnn_input_frame_size: %d\n", cnn_input_frame_size);
  //printf("cnn_output_frame_size: %d\n", cnn_output_frame_size);
#endif

  //-- STATIC VARIABLES (with RESET) ------------------------------------------

  //-- STATIC DATAFLOW VARIABLES ------------------------------------------
#ifdef WRAPPER_TEST
  static stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> > sToHaddocBuffer_chan1 ("sToHaddocBuffer_chan1");
#pragma HLS STREAM variable=sToHaddocBuffer_chan1   depth=cnn_input_frame_size
  static stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> > sToHaddocBuffer_chan2 ("sToHaddocBuffer_chan2");
#pragma HLS STREAM variable=sToHaddocBuffer_chan2   depth=cnn_input_frame_size
  static stream<Axis<DOSA_WRAPPER_INPUT_IF_BITWIDTH> > sToHaddocBuffer_chan3 ("sToHaddocBuffer_chan3");
#pragma HLS STREAM variable=sToHaddocBuffer_chan3   depth=cnn_input_frame_size
  //static ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> g_chan1_buffer_0[CNN_OUTPUT_FRAME_SIZE];
  //static ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> g_chan1_buffer_1[CNN_OUTPUT_FRAME_SIZE];
  //static ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> g_chan2_buffer_0[CNN_OUTPUT_FRAME_SIZE];
  //static ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> g_chan2_buffer_1[CNN_OUTPUT_FRAME_SIZE];
  //static ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> g_chan3_buffer_0[CNN_OUTPUT_FRAME_SIZE];
  //static ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> g_chan3_buffer_1[CNN_OUTPUT_FRAME_SIZE];
  //static ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> g_chan4_buffer_0[CNN_OUTPUT_FRAME_SIZE];
  //static ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> g_chan4_buffer_1[CNN_OUTPUT_FRAME_SIZE];
  ////potential VIVADO HLS bug...with partition complete, the wrong error of two write access is issued
  ////#pragma HLS ARRAY_PARTITION variable=g_chan1_buffer_0 complete
  ////#pragma HLS ARRAY_PARTITION variable=g_chan1_buffer_1 complete
  ////#pragma HLS ARRAY_PARTITION variable=g_chan2_buffer_0 complete
  ////#pragma HLS ARRAY_PARTITION variable=g_chan2_buffer_1 complete
  ////#pragma HLS ARRAY_PARTITION variable=g_chan3_buffer_0 complete
  ////#pragma HLS ARRAY_PARTITION variable=g_chan3_buffer_1 complete
  ////#pragma HLS ARRAY_PARTITION variable=g_chan4_buffer_0 complete
  ////#pragma HLS ARRAY_PARTITION variable=g_chan4_buffer_1 complete
  ////ERROR: [XFORM 203-711] Internal global variable 'g_chan1_buffer_0.V.0'  failed dataflow checking: it can only be written in one process function.
  ////WARNING: [XFORM 203-713] Internal global variable 'g_chan1_buffer_0.V.0' has write operations in process function 'pFromHaddocFlatten'.
  ////WARNING: [XFORM 203-713] Internal global variable 'g_chan1_buffer_0.V.0' has read and write operations in process function 'pFromHaddocDeq'.
  ////using cyclic instead
  //#pragma HLS ARRAY_PARTITION variable=g_chan1_buffer_0 cyclic factor=2*wrapper_output_if_haddoc_words_cnt_ceil
  //#pragma HLS ARRAY_PARTITION variable=g_chan1_buffer_1 cyclic factor=2*wrapper_output_if_haddoc_words_cnt_ceil
  //#pragma HLS ARRAY_PARTITION variable=g_chan2_buffer_0 cyclic factor=2*wrapper_output_if_haddoc_words_cnt_ceil
  //#pragma HLS ARRAY_PARTITION variable=g_chan2_buffer_1 cyclic factor=2*wrapper_output_if_haddoc_words_cnt_ceil
  //#pragma HLS ARRAY_PARTITION variable=g_chan3_buffer_0 cyclic factor=2*wrapper_output_if_haddoc_words_cnt_ceil
  //#pragma HLS ARRAY_PARTITION variable=g_chan3_buffer_1 cyclic factor=2*wrapper_output_if_haddoc_words_cnt_ceil
  //#pragma HLS ARRAY_PARTITION variable=g_chan4_buffer_0 cyclic factor=2*wrapper_output_if_haddoc_words_cnt_ceil
  //#pragma HLS ARRAY_PARTITION variable=g_chan4_buffer_1 cyclic factor=2*wrapper_output_if_haddoc_words_cnt_ceil
  static stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> > sFromHaddocBuffer_chan1 ("sFromHaddocBuffer_chan1");
#pragma HLS STREAM variable=sFromHaddocBuffer_chan1 depth=cnn_output_frame_size
  static stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> > sFromHaddocBuffer_chan2 ("sFromHaddocBuffer_chan2");
#pragma HLS STREAM variable=sFromHaddocBuffer_chan2 depth=cnn_output_frame_size
  static stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> > sFromHaddocBuffer_chan3 ("sFromHaddocBuffer_chan3");
#pragma HLS STREAM variable=sFromHaddocBuffer_chan3 depth=cnn_output_frame_size
  static stream<ap_uint<DOSA_HADDOC_GENERAL_BITWIDTH> > sFromHaddocBuffer_chan4 ("sFromHaddocBuffer_chan4");
#pragma HLS STREAM variable=sFromHaddocBuffer_chan4 depth=cnn_output_frame_size
  static stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> > sOutBuffer_chan1("sOutBuffer_chan1");
#pragma HLS STREAM variable=sOutBuffer_chan1 depth=cnn_output_frame_size
  static stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> > sOutBuffer_chan2("sOutBuffer_chan2");
#pragma HLS STREAM variable=sOutBuffer_chan2 depth=cnn_output_frame_size
  static stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> > sOutBuffer_chan3("sOutBuffer_chan3");
#pragma HLS STREAM variable=sOutBuffer_chan3 depth=cnn_output_frame_size
  static stream<Axis<DOSA_WRAPPER_OUTPUT_IF_BITWIDTH> > sOutBuffer_chan4("sOutBuffer_chan4");
#pragma HLS STREAM variable=sOutBuffer_chan4 depth=cnn_output_frame_size
#else
  //DOSA_ADD_haddoc_buffer_instantiation
#endif
  static stream<bool> sHaddocUnitProcessing ("sHaddocUnitProcessing");
  const int haddoc_processing_fifo_depth = HADDOC_AVG_LAYER_LATENCY * DOSA_HADDOC_LAYER_CNT;
  #pragma HLS STREAM variable=sHaddocUnitProcessing depth=2*haddoc_processing_fifo_depth
  static stream<ap_uint<DOSA_HADDOC_OUTPUT_BITDIWDTH> > sFromHaddocBuffer ("sFromHaddocBuffer");
  #pragma HLS STREAM variable=sFromHaddocBuffer depth=3*haddoc_processing_fifo_depth  //so that we can receive, what we send out...
  //static stream<bool> sFromHaddocFrameComplete ("sFromHaddocFrameComplete");
  //#pragma HLS STREAM variable=sFromHaddocFrameComplete depth=2  //2! since arrays are double-buffers
  //static stream<bool> sFromHaddocIfLineComplete ("sFromHaddocIfLineComplete");
  //#pragma HLS STREAM variable=sFromHaddocIfLineComplete depth=2*wrapper_output_if_haddoc_words_cnt_ceil  //2 since arrays are double-buffers

  //-- LOCAL VARIABLES ------------------------------------------------------------
  uint16_t debug0 = 0;
  uint16_t debug1 = 0;
  uint16_t debug2 = 0;
  uint16_t debug3 = 0;
  //ap_uint<DOSA_HADDOC_INPUT_BITDIWDTH+1>      output_vector = 0;
  //-- PROCESS INSTANTIATION ------------------------------------------------------

  // with the current strategy [INPUT] -> [CHANEL, with same bitwidth] -> [BATCH_PIXEL], we can maintain
  // ~ (DOSA_HADDOC_INPUT_CHAN_NUM * DOSA_HADDOC_GENERAL_BITWIDTH)/DOSA_WRAPPER_INPUT_IF_BITWIDTH of the bandwidth
  pToHaddocEnq(
#ifdef WRAPPER_TEST
      sToHaddocBuffer_chan1, sToHaddocBuffer_chan2, sToHaddocBuffer_chan3,
#else
      //DOSA_ADD_toHaddoc_buffer_list
#endif
      siData, &debug0);

  *po_haddoc_frame_valid = 0x1;

  pToHaddocDeq(
#ifdef WRAPPER_TEST
      sToHaddocBuffer_chan1, sToHaddocBuffer_chan2, sToHaddocBuffer_chan3,
#else
      //DOSA_ADD_toHaddoc_buffer_list
#endif
      //po_haddoc_data_valid,
      //po_haddoc_frame_valid,
      //po_haddoc_data_vector,
      //&output_vector,
      po_haddoc_data,
      sHaddocUnitProcessing, &debug1);

  pFromHaddocEnq(
      //pi_haddoc_data_valid, pi_haddoc_frame_valid, pi_haddoc_data_vector,
      pi_haddoc_data,
      sHaddocUnitProcessing, sFromHaddocBuffer);

  pFromHaddocFlatten(
#ifdef WRAPPER_TEST
      sFromHaddocBuffer_chan1, sFromHaddocBuffer_chan2, sFromHaddocBuffer_chan3, sFromHaddocBuffer_chan4,
#else
      //DOSA_ADD_from_haddoc_stream_list
#endif
      sFromHaddocBuffer, &debug2);


  pFromHaddocDeq(
#ifdef WRAPPER_TEST
      sOutBuffer_chan1, sOutBuffer_chan2, sOutBuffer_chan3, sOutBuffer_chan4,
#else
      //DOSA_ADD_from_haddoc_out_list
#endif
      soData, &debug3);

  //currently the slowest --> last
  pFromHaddocWiden(
#ifdef WRAPPER_TEST
      sFromHaddocBuffer_chan1, sFromHaddocBuffer_chan2, sFromHaddocBuffer_chan3, sFromHaddocBuffer_chan4,
      sOutBuffer_chan1, sOutBuffer_chan2, sOutBuffer_chan3, sOutBuffer_chan4,
#else
      //DOSA_ADD_from_haddoc_stream_list
      //DOSA_ADD_from_haddoc_out_list
#endif
      0);

  *debug_out = (uint64_t) debug0;
  *debug_out |= ((uint64_t) debug1) << 16;
  *debug_out |= ((uint64_t) debug2) << 32;
  *debug_out |= ((uint64_t) debug3) << 48;

  //un-bundle output
  //*po_haddoc_data_valid = (ap_uint<1>) (output_vector >> DOSA_HADDOC_INPUT_BITDIWDTH);
  //*po_haddoc_data_vector = (ap_uint<DOSA_HADDOC_INPUT_BITDIWDTH>) output_vector;

}



