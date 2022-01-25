/*******************************************************************************
 * Copyright 2016 -- 2022 IBM Corporation
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
//  *     Created: May 2019
//  *     Authors: FAB, WEI, NGL
//  *
//  *     Description:
//  *        The Role for a Triangle Example application (UDP or TCP)
//  *

#include "triangle_app.hpp"


void pPortAndDestionation(
    ap_uint<32>             *pi_rank,
    ap_uint<32>             *pi_size,
    stream<NodeId>          &sDstNode_sig,
    ap_uint<32>                 *po_rx_ports
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
//#pragma HLS pipeline II=1 //not necessary
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static PortFsmType port_fsm = FSM_WRITE_NEW_DATA;
#pragma HLS reset variable=port_fsm


  switch(port_fsm)
  {
    default:
    case FSM_WRITE_NEW_DATA:
        //Triangle app needs to be reset to process new rank
        if(!sDstNode_sig.full())
        {
          NodeId dst_rank = (*pi_rank + 1) % *pi_size;
          printf("rank: %d; size: %d; \n", (int) *pi_rank, (int) *pi_size);
          sDstNode_sig.write(dst_rank);
          port_fsm = FSM_DONE;
        }
        break;
    case FSM_DONE:
        *po_rx_ports = 0x1; //currently work only with default ports...
        break;
  }
}


void pEnq(
    stream<NetworkMetaStream>   &siNrc_meta,
    stream<NetworkWord>         &siNrc_data,
    stream<NetworkMetaStream>   &sRxtoTx_Meta,
    stream<NetworkWord>         &sRxpToTxp_Data
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static PacketFsmType enqueueFSM = WAIT_FOR_META;
#pragma HLS reset variable=enqueueFSM
  //-- LOCAL VARIABLES ------------------------------------------------------
  NetworkWord udpWord = NetworkWord();
  NetworkMetaStream meta_tmp = NetworkMetaStream();

  switch(enqueueFSM)
  {
    default:
    case WAIT_FOR_META:
      if ( !siNrc_meta.empty() && !sRxtoTx_Meta.full() )
      {
        meta_tmp = siNrc_meta.read();
        //meta_tmp.tlast = 1; //just to be sure...
        sRxtoTx_Meta.write(meta_tmp);
        enqueueFSM = PROCESSING_PACKET;
      }
      break;

    case PROCESSING_PACKET:
      if ( !siNrc_data.empty() && !sRxpToTxp_Data.full() )
      {
        udpWord = siNrc_data.read();
        sRxpToTxp_Data.write(udpWord);
        if(udpWord.tlast == 1)
        {
          enqueueFSM = WAIT_FOR_META;
        }
      }
      break;
  }
}


void pDeq(
    stream<NodeId>          &sDstNode_sig,
    stream<NetworkMetaStream>   &sRxtoTx_Meta,
    stream<NetworkWord>         &sRxpToTxp_Data,
    stream<NetworkMetaStream>   &soNrc_meta,
    stream<NetworkWord>         &soNrc_data
    )
{
  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS INLINE off
#pragma HLS pipeline II=1
  //-- STATIC VARIABLES (with RESET) ------------------------------------------
  static PacketFsmType dequeueFSM = WAIT_FOR_META;
#pragma HLS reset variable=dequeueFSM
  //-- STATIC DATAFLOW VARIABLES ------------------------------------------
  //static NetworkMeta meta_out;
  static NodeId dst_rank;
  //-- LOCAL VARIABLES ------------------------------------------------------
  NetworkWord udpWordTx = NetworkWord();

  switch(dequeueFSM)
  {
    default:
    case WAIT_FOR_META:
      if(!sDstNode_sig.empty())
      {
        dst_rank = sDstNode_sig.read();
        dequeueFSM = WAIT_FOR_STREAM;
        //Triangle app needs to be reset to process new rank
      }
      break;
    case WAIT_FOR_STREAM:
      //-- Forward incoming chunk to SHELL
      if ( //!sRxpToTxp_Data.empty() && 
          !sRxtoTx_Meta.empty() 
          //&& !soNrc_data.full() 
          && !soNrc_meta.full() 
        )
      {
        //udpWordTx = sRxpToTxp_Data.read();
        //soNrc_data.write(udpWordTx);

        NetworkMeta meta_in = sRxtoTx_Meta.read().tdata;
        NetworkMeta meta_out = NetworkMeta();
        //meta_out_stream.tlast = 1;
        //meta_out_stream.tkeep = 0xFF; //just to be sure!

        //meta_out.dst_rank = (*pi_rank + 1) % *pi_size;
        meta_out.dst_rank = dst_rank;
        //printf("meat_out.dst_rank: %d\n", (int) meta_out_stream.tdata.dst_rank);
        meta_out.dst_port = DEFAULT_TX_PORT;
        //meta_out.src_rank = (NodeId) *pi_rank;
        meta_out.src_rank = NAL_THIS_FPGA_PSEUDO_NID; //will be ignored, it is always this FPGA...
        meta_out.src_port = DEFAULT_RX_PORT;
        meta_out.len = meta_in.len;

       soNrc_meta.write(NetworkMeta(meta_out));

        //if(udpWordTx.tlast != 1)
        //{
          dequeueFSM = PROCESSING_PACKET;
        //}
        //dequeueFSM = WRITE_META;
      }
      break;
    //case WRITE_META:
    //  if(!soNrc_meta.full())
    //  {
    //   soNrc_meta.write(NetworkMeta(meta_out));
    //   dequeueFSM = PROCESSING_PACKET;
    //  }
    //  break;

    case PROCESSING_PACKET:
      if( !sRxpToTxp_Data.empty() && !soNrc_data.full())
      {
        udpWordTx = sRxpToTxp_Data.read();
        soNrc_data.write(udpWordTx);

        if(udpWordTx.tlast == 1)
        {
          dequeueFSM = WAIT_FOR_STREAM;
        }

      }
      break;
  }

}



/*****************************************************************************
 * @brief   Main process of the UDP/TCP Triangle Application. 
 *          This HLS IP receives a packet and forwards it to the next node
 *          in the cluster. The last forwards it to 0.
 * @ingroup ROLE
 *
 * @return Nothing.
 *****************************************************************************/
void triangle_app(
    ap_uint<32>             *pi_rank,
    ap_uint<32>             *pi_size,
    //------------------------------------------------------
    //-- SHELL / This / UDP/TCP Interfaces
    //------------------------------------------------------
    stream<NetworkWord>         &siNrc_data,
    stream<NetworkWord>         &soNrc_data,
    stream<NetworkMetaStream>   &siNrc_meta,
    stream<NetworkMetaStream>   &soNrc_meta,
    ap_uint<32>                 *po_rx_ports
    )
{

  //-- DIRECTIVES FOR THE BLOCK ---------------------------------------------
#pragma HLS INTERFACE ap_ctrl_none port=return

#pragma HLS INTERFACE axis register both port=siNrc_data
#pragma HLS INTERFACE axis register both port=soNrc_data

#pragma HLS INTERFACE axis register both port=siNrc_meta
#pragma HLS INTERFACE axis register both port=soNrc_meta

#pragma HLS INTERFACE ap_vld register port=po_rx_ports name=poROL_NRC_Rx_ports
#pragma HLS INTERFACE ap_vld register port=pi_rank name=piFMC_ROL_rank
#pragma HLS INTERFACE ap_vld register port=pi_size name=piFMC_ROL_size


  //-- DIRECTIVES FOR THIS PROCESS ------------------------------------------
#pragma HLS DATAFLOW

  //-- STATIC VARIABLES (with RESET) ------------------------------------------

  //-- STATIC DATAFLOW VARIABLES ------------------------------------------
  static stream<NetworkWord>       sBuffer_Data   ("sBuffer_Data");
  static stream<NetworkMetaStream> sBuffer_Meta   ("sBuffer_Meta");
  static stream<NodeId>            sDstNode_sig   ("sDstNode_sig");

#pragma HLS STREAM variable=sBuffer_Data     depth=252
#pragma HLS STREAM variable=sBuffer_Meta     depth=32
#pragma HLS STREAM variable=sDstNode_sig     depth=1


  //-- LOCAL VARIABLES ------------------------------------------------------

  pPortAndDestionation(pi_rank, pi_size, sDstNode_sig, po_rx_ports);

  pEnq(siNrc_meta, siNrc_data, sBuffer_Meta, sBuffer_Data);

  pDeq(sDstNode_sig, sBuffer_Meta, sBuffer_Data, soNrc_meta, soNrc_data);

}

