#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Feb 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Python module to export dosa_infer C function in Python
#  *
#  *
import os
import ctypes
import numpy as np
import time
import json
import requests
import subprocess

# from cFSPlib import cFSP

__filedir__ = os.path.dirname(os.path.abspath(__file__))
__so_lib_name__ = 'dosa_infer_pass.so'

__openstack_user_template__ = {'credentials': {'username': "your user name", 'password': "your user password"},
                               'project': "default"}
__cf_manager_url__ = "10.12.0.132:8080"
__NON_FPGA_IDENTIFIER__ = "NON_FPGA"


def errorReqExit(msg, code):
    print("Request "+msg+" failed with HTTP code "+str(code)+".\n")
    exit(1)


def load_user_credentials(json_file):
    __openstack_user__ = 'X'
    __openstack_pw__ = 'Y'
    __openstack_project__ = 'Z'

    try:
        with open(json_file, 'r') as infile:
            data = json.load(infile)
        __openstack_user__ = data['credentials']['username']
        __openstack_pw__ = data['credentials']['password']
        if 'project' in data:
            __openstack_project__ = data['project']
        ret_dict = {'user': __openstack_user__, 'pw': __openstack_pw__, 'proj': __openstack_project__}
        return 0, ret_dict
    except Exception as e:
        print(e)
        print("Writing credentials template to {}\n".format(json_file))

    with open(json_file, 'w') as outfile:
        json.dump(__openstack_user_template__, outfile)
    return -1, {}


def get_cluster_data(cluster_id, user_dict):

    print("Requesting cluster data...")

    r1 = requests.get("http://"+__cf_manager_url__+"/clusters/"+str(cluster_id)+"?username={0}&password={1}"
                      .format(user_dict['user'], user_dict['pw']))

    if r1.status_code != 200:
        # something went horrible wrong
        return errorReqExit("GET clusters", r1.status_code)

    cluster_data = json.loads(r1.text)
    return cluster_data


class DosaRoot:

    def __init__(self, used_bitwidth, signed=True):
        libname = os.path.abspath(__filedir__ + '/' + __so_lib_name__)
        self.c_lib = ctypes.CDLL(libname)
        self.c_lib.infer.restype = ctypes.c_int
        self.nbits = used_bitwidth
        self.n_bytes = int((used_bitwidth + 7) / 8)
        if used_bitwidth == 8:
            if signed:
                self.ndtype = np.int8
                self.ctype = ctypes.c_int8
            else:
                self.ndtype = np.uint8
                self.ctype = ctypes.c_uint8
        elif used_bitwidth == 16:
            if signed:
                self.ndtype = np.int16
                self.ctype = ctypes.c_int16
            else:
                self.ndtype = np.uint16
                self.ctype = ctypes.c_uint16
        else:
            if signed:
                self.ndtype = np.int32
                self.ctype = ctypes.c_int32
            else:
                self.ndtype = np.uint32
                self.ctype = ctypes.c_uint32
        self.user_dict = {}

    # def _prepare_data(self, tmp_array):
    #     bin_str = ''
    #     msg_len = 0
    #     with np.nditer(tmp_array) as it:
    #         for x in it:
    #             br = np.binary_repr(x, width=self.nbits)
    #             bin_str += br
    #             msg_len += 1
    #     ret = bytes(br, 'ascii')
    #     return ret, msg_len

    def init(self, mpi_arg_list):
        argc = len(mpi_arg_list) + 1
        argv = __so_lib_name__ + ' ' + ' '.join(mpi_arg_list)
        argv += ' \0'
        # print(argv)
        arglist = [__so_lib_name__]
        arglist.extend(mpi_arg_list)
        blist = []
        for a in arglist:
            blist.append(bytes(a, 'ascii'))
        cargs = (ctypes.c_char_p * len(blist))(*blist)
        #self.c_lib.init(ctypes.c_int(argc), ctypes.c_char_p(bytes(argv, 'ascii')))
        self.c_lib.init(ctypes.c_int(argc), cargs)

    def init_from_cluster(self, cluster_id, host_address, json_file='./user.json'):
        _, user_dict = load_user_credentials(json_file)
        self.user_dict = user_dict
        cluster = get_cluster_data(cluster_id, user_dict)
        number_of_nodes = len(cluster['nodes'])
        slot_ip_list = [0]*number_of_nodes
        print("Ping all nodes, build ARP table...")
        sw_node_id = 0
        # host_address = 'localhost'
        for node in cluster['nodes']:
            if node['image_id'] == __NON_FPGA_IDENTIFIER__:
                sw_node_id = node['node_id']
                # if host_address != 'localhost':
                #     print("Warning: More than one CPU host in cluster, that's unexpected...")
                # host_address = node['node_ip']
                slot_ip_list[sw_node_id] = __NON_FPGA_IDENTIFIER__
                continue
            subprocess.call(["/usr/bin/ping","-I{}".format(host_address), "-c3", "{0}".format(node['node_ip'])],
                            stdout=subprocess.PIPE, cwd=os.getcwd())
            # print("/usr/bin/ping","-I{}".format(str(host_address)) , "-c3", "{0}".format(node['node_ip']))
            slot_ip_list[node['node_id']] = str(node['node_ip'])
        # init
        # Usage: ./zrlmpi_cpu_app <tcp|udp> <host-address> <cluster-size> <own-rank> <ip-rank-1> <ip-rank-2> <...>
        arg_list = ['udp', str(host_address), str(number_of_nodes), str(sw_node_id)]
        for e in slot_ip_list:
            if e != __NON_FPGA_IDENTIFIER__:
                arg_list.append(e)
        # print(arg_list)
        self.init(arg_list)

    def infer(self, x: np.ndarray, output_shape, debug=False):
        # input_data, input_length = self._prepare_data(x)
        input_data = x.astype(self.ndtype)
        input_length = int(self.n_bytes * input_data.size)
        output_length = int(self.n_bytes)
        for d in output_shape:
            output_length *= d
        # out_length_32bit = np.ceil(output_length * (self.nbits / 32))
        # output_placeholder = bytearray(out_length_32bit)
        output_placeholder = bytearray(output_length)
        output_array = self.ctype * output_length

        infer_start = time.time()
        # int infer(int *input, uint32_t input_length, int *output, uint32_t output_length);
        rc = self.c_lib.infer(input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), ctypes.c_uint32(input_length),
                              output_array.from_buffer(output_placeholder), ctypes.c_uint32(output_length))
        infer_stop = time.time()
        infer_time = infer_stop - infer_start
        if debug:
            print(f'DOSA inference run returned {rc} after {infer_time}s.')

        output_deserialized = np.frombuffer(output_placeholder, dtype=self.ndtype)
        output = np.reshape(output_deserialized, newshape=output_shape)
        return output

    def reset_sw_sate(self):
        self.c_lib.reset_state()
