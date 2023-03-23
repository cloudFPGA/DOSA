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
import random

# from cFSPlib import cFSP

__filedir__ = os.path.dirname(os.path.abspath(__file__))
__so_lib_name__ = 'dosa_infer_pass.so'

__openstack_user_template__ = {'credentials': {'username': "your user name", 'password': "your user password"},
                               'project': "default"}
__cf_manager_url__ = "10.12.0.132:8080"
__NON_FPGA_IDENTIFIER__ = "NON_FPGA"
__rank_calling_FPGA_resets__ = 0


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

    print("Requesting FPGA cluster data...")

    r1 = requests.get("http://"+__cf_manager_url__+"/clusters/"+str(cluster_id)+"?username={0}&password={1}"
                      .format(user_dict['user'], user_dict['pw']))

    if r1.status_code != 200:
        # something went horrible wrong
        return errorReqExit("GET clusters", r1.status_code)

    cluster_data = json.loads(r1.text)
    return cluster_data


def get_action_data(action_name, user_dict):

    print("Requesting action data...")

    r1 = requests.get("http://"+__cf_manager_url__+"/actions/"+str(action_name)+"?username={0}&password={1}"
                      .format(user_dict['user'], user_dict['pw']))

    if r1.status_code != 200:
        # something went horrible wrong
        return errorReqExit("GET action", r1.status_code)

    action_data = json.loads(r1.text)
    return action_data


def restart_app(cluster_id, user_dict):
    print("Reset all ROLEs (FPGAs)...")
    r1 = requests.patch("http://"+__cf_manager_url__+"/clusters/"+str(cluster_id)+"/restart?username={0}&password={1}"
                        .format(user_dict['user'], user_dict['pw']))

    if r1.status_code != 200:
        # something went horrible wrong
        return errorReqExit("PATCH cluster restart", r1.status_code)

    print("...done.")
    return


class DosaRoot:

    def __init__(self, used_bitwidth, signed=True):
        libname = os.path.abspath(__filedir__ + '/' + __so_lib_name__)
        self.c_lib = ctypes.CDLL(libname)
        # self.c_lib.infer.restype = ctypes.c_int
        # self.c_lib.malloc.restype = ctypes.c_void_p
        self.c_lib.infer_batch.restype = ctypes.c_int
        self.c_lib.reset_state.restype = ctypes.c_void_p
        self.c_lib.get_pipeline_store_depth.restype = ctypes.c_uint32
        self.c_lib.get_batch_input_size.restype = ctypes.c_uint32
        self.c_lib.get_batch_output_size.restype = ctypes.c_uint32
        self.c_lib.get_pipeline_full_batch_size.restype = ctypes.c_uint32
        self.c_lib.are_processing_pipelines_filled.restype = ctypes.c_bool
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
        self.cluster = {}
        self.cluster_id = -1
        self.pipeline_store_depth = -1
        self.minimum_input_batch_size = -1
        self.minimum_output_batch_size = -1
        self.pipeline_full_batch_size = -1
        self.number_of_fpga_nodes = 0
        self.number_of_cpu_nodes = 0
        self.own_rank = -1
        self._root_rank = 0

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

    def _get_host_address_and_cluster_list(self):
        cpu_ranks = []
        ip_dict = {}
        possible_cpu_ip_addresses = []
        for node in self.cluster['nodes']:
            if node['image_id'] == __NON_FPGA_IDENTIFIER__:
                cpu_ranks.append(node['node_id'])
                possible_cpu_ip_addresses.append(node['node_ip'])
                self.number_of_cpu_nodes += 1
            else:
                # other_ip_addresses.append(node['node_ip'])
                self.number_of_fpga_nodes += 1
            # in all cases
            ip_dict[node['node_id']] = node['node_ip']
        correct_line = ''
        proc = subprocess.run('/usr/sbin/ip addr', shell=True, capture_output=True, text=True, check=True)
        break_fors = False
        for line in proc.stdout.splitlines():
            for cip in possible_cpu_ip_addresses:
                if cip in line:
                    correct_line = line
                    break_fors = True
                    break
            if break_fors:
                break
        host_address = correct_line.split()[1].split('/')[0]
        me_id = possible_cpu_ip_addresses.index(host_address)
        own_rank = cpu_ranks[me_id]
        # del possible_cpu_ip_addresses[me_id]
        # other_ip_addresses.extend(possible_cpu_ip_addresses)
        other_ip_addresses = []
        for rank in ip_dict:
            if rank == own_rank:
                continue
            other_ip_addresses.append(ip_dict[rank])
        self.own_rank = own_rank
        self._root_rank = min(cpu_ranks)
        return host_address, own_rank, other_ip_addresses

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
        # self.c_lib.init(ctypes.c_int(argc), ctypes.c_char_p(bytes(argv, 'ascii')))
        self.c_lib.init(ctypes.c_int(argc), cargs)
        self.pipeline_store_depth = self.c_lib.get_pipeline_store_depth()
        self.minimum_input_batch_size = self.c_lib.get_batch_input_size()
        self.minimum_output_batch_size = self.c_lib.get_batch_output_size()
        self.pipeline_full_batch_size = self.c_lib.get_pipeline_full_batch_size()
        # print("cluster properties: {} {} {} {}".format(self.pipeline_store_depth, self.minimum_input_batch_size,
        #                                             self.minimum_output_batch_size, self.pipeline_full_batch_size))

    def init_from_action(self, action_name, json_file='./user.json', debug=False):
        _, user_dict = load_user_credentials(json_file)
        self.user_dict = user_dict
        action = get_action_data(action_name, user_dict)
        # print(action)
        if len(action['deployed_cluster_ids']) == 0:
            print("ERROR: Requested action is not defined.\nFAILED DEPENDENCY. STOP.")
            exit(1)
        cluster_id = action['deployed_cluster_ids'][0]
        if len(action['deployed_cluster_ids']) > 1:
            cluster_id = action['deployed_cluster_ids'][random.randint(0, len(action['deployed_cluster_ids']) - 1)]
            print(f"INFO: Multimple clusters for the action {action_name} available, selected {cluster_id} at random.")
        else:
            print(f"INFO: Action {action_name} is deployed with cluster {cluster_id}.")
        return self.init_from_cluster(cluster_id, json_file=json_file, debug=debug)

    def init_from_cluster(self, cluster_id, json_file='./user.json', debug=False):
        _, user_dict = load_user_credentials(json_file)
        self.user_dict = user_dict
        cluster = get_cluster_data(cluster_id, user_dict)
        self.cluster = cluster
        self.cluster_id = cluster_id
        number_of_nodes = len(cluster['nodes'])
        host_address, sw_node_id, other_ip_addresses = self._get_host_address_and_cluster_list()
        print("Initialize ZRLMPI...")
        print(f"Using {host_address} as local IP address.")
        # if debug:
        #     print(f"DEBUG: other ip addresses are: {other_ip_addresses}")
        #     print(f"DEBUG: root rank {self._root_rank}")
        ping_cnt = 2
        if debug:
            print("Ping all nodes, build ARP table...")
            ping_cnt = 3
        # host_address = 'localhost'
        for node in cluster['nodes']:
            # if node['image_id'] == __NON_FPGA_IDENTIFIER__:
            if node['node_ip'] == sw_node_id:
                    continue
            subprocess.call(["/usr/bin/ping","-I{}".format(host_address), "-c{}".format(ping_cnt), "{0}".format(node['node_ip'])],
                            stdout=subprocess.PIPE, cwd=os.getcwd())
        # init
        # Usage: ./zrlmpi_cpu_app <tcp|udp> <host-address> <cluster-size> <own-rank> <ip-rank-1> <ip-rank-2> <...>
        arg_list = ['udp', str(host_address), str(number_of_nodes), str(sw_node_id)]
        for e in other_ip_addresses:
            arg_list.append(e)
        # print(arg_list)
        self.init(arg_list)
        print("...done.")

    # def infer(self, x: np.ndarray, output_shape, debug=False):
    def infer_batch(self, x: np.ndarray, output_shape: tuple, debug=False):
        if len(x.shape) < 2:
            print("[DOSA:infer_batch:ERROR] input array must be an array of arrays (i.e. [[1,2], [3,4]]).")
            return -1
        processing_pipelines_filled = bool(self.c_lib.are_processing_pipelines_filled())
        if debug:
            print("[DOSA:runtime:INFO] processing pipeline of DOSA is filled: {}".format(processing_pipelines_filled))
        za = np.zeros(x[0].shape, self.ndtype)
        batch_size = len(x)
        input_data = x.astype(self.ndtype)
        single_input_length = int(self.n_bytes * input_data[0].size)
        single_output_length = int(self.n_bytes)
        for d in output_shape:
            single_output_length *= d
        if not processing_pipelines_filled:
            # now, adapt to minimum length requirements
            # added_zero_tensors = 0
            # batch_input = input_data
            added_zero_tensors = self.pipeline_store_depth
            batch_input = np.vstack([input_data, [za]*added_zero_tensors])
            if (batch_size + added_zero_tensors - self.minimum_input_batch_size) % self.pipeline_full_batch_size != 0:
                add_zt = self.pipeline_full_batch_size - ((batch_size + added_zero_tensors
                                                           - self.minimum_input_batch_size)
                                                          % self.pipeline_full_batch_size)
                batch_input = np.vstack([batch_input, [za]*add_zt])
                added_zero_tensors += add_zt
            output_batch_num = len(batch_input) - self.pipeline_store_depth
            # if batch_size % self.minimum_input_batch_size != 0:
            #     # added_zero_tensors = self.minimum_input_batch_size - (batch_size % self.minimum_input_batch_size)
            #     batch_input = np.vstack([input_data, [za]*added_zero_tensors])
            # if added_zero_tensors < self.pipeline_store_depth:
            #     # add another batch to get results back
            #     added_zero_tensors += self.minimum_input_batch_size
            #     batch_input = np.vstack([batch_input, [za]*self.minimum_input_batch_size])
            # batch_rounds = (batch_size + added_zero_tensors) / self.minimum_input_batch_size
        else:
            # if batch_size < self.pipeline_full_batch_size:
            if batch_size % self.pipeline_full_batch_size != 0:
                added_zero_tensors = self.pipeline_full_batch_size - (batch_size % self.pipeline_full_batch_size)
                batch_input = np.vstack([input_data, [za]*added_zero_tensors])
            else:
                added_zero_tensors = 0
                batch_input = input_data
            output_batch_num = int(batch_size + added_zero_tensors)

        input_num = len(batch_input)
        # add one "line" to avoid SEGFAULT
        np.vstack([batch_input, [za]])
        # output_batch_num = int(self.minimum_output_batch_size * batch_rounds)
        expected_num_output = batch_size
        single_output_shape = output_shape
        if len(output_shape) > 1:
            # expected_num_output = output_shape[0]
            assert output_shape[0] == batch_size
            single_output_shape = output_shape[1:]
        output_overhead_length = output_batch_num - expected_num_output
        total_output_shape = [output_batch_num]
        output_total_length = output_batch_num * self.n_bytes
        for d in single_output_shape:
            total_output_shape.append(d)
            output_total_length *= d

        # output_placeholder = bytearray(single_output_length)
        # output_array = self.ctype * single_output_length
        c_input = batch_input.ctypes.data_as(ctypes.POINTER(ctypes.c_char))
        c_input_num = ctypes.c_uint32(input_num)
        # +4 to avoid SEGFAULT
        output_placeholder = bytearray(output_total_length + 4*self.n_bytes)
        output_array = self.ctype * int((output_total_length/self.n_bytes) + 4)
        c_output = output_array.from_buffer(output_placeholder)
        # output_placeholder = self.c_lib.malloc((output_total_length + 4) * ctypes.sizeof(self.ctype))
        c_output = ctypes.cast(c_output, ctypes.POINTER(ctypes.c_char))
        c_output_num = ctypes.c_uint32(output_batch_num)
        if debug:
            print("[DOSA:DEBUG] c_input size {}; c_input_num: {}; c_output size {}; c_output_num {}; n_bytes {};".format(len(batch_input)*single_input_length, input_num, output_total_length + 4*self.n_bytes, output_batch_num, self.n_bytes))

        infer_start = time.time()

        # int infer(int *input, uint32_t input_length, int *output, uint32_t output_length);
        # rc = self.c_lib.infer(input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        #                       ctypes.c_uint32(single_input_length),
        #                       output_array.from_buffer(output_placeholder), ctypes.c_uint32(single_output_length))

        # extern "C" int infer_batch(int *input, uint32_t input_num, int *output, uint32_t output_num);
        rc = self.c_lib.infer_batch(c_input, c_input_num, c_output, c_output_num)

        infer_stop = time.time()
        infer_time = infer_stop - infer_start
        if debug:
            print(f'DOSA inference run returned {rc} after {infer_time}s.')

        output_deserialized = np.frombuffer(output_placeholder, dtype=self.ndtype)
        try:
            output = np.reshape(output_deserialized[:-4], newshape=total_output_shape)  # -4 is sufficient, due to data-type!
        except Exception as e:
            print('[DOSA:runtime:ERROR] {}.\n'.format(e))
            print(output_deserialized)
            output = output_deserialized[:-4]

        expected_output = output[0:expected_num_output]
        return expected_output

    def reset(self, force=False):
        self.c_lib.reset_state()
        # if force or self.own_rank == self._root_rank:
        #     restart_app(self.cluster_id, self.user_dict)

    def cleanup(self):
        self.c_lib.cleanup()


