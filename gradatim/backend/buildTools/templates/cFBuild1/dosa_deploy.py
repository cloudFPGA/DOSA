#  /*******************************************************************************
#   * Copyright 2019 -- 2024 IBM Corporation
#   *
#   * Licensed under the Apache License, Version 2.0 (the "License");
#   * you may not use this file except in compliance with the License.
#   * You may obtain a copy of the License at
#   *
#   *     http://www.apache.org/licenses/LICENSE-2.0
#   *
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS,
#   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   * See the License for the specific language governing permissions and
#   * limitations under the License.
#  *******************************************************************************/
#

#  *
#  *                       cloudFPGA
#  *    =============================================
#  *     Created: Mar 2023
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Python module to automatically upload the bitfiles
#  *         and creating a cluster
#  *
#  *
import sys
import os
import json
import requests
import glob
import urllib.parse
from datetime import datetime

# from cFSPlib import cFSP
from cFSPlib.cfsp_image import ImagesApi, ApiException, ApiClient, Configuration

__filedir__ = os.path.dirname(os.path.abspath(__file__))
__cluster_filename__ = 'cluster.json'

__openstack_user_template__ = {'credentials': {'username': "your user name", 'password': "your user password"},
                               'project': "default"}
__cf_manager_url__ = "10.12.0.132:8080"
__NON_FPGA_IDENTIFIER__ = "NON_FPGA"

__cf_device_name__ = 'cF_FMKU60_Themisto-Role_1'
__cpu_device_name__ = 'CPU_dummy_x86-1'


def errorReqExit(msg, code):
    print("Request " + msg + " failed with HTTP code " + str(code) + ".\n")
    exit(1)


def load_user_credentials(json_file):
    __openstack_user__ = 'X'
    __openstack_pw__ = 'Y'
    __openstack_project__ = 'Z'

    try:
        with open(json_file, 'r') as infile:
            data = json.load(infile)
        __openstack_user__ = data['credentials']['username']
        __openstack_pw__ = urllib.parse.unquote(data['credentials']['password'])
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


def upload_image(dcp_folder, user_dict, node_name, app_name, api_instance):
    lt = glob.glob(dcp_folder + '/4_*{}.bit'.format(node_name))
    if len(lt) == 0:
        # monolithic case?
        lt = glob.glob(dcp_folder + '/4_*{}_monolithic.bit'.format(node_name))
    if len(lt) != 1:
        print(f"ERROR: unexpected or missing files in folder {dcp_folder}: {lt}")
        exit(1)
    bit_file_name = os.path.basename(lt[0])
    assert 'partial' not in bit_file_name
    bit_file_path = os.path.abspath(dcp_folder + '/' + bit_file_name)
    static_json_file = dcp_folder + '/3_topFMKU60_STATIC.json'
    with open(static_json_file, 'r') as f:
        cl_data = json.load(f)
    tstmp = datetime.today().strftime('%Y-%m-%d_%H:%M')
    image_details = '{"breed": "SHELL", "fpga_board": "' + str(cl_data['fpga_board']) + \
                    '", "shell_type": "' + str(cl_data['shell']) + '", "comment": "DOSA automatic deploy of ' + \
                    node_name + ' from ' + app_name + ' on ' + tstmp + '"}'
    pr_verify_rpt = ""
    api_response = api_instance.cf_manager_rest_api_post_images(image_details, bit_file_path, pr_verify_rpt,
                                                                str(user_dict['user']), str(user_dict['pw']))
    return api_response.id


def upload_app_logic(dcp_folder, user_dict, node_name, app_name, api_instance):
    lt = glob.glob(dcp_folder + '/4_*partial.bin')
    assert len(lt) == 1
    bin_file_name = os.path.basename(lt[0])
    bin_file_path = os.path.abspath(dcp_folder + '/' + bin_file_name)
    sig_file_name = bin_file_name + '.sig'
    sig_file_path = os.path.abspath(dcp_folder + '/' + sig_file_name)
    lt = glob.glob(dcp_folder + '/5_*partial.rpt')
    assert len(lt) == 1
    rpt_file_name = os.path.basename(lt[0])
    rpt_file_path = os.path.abspath(dcp_folder + '/' + rpt_file_name)
    static_json_file = dcp_folder + '/3_topFMKU60_STATIC.json'
    with open(static_json_file, 'r') as f:
        cl_data = json.load(f)
    tstmp = datetime.today().strftime('%Y-%m-%d_%H:%M')
    image_details = '{"cl_id": "' + str(cl_data['id']) + '", "fpga_board": "' + str(cl_data['fpga_board']) + \
                    '", "shell_type": "' + str(cl_data['shell']) + '", "comment": "DOSA automatic deploy of ' \
                    + node_name + ' from ' + app_name + ' on ' + tstmp + '"}'
    api_response = api_instance.cf_manager_rest_api_post_app_logic(image_details, bin_file_path, sig_file_path,
                                                                   rpt_file_path, str(user_dict['user']), str(user_dict['pw']))
    return api_response.id


def create_new_cluster(cluster_req, user_dict):
    # build cluster_req structure
    print("Creating FPGA cluster...")
    print(json.dumps(cluster_req, indent=2))

    print("Executing POST ...")
    r1 = requests.post("http://" + __cf_manager_url__ + "/clusters?username={0}&password={1}&project_name={2}"
                                                        "&dont_verify_memory=1".format(
        user_dict['user'], urllib.parse.quote(user_dict['pw']),
        # user_dict['proj']
        'cf_Test_sled3'
        ),
                       json=cluster_req)

    if r1.status_code != 200:
        # something went wrong
        return errorReqExit("POST cluster", r1.status_code)

    cluster_data = json.loads(r1.text)
    print("...done.")
    # print("Id of new cluster: {}".format(cluster_data['cluster_id']))
    return cluster_data


def main(path_to_build_folder, user_file, host_addresses, use_pr_flow=True):
    cluster_json_path = path_to_build_folder + '/' + __cluster_filename__
    with open(cluster_json_path, 'r') as infile:
        cluster_data = json.load(infile)
    _, user_dict = load_user_credentials(user_file)

    # check for enoug cpu addresses
    cpu_addr = host_addresses.split(',')
    necessary_cpu_ids = 0
    for n in cluster_data['nodes']:
        if n['type'] == __cpu_device_name__:
            necessary_cpu_ids += len(n['ranks'])
    if len(cpu_addr) != necessary_cpu_ids:
        print(f'ERROR: {len(cpu_addr)} ip addresses for cpu nodes were given, but {necessary_cpu_ids} '
              f'are required. More details can be found in {cluster_json_path}. STOP.')
        exit(1)
    conf = Configuration()
    conf.host = __cf_manager_url__
    api_client = ApiClient(conf)
    api_instance = ImagesApi(api_client=api_client)
    print_dict = {}
    for n in cluster_data['nodes']:
        if n['type'] == __cf_device_name__:
            dcp_folder = os.path.abspath(path_to_build_folder + '/' + n['folder'] + '/dcps/')
            if use_pr_flow:
                bitfile_id = upload_app_logic(dcp_folder, user_dict, n['folder'], cluster_data['name'], api_instance)
            else:
                bitfile_id = upload_image(dcp_folder, user_dict, n['folder'], cluster_data['name'], api_instance)
            n['image-id'] = bitfile_id
            print_dict[n['folder']] = bitfile_id
        # elif n['type'] == __cpu_device_name__:
        #     sw_rank = n['ranks'][0]

    print('Images uploaded:')
    print(json.dumps(print_dict, indent=2))

    # create cluster dict
    cluster_req = []
    cpu_id = 0
    for n in cluster_data['nodes']:
        if n['type'] == __cf_device_name__:
            for r in n['ranks']:
                fpgaNode = {
                    'image_id': n['image-id'],
                    'node_id': int(r)
                }
                cluster_req.append(fpgaNode)
        elif n['type'] == __cpu_device_name__:
            for r in n['ranks']:
                try:
                    cpu_node = {'image_id': __NON_FPGA_IDENTIFIER__,
                                 'node_id': int(r),
                                 'node_ip': cpu_addr[cpu_id]}
                except IndexError:
                    print(f'ERROR: Only {len(cpu_addr)} ip addresses for cpu nodes were given, but at least '
                          f'{len(cpu_addr) + 1} are required. More details can be found in {cluster_json_path}.')
                    exit(1)
                else:
                    cpu_id += 1
                    cluster_req.append(cpu_node)
    assert cpu_id == len(cpu_addr)

    cfrm_data = create_new_cluster(cluster_req, user_dict)

    print('Cluster details:')
    print(json.dumps(cfrm_data, indent=2))
    print("Id of new cluster: {}".format(cfrm_data['cluster_id']))
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("USAGE: {} <path/to/dosa/build_dir/> <path/to/user.json> <cpu-ip-addresses,comma,separated>".format(sys.argv[0]))
        exit(1)
    main(os.path.abspath(sys.argv[1]), sys.argv[2], sys.argv[3], use_pr_flow=False)

