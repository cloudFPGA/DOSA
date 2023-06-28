#  /*******************************************************************************
#   * Copyright 2019 -- 2023 IBM Corporation
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
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Mar 2022
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Python script to save reports of DOSA compilation
#  *         with relevant annotations
#  *
#  *
import sys
import os
import json
from datetime import datetime


__filedir__ = os.path.dirname(os.path.abspath(__file__))
__cluster_filename__ = 'arch_info.json'
__cf_device_name__ = 'cF_FMKU60_Themisto-Role_1'
__cpu_device_name__ = 'CPU_dummy_x86-1'


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("USAGE: {} <path/to/save_dir/outside_build_dirs/>".format(sys.argv[0]))
        exit(1)

    global_rpt_dir = os.path.abspath(sys.argv[1])
    tstmp = datetime.today().strftime('%Y-%m-%d_%H%M')
    my_rpt_dir = '{}/rpt_{}'.format(global_rpt_dir, tstmp)

    cluster_json_path = __filedir__ + '/' + __cluster_filename__
    with open(cluster_json_path, 'r') as infile:
        cluster_data = json.load(infile)

    for n in cluster_data['nodes']:
        if n['type'] == __cf_device_name__:
            dcp_folder = os.path.abspath(__filedir__ + '/' + n['folder'] + '/dcps/')
            os.system('mkdir -p {}/{}'.format(my_rpt_dir, n['folder']))
            os.system('cp {}/7_* {}/{}/'.format(dcp_folder, my_rpt_dir, n['folder']))
    os.system('mkdir -p {}/other_rpts/'.format(my_rpt_dir,))
    os.system('cp -R {}/* {}/other_rpts/'.format(__filedir__ + '/tmp_rpt_dir', my_rpt_dir,))
    os.system('cp {} {}/'.format(__cluster_filename__, my_rpt_dir))

    exit(0)

