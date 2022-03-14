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
__cluster_filename__ = 'draft_info.json'
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

