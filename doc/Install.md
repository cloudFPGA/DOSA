Installation and Dependencies of DOSA (gradatim)
===================================================

Requirements
--------------
### for RHEL 

```bash
yum install python3.8 tmux python3-virtualenv
yum groupinstall 'Development Tools'
# for tvm
yum install llvm-toolset llvm-devel llvm-libs cmake
```

### Python virtual environment
```bash
virtualenv venv -p /usr/bin/python3.8
. venv/bin/activate
pip install -r requirements.txt --no-dependencies
```

(Please note, the `--no-dependencies` is necessary to be able to install all required packages. All packages and their 
necessary dependencies are defined in the `requirements.txt`, but `pip` sometimes/often tries to solve it on its own and fails). 

### TVM

Dosa requires an installation of TVM, including python bindings.
To be able to use the post-quantization feature of DOSA, our custom version of TVM is required: https://github.com/cloudFPGA/tvm-for-dosa

1. Please follow the steps in [TVM installation](https://tvm.apache.org/docs/install/from_source.html#) (some hints are also given in [./TVM-notes.md](./TVM-notes.md)).
2. add local tvm installation to environment. You have two *alternatives*:
   1. updated `PYHTHONPATH` as it is described [here](https://tvm.apache.org/docs/install/from_source.html#tvm-package). (But we made the experience this is not always working with virutalenv or IDEs.)
   2. *OR*: add it as `.pth` to venv:
       - if `virtualenvwrapper` is installed: 
          ```bash
          source venv/bin/activate
          add2virtualenv path/to/tvm/python path/to/tvm/vta/python
          ```
        - otherwise:
          ```bash
          cp ./setup/_virtualenv_path_extensions.pth ./venv/lib/python3.8/site-packages/
          ```
          and **update** the absolute paths in the `.pth` file!

### FPGA build tools 

To generate FPGA binaries, the corresponding HLS, synthesis, and place and route tools must be installed. 
In case the Xilinx Vivado suite is used, ensure that the [Y2K22 patch](https://support.xilinx.com/s/article/76960?language=en_US
) is installed. 

Installation and Setup
------------------------------------

```bash
git clone --recurse-submodules https://github.com/cloudFPGA/DOSA.git
```

DOSA runs purely in python, so besides the virtual environment (see above) only environment variables need to be set:
```bash
export TVM_LOG_DEBUG=0
# only necessary for builds on cloudFPGA
export DOSA_cFBuild1_used_dcps_path=/path/to/folder/with/shell_STATIC.dcp/
# sometimes necessary
export PYTHONPATH=.
```

The `DOSA_cFBuild1_used_dcps_path` is only required if [IBM cloudFPGA](https://github.com/cloudFPGA) is used as build platform. 
It is not required if DOSA is invoked with the `--no-build` flag. 
How to retreive the latest static Shells for the FPGAs (i.e. the `.dcp` files) is explained [here](https://cloudfpga.github.io/Doc/pages/GETTING_STARTED/getting_started.html#id6).

Git submodules contained in DOSA
-------------------------------------

- [hls4ml](https://github.com/cloudFPGA/hls4ml-for-dosa) in `gradatim/backend/third_party_libs/`
- [vhdl4cnn](https://github.com/cloudFPGA/VHDL4CNN) in `gradatim/backend/third_party_libs/`
- [cFDK](https://github.com/cloudFPGA/cFDK) in `gradatim/backend/buildTools/lib/cFBuild1/`
- [cFCreate](https://github.com/cloudFPGA/cFCreate) in `gradatim/backend/buildTools/lib/cFBuild1/`
- [ZRLMPI](https://github.com/cloudFPGA/ZRLMPI) in `gradatim/backend/codeGen/templates/`


Docker
---------------------------------------

DOSA, without the FPGA build tools, can also be build and run inside a docker container:
```commandline
cd DOSA/doc
docker build -f Dockerfile -t dosa-build .
```

Afterwards, DOSA CLI can be started via:
```commandline
docker run -it -v ./my_in_and_out_dir/:/scratch:Z -v ./folder_with_current_shell_STATIC/:/current_dcps/:Z dosa-build
# use DOSA CLI as described in usage
./gradatim -h
# for example, build one design with an example ONNX
./gradatim.sh onnx ./config/dosa_config_default.json ./examples/PTTCNN_int8.onnx ./examples/PTTCNN_meta.json /app/scratch/pttcnn/ --no-roofline
```
If building for the cloudFPGA target, the required `Shell_STATIC.dcp` files (as described above) should then be mounted to the `/current_dcps/` folder inside the container. 
(Naturally, this disables the roofline diagram feature of DOSA, which requires a GUI.)

