Installation and Dependencies of DOSA (dimidium)
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
pip3 install -r requirements.txt
```

### TVM

Dosa requires an installation of TVM, including python bindings. 

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

- [hls4ml](https://github.com/cloudFPGA/hls4ml-for-dosa) in `dimidium/backend/third_party_libs/`
- [vhdl4cnn](https://github.com/cloudFPGA/VHDL4CNN) in `dimidium/backend/third_party_libs/`
- [cFDK](https://github.com/cloudFPGA/cFDK) in `dimidium/backend/buildTools/lib/cFBuild1/`
- [cFCreate](https://github.com/cloudFPGA/cFCreate) in `dimidium/backend/buildTools/lib/cFBuild1/`
- [ZRLMPI](https://github.com/cloudFPGA/ZRLMPI) in `dimidium/backend/codeGen/templates/`

