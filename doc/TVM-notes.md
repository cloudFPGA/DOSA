TVM notes
=================

Install from source: https://tvm.apache.org/docs/install/from_source.html

To be able to use the post-quantization feature of DOSA, our custom version of TVM is required: https://github.com/cloudFPGA/tvm-for-dosa

Usually, it means the following steps:

```
$ git clone --recurse-submodules https://github.com/cloudFPGA/tvm-for-dosa.git tvm
$ mkdir -p tvm/build
$ cd tvm/build
$ cp ../cmake/config.cmake ./
$ which llvm-config  # update config.cmake with the respective llvm path
# maybe add further customations to build/config.cmake, see how-to above
$ cmake ..
$ make -j8
$ sudo make install
```

*Note: GCC 11 is required for a build* (C++17 support and bugs in linking).

On RHEL, this requires the following:
```
# yum install gcc-toolset-11
$ scl enable gcc-toolset-11 "cmake .."
$ scl enable gcc-toolset-11 "make -j 16"
# install....
```



