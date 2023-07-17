TVM notes
=================

Install from source: https://tvm.apache.org/docs/install/from_source.html

Usually, it means the following steps:

```
$ git clone --recurse-submodules https://github.com/apache/tvm tvm
$ git submodule init
$ git submodule update
$ mkdir tvm/build
$ cd tvm/build
$ cp ../cmake/config.cmake ./
$ which llvm-config  # update config.cmake with the respective llvm path
# maybe add further customations to build/config.cmake, see how-to above
$ cmake ..
$ make -j8
```

*Note: GCC 11 is required for a build* (C++17 support and bugs in linking).

On RHEL, this requires the following:
```
# yum install gcc-toolset-11
$ scl enable gcc-toolset-11 "cmake .."
$ scl enable gcc-toolset-11 "make -j 16"
```



