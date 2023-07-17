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


