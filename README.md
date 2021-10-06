DOSA
=========
**Draico OSA**


Installation
----------------

1. [TVM installation](https://tvm.apache.org/docs/install/from_source.html#)
    - use the **internal** fork of TVM as source, see below
2.  ```bash
    $ virtualenv venv -p /usr/bin/python3.8
    $ . venv/bin/activate
    $ pip3 install -r requirements.txt
    ```
3. add local tvm installation to environment
    - updated `PYHTHONPATH` as it is described [here](https://tvm.apache.org/docs/install/from_source.html#tvm-package), is not always working with virutalenv or pycharm.
    - instead, add it as `.pth` to venv:
        - if `virtualenvwrapper` is installed: 
          ```bash
          $ add2virtualenv /home/ngl/gitrepos/tvm/python /home/ngl/gitrepos/tvm/vta/python
          ```
        - otherwise:
          ```bash
          $ cp ./setup/_virtualenv_path_extensions.pth ./venv/lib/python3.8/site-packages/
          ```
          and **update** the absolute paths in the `.pth` file. 
          


Structure
-------------

- learning
- dimidium:
    - first (alpha) version of DOSA
    - https://en.wikipedia.org/wiki/Dimidium

t.b.c....


Private TVM fork
--------------------

To store internal configurations, patches and changes, an IBM internal fork exists: https://github.ibm.com/cloudFPGA/tvm-for-dosa

The internal repo can be added like this:
```
$ git remote add internal git@github.ibm.com:cloudFPGA/tvm-for-dosa.git
$ git push -u internal main
```


On changes, just run `make -j4` again to build TVM (`cmake ..` not really necessary).
From time to time a `git pull origin main` is recommended to pull latest changes from TVM.


DOSA examples
--------------------

- `cifar10` example with CNN from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- `crnn` example from https://codingvision.net/pytorch-crnn-seq2seq-digits-recognition-ctc




