DOSA
=========
**Draico OSA** 

Installation
----------------

1. [TVM installation](https://tvm.apache.org/docs/install/from_source.html#)
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
t.b.c....


