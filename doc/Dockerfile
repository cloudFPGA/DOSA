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

FROM ubuntu:22.04 AS dev

RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    curl \
    g++ \
    gdb \
    git \
    graphviz \
    libcurl4-openssl-dev \
    libssl-dev \
    libtinfo-dev \
    libz-dev \
    make \
    ninja-build \
    pkg-config \
    sudo \
    unzip \
    wget \
    python3 \
    python3-dev \
    python3-setuptools \
    python3-pip \
    gcc \
    libtinfo-dev \
    zlib1g-dev \
    build-essential \
    cmake \
    libedit-dev \
    libxml2-dev \
    llvm-15  \
    clang-15 libclang-15-dev


### install TVM ###
FROM dev AS tvm-build

ARG max_jobs=8
ENV MAX_JOBS=${max_jobs}

WORKDIR /app

RUN git clone --recurse-submodules https://github.com/cloudFPGA/tvm-for-dosa.git tvm
RUN cd tvm && \
  cd build && \
  cp ../cmake/config.cmake ./ && \
  cmake .. && \
  make -j$MAX_JOBS && \
  make install


### install DOSA ###
FROM tvm-build AS dosa-base

WORKDIR /app

RUN git clone --recurse-submodules https://github.com/cloudFPGA/DOSA.git
RUN cd DOSA && pip install -r requirements.txt --no-dependencies
RUN pip install tvm/python


### environement ###
FROM dosa-base AS dosa-run

ENV DOSA_cFBuild1_used_dcps_path=/current_dcps/
ENV VIRTUAL_ENV=/usr/bin/python3

WORKDIR /app/DOSA


