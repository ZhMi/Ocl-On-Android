#!/usr/bin/bash
set -ex

# build
make clean
make

# push
adb.exe push --sync gemm.cl /data/local/tmp/
adb.exe push --sync gemm /data/local/tmp/

# run
adb.exe shell chmod 555 /data/local/tmp/gemm
adb.exe shell /data/local/tmp/gemm
