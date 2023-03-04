#!/usr/bin/bash
set -ex

# build
make clean
make

# push
adb.exe push --sync reduce.cl /data/local/tmp/
adb.exe push --sync reduce /data/local/tmp/

# run
adb.exe shell chmod 555 /data/local/tmp/reduce
adb.exe shell /data/local/tmp/reduce
