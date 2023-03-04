#!/usr/bin/bash
set -ex

# build
make clean
make

# push
adb.exe push --sync elementwise_add.cl /data/local/tmp/
adb.exe push --sync elementwise_add /data/local/tmp/

# run
adb.exe shell chmod 555 /data/local/tmp/elementwise_add
adb.exe shell /data/local/tmp/elementwise_add