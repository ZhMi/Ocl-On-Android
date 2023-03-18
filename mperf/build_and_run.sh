#!/usr/bin/bash
set -ex

# push
adb.exe push --sync mali_compare_s32_s16_s8_compute_test /data/local/tmp/
adb.exe push --sync compare_s32_s16_s8_compute_test_kernel.cl /data/local/tmp/

adb.exe shell < adb_cmd.txt