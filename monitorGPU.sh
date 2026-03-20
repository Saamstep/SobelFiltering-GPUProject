#!/bin/bash

# Script adapted from https://gist.github.com/omerfsen/8ecb620675525ac724a92bdf5a31a4b3

LOGFILE="nvidia_smi_monitor.csv"

echo "timestamp,gpu_index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw" > $LOGFILE

while true; do
  nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw              --format=csv,noheader >> $LOGFILE
  sleep 0.2
done