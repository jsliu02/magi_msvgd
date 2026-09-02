#!/bin/bash
cd /home/jamie/storage-1/github-repos/magi_msvgd/investigation7
while pgrep -f "exp02_coldstart" > /dev/null; do sleep 20; done
export JAX_PLATFORMS=cpu
P=/home/jamie/miniforge3/envs/magi/bin/python
nohup $P exp09_profiled_svgd.py fn > exp09_fn.log 2>&1 &
nohup $P exp12_bandwidth_magi.py fn > exp12_fn.log 2>&1 &
nohup $P exp12_bandwidth_magi.py lorenz > exp12_lorenz.log 2>&1 &
nohup $P exp12_bandwidth_magi.py hiv > exp12_hiv.log 2>&1 &
wait
nohup $P exp09_profiled_svgd.py lorenz > exp09_lorenz.log 2>&1 &
nohup $P exp05_whatchanged.py > exp05.log 2>&1 &
wait
echo CHAIN4DONE
