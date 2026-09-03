#!/bin/bash
cd /home/jamie/storage-1/github-repos/magi_msvgd/investigation8
export JAX_PLATFORMS=cpu
for s in fn lorenz hiv; do
  nohup /home/jamie/miniforge3/envs/magi/bin/python exp01_bwrule.py $s > exp01_$s.log 2>&1 &
done
wait
echo EXP01DONE
