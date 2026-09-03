#!/bin/bash
# Serialise a list of scripts onto one GPU. $1 = physical GPU index (PCI order), rest = commands.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
shift
cd /home/jamie/storage-1/github-repos/magi_msvgd/investigation8
while [ $# -gt 0 ]; do
  eval "$1"
  shift
done
