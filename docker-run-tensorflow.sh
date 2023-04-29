#!/bin/sh
docker run -ti \
  --gpus all \
  -v $HOME:$HOME -u $UID \
  -v $PWD:$PWD -w $PWD \
  -e HOME=$HOME \
  tensorflow/tensorflow:latest-gpu
