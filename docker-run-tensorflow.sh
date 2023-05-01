#!/bin/sh
docker run -ti \
  --gpus all \
  --publish 8889:8888 \
  -v $HOME:$HOME -u $UID \
  -v $PWD:$PWD -w $PWD \
  -e HOME=$HOME \
  tensorflow/tensorflow:latest-gpu
