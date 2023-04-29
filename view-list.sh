#!/bin/sh
set -eu

# View all images from a list, putting them on thumbnail pages.
# Specify the list as an argument.

BATCH_SIZE=35
INPUT_LIST="$1"

PAGE=0
while [ "$(($PAGE*$BATCH_SIZE))" -lt "$(wc -l < "$INPUT_LIST")" ]; do
  tail -n "+$(($PAGE*$BATCH_SIZE+1))" "$INPUT_LIST" | head -n "$BATCH_SIZE" | 
    tr "\n" "\0" | xargs -0 feh \
      --fullscreen --index \
      --limit-width=3840 --limit-height=2160 \
      --thumb-height=400 --thumb-width=512 \
      #
  echo "Press ENTER for next page, or CTRL-C to abort."
  read
  PAGE=$(($PAGE+1))
done