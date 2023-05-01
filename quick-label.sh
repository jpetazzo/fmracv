#!/bin/sh
set -e

# This is a script to quickly classify a bunch of images.
# Call it like this:
# ./quick-label.sh image-list label1 label2 label3
# It will display all the images in the file "image-list"
# one at a time. You can then press 1 to assign label1,
# 2 to assign label2, etc.; and it will then put the file
# in "image-list.label1" (or image-list.label2, etc.) 
# Note that it currently only supports up to 3 labels.
# Also, for simplicity, it must be executed from the imgroot
# directory.

INPUT_LIST="$1"
shift

if [ "$(basename "$(readlink -f .)")" != "imgroot" ]; then
  echo "Please run this script from the imgroot directory,"
  echo "specifying list files as relative paths. Thanks."
  exit 1
fi

feh \
  --fullscreen \
  --filelist "$INPUT_LIST" \
  --on-last-slide quit \
  --draw-actions \
  --action1 [$1]"echo %F >> $INPUT_LIST.$1" \
  --action2 [$2]"echo %F >> $INPUT_LIST.$2" \
  --action3 [$3]"echo %F >> $INPUT_LIST.$3" \
  #

grep --fixed-strings -v \
  --file $INPUT_LIST.$1 \
  --file $INPUT_LIST.$2 \
  --file $INPUT_LIST.$3 \
  $INPUT_LIST > $INPUT_LIST.todo \
  || true

mv "$INPUT_LIST.todo" "$INPUT_LIST"

wc -l $INPUT_LIST*
