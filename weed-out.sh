#!/bin/sh
set -eu

# This script is meant to be used when you have a list of images that are
# all of the same kind (e.g. "postcards with a stamp") except for a few of
# them, and you want to weed them out of the list.
#
# Specify the list as the first argument. The script will use "feh" to display
# a mosaic of images. Click on the images that don't belong to the list, then
# hit "q" when done to exit "feh".
#
# The script places the "good" images in a ".ok" file, and the ones that don't
# belong in a ".err" file. It progressively drains the input list.

INPUT_LIST="$1"
OUTPUT_OK="$1.ok"
OUTPUT_ERR="$1.err"
OUTPUT_TMP="$1.tmp"
BATCH_SIZE=35

while [ -s "$INPUT_LIST" ]; do
  echo "$(wc -l "$INPUT_LIST") image(s) remaining in '$INPUT_LIST'."
  truncate --size=0 "$OUTPUT_TMP"
  head -n "$BATCH_SIZE" "$INPUT_LIST" |
  tr "\n" "\0" |
  xargs -0 feh \
    --fullscreen --thumbnails \
    --limit-width=3840 --limit-height=2160 \
    --thumb-height=400 --thumb-width=512 \
    --action "echo %F >> $OUTPUT_TMP" \
    #
  echo "Press ENTER to commit and continue; Ctrl-C to abort."
  read foo
  cat "$OUTPUT_TMP" >> "$OUTPUT_ERR"
  head -n "$BATCH_SIZE" "$INPUT_LIST" | grep -v -F -f "$OUTPUT_TMP" >> "$OUTPUT_OK"
  tail -n "+$(($BATCH_SIZE+1))" "$INPUT_LIST" > "$OUTPUT_TMP"
  mv "$OUTPUT_TMP" "$INPUT_LIST"
done
