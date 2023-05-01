#!/bin/sh
REMOTE=s3://ephemera-scan-backups
LOCAL=./imgroot

if ! [ "$1" ]; then
  echo "Please give a file containing a list of images to fetch."
  exit 1
fi

cat "$1" | while IFS= read key; do
  [ -f "$LOCAL/$key" ] || echo "$key"
done | xargs -P 10 -I @@KEY@@ -n1 aws --profile ephemerasearch s3 cp "$REMOTE/@@KEY@@" "$LOCAL/@@KEY@@"
