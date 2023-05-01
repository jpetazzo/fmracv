#!/bin/sh

REMOTE_HOST=fmracv
REMOTE_PATH=fmracv
REMOTE_FQDN=fmracv.tinyshellscript.com

rsync -Pav \
  --exclude databases \
  --exclude imgroot \
  --exclude lists \
  --exclude notebooks \
  --exclude __pycache__ \
  --exclude .git \
  --exclude .ipynb_checkpoints \
  ./ $REMOTE_HOST:$REMOTE_PATH/

ssh $REMOTE_HOST "cd $REMOTE_PATH && ./run.sh"

for PORT in 8001 8005; do
  for TRY in $(seq 5); do
    curl -f http://$REMOTE_FQDN:$PORT && break
    sleep 2
  done
done
