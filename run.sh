#!/bin/sh
set -eu

[ -d venv ] || python -m venv venv
venv/bin/pip install --quiet --upgrade --requirement requirements.txt

for MODEL in front-vs-back with-address-or-blank; do
  UNIT=fmracv.$MODEL
  systemctl --user stop $UNIT || true
  systemctl --user reset-failed $UNIT || true
  systemd-run --user --unit=$UNIT --same-dir venv/bin/python fmracv.py serve $MODEL.yaml
  systemctl --user status $UNIT --no-pager
done
