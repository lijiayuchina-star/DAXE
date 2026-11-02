#!/usr/bin/env bash
# Copyright (c) 2026 Jiayu Li, WELAB, Zhejiang University.
# Author : Jiayu Li
# Unit   : WELAB, Zhejiang University
# Email  : li.jia.yu@foxmail.com
# Date   : 2026-11-02
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}

"$PY" daxe.py --mode both --iters 60 --eta 3e-4 \
    --raw raw/vessel.tif --seq seq/mask.seq --out rlt
echo "[DAXE] all done, results in rlt/"
