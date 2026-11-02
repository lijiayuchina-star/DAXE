# Copyright (c) 2026 Jiayu Li, WELAB, Zhejiang University.
# Author : Jiayu Li
# Unit   : WELAB, Zhejiang University
# Email  : li.jia.yu@foxmail.com
# Date   : 2026-11-02
"""Command line entry of the DAXE solver.

Examples
--------
python daxe.py                                  # ADMM + DL, defaults
python daxe.py --mode admm --iters 60           # ADMM only
python daxe.py --mode dl                        # DL only
python daxe.py --raw data/a.tif --seq data/m.seq --out results
"""
import argparse
import os
import pickle

import numpy as np
import tifffile
import torch

from admm_base import Mask, DAXE_admm
from dl_base import DAXE_dl

ROOT = os.path.dirname(os.path.abspath(__file__))
CR = 9
SR = 2
DIM = 40
STP = 5
ETA = 3e-4
ITERS = 60


def load_img(path, cr):
    a = np.float32(tifffile.imread(path))
    a = a.transpose(2, 1, 0).transpose(1, 0, 2)
    a = (a - a.min()) / (a.max() - a.min())
    return torch.from_numpy(a[..., :a.shape[-1] // cr * cr]).cuda()


def load_seq(path):
    with open(path, 'rb') as f:
        return torch.from_numpy(np.float32(pickle.load(f))).cuda()


def save(path, x):
    tifffile.imwrite(path, x.detach().cpu().numpy().transpose(2, 0, 1))
    print('[DAXE] saved %s' % path)


def parse_args():
    p = argparse.ArgumentParser(description='DAXE reconstruction')
    p.add_argument('--mode', choices=['admm', 'dl', 'both'], default='both')
    p.add_argument('--iters', type=int, default=ITERS)
    p.add_argument('--eta', type=float, default=ETA)
    p.add_argument('--raw', default=os.path.join(ROOT, 'raw', 'vessel.tif'))
    p.add_argument('--seq', default=os.path.join(ROOT, 'seq', 'mask.seq'))
    p.add_argument('--out', default=os.path.join(ROOT, 'rlt'))
    p.add_argument('--weights',
                   default=os.path.join(ROOT, 'weights', 'BestModel_NRS-40.pth'))
    p.add_argument('--dim', type=int, default=DIM)
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    print('[DAXE] mode=%s iters=%d eta=%g raw=%s seq=%s out=%s'
          % (a.mode, a.iters, a.eta, a.raw, a.seq, a.out))

    x0 = load_img(a.raw, CR)
    seq = load_seq(a.seq)
    m = Mask(CR, seq)

    if a.mode in ('admm', 'both'):
        yf = m.Phi(x0)
        yb, _ = m.backward(x0, SR)
        save(os.path.join(a.out, 'yf.tif'), yf)
        save(os.path.join(a.out, 'yb.tif'), yb)
        x = DAXE_admm(m, yf, yb, SR, a.iters, a.eta, STP)
        save(os.path.join(a.out, 'DAXE_admm.tif'), x)

    if a.mode in ('dl', 'both'):
        x = DAXE_dl(m, x0, seq.view(1, -1), SR, a.dim, a.weights)
        save(os.path.join(a.out, 'DAXE_dl.tif'), x)

    print('[DAXE] finished')


if __name__ == '__main__':
    main()
