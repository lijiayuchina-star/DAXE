# Copyright (c) 2026 Jiayu Li, WELAB, Zhejiang University.
# Author : Jiayu Li
# Unit   : WELAB, Zhejiang University
# Email  : li.jia.yu@foxmail.com
# Date   : 2026-11-02
"""ADMM solver base: mask operator, backward scan, BM4D/Hessian priors.

Configuration follows E:\\CS\\OMMT_re (Run.py -> DoubleScan_CS_Solver,
method 6 "BM4D+Hes"): ITERNUM 60, ETA 3e-4, CR 9, SR 2, RSF 5,
priors [(0.09,), (1, 100, 1, 2)] with weights [0.8, 0.2],
backward scan lambda 4e-3, block size 128.
"""
import math

import numpy as np
import torch
from bm4d import bm4d


def BM4D(x, i, sigma=0.2):
    a = bm4d(x.detach().cpu().numpy(), sigma).astype(np.float32)
    return torch.from_numpy(a).cuda()


def _fconj(a, shape):
    f = torch.fft.fftn(a, shape)
    return f * torch.conj(f)


def _fft_of_diff(shape, sigma):
    v0 = torch.tensor([1, -2, 1])
    v1 = torch.tensor([[1, -1], [-1, 1]])
    v3 = torch.tensor([1, -1])
    d = (
        _fconj(v0.reshape((1, 1, 3)), shape)
        + _fconj(v0.reshape((1, 3, 1)), shape)
        + sigma ** 2 * _fconj(v0.reshape((3, 1, 1)), shape)
        + 2 * _fconj(v1.reshape((1, 2, 2)), shape)
        + 2 * sigma * _fconj(v1.reshape((2, 1, 2)), shape)
        + 2 * sigma * _fconj(v1.reshape((2, 2, 1)), shape)
        + _fconj(v3.reshape((1, 1, 2)), shape)
        + _fconj(v3.reshape((1, 2, 1)), shape)
        + sigma * _fconj(v3.reshape((2, 1, 1)), shape)
    )
    return d.real


def _fdiff(d, dim):
    if dim == 0:
        return torch.diff(d, dim=0, append=torch.zeros([1, d.shape[1], d.shape[2]]).cuda())
    if dim == 1:
        return torch.diff(d, dim=1, append=torch.zeros([d.shape[0], 1, d.shape[2]]).cuda())
    return torch.diff(d, dim=2, append=torch.zeros([d.shape[0], d.shape[1], 1]).cuda())


def _bdiff(d, dim):
    if dim == 0:
        return torch.diff(d, dim=0, prepend=torch.zeros([1, d.shape[1], d.shape[2]]).cuda())
    if dim == 1:
        return torch.diff(d, dim=1, prepend=torch.zeros([d.shape[0], 1, d.shape[2]]).cuda())
    return torch.diff(d, dim=2, prepend=torch.zeros([d.shape[0], d.shape[1], 1]).cuda())


def _middle(u, b, lam):
    s = torch.clip(abs(u + b) - 1 / lam, min=0)
    s = s * torch.sign(u + b)
    b += u - s
    return s


def _lsparse(b, x, lam):
    return _middle(x, b, lam) - b


def _bfbf(b, c0, c1, x, lam):
    u = _bdiff(_fdiff(x, c0), c1)
    return _bdiff(_fdiff(_middle(u, b, lam) - b, c1), c0)


def _ffbb(b, c0, c1, x, lam):
    u = _fdiff(_fdiff(x, c0), c1)
    return 2 * _bdiff(_bdiff(_middle(u, b, lam) - b, c1), c0)


def Hessian(x, i, iters=30, mu=150, sigma=1, lamda=1):
    if x.ndim == 2 or x.shape[0] < 3:
        sigma = 0
    m = mu / lamda
    ym = x.max()
    x = x / ym
    shape = x.shape
    div = (_fft_of_diff(shape, sigma) + m).type(torch.float32).cuda()
    bs = torch.zeros((7,) + shape, dtype=torch.float32).cuda()
    y = torch.zeros(shape, dtype=torch.float32).cuda()
    f = m * x
    for k in range(iters):
        f = torch.fft.fftn(f)
        y = torch.fft.ifftn(f / (div if k >= 1 else m)).real
        f = m * x
        f = f + _bfbf(bs[0], 1, 1, y, lamda)
        f = f + _bfbf(bs[1], 2, 2, y, lamda)
        f = f + _bfbf(bs[2], 0, 0, y, lamda)
        f = f + _ffbb(bs[3], 1, 2, y, lamda)
        f = f + sigma * _ffbb(bs[4], 1, 0, y, lamda)
        f = f + sigma * _ffbb(bs[5], 2, 0, y, lamda)
        f = f + 0.05 * _lsparse(bs[6], y, lamda)
    y = torch.clip(y, min=0)
    if y.shape != shape:
        y = y[:shape[0]]
    return y * ym


PRIOR_FUC = [BM4D, Hessian]
IT_DEFAULT = 60
ETA_DEFAULT = 3e-4
RSF = 5
BACK_LAMDA = 4e-3
BLOCK_MAX = 128
PRIOR_PARA = [(0.09,), (1, 100, 1, 2)]
PRIOR_WGT = [0.8, 0.2]


class Mask:
    def __init__(self, cr, seq):
        self.CR = cr
        self.M = seq.view(1, 1, -1)[..., :cr]

    def Phi(self, x, sr=1, flip=False):
        p = self.M
        c = self.CR
        if sr != 1:
            c = self.CR * sr
            p = p.repeat(1, 1, 1, sr).view(1, 1, 1, c)
        if flip:
            p = p.flip(-1)
        h, w, s = x.shape
        st = s // c
        return (x[..., :st * c].view(h, w, st, c) * p).sum(-1)

    def Phi_T(self, y, sr=1, flip=False):
        p = self.M
        c = self.CR
        if sr != 1:
            c = self.CR * sr
            p = p.repeat(1, 1, 1, sr).view(1, 1, 1, c)
        if flip:
            p = p.flip(-1)
        h, w, st = y.shape
        r = y.view(h, w, st, 1).repeat(1, 1, 1, c) * p
        return r.view(h, w, st * c)

    def sum(self):
        return (self.M ** 2).sum(-1).cuda()

    def backward(self, img, sr):
        h, w, c = img.shape
        t = c - (c // self.CR) * self.CR
        rsf = RSF
        r = torch.zeros((h, w, c + t + self.CR)).cuda()
        r[..., rsf:rsf + c] = img
        return allpass(r, self.CR * sr), rsf


def allpass(img, cr):
    h, w, c = img.shape
    sp = c // cr
    return img[..., :sp * cr].view(h, w, sp, cr).sum(-1)


def iallpass(img, cr):
    h, w, c = img.shape
    return img.view(h, w, c, 1).repeat(1, 1, 1, cr).view(h, w, -1)


class BackScan:
    def __init__(self, yb, sr, cr, it, st, lam=BACK_LAMDA):
        self.y = yb
        self.fb = sr * cr
        self.it = it
        self.st = st
        self.lam = lam
        self.ready = False

    def match(self, r):
        mc = rc = r.shape[-1]
        st = -self.st
        l0 = max(-st, 0)
        r0 = min(rc - st, mc)
        ml = math.ceil(l0 / self.fb)
        mr = math.floor(r0 / self.fb)
        self.il = max(st, 0) + (ml * self.fb - l0)
        self.sp = (mr - ml) * self.fb
        self.ref = self.y[..., ml:mr]
        self.ready = True

    def __call__(self, img, i):
        if i < self.it // 10:
            return img
        if i == self.it // 10:
            self.match(img)
            return img
        if not self.ready:
            self.match(img)
        z = img[..., self.il:self.il + self.sp].clone()
        a = self.fb + self.lam / 2
        img[..., self.il:self.il + self.sp] = z + iallpass((self.ref - allpass(z, self.fb)) / a, self.fb)
        return img


def Block_Prior(P, X, i, para, mx=BLOCK_MAX):
    h, w, s = X.shape
    if s < mx:
        return P(X, i, *para)
    step = s // mx
    tail = 0 if s % mx == 0 else 1
    c = torch.zeros_like(X)
    r = torch.zeros_like(X)
    for k in range(step):
        b = X[..., k * mx:(k + 1) * mx]
        lo, hi = b.min(), b.max()
        b = P((b - lo) / (hi - lo), i, *para)
        r[..., k * mx:(k + 1) * mx] += b * (hi - lo) + lo
        c[..., k * mx:(k + 1) * mx] += 1
        if tail:
            b = X[..., -mx:]
            lo, hi = b.min(), b.max()
            b = P((b - lo) / (hi - lo), i, *para)
            r[..., -mx:] += b * (hi - lo) + lo
            c[..., -mx:] += 1
    return r / c


def DAXE_admm(m, yf, yb, sr, it=IT_DEFAULT, eta=ETA_DEFAULT, st=RSF):
    x = m.Phi_T(yf)
    ms = m.sum()
    h, w, c = x.shape
    bs = BackScan(yb, sr, m.CR, it, st)
    pf = list(PRIOR_FUC)
    pp = list(PRIOR_PARA)
    n = len(pf)
    # OMMT_re DoubleScan_CS_Solver: given weights are scaled by n/(n+1),
    # then the backward-scan prior gets 1/(n+1).
    wgt = [p * n / (n + 1) for p in PRIOR_WGT] + [1 / (n + 1)]
    u = x.unsqueeze(-1).repeat(1, 1, 1, n + 1).clone()
    v = torch.zeros((h, w, c, n + 1)).cuda()
    with torch.no_grad():
        for i in range(it):
            us = 0
            for p in range(n + 1):
                us = us + (u[..., p] - v[..., p] / eta) * wgt[p]
            mt = (yf - m.Phi(us)) / (ms.unsqueeze(-1).repeat(1, 1, yf.shape[-1]) + eta)
            x = us + m.Phi_T(mt)
            for p in range(n):
                u[..., p] = Block_Prior(pf[p], x + v[..., p] / eta, i, pp[p], BLOCK_MAX)
                v[..., p] = v[..., p] + eta * (x - u[..., p])
            u[..., -1] = bs(x + v[..., -1] / eta, i)
            v[..., -1] = v[..., -1] + eta * (x - u[..., -1])
            u = u.detach()
            v = v.detach()
            print('it %d' % i, flush=True)
    return x
