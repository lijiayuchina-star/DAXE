# Copyright (c) 2026 Jiayu Li, WELAB, Zhejiang University.
# Author : Jiayu Li
# Unit   : WELAB, Zhejiang University
# Email  : li.jia.yu@foxmail.com
# Date   : 2026-11-02
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from functools import partial
from einops import rearrange, reduce
import einops

from torch.autograd import Function
from torch.nn import init
from torch.nn.modules.utils import _triple
from torch.autograd.function import once_differentiable

from   torch import nn, einsum
import torch.nn.functional as F
import torch
import types
import os

#-----------------------------------

from   inspect import isfunction
import math
from   einops.layers.torch import Rearrange

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, *args, **kwargs):
        nx = self.fn(x, *args, **kwargs) 
        if isinstance(x,tuple):
            x_f,x_b   = x
            nx_f,nx_b = nx
            x         = (x_f+nx_f,x_b+nx_b)
        else:
            x         = x + nx
        return x

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim   = dim
    def forward(self, time):
        device     = time.device
        dtype      = time.dtype
        half_dim   = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim,device=device) * -embeddings)
        embeddings = time[:,None] * embeddings[None,:]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings.to(torch.float16 if dtype == torch.int16 else torch.float32)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def group_cat(x,h):
    if isinstance(x,tuple):
        x_f,x_b = x
        h_f,h_b = h
        x_f = torch.cat((x_f, h_f), dim=1)
        x_b = torch.cat((x_b, h_b), dim=1)
        return (x_f,x_b)
    else:
        x = torch.cat((x, h), dim=1)
        return x

class Upsample(nn.Module):
    def __init__(self,dim,dim_out=None):
        super().__init__()
        self.up    = nn.Upsample(scale_factor=(1,2,2), mode="nearest")   
        self.cov   = nn.Conv3d(dim, default(dim_out, dim), (1,3,3), padding=(0,1,1))
    def forward(self,x):
        if not isinstance(x,tuple):
            x_f    = x
            x_b    = None
        else:
            x_f,x_b= x
            
        if not x_b is None:
            f_h = x_f.shape[2]
            x_f = torch.cat((x_f,x_b),dim=2)
        x_f = self.up(x_f)
        x_f = self.cov(x_f)
        if not x_b is None:
            x_b = x_f[:,:,f_h:,:,:].clone()
            x_f = x_f[:,:,:f_h,:,:]
            return (x_f,x_b)
        else:
            return x_f
        
class Downsample(nn.Module):
    def __init__(self,dim,dim_out=None):
        super().__init__()
        self.down  = Rearrange("b c h (w p1) (k p2) -> b (c p1 p2) h w k", p1=2,p2=2)   
        self.cov   = nn.Conv3d(dim * 4, default(dim_out, dim), 1)
    def forward(self,x):
        if not isinstance(x,tuple):
            x_f    = x
            x_b    = None
            x_f = self.down(x_f)
            x_f = self.cov(x_f)
            return x_f
        else:
            x_f,x_b= x
            x_f = self.down(x_f)
            x_f = self.cov(x_f)
            x_b = self.down(x_b)
            x_b = self.cov(x_b)
            return (x_f,x_b)
     
#-----------------------------------

class group_conv(nn.Module):
    def __init__(self,dim,out_dim,kernel=1,padding=0):
        super().__init__()
        self.init_conv = nn.Conv3d(dim, out_dim,\
            kernel_size=kernel,padding=padding)
        
    def forward(self,x):
        if not isinstance(x,tuple):
            x_f     = x
            return self.init_conv(x_f)
        else:
            x_f,x_b = x
            x_f     = self.init_conv(x_f)
            x_b     = self.init_conv(x_b)
            return (x_f,x_b)
        
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)
    def forward(self,x):
        if not isinstance(x,tuple):
            x     = x
            groupflag = False
        else:
            f_h   = x[0].shape[2]
            x     = torch.cat(x,dim = 2)
            groupflag = True
            
        b, c, h, w, s = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y z -> b h x (c y z)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        x = einsum("b h i j, b h d j -> b h i d", attn, v)
        x = rearrange(x, "b h (c y z) x -> b (h c) x y z",x=h,y=w,z=s)
        x = self.to_out(x)
        
        if groupflag:
            x_b = x[:,:,f_h:,:,:].clone()
            x_f = x[:,:,:f_h,:,:]
            return (x_f,x_b)
        else:
            return x

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv3d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self,x):
        if not isinstance(x,tuple):
            x     = x
            groupflag = False
        else:
            f_h     = x[0].shape[2]
            x     = torch.cat(x,dim = 2)
            groupflag = True
            
        b, c, h, w, s = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y z -> b h x (c y z)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        x = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        x = rearrange(x, "b h x (c y z) -> b (h c) x y z", h=self.heads, x=h, y=w, z=s)
        x = self.to_out(x)
        if groupflag:
            x_b = x[:,:,f_h:,:,:].clone()
            x_f = x[:,:,:f_h,:,:]
            return (x_f,x_b)
        else:
            return x

class WeightStandardizedConv3d(nn.Conv3d):

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1 1", "mean")
        var  = reduce(weight, "o ... -> o 1 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv3d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

class SeqEmd_Block_3D_DW(nn.Module):
    def __init__(self, dim, dim_out,Zrange=3,groups=8):
        super(SeqEmd_Block_3D_DW,self).__init__()
        #-------------------------
        self.proj1  = WeightStandardizedConv3d(dim, dim_out,(1,3,3),\
                                                    padding=(0,1,1))
        self.norm1  = nn.GroupNorm(groups, dim_out)
        self.act1   = nn.SiLU()
        #-------------------------
        self.proj2  = WeightStandardizedConv3d(dim_out, dim_out,(Zrange,1,1),\
                                                    padding=((Zrange//2),0,0))
        self.norm2  = nn.GroupNorm(groups, dim_out)
        self.act2   = nn.SiLU()
        #-------------------------

    def forward(self, x,shift=None):
        #-------------------------
        x = self.proj1(x)
        x = self.norm1(x)
        if exists(shift):
            scale, shift = shift
            x = x*(scale + 1) + shift
        x = self.act1(x)
        #-------------------------
        x = self.proj2(x)
        x = self.norm2(x)
        x = self.act2(x)
        #-------------------------
        return x

#----------------------------------
class Dual_ResnetBlock_3D(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, emb_dim=None, groups=8):
        super(Dual_ResnetBlock_3D,self).__init__()
        # B C E
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, dim_out * 2))
            if exists(emb_dim)
            else None
        )

        self.block1   = SeqEmd_Block_3D_DW(dim    ,dim_out,Zrange=3,groups=groups)
        #self.block2   = SeqEmd_Block_3D_DW(dim_out,dim_out,Zrange=3,groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self,x,e):
        if not isinstance(x,tuple):
            x_f    = x
            e_f    = e
            x_b    = None
        else:
            x_f,x_b= x
            e_f,e_b= e
            
        B          = x_f.shape[0]
            
        if exists(self.mlp) and exists(e_f):
            e_f = rearrange(e_f,"b c h -> (b c) h")
            e_f = self.mlp (e_f)
            e_f = rearrange(e_f,"(b c) h -> b h c 1 1",b=B)
            e_f = e_f.chunk(2, dim=1)

        h_f = self.block1(x_f,shift=e_f)
        #h_f = self.block2(h_f)
        x_f = h_f + self.res_conv(x_f)
    
        if exists(x_b):  
            e_b = None
            if exists(self.mlp) and exists(e_b):
                e_b = rearrange(e_b,"b c h -> (b c) h")
                e_b = self.mlp (e_b)
                e_b = rearrange(e_b,"(b c) h -> b h c 1 1",b=B)
                e_b = e_b.chunk(2, dim=1)
          
            h_b = self.block1(x_b,shift=e_b)
            #h_b = self.block2(h_b)
            x_b = h_b + self.res_conv(x_b)
            return (x_f,x_b)
        else:
            return x_f

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        if not isinstance(x,tuple):
            x_f    = x
            x_b    = None
        else:
            x_f,x_b= x
        x_f = self.norm(x_f)
        if exists(x_b): 
            x_b = self.norm(x_b)
            x   =  (x_f,x_b)
        else:
            x   =  x_f
            
        return self.fn(x)

#-----------------------------------
class Net(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1,2,4,8),
        channels=1,
        self_condition=False,
        resnet_block_groups=4,
        
    ):
        super().__init__()

        # determine dimensions
        self.self_condition = self_condition

        init_dim = default(init_dim, dim)
        self.init_conv = group_conv(channels, init_dim) # changed to 1 and 0 from 7,3

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(Dual_ResnetBlock_3D, groups=resnet_block_groups)

        # time embeddings
        emd_dim              = dim
        self.emd_dim         = emd_dim

        # layers
        self.downs = nn.ModuleList([])
        self.ups   = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, emb_dim=emd_dim),
                        block_klass(dim_in, dim_in, emb_dim=emd_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else group_conv(dim_in, dim_out, (1,3,3),(0,1,1)),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, emb_dim=emd_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, emb_dim=emd_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, emb_dim=emd_dim),
                        block_klass(dim_out + dim_in, dim_out, emb_dim=emd_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else group_conv(dim_out, dim_in, (1,3,3),(0,1,1)),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, emb_dim=emd_dim)
        self.final_conv      = group_conv(dim, self.out_dim) 

    def Seq_emb(self,x,AOTF_Seq):
        RL_Flag = isinstance(x,tuple)
        if not RL_Flag:
            x_f    = x
            FL     = x_f.shape[2]
        else:
            x_f,x_b= x
            FL     = x_f.shape[2]
            RL     = x_b.shape[2]
        B      = x_f.shape[0]
        CR         = AOTF_Seq.shape[1]
        AOTF_Seq  *= 8000
        Base_Frq   = AOTF_Seq.repeat(1,(FL//CR)+1)
        Base_Frq   = Base_Frq[:,:FL].view(B,FL,1)
        freqs      = torch.linspace(0,1,self.emd_dim//4).to(x_f.device)
        freqs      = freqs.view(B,1,-1).repeat(1,FL,1)
        freqs      = torch.pow(10000,freqs)
        PosSeq     = torch.linspace(0,10000,FL).to(x_f.device)
        PosSeq     = PosSeq.repeat(B,1,1).view(B,FL,-1)
        for_emb    = torch.cat([
                                torch.sin(Base_Frq/freqs),\
                                torch.cos(Base_Frq/freqs),\
                                torch.sin(PosSeq/freqs),\
                                torch.cos(PosSeq/freqs),\
                                ], dim=-1).to(x_f.dtype).to(x_f.device)
        #for_emb    = for_emb.view(B,-1)
        if RL_Flag:
            freqs      = torch.linspace(0,1,self.emd_dim//4).to(x_f.device)
            freqs      = freqs.view(B,1,-1).repeat(1,FL,1)
            freqs      = torch.pow(10000,freqs)
            PosSeq     = torch.linspace(0,10000,RL).to(x_f.device)
            PosSeq     = PosSeq.repeat(B,1,1).view(B,RL,-1)
            back_emb   = torch.cat([
                                    torch.sin(9000/freqs),\
                                    torch.cos(9000/freqs),\
                                    torch.sin(PosSeq/freqs),\
                                    torch.cos(PosSeq/freqs),\
                                    ], dim=-1).to(x_f.dtype).to(x_f.device)
            #back_emb = back_emb.view(B,-1)
        if RL_Flag: 
            return (for_emb,back_emb)
        else:
            return for_emb

    def forward(self,x,AOTF_Seq):
        
        if not isinstance(x,list):
            x_f    = x
            x_b    = None
        else:
            x_f,x_b= x
            
        B,H,W,S = x_f.shape
        CR      = AOTF_Seq.shape[0]
        x_f     = x_f.permute(0,3,1,2) # B H W S -> B S H W
        x_f     = x_f.view(B,1,S,H,W)

        if not x_b is None:
            _B,_H,_W,_S = x_b.shape
            x_b     = x_b.permute(0,3,1,2) # B H W S -> B S H W
            x_b     = x_b.view(_B,1,_S,_H,_W)
            
        if  x_b is None:
            x       = x_f
        else:
            x       = (x_f,x_b)
            
        e       = self.Seq_emb(x,AOTF_Seq)
  
        x       = self.init_conv(x) 
            
        if not x_b is None:
            x_f,x_b = x
            r       = (x_f.clone(),x_b.clone())
        else:
            r       = x.clone()
            
        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, e)
            h.append(x)
            x = block2(x, e)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, e)
        x = self.mid_attn(x)
        x = self.mid_block2(x, e)

        for block1, block2, attn, upsample in self.ups:
            x = group_cat(x, h.pop())
            x = block1(x, e)
            x = group_cat(x, h.pop())
            x = block2(x, e)
            x = attn(x)
            x = upsample(x)

        x = group_cat(x, r)
        x = self.final_res_block(x, e)
        
        x = self.final_conv(x)
        if isinstance(x,tuple):
            x = x[0] # B S H W -> B H W S
        return x.view(B,S,H,W).permute(0,2,3,1)
#-----------------------------------

# ----------------------------------------
# Solver
# ----------------------------------------
def DAXE_dl(m, x0, s, sr=2, dim=40, wgt=None, device='cuda'):
    """Deep learning branch of DAXE.

    m    : Mask instance from admm_base
    x0   : raw volume, shape (H, W, C)
    s    : full mask sequence, shape (1, L)
    sr   : backward scan down-sampling factor
    dim  : network width, must match the checkpoint
    wgt  : checkpoint path
    """
    rate = m.CR * sr
    h, w, c = x0.shape
    x0 = x0[:(h // 8) * 8, :(w // 8) * 8, :(c // rate) * rate]

    yf = m.Phi(x0)
    yb = x0.reshape(x0.shape[0], x0.shape[1], -1, rate).sum(-1)

    a = yf.repeat_interleave(m.CR, dim=2)
    b = yb.repeat_interleave(rate, dim=2)
    a = (a - a.min()) / (a.max() - a.min())
    b = (b - b.min()) / (b.max() - b.min())

    net = Net(dim=dim, dim_mults=(1, 2, 4, 8), channels=1, resnet_block_groups=4)
    net = net.to(device)
    try:
        sd = torch.load(wgt, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(wgt, map_location=device)
    except Exception:
        sd = torch.load(wgt, map_location=device)
    try:
        net.load_state_dict(sd)
    except Exception:
        net.load_state_dict({k[7:]: v for k, v in sd.items()})
    net.eval()

    with torch.no_grad():
        y = net([a.unsqueeze(0), b.unsqueeze(0)], s.clone())
    y = y.squeeze(0)
    return (y - y.min()) / (y.max() - y.min())
