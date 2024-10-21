# import math
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from timm.models.layers import drop_path
import math
from functools import partial

import einops
        

def swish(x):
    return x * torch.sigmoid(x)

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        return x * torch.sigmoid(x)

class GELU2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

act = nn.GELU

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class Attention(nn.Module):
    def __init__(self, H):
        super().__init__()
        qkv_bias=True
        self.num_heads = H.bert_n_head
        head_dim = H.bert_n_emb // self.num_heads

        self.scale = head_dim ** -0.5 # not needed as same with the default

        self.qkv = nn.Linear(H.bert_n_emb, H.bert_n_emb * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(H.attn_pdrop)
        self.proj = nn.Linear(H.bert_n_emb, H.bert_n_emb)
        self.proj_drop = nn.Dropout(H.resid_pdrop)

        if self.qkv.bias is not None:
            self.qkv.bias.data.zero_()
    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
        x = F.scaled_dot_product_attention(q, k, v)
        x = einops.rearrange(x, 'B H L D -> B L (H D)', H=self.num_heads)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):

    def __init__(self, H, dim=None):
        super().__init__()
        assert H.bert_n_emb % H.bert_n_head == 0
        # key, query, value projections for all heads
        if dim is None:
            dim = H.text_emb
        self.kv = nn.Linear(dim, H.bert_n_emb*2)
        self.query = nn.Linear(H.bert_n_emb, H.bert_n_emb)
        # regularization
        self.attn_drop = nn.Dropout(H.attn_pdrop)
        self.resid_drop = nn.Dropout(H.resid_pdrop)
        # output projection
        self.proj = nn.Linear(H.bert_n_emb, H.bert_n_emb)
        self.n_head = H.bert_n_head
        
        self.null_emb = nn.Parameter(torch.zeros(1, 1, dim))
        self.null_emb.data.normal_(0, 0.01)

    def forward(self, x, c, layer_past=None):
        # pdb.set_trace()
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        if c is None:
            c = self.null_emb.repeat(B, 1, 1)
        kv = self.kv(c)
        kv = einops.rearrange(kv, 'B L (K H D) -> K B H L D', K=2, H=self.n_head)
        k, v = kv[0], kv[1]
        
        q = self.query(x)
        q = einops.rearrange(q, 'B L (H D) -> B H L D', H=self.n_head)

        y = F.scaled_dot_product_attention(q,k,v)

        y = einops.rearrange(y, 'B H L D -> B L (H D)', H=self.n_head)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, H, drop_path=0.0):
        super().__init__()

        self.ln1 = nn.LayerNorm(H.bert_n_emb)
        self.ln2 = nn.LayerNorm(H.bert_n_emb)
        # self.attn = CausalSelfAttention(H)
        self.attn = Attention(H)
        self.mlp = nn.Sequential(
            nn.Linear(H.bert_n_emb, 4 * H.bert_n_emb),
            act(),  # nice
            nn.Linear(4 * H.bert_n_emb, H.bert_n_emb),
            nn.Dropout(H.resid_pdrop),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        init_values = 1.0
        self.gamma_1 = nn.Parameter(init_values * torch.ones((H.bert_n_emb)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((H.bert_n_emb)), requires_grad=True)

    def forward(self, x):

        x = x + self.drop_path(self.gamma_1 * self.attn(self.ln1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.ln2(x)))

        return x

class CrossBlock(nn.Module):
    def __init__(self, H, drop_path=0.0):
        super().__init__()

        self.ln1 = nn.LayerNorm(H.bert_n_emb)
        self.ln1_5 = nn.LayerNorm(H.bert_n_emb)
        self.ln2 = nn.LayerNorm(H.bert_n_emb)
        # self.attn = CausalSelfAttention(H)
        self.attn = Attention(H)

        self.cross = CrossAttention(H)

        self.mlp = nn.Sequential(
            nn.Linear(H.bert_n_emb, 4 * H.bert_n_emb),
            act(),  # nice
            nn.Linear(4 * H.bert_n_emb, H.bert_n_emb),
            nn.Dropout(H.resid_pdrop),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        init_values = 1.0
        self.gamma_1 = nn.Parameter(init_values * torch.ones((H.bert_n_emb)), requires_grad=True)
        self.gamma_1_5 = nn.Parameter(init_values * torch.ones((H.bert_n_emb)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((H.bert_n_emb)), requires_grad=True)

    def forward(self, x, c):


        x = x + self.drop_path(self.gamma_1 * self.attn(self.ln1(x)))
        x = x + self.drop_path(self.gamma_1_5 * self.cross(self.ln1_5(x), c))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.ln2(x)))
        return x

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_steps, dim, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class AdaLayerNorm_Time(nn.Module):
    def __init__(self, n_embd, diffusion_step, emb_type="adalayernorm_abs"):
        super().__init__()
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(diffusion_step, n_embd)
        else:
            self.emb = nn.Embedding(diffusion_step, n_embd)
        self.silu = nn.SiLU()
        self.l0 = nn.Linear(n_embd, n_embd)
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep):
        # emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        emb = self.linear(self.silu(self.l0(self.emb(timestep)))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x

class AdaLayerNorm_Cls(nn.Module):
    def __init__(self, n_embd, num_classes):
        super().__init__()
        self.emb = nn.Embedding(num_classes, n_embd)
        self.silu = nn.SiLU()
        self.l0 = nn.Linear(n_embd, n_embd)
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep):
        # emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        emb = self.linear(self.silu(self.l0(self.emb(timestep)))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x

class AdaEMB_Cls(nn.Module):
    def __init__(self, n_embd, num_classes):
        super().__init__()
        self.emb = nn.Embedding(num_classes, n_embd)
        self.silu = nn.SiLU()
        self.l0 = nn.Linear(n_embd, n_embd)
        self.linear = nn.Linear(n_embd, n_embd)
        self.layernorm = nn.LayerNorm(n_embd)
    def forward(self, timestep):
        # emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        emb = self.linear(self.silu(self.layernorm(self.l0(self.emb(timestep))))).unsqueeze(1)
        return emb

class AdaTkn_Time(nn.Module):
    def __init__(self, n_embd, diffusion_step, emb_type="adalayernorm_abs"):
        super().__init__()
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(diffusion_step, n_embd)
        else:
            self.emb = nn.Embedding(diffusion_step, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd)
        self.l0 = nn.Linear(n_embd, n_embd)

    def forward(self, timestep):
        # emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        emb = self.linear(self.silu(self.l0(self.emb(timestep)))).unsqueeze(1)
        return emb

class AdaLayerNorm_Spatial(nn.Module):
    def __init__(self, n_embd, spatial_shape, emb_type="adalayernorm_abs"):
        super().__init__()
        if "abs" in emb_type:
            # self.emb = get_2d_sincos_pos_embed(n_embd, spatial_shape)
            self.register_buffer('emb', torch.Tensor(get_2d_sincos_pos_embed(n_embd, spatial_shape)))
        else:
            self.emb = nn.Embedding(spatial_shape, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x):
        emb = self.linear(self.silu(self.emb)).unsqueeze(0)
        scale, shift = torch.chunk(emb, 2, dim=-1)
        x = self.layernorm(x) * (1 + scale) + shift
        return x



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=256):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb




if __name__ == '__main__':


    grid = np.arange(20, dtype=float)
    data = get_1d_sincos_pos_embed_from_grid(64, grid)
    pdb.set_trace()
