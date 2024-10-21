# Modified from:
#   VQGAN:    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py  
#   nanoGPT:  https://github.com/karpathy/nanoGPT/blob/master/model.py
#   llama:    https://github.com/facebookresearch/llama/blob/main/llama/model.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
#   PixArt:   https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py

# cat_tokens: with linear projection to project catted tokens() to dim

from dataclasses import dataclass
from typing import Optional, List


import torch
import torch.nn as nn
from torch.nn import functional as F
from .drop_path import DropPath
import numpy as np
import math
from einops import rearrange, reduce, pack, unpack

def cosine_sim_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
    L, S = query.size(-2), key.size(-2)
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias = attn_bias.to(query.device)
    
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    
    query = F.normalize(query, p=2.0, dim=-1)
    key = F.normalize(key, p=2.0, dim=-1)
    attn_weight = query @ key.transpose(-2, -1)
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

@dataclass
class ModelArgs:
    use_adaLN: bool = False
    attn_type: str = 'sdp' ##sdp for scaled dot-product; or cs for cosine similarity
    pause: bool = False
    pause_num: int = 0
    token_each: int = 4
    split_embder: bool = False  ##not use
    cls_row: bool = False
    pos_type: str = 'rope2d'
    code_dim: int = 16
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    
    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    num_classes: int = 1000
    caption_dim: int = 2048
    class_dropout_prob: float = 0.1
    model_type: str = 'c2i'

    vocab_size: int = 256
    cls_token_num: int = 1
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048


#################################################################################
#                      Embedding Layers for Class Labels                        #
#################################################################################
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#################################################################################
#                                  GPT Model                                    #
#################################################################################
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.config = config
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor = None, 
        input_pos: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None
    ):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)
        # breakpoint()
        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        
        if 'rope' in self.config.pos_type:
            xq = apply_rotary_emb(xq, freqs_cis)
            xk = apply_rotary_emb(xk, freqs_cis)
        
        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))
        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        
        if self.config.attn_type == 'sdp':
            # print('we are using scaled dot-product attention')
            output = F.scaled_dot_product_attention(
                xq, keys, values, 
                attn_mask=mask, 
                is_causal=True if mask is None else False, # is_causal=False is for KV cache
                dropout_p=self.attn_dropout_p if self.training else 0)
        else:
            # print('we are using cosine similarity attention')
            output = cosine_sim_attention(
                xq, keys, values, 
                attn_mask=mask, 
                is_causal=True if mask is None else False, # is_causal=False is for KV cache
                dropout_p=self.attn_dropout_p if self.training else 0)
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_adaLN = config.use_adaLN
        
        if self.use_adaLN:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(config.dim, 6 * config.dim, bias=True)
            )
            # print("We are using adaLN!")

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None):
        if self.use_adaLN:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=2)
            h = x + gate_msa * self.drop_path(self.attention(modulate(self.attention_norm(x), shift_msa, scale_msa), freqs_cis, start_pos, mask))
            out = h + gate_mlp * self.drop_path(self.feed_forward(modulate(self.ffn_norm(h), shift_mlp, scale_mlp)))
        else:
            h = x + self.drop_path(self.attention(self.attention_norm(x), freqs_cis, start_pos, mask))
            out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.vocab_size = 2 ** config.code_dim
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num
        self.token_each = config.token_each
        self.cls_row = config.cls_row
        self.pos_type = config.pos_type
        self.pause = config.pause
        self.pause_num = config.pause_num
        self.use_adaLN = config.use_adaLN 
        
   
        if self.cls_row:
            rope_cls_token_num = int(self.block_size ** 0.5)
        else:
            rope_cls_token_num = self.cls_token_num
        
        self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
                
        if self.token_each > 1:
            self.tok_embedder = torch.nn.ModuleList()
            for i in range(self.token_each):
                self.tok_embedder.append(nn.Embedding(self.vocab_size, config.dim))
            self.proj = nn.Linear(self.token_each*config.dim, config.dim, bias=False)
        else:
            self.tok_embedder = nn.Embedding(self.vocab_size, config.dim)
        
        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(config, dpr[layer_id]))

        if self.config.use_adaLN:
            self.final_adaLN = nn.Sequential(
                nn.SiLU(),
                nn.Linear(config.dim, 2 * config.dim, bias=True)
            )
        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        if self.token_each > 1:
            self.output = torch.nn.ModuleList()
            for i in range(self.token_each):
                self.output.append(nn.Linear(config.dim, self.vocab_size, bias=False))
        else:
            self.output = nn.Linear(config.dim, self.vocab_size, bias=False)

        
        # rotary pos embedding
        grid_size = int(self.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        if 'rope' in self.pos_type:
            if self.pos_type == 'rope1d':
                self.freqs_cis = precompute_freqs_cis(int((self.block_size+1)*self.token_each), self.config.dim // self.config.n_head, self.config.rope_base, rope_cls_token_num, self.cls_row, self.token_each)
            elif self.pos_type == 'rope2d':
                self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, rope_cls_token_num, self.cls_row)
            else:
                print("Unidentified position embedding type!!!")
    
        # breakpoint()
        
        if self.pause:
            self.pause_idxs = torch.randint(0, self.block_size, (self.pause_num,))
        
        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1

        self.initialize_weights()

    def initialize_weights(self):        
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)
        # Zero-out adaLN modulation layers in GPT blocks:
        if self.config.use_adaLN:
            for layer in self.layers:
                nn.init.constant_(layer.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.final_adaLN[-1].weight, 0)
            nn.init.constant_(self.final_adaLN[-1].bias, 0)
        # Zero-out output layers:
        if self.token_each > 1:
            for output in self.output:
                nn.init.constant_(output.weight, 0)
        else:
            nn.init.constant_(self.output.weight, 0)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        # if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
        #     return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)

        causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        grid_size = int(self.config.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        # if self.cls_row:
        #     grid_size = grid_size + 1
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num, self.cls_row)

    
    def forward(
        self, 
        idx: torch.Tensor, 
        cond: torch.Tensor,  # cond_idx_or_embed
        input_pos:  Optional[torch.Tensor] = None, 
        targets: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ):
        device = idx.device if not idx is None else cond.device
        bs, l, t_n = idx.size() if not idx is None else (cond.size()[0], 0, self.token_each)
        l = l+1 #cls token
        t = int(l * t_n)
        pos = torch.arange(0, l, dtype=torch.long, device=device) # shape (t)
        
        if idx is not None and cond is not None: # training or naive inference
            cond_embeddings = self.cls_embedding(cond, train=self.training)[:,:self.cls_token_num]
            
            if self.token_each == 1:
                token_embeddings = self.tok_embedder(idx).squeeze(-2)
            else:
                token_embeddings = []
                for i in range(self.token_each):
                    token_embeddings.append(self.tok_embedder[i](idx[:,:,i]))
                
                token_embeddings = self.proj(torch.cat(token_embeddings, dim=-1))
            
            token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
                
            self.freqs_cis = self.freqs_cis.to(device)
            
            h = self.tok_dropout(token_embeddings)
            
        else:
            if cond is not None: # prefill in inference
                token_embeddings = self.cls_embedding(cond, train=self.training)[:,:self.cls_token_num]
                cond_embeddings = self.cls_embedding(cond, train=self.training)[:,:self.cls_token_num]
                self.freqs_cis = self.freqs_cis.to(device)
                
            else: # decode_n_tokens(kv cache) in inference ###not happpen without kv cache
                token_embeddings = self.tok_embedder(idx).squeeze(-2)
                cond_embeddings = None
                
            h = self.tok_dropout(token_embeddings)
            
        if 'rope' in self.pos_type:
            self.freqs_cis = self.freqs_cis.to(device)
            freqs_cis = self.freqs_cis[:token_embeddings.shape[1]]
            if self.pause:
                freqs_cis = torch.cat([freqs_cis, torch.zeros(num_pause,  self.config.dim // self.config.n_head // 2, 2).to(device)])
        else:
            freqs_cis = None
        
        for layer in self.layers:
            h = layer(h, cond_embeddings, freqs_cis, input_pos, mask)
        
        if self.use_adaLN:
            shift, scale = self.final_adaLN(cond_embeddings).chunk(2, dim=2)
            h = modulate(self.norm(h), shift, scale)
        else:
            h = self.norm(h)
        
        if self.token_each > 1:
            logits = []
            for output in self.output:
                logits.append(output(h).float())
            logits = torch.stack(logits, 2).reshape(bs, -1, self.tok_embedder[0].weight.shape[0])    
        else:
            logits = self.output(h).float() # [bs, total_length, num_clases]

        acc_img = 0.0
        if self.training:
            logits = logits[:, self.cls_token_num - 1:].contiguous()
            true_nums_img = (torch.argmax(logits, dim=-1) == targets.reshape(targets.shape[0], -1)).sum()
            acc_img = true_nums_img / logits.shape[0] / logits.shape[1]
            # print('true num:', true_nums_img)
        # if we are given some desired targets also calculate the loss
        loss = None
        if valid is not None:
            loss_all = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            valid_all = valid[:,None].repeat(1, targets.shape[1]).view(-1)
            loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
        elif weights is not None:
            pred_prob = F.log_softmax(logits, dim=-1)
            loss = (-(weights * pred_prob.reshape(-1, self.vocab_size))).sum(dim=-1).mean()
        elif targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss, acc_img

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)
    
    @torch.no_grad()
    def generate_cfg(self, idx, cond, num_iter, temperature=1.0, top_k=None, remask=False, cfg_scale=10.0, cfg_schedule='constant', scale_pow=1.0):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence num_iter times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        cond_null = torch.ones_like(cond) * self.num_classes if cond is not None else None
        cond = torch.cat([cond, cond_null]) if cond is not None else None
        for step in range(num_iter):
            ratio = 1. * (step+1) / num_iter
            if cfg_schedule == 'linear':
                # cfg = 1. + (cfg_scale - 1.) * (1. - mask_rate)
                cfg = 1. + (cfg_scale - 1.) * ratio
            elif cfg_schedule == 'constant':
                cfg = cfg_scale
            elif cfg_schedule == 'cos':
                # scale_step = (1-np.cos((((step+1)/num_iter)**scale_pow)*math.pi)) *1/2
                # cfg = (cfg_scale-1)*scale_step + 1
                x_mapped = -np.pi / 2 + ratio * (np.pi / 2)
                cfg = 1. + (cfg_scale - 1.) * np.cos(x_mapped)
            elif cfg_schedule == 'log':
                cfg = 1. + (cfg_scale - 1.) * np.log1p(ratio * (np.e -1.)) / np.log1p(np.e -1.)
            elif cfg_schedule == 'square':
                cfg = 1. + (cfg_scale - 1) * ratio**2
            elif cfg_schedule == 'square_root':
                ratio = 1. * (step) / num_iter
                cfg = 1. + (cfg_scale - 1.) * (ratio**0.5)
            
            idx_cond = idx if idx is None else torch.cat([idx, idx])
            # forward the model to get the logits for the index in the sequence
            logits, _, _ = self(idx=idx_cond, cond=cond,)
            # pluck the logits at the final step and scale by desired temperature
            logits_combined = logits
            cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0) 
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg
            
            logits = logits[:, -self.token_each:, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, :, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = []
            for i in range(self.token_each):
                idx_next.append(torch.multinomial(probs[:, i, :], num_samples=1))
            idx_next = torch.cat(idx_next, dim=1).unsqueeze(1)
            # append sampled index to the running sequence and continue
            if idx is not None:
                idx = torch.cat((idx, idx_next), dim=1)
            else:
                idx = idx_next
            
        return idx, logits


#################################################################################
#                      Rotary Positional Embedding Functions                    #
#################################################################################
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py 
def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs) # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1) # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache 

def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120, cls_row=False):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs) # (grid_size, head_dim // 2)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1) # (grid_size, grid_size, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+grid_size**2, head_dim // 2, 2)
    
    return cond_cache 

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    if not x.shape[1] == freqs_cis.shape[0]:
        freqs_cis = freqs_cis.repeat_interleave(x.shape[1] // freqs_cis.shape[0], dim=0)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2)
    # breakpoint()
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2) # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)



#################################################################################
#                                GPT Configs                                    #
#################################################################################

### class-conditional
def GPT_XXXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs)) # 3.9B

def GPT_2B(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=28, dim=1792, **kwargs)) # 2.0B

def GPT_XXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs)) # 1.4B

def GPT_XL(**kwargs):
    return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs)) # 775M

def GPT_L(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs)) # 343M

def GPT_B(**kwargs):
    return Transformer(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs)) # 111M
        

GPT_models = {
    'GPT-B': GPT_B, 'GPT-L': GPT_L, 'GPT-XL': GPT_XL, 'GPT-XXL': GPT_XXL, 'GPT-XXXL': GPT_XXXL,
    'GPT-2B': GPT_2B, 
}

  
#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
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
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb