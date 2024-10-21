# Modified from:
#   VQGAN:    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py  
#   nanoGPT:  https://github.com/karpathy/nanoGPT/blob/master/model.py
#   llama:    https://github.com/facebookresearch/llama/blob/main/llama/model.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
#   PixArt:   https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py


from dataclasses import dataclass
from typing import Optional, List


import torch
import torch.nn as nn
from torch.nn import functional as F
from .drop_path import DropPath
import numpy as np
import math
from einops import rearrange, reduce, pack, unpack
import scipy.stats as stats
import random

def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)


class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss
    
@dataclass
class ModelArgs:
    ### mlm
    mask_ratio_min: float = 0.5
    mask_ratio_max: float = 1.0
    mask_ratio_mu: float = 0.55
    mask_ratio_std: float = 0.25
    pause: bool = False
    pause_num: int = 10
    
    use_adaLN: bool = False
    smoothing: float = 0.1
    gen_iter: int = 10
    maskpersample: bool = False
    mask_schstep: int = 10000
    mask_schtype: str = 'cos'
    norm_type: str = 'RMS'
    pos_type: str = 'rope2d'
    token_each: int = 4
    cls_row: bool = False
    
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

    # vocab_size: int = 64
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
        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        
        if 'rope' in self.config.pos_type:
            xq = apply_rotary_emb(xq, freqs_cis)
            xk = apply_rotary_emb(xk, freqs_cis)
        
        # breakpoint()
        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))
        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        
        output = F.scaled_dot_product_attention(
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

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None):
        h = x + self.drop_path(self.attention(self.attention_norm(x), freqs_cis, start_pos, mask))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out

class MlmLayer(nn.Module):

    def __init__(self, feat_emb_dim, word_emb_dim, vocab_size):
        super().__init__()
        self.fc = nn.Linear(feat_emb_dim, word_emb_dim)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(word_emb_dim)
        self.bias = nn.Parameter(torch.zeros(1, 1, vocab_size))

    def forward(self, x, word_embeddings):
        mlm_hidden = self.fc(x)
        mlm_hidden = self.gelu(mlm_hidden)
        mlm_hidden = self.ln(mlm_hidden)
        word_embeddings = word_embeddings.transpose(0, 1)
        logits = torch.matmul(mlm_hidden, word_embeddings)
        logits = logits + self.bias
        return logits

class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        # self.vocab_size = config.vocab_size
        self.codebook_size = 2 ** config.code_dim
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num
        self.token_each = config.token_each
        self.cls_row = config.cls_row
        self.smoothing = config.smoothing
        self.pos_type = config.pos_type
        self.gen_iter = config.gen_iter
        self.maskpersample = config.maskpersample
        self.mask_schstep = config.mask_schstep
        self.mask_schtype = config.mask_schtype
        self.pause = config.pause
        self.pause_num = config.pause_num
        self.use_adaLN = config.use_adaLN 
        
        self.row_num = int(self.block_size ** 0.5)
        
        rope_cls_token_num = self.cls_token_num
        
        self.vocab_size = self.codebook_size + 1 ## mask token
        self.mask_token_label = self.vocab_size - 1
        if self.pause:
            self.vocab_size = self.vocab_size + 1 ## mask token
            
        self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
        
        self.tok_embedder = nn.Embedding(self.vocab_size, config.dim)
        self.tok_dropout = nn.Dropout(config.token_dropout_p)
        self.pos_embed = nn.Embedding(self.block_size, self.config.dim, _freeze=True)
        if self.pos_type == 'sincos':
            self.pos_embed = nn.Parameter(torch.zeros(self.block_size+1, self.config.dim), requires_grad=False)            
                
        
        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(config, dpr[layer_id]))

        # linear project catted tokens
        if self.token_each > 1:
            self.proj = nn.Linear(self.token_each*config.dim, config.dim, bias=False)
        
        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        if self.token_each > 1:
            self.output = torch.nn.ModuleList()
            for i in range(self.token_each):
                self.output.append(nn.Linear(config.dim, self.vocab_size, bias=False))
        else:
            self.output = nn.Linear(config.dim, self.vocab_size, bias=False)
        
        self.mlm_layer = MlmLayer(feat_emb_dim=config.dim, word_emb_dim=config.dim, vocab_size=self.vocab_size)
        if self.smoothing > 0.0:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=self.smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        # self.output.weight = self.tok_embedder.weight

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

        
        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1

        self.initialize_weights()

    def initialize_weights(self):        
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Zero-out output layers:
        if self.token_each > 1:
            for output in self.output:
                nn.init.constant_(output.weight, 0)
        else:
            nn.init.constant_(self.output.weight, 0)
            
        if self.pos_type =='sincos':
            if self.cls_row:
                pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.row_num, self.row_num+1)
            else:
                pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.row_num, self.row_num, cls_token=True)
            # breakpoint()
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())
            # self.pos_embed.weight = torch.from_numpy(pos_embed).float()
            
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
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num, self.cls_row)

    
    def forward(
        self, 
        idx: torch.Tensor, 
        cond: torch.Tensor,  # cond_idx_or_embed
        token_all_mask: Optional[torch.Tensor] = None, 
        train_iter_loc: Optional[torch.Tensor] = None,
        input_pos:  Optional[torch.Tensor] = None, 
        targets: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ):
        device = idx.device if not idx is None else cond.device
        # masking
        bsz, seq_len, t_n = idx.size() if not idx is None else (cond.size()[0], 0, self.token_each)  
        t = int(seq_len * t_n)
        
        ################### Masking #####################
        start_iter = 0
        if token_all_mask is None: ### training process 
            if self.maskpersample:
                step = torch.randint(start_iter, self.gen_iter, (bsz,))
                ratio = 1. * (step) / self.gen_iter
                if train_iter_loc is not None:
                    if train_iter_loc < self.mask_schstep:
                        mask_rate = torch.cos(math.pi / 2. * ratio)
                    else:
                        mask_rate = 1. - torch.cos(math.pi / 2. * ratio)
                num_masked_tokens = torch.floor(seq_len * mask_rate)
                
                num_masked_tokens = torch.maximum(torch.Tensor([1]), num_masked_tokens).long().to(device)

                while True:
                    noise = torch.rand(bsz, seq_len, device=device)  # noise in [0, 1]
                    sorted_noise, _ = torch.sort(noise, dim=1)  # ascend: small is remove, large is keep
                    # cutoff_mask = sorted_noise[:, num_masked_tokens-1:num_masked_tokens]
                    cutoff_mask = torch.cat([sorted_noise[i][num_masked_tokens[i]-1:num_masked_tokens[i]] for i in range(bsz)]).unsqueeze(1)
                    token_all_mask = (noise <= cutoff_mask).float()
                    if token_all_mask.sum() == num_masked_tokens.sum():
                        break
                    else:
                        print("Rerandom the noise!")
            else:
                step = torch.randint(start_iter, self.gen_iter, (1,))
                ratio = 1. * (step) / self.gen_iter
                # breakpoint()
                if self.mask_schtype == 'cos':
                    mask_rate = torch.cos(math.pi / 2. * ratio)
                if self.mask_schtype == 'linear':
                    mask_rate = 1. - ratio
                num_masked_tokens = torch.floor(seq_len * mask_rate)
                num_masked_tokens = int(torch.maximum(torch.Tensor([1]), num_masked_tokens).item())
                while True:
                    noise = torch.rand(bsz, seq_len, device=device)  # noise in [0, 1]
                    sorted_noise, _ = torch.sort(noise, dim=1)  # ascend: small is remove, large is keep
                    cutoff_mask = sorted_noise[:, num_masked_tokens-1:num_masked_tokens]
                    token_all_mask = (noise <= cutoff_mask).float()
                    if token_all_mask.sum() == bsz*num_masked_tokens:
                        break
                    else:
                        print("Rerandom the noise!")
                    
        idx[token_all_mask.nonzero(as_tuple=True)] = self.mask_token_label  #1 -> mask token
        idx = idx.long()
        token_embeddings = self.tok_embedder(idx)
        
        # concate class token
        cond_embeddings = self.cls_embedding(cond, train=self.training)[:,:self.cls_token_num]
        cond_embeddings = cond_embeddings.repeat_interleave(self.token_each, dim=1).unsqueeze(1)
        
        token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
        
        if 'rope' not in self.pos_type:
            token_embeddings = token_embeddings + self.pos_embed.data.unsqueeze(1).repeat_interleave(self.token_each, dim=1)
        
        token_embeddings = token_embeddings.reshape(bsz, -1, token_embeddings.shape[-1])
        
        h = self.tok_dropout(token_embeddings)
        if 'rope' in self.pos_type:
            self.freqs_cis = self.freqs_cis.to(device)
            freqs_cis = self.freqs_cis
        else:
            freqs_cis = None
            
        mask = torch.ones((token_embeddings.shape[-2], token_embeddings.shape[-2]), dtype=torch.bool).to(device)
        
        # transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis, input_pos, mask)
        
        # output layers
        h = self.norm(h)
        
        logits = self.mlm_layer(h, self.tok_embedder.weight.data.detach()) #bsz, l, vocab_size
        logits = logits[:, self.token_each:, :self.codebook_size]
        
        # if we are given some desired targets also calculate the loss
        loss = None
        if weights is not None:
            pred_prob = F.log_softmax(logits, dim=-1)
            loss = (-(weights * pred_prob.reshape(-1, self.codebook_size))).sum(dim=-1).mean()
            
        elif targets is not None:
            loss = self.criterion(logits.reshape(bsz*seq_len*self.token_each, -1), targets.reshape(bsz*seq_len*self.token_each))
            loss = loss.reshape(bsz, seq_len, self.token_each)
            token_all_mask = token_all_mask.repeat_interleave(self.token_each, dim=1).reshape(bsz, seq_len, self.token_each)
            loss = (loss * token_all_mask).sum() / token_all_mask.sum()  # mean loss on removed patches
        
        return logits, loss, token_all_mask.sum()/bsz

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)
    
    
    @torch.no_grad()
    def generate_cfg(self, idx, cond, num_iter=10, temperature=1.0, top_k=None, cfg_scale=10.0, remask=False, cfg_schedule='constant', scale_pow=1.0):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        bsz = cond.shape[-1]
        cond_null = torch.ones_like(cond) * self.num_classes
        cond = torch.cat([cond, cond_null])
        
        unknown_number_in_the_beginning = self.block_size
        _CONFIDENCE_OF_KNOWN_TOKENS = +np.inf
        
        initial_token_indices = self.mask_token_label * torch.ones(bsz, unknown_number_in_the_beginning, self.token_each)
        device = cond.device

        idx = initial_token_indices.to(device)
        for step in range(num_iter):
            cur_idx = idx.clone().long()
            idx = torch.cat([cur_idx, cur_idx])
            token_all_mask = idx == self.mask_token_label
            
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1. * (step + 1) / num_iter
            if cfg_schedule == 'linear':
                # cfg = 1. + (cfg_scale - 1.) * (1. - mask_rate)
                cfg = 1. + (cfg_scale - 1.) * ratio
            elif cfg_schedule == 'constant':
                cfg = cfg_scale
            elif cfg_schedule == 'powercos':
                scale_step = (1-np.cos((((step+1)/num_iter)**scale_pow)*math.pi)) *1/2
                cfg = (cfg_scale-1)*scale_step + 1
                print(f"step: {step}; scale step: {scale_step}; cfg: {cfg}")
                    
            if self.mask_schtype == 'cos':
                mask_ratio = np.cos(math.pi / 2. * ratio)
            if self.mask_schtype == 'linear':
                mask_ratio = 1. - ratio
                
            # forward the model to get the logits for the index in the sequence
            logits, _, _ = self(idx=idx, cond=cond, token_all_mask=token_all_mask)
            # pluck the logits at the final step and scale by desired temperature
            logits_combined = logits
            cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0) 
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg
            
            sample_dist = torch.distributions.categorical.Categorical(logits=logits)
            sampled_ids = sample_dist.sample()
            sampled_ids_ori = sampled_ids.reshape(bsz, -1, self.token_each)
            
            # get ids for next step
            unknown_map = (cur_idx == self.mask_token_label)
            sampled_ids = torch.where(unknown_map, sampled_ids_ori, cur_idx)
            
            probs = F.softmax(logits, dim=-1)  #bs, seq_len, vocab_len
            
            probs = probs.reshape(bsz, self.block_size, self.token_each, self.codebook_size)
            selected_probs = torch.gather(probs, dim=-1, index=sampled_ids.unsqueeze(-1)).squeeze(-1)
            
            mask_prob = (selected_probs * unknown_map).sum() / unknown_map.sum()
            unmask_prob = (selected_probs * (1-unknown_map.float())).sum() / (1-unknown_map.float()).sum()
            selected_probs_ori = torch.gather(probs, dim=-1, index=sampled_ids_ori.unsqueeze(-1)).squeeze(-1)
            mask_prob_ori = (selected_probs_ori * unknown_map).sum() / unknown_map.sum()
            unmask_prob_ori = (selected_probs_ori * (1-unknown_map.float())).sum() / (1-unknown_map.float()).sum()
            
            selected_probs = torch.where(unknown_map, selected_probs.double(), _CONFIDENCE_OF_KNOWN_TOKENS).float()
            
            if remask and step < int(num_iter*0.5):
                remask_idx = torch.nonzero(sampled_ids_ori!=sampled_ids, as_tuple=False)
                print('step:', step, 'unmask num:', remask_idx.shape[0])
                for _id in remask_idx:
                    selected_probs[[i for i in _id]] = selected_probs_ori[[i for i in _id]]
            
                
            mask_len = torch.Tensor([np.floor(unknown_number_in_the_beginning * mask_ratio)]).to(device)
            # Keeps at least one of prediction in this round and also masks out at least
            mask_len = torch.maximum(torch.Tensor([1]).to(device),
                                    torch.minimum(torch.sum(unknown_map[:,:,:1].squeeze(-1), dim=-1, keepdims=True) - 1, mask_len))
            
            # Sample masking tokens for next iteration
            masking = mask_by_random_topk(mask_len[0], selected_probs, temperature * (1 - ratio))
            # Masks tokens with lower confidence.
            idx = torch.where(masking, self.mask_token_label, sampled_ids) #masking True的地方是mask_token_id, False是sampled_ids

        return sampled_ids, logits

def mask_by_random_topk(mask_len, probs, temperature=1.0):
    mask_len = mask_len.squeeze()
    confidence = torch.log(probs) + torch.Tensor(temperature * np.random.gumbel(size=probs.shape)).to(probs.device)
    pos_confidence = confidence.mean(dim=-1).unsqueeze(-1)
    sorted_confidence, _ = torch.sort(pos_confidence, axis=-2)
    # Obtains cut off threshold given the mask lengths.
    cut_off = sorted_confidence[:, mask_len.long()-1:mask_len.long(), :]
    # Masks tokens with lower confidence.
    masking = (pos_confidence <= cut_off)
    return masking

#################################################################################
#                      Rotary Positional Embedding Functions                    #
#################################################################################
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py 
def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120, cls_row=False, token_each=1):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs) # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1) # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = cache
    return cond_cache 


def precompute_freqs_cis_2d_cls0(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120, cls_row=False):
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
}

  
#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size1, grid_size2, cls_token=False, extra_tokens=1):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size1, dtype=np.float32)
    grid_w = np.arange(grid_size2, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    # breakpoint()
    grid = np.stack(grid, axis=0)
    # breakpoint()
    grid = grid.reshape([2, 1, grid_size1, grid_size2])
    # breakpoint()
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    # breakpoint()
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