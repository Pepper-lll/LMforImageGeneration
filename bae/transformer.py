import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path

from bae.transformer_utils import *


class Transformer(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, H, avg_pooling=False):
        super().__init__()

        self.vocab_size = H.codebook_size + 1
        self.n_embd = H.bert_n_emb
        self.block_size = H.block_size
        self.n_layers = H.bert_n_layers
        self.codebook_size = H.codebook_size
        self.causal = H.sampler in ['autoregressive', 'arebm']
        self.avg_pooling = avg_pooling
        if self.causal:
            self.vocab_size = H.codebook_size

        self.tok_emb = nn.Embedding(self.vocab_size, self.n_embd)
        self.pos_emb = nn.Parameter(
            torch.zeros(1, self.block_size, self.n_embd))
        self.start_tok = nn.Parameter(torch.zeros(1, 1, self.n_embd))
        self.drop = nn.Dropout(H.embd_pdrop)

        self.blocks = nn.Sequential(*[Block(H, 0) for _ in range(self.n_layers)])
        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.head = nn.Linear(self.n_embd, self.codebook_size, bias=False)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, t=None):
        # each index maps to a (learnable) vector
        token_embeddings = self.tok_emb(idx)

        if self.causal:
            token_embeddings = torch.cat(
                (self.start_tok.repeat(token_embeddings.size(0), 1, 1), token_embeddings),
                dim=1
            )

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # each position maps to a (learnable) vector

        position_embeddings = self.pos_emb[:, :t, :]

        x = token_embeddings + position_embeddings

        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        if self.avg_pooling:
            x = x.mean(1)
        x = self.ln_f(x)
        logits = self.head(x)

        return logits

class TransformerBD(nn.Module):

    def __init__(self, H, avg_pooling=False):
        super().__init__()

        self.vocab_size = H.codebook_size
        self.n_embd = H.bert_n_emb
        self.block_size = H.block_size
        self.n_layers = H.bert_n_layers
        self.codebook_size = H.codebook_size

        self.tok_emb = nn.Embedding(self.vocab_size, self.n_embd)
        self.tok_emb.weight.data.normal_(0.0, 0.02)
        
        self.drop = nn.Dropout(H.embd_pdrop)

        # transformer
        block_type = Block
        dpr = [x.item() for x in torch.linspace(0, H.drop_path, self.n_layers)]

        self.exp_type = 'unconditional'
        if not H.cross and H.dataset.startswith('laion'):
            self.cls_embedding = nn.Linear(768, H.bert_n_emb)
            self.exp_type = 't2i_tkn'
        if H.cross and H.dataset.startswith('laion'):
            self.exp_type = 't2i_cross'
            block_type = CrossBlock
        if H.dataset.startswith('imagenet'):
            self.cls_embedding = nn.Embedding(H.num_classes, self.n_embd)
            self.exp_type = 'class_tkn'

        self.blocks = nn.Sequential(*[block_type(H, dpr[i]) for i in range(self.n_layers)])

        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.head = nn.Linear(self.n_embd, self.codebook_size, bias=True)

        self.sample_steps = H.sample_steps

        self.time_step_embedding = AdaTkn_Time(self.n_embd, self.sample_steps)

        self.pos_emb = nn.Parameter(torch.Tensor(get_2d_sincos_pos_embed(self.n_embd, H.latent_shape[-1])).unsqueeze(0))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx, label=None, time_steps=None,):
        # pdb.set_trace()
        # each index maps to a (learnable) vector
        # token_embeddings = self.tok_emb(idx)
        if idx.shape[1] == 0:
            token_embeddings = torch.zeros(idx.shape[0], 0, self.n_embd).to('cuda')
        else:
            # token_embeddings = (idx*1.0) @ self.tok_emb.weight #/ float(self.n_embd)
            token_embeddings = (idx*1.0 - 0.5) * 2.0 @ self.tok_emb.weight #/ float(self.n_embd)

        t = token_embeddings.shape[1]

        position_embeddings = self.pos_emb[:, :t, :]

        x = token_embeddings + position_embeddings

        time_tkn = True
        # time_tkn = False
        time_emb = self.time_step_embedding(time_steps)
        if time_tkn:
            x = torch.cat([x, time_emb], 1)
        else:
            x = x + time_emb

        if self.exp_type.endswith('tkn') and label != None:
            # pdb.set_trace()
            cls_emb = self.cls_embedding(label).unsqueeze(1)
            # x = x + cls_emb
            x = torch.cat([x, cls_emb], 1)

        x = self.drop(x)
        for i, block in enumerate(self.blocks):
            if self.exp_type == 't2i_cross':
                x = block(x, label)
            else:
                x = block(x)

        x = x[:, :self.block_size, :]
        logits = self.head(self.ln_f(x))
        return logits

if __name__ == '__main__':
    pass
