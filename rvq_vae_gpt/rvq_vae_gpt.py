import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat, pack, unpack

from local_attention import LocalMHA
from vector_quantize_pytorch import VectorQuantize

from beartype import beartype
from beartype.typing import Tuple

# helpers

def exists(val):
    return val is not None

def divisible_by(numer, denom):
    return (numer % denom) == 0

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult * 2 / 3)

    return nn.Sequential(
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Linear(dim_inner, dim)
    )

# local attention

class LocalTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        dim_head,
        window_size
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LocalMHA(
                    dim = dim,
                    heads = heads,
                    dim_head = dim_head,
                    qk_rmsnorm = True,
                    window_size = window_size,
                    use_rotary_pos_emb = True,
                    use_xpos = True,
                    causal = True
                ),
                FeedForward(dim = dim)
            ]))

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x

# modules

@beartype
class TextVQVAE(nn.Module): # or genomics, eventually, with num_tokens set to 4
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        strides: Tuple[int],
        codebook_size = 1024,
        local_attn_window_size = 32,
        local_attn_heads = 8,
        local_attn_dim_head = 64,
        num_codebooks = 4
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.total_strides = torch.tensor(list(strides)).cumsum(dim = -1).item()

        self.encoder = nn.ModuleList([])

        for _ in strides:
            self.encoder.append(LocalTransformer(
                dim = dim,
                depth = depth,
                heads = local_attn_heads,
                dim_head = local_attn_dim_head,
                window_size = local_attn_window_size
            ))

        self.vq = VectorQuantize(
            dim = dim,
            codebook_size = codebook_size,
            heads = num_codebooks  # use multi-headed vq, product quantization like
        )

        self.decoder = nn.ModuleList([])

        for _ in strides:
            self.decoder.append(LocalTransformer(
                dim = dim,
                depth = depth,
                heads = local_attn_heads,
                dim_head = local_attn_dim_head,
                window_size = local_attn_window_size
            ))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        ids,
        return_codebook_indices = False
    ):
        batch, seq = ids.shape
        assert divisible_by(seq, self.total_strides)

        ids = ids.to(self.device)

        tokens = self.token_emb(ids)

        for local_attn in self.encoder:
            tokens = local_attn(tokens)

        tokens, indices, commit_loss = self.vq(tokens)

        if return_codebook_indices:
            return indices

        for local_attn in self.encoder:
            tokens = local_attn(tokens)

        logits = self.to_logits(tokens)

        logits = rearrange(logits, 'b n c -> b c n')

        loss = F.cross_entropy(
            logits,
            ids
        )

        return loss + commit_loss

# hierarchical transformer

class Transformer(nn.Module):
    pass
