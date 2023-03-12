import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from local_attention import LocalMHA
from vector_quantize_pytorch import VectorQuantize

from beartype import beartype
from beartype.typing import Tuple, Optional, Union

from pathlib import Path
import pickle

# helpers

def exists(val):
    return val is not None

def first(it):
    return it[0]

def default(*vals):
    for val in vals:
        if exists(val):
            return val
    return None

def divisible_by(numer, denom):
    return (numer % denom) == 0

def cast_tuple(t, len = 1):
    return ((t,) * len) if not isinstance(t, tuple) else t

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

# the best kind of down and upsampling

class Upsample(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        factor = 2
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        linear = nn.Linear(dim, dim_out * factor)

        self.net = nn.Sequential(
            linear,
            nn.SiLU(),
            Rearrange('b n (p d) -> b (n p) d', p = factor)
        )

        self.factor = factor
        self.init_(linear)

    def init_(self, linear):
        o, i = linear.weight.shape

        linear_weight = torch.empty(o // self.factor, i)
        nn.init.kaiming_uniform_(linear_weight)

        linear_weight = repeat(linear_weight, 'o ... -> (o r) ...', r = self.factor)

        linear_weight.data.copy_(linear_weight)
        nn.init.zeros_(linear.bias.data)

    def forward(self, x):
        return self.net(x)

def Downsample(
    dim,
    dim_out = None,
    factor = 2
):
    dim_out = default(dim_out, dim)
    return nn.Sequential(
        Rearrange('b (n p) d -> b n (p d)', p = factor),
        nn.Linear(dim * factor, dim_out)
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
        strides: Union[int, Tuple[int, ...]],
        codebook_size = 1024,
        local_attn_window_size = 32,
        local_attn_heads = 8,
        local_attn_dim_head = 64,
        num_codebooks = 4,
        vq_decay = 0.9
    ):
        super().__init__()

        config = locals()
        config.pop('self')
        config.pop('__class__')
        self._config = config

        assert 0 < vq_decay <= 1.
        assert divisible_by(dim, num_codebooks)

        strides = cast_tuple(strides)
        num_layers = len(strides)

        depth = cast_tuple(depth, num_layers)
        local_attn_window_size = cast_tuple(local_attn_window_size, num_layers)

        assert num_layers == len(depth) == len(local_attn_window_size)

        self.token_emb = nn.Embedding(num_tokens, dim)

        self.total_strides = torch.tensor(list(strides)).cumprod(dim = -1)[-1].item()

        self.encoder = nn.ModuleList([])

        layer_params = tuple(zip(
            strides,
            depth,
            local_attn_window_size
        ))

        self.init_transformer = LocalTransformer(
            dim = dim,
            depth = first(depth),
            heads = local_attn_heads,
            dim_head = local_attn_dim_head,
            window_size = first(local_attn_window_size)
        )

        self.final_transformer = LocalTransformer(
            dim = dim,
            depth = first(depth),
            heads = local_attn_heads,
            dim_head = local_attn_dim_head,
            window_size = first(local_attn_window_size)
        )

        for layer_stride, layer_depth, layer_local_attn_window_size in layer_params:
            self.encoder.append(nn.ModuleList([
                Downsample(dim = dim, factor = layer_stride),
                LocalTransformer(
                    dim = dim,
                    depth = layer_depth,
                    heads = local_attn_heads,
                    dim_head = local_attn_dim_head,
                    window_size = layer_local_attn_window_size
                )
            ]))

        self.vq = VectorQuantize(
            dim = dim,
            codebook_size = codebook_size,
            heads = num_codebooks,  # use multi-headed vq, product quantization like
            decay = vq_decay,
            commitment_weight = 1.   # the weight on the commitment loss
        )

        self.decoder = nn.ModuleList([])

        for layer_stride, layer_depth, layer_local_attn_window_size in reversed(layer_params):
            self.decoder.append(nn.ModuleList([
                Upsample(dim = dim, factor = layer_stride),
                LocalTransformer(
                    dim = dim,
                    depth = layer_depth,
                    heads = local_attn_heads,
                    dim_head = local_attn_dim_head,
                    window_size = layer_local_attn_window_size
                )
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def save(self, path):
        path = Path(path)
        pkg = dict(
            model = self.state_dict(),
            config = pickle.dumps(self._config)
        )
        torch.save(pkg, str(path))

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path))
        self.load_state_dict(pkg['model'])

    @classmethod
    def init_and_load(cls, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path))
        model = cls(**pickle.loads(pkg['config']))
        model.load(path)
        return model

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, ids):
        tokens = self.token_emb(ids)

        tokens = self.final_transformer(tokens)

        for downsample, local_attn in self.encoder:
            tokens = downsample(tokens)
            tokens = local_attn(tokens)
        return tokens

    def decode(self, codes):
        tokens = codes

        for upsample, local_attn in self.decoder:
            tokens = local_attn(tokens)
            tokens = upsample(tokens)

        tokens = self.final_transformer(tokens)

        logits = self.to_logits(tokens)
        return logits

    @torch.no_grad()
    def decode_from_codebook_ids(self, codebook_ids):
        codes = self.vq.get_codes_from_indices(codebook_ids)
        return self.decode(codes)

    def forward(
        self,
        ids,
        return_codebook_indices = False,
        return_reconstruction = False
    ):
        batch, seq = ids.shape
        assert divisible_by(seq, self.total_strides)

        ids = ids.to(self.device)

        tokens = self.encode(ids)

        tokens, indices, commit_loss = self.vq(tokens)

        if return_codebook_indices:
            return indices

        logits = self.decode(tokens)

        logits = rearrange(logits, 'b n c -> b c n')

        loss = F.cross_entropy(
            logits,
            ids
        )

        loss =  loss + commit_loss

        if return_reconstruction:
            return loss, logits.argmax(dim = 1)

        return loss

# hierarchical transformer

class Transformer(nn.Module):
    pass
