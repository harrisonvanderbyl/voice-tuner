from contextlib import nullcontext
import math
import os
from dataclasses import dataclass
from functools import partial, wraps
from math import ceil, prod
import random
from typing import Callable, List, Tuple

import huggingface_hub
import inquirer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange, reduce
from einx import get_at
from torch import Tensor, autocast, int32, nn
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from torch.utils.checkpoint import checkpoint


from einops import rearrange, pack, unpack

# helper functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# tensor helpers

def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

# main class

class FSQ(nn.Module):
    def __init__(
        self,
        levels: List[int],
        dim: int | None = None,
        num_codebooks = 1,
        keep_num_codebooks_dim: bool | None = None,
        scale: float | None = None,
        allowed_dtypes: Tuple[torch.dtype, ...] = (torch.float32, torch.float64),
        channel_first: bool = False,
        projection_has_bias: bool = True,
        return_indices = True,
        force_quantization_f32 = True
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent = False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent = False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        self.channel_first = channel_first

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim, bias = projection_has_bias) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim, bias = projection_has_bias) if has_projections else nn.Identity()

        self.has_projections = has_projections

        self.return_indices = return_indices
        if return_indices:
            self.codebook_size = self._levels.prod().item()
            implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
            self.register_buffer("implicit_codebook", implicit_codebook, persistent = False)

        self.allowed_dtypes = allowed_dtypes
        self.force_quantization_f32 = force_quantization_f32

    def bound(self, z, eps: float = 1e-3):
        """ Bound `z`, an array of shape (..., d). """
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z):
        """ Quantizes z, returns quantized zhat, same shape as z. """
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2 # Renormalize to [-1, 1].
        return quantized / half_width
    
    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width
    
    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def codes_to_indices(self, zhat):
        """ Converts a `code` to an index in the codebook. """
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_level_indices(self, indices):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def indices_to_codes(self, indices):
        """ Inverse of `codes_to_indices`. """
        assert exists(indices)

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        codes = self._indices_to_codes(indices)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        codes = self.project_out(codes)

        if is_img_or_video or self.channel_first:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def forward(self, z):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

        is_img_or_video = z.ndim >= 4
        need_move_channel_last = is_img_or_video or self.channel_first

        # standardize image or video into (batch, seq, dimension)

        if need_move_channel_last:
            z = rearrange(z, 'b d ... -> b ... d')
            z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        z = self.project_in(z)

        z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks)

        # whether to force quantization step to be full precision or not

        force_f32 = self.force_quantization_f32
        quantization_context = partial(autocast, 'cuda', enabled = False) if force_f32 else nullcontext

        with quantization_context():
            orig_dtype = z.dtype

            if force_f32 and orig_dtype not in self.allowed_dtypes:
                z = z.float()

            codes = self.quantize(z)

            # returning indices could be optional

            indices = None

            if self.return_indices:
                indices = self.codes_to_indices(codes)

            codes = rearrange(codes, 'b n c d -> b n (c d)')

            codes = codes.type(orig_dtype)

        # project out

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if need_move_channel_last:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')

            indices = maybe(unpack_one)(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim and self.return_indices:
            indices = maybe(rearrange)(indices, '... 1 -> ...')

        # return quantized output and indices

        return out, indices

# helper functions

def exists(val):
    return val is not None

def first(l):
    return l[0]

def default(val, d):
    return val if exists(val) else d

def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

# main class

class ResidualFSQ(nn.Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """

    def __init__(
        self,
        *,
        dim,
        levels: List[int],
        num_quantizers,
        quantize_dropout = False,
        quantize_dropout_cutoff_index = 0,
        quantize_dropout_multiple_of = 1,
        **kwargs
    ):
        super().__init__()
        codebook_dim = len(levels)

        requires_projection = codebook_dim != dim
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection

        self.num_quantizers = num_quantizers

        self.levels = levels
        self.layers = nn.ModuleList([])

        levels_tensor = torch.Tensor(levels)

        scales = []

        for ind in range(num_quantizers):
            scales.append((levels_tensor - 1) ** -ind)

            fsq = FSQ(
                levels = levels,
                dim = codebook_dim,
                **kwargs
            )

            self.layers.append(fsq)

        assert all([not fsq.has_projections for fsq in self.layers])

        self.codebook_size = self.layers[0].codebook_size

        self.register_buffer('scales', torch.stack(scales), persistent = False)

        self.quantize_dropout = quantize_dropout and num_quantizers > 1

        assert quantize_dropout_cutoff_index >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of  # encodec paper proposes structured dropout, believe this was set to 4

    @property
    def codebooks(self):
        codebooks = [layer.implicit_codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim = 0)
        return codebooks

    def get_codes_from_indices(self, indices):

        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # may also receive indices in the shape of 'b h w q' (accept_image_fmap)

        indices, ps = pack([indices], 'b * q')

        # because of quantize dropout, one can pass in indices that are coarse
        # and the network should be able to reconstruct

        if quantize_dim < self.num_quantizers:
            assert self.quantize_dropout > 0., 'quantize dropout must be greater than 0 if you wish to reconstruct from a signal with less fine quantizations'
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value = -1)

        # take care of quantizer dropout

        mask = indices == -1
        indices = indices.masked_fill(mask, 0) # have it fetch a dummy code to be masked out later

        all_codes = get_at('q [c] d, b n q -> q b n d', self.codebooks, indices)

        # mask out any codes that were dropout-ed

        all_codes = all_codes.masked_fill(rearrange(mask, 'b n q -> q b n 1'), 0.)

        # scale the codes

        scales = rearrange(self.scales, 'q d -> q 1 1 d')
        all_codes = all_codes * scales

        # if (accept_image_fmap = True) then return shape (quantize, batch, height, width, dimension)

        all_codes, = unpack(all_codes, ps, 'q b * d')

        return all_codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        codes_summed = reduce(codes, 'q ... -> ...', 'sum')
        return self.project_out(codes_summed)

    def forward(
        self,
        x,
        return_all_codes = False,
        rand_quantize_dropout_fixed_seed = None
    ):
        num_quant, quant_dropout_multiple_of, device = self.num_quantizers, self.quantize_dropout_multiple_of, x.device

        x = self.project_in(x)

        quantized_out = 0.
        residual = first(self.layers).bound(x)

        all_indices = []

        should_quantize_dropout = self.training and self.quantize_dropout

        # sample a layer index at which to dropout further residual quantization
        # also prepare null indices

        if should_quantize_dropout:
            rand = random.Random(rand_quantize_dropout_fixed_seed) if exists(rand_quantize_dropout_fixed_seed) else random

            rand_quantize_dropout_index = rand.randrange(self.quantize_dropout_cutoff_index, num_quant)

            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = round_up_multiple(rand_quantize_dropout_index + 1, quant_dropout_multiple_of) - 1

            null_indices = torch.full(x.shape[:2], -1., device = device, dtype = torch.long)

        # go through the layers

        with autocast('cuda', enabled = False):
            for quantizer_index, (layer, scale) in enumerate(zip(self.layers, self.scales)):

                if should_quantize_dropout and quantizer_index > rand_quantize_dropout_index:
                    all_indices.append(null_indices)
                    continue

                quantized, indices = layer(residual / scale)
                quantized = quantized * scale

                residual = residual - quantized.detach()
                quantized_out = quantized_out + quantized

                all_indices.append(indices)

        # project out, if needed

        quantized_out = self.project_out(quantized_out)

        # stack all indices

        all_indices = torch.stack(all_indices, dim = -1)

        ret = (quantized_out, all_indices)

        if not return_all_codes:
            return ret

        # whether to return all codes from all codebooks across layers

        all_codes = self.get_codes_from_indices(all_indices)

        # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)

        return (*ret, all_codes)

class GroupedResidualFSQ(nn.Module):
    def __init__(
        self,
        *,
        dim,
        groups = 1,
        accept_image_fmap = False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.groups = groups
        assert (dim % groups) == 0
        dim_per_group = dim // groups

        self.accept_image_fmap = accept_image_fmap

        self.rvqs = nn.ModuleList([])

        for _ in range(groups):
            self.rvqs.append(ResidualFSQ(
                dim = dim_per_group,
                **kwargs
            ))

        self.codebook_size = self.rvqs[0].codebook_size

    @property
    def codebooks(self):
        return torch.stack(tuple(rvq.codebooks for rvq in self.rvqs))

    @property
    def split_dim(self):
        return 1 if self.accept_image_fmap else -1

    def get_codes_from_indices(self, indices):
        codes = tuple(rvq.get_codes_from_indices(chunk_indices) for rvq, chunk_indices in zip(self.rvqs, indices))
        return torch.stack(codes)

    def get_output_from_indices(self, indices):
        outputs = tuple(rvq.get_output_from_indices(chunk_indices) for rvq, chunk_indices in zip(self.rvqs, indices))
        return torch.cat(outputs, dim = self.split_dim)

    def forward(
        self,
        x,
        return_all_codes = False
    ):
        shape, split_dim = x.shape, self.split_dim
        assert shape[split_dim] == self.dim

        # split the feature dimension into groups

        x = x.chunk(self.groups, dim = split_dim)

        forward_kwargs = dict(
            return_all_codes = return_all_codes,
            rand_quantize_dropout_fixed_seed = random.randint(0, int(1e7))
        )

        # invoke residual vq on each group

        out = tuple(rvq(chunk, **forward_kwargs) for rvq, chunk in zip(self.rvqs, x))
        out = tuple(zip(*out))

        # otherwise, get all the zipped outputs and combine them

        quantized, all_indices, *maybe_all_codes = out

        quantized = torch.cat(quantized, dim = split_dim)
        all_indices = torch.stack(all_indices)

        ret = (quantized, all_indices, *maybe_all_codes)
        return ret


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv1D") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


def unpad1d(x: torch.Tensor, paddings: tuple[int, int]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


def get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    """See `pad_for_conv1d`."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad1d(
    x: torch.Tensor,
    paddings: tuple[int, int],
    mode: str = "zeros",
    value: float = 0.0,
):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right
    before the reflection happen.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


class PineConvNet(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, dilation=1, stride=1, groups=1
    ):
        super(PineConvNet, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.dilation = dilation

    def forward(self, x):
        pad = self.kernel_size - self.stride
        extra_padding = get_extra_padding_for_conv1d(
            x, self.kernel_size, self.stride, pad
        )
        x = pad1d(x, (pad, extra_padding), mode="constant", value=0)
        return self.conv(x).contiguous()

    def weight_norm(self, name="weight", dim=0):
        self.conv = weight_norm(self.conv, name=name, dim=dim)
        return self

    def remove_weight_norm(self):
        self.conv = remove_parametrizations(self.conv)
        return self


class PineTransConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1):
        super(PineTransConvNet, self).__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation
        )
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        x = self.conv(x)
        pad = self.kernel_size - self.stride
        padding_right = math.ceil(pad)
        padding_left = pad - padding_right
        x = unpad1d(x, (padding_left, padding_right))
        return x.contiguous()

    def weight_norm(self, name="weight", dim=0):
        self.conv = weight_norm(self.conv, name=name, dim=dim)
        return self

    def remove_weight_norm(self):
        self.conv = remove_parametrizations(self.conv)
        return self


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()

        self.convs1 = nn.ModuleList(
            [
                PineConvNet(
                    channels, channels, kernel_size, stride=1, dilation=dilation[0]
                ).weight_norm(),
                PineConvNet(
                    channels, channels, kernel_size, stride=1, dilation=dilation[1]
                ).weight_norm(),
                PineConvNet(
                    channels, channels, kernel_size, stride=1, dilation=dilation[2]
                ).weight_norm(),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                PineConvNet(
                    channels, channels, kernel_size, stride=1, dilation=dilation[0]
                ).weight_norm(),
                PineConvNet(
                    channels, channels, kernel_size, stride=1, dilation=dilation[1]
                ).weight_norm(),
                PineConvNet(
                    channels, channels, kernel_size, stride=1, dilation=dilation[2]
                ).weight_norm(),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.silu(x)
            xt = c1(xt)
            xt = F.silu(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_parametrizations(self):
        for conv in self.convs1:
            remove_parametrizations(conv, tensor_name="weight")
        for conv in self.convs2:
            remove_parametrizations(conv, tensor_name="weight")


class ParallelBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_sizes: tuple[int] = (3, 7, 11),
        dilation_sizes: tuple[tuple[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()

        assert len(kernel_sizes) == len(dilation_sizes)

        self.blocks = nn.ModuleList()
        for k, d in zip(kernel_sizes, dilation_sizes):
            self.blocks.append(ResBlock1(channels, k, d))

    def forward(self, x):
        return torch.stack([block(x) for block in self.blocks], dim=0).mean(dim=0)

    def remove_parametrizations(self):
        for block in self.blocks:
            block.remove_parametrizations()


class HiFiGANGenerator(nn.Module):
    def __init__(
        self,
        *,
        hop_length: int = 512,
        upsample_rates: tuple[int] = (8, 8, 2, 2, 2),
        upsample_kernel_sizes: tuple[int] = (16, 16, 8, 2, 2),
        resblock_kernel_sizes: tuple[int] = (3, 7, 11),
        resblock_dilation_sizes: tuple[tuple[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        num_mels: int = 128,
        upsample_initial_channel: int = 512,
        pre_conv_kernel_size: int = 7,
        post_conv_kernel_size: int = 7,
        post_activation: Callable = partial(nn.SiLU, inplace=True),
    ):
        super().__init__()

        assert (
            prod(upsample_rates) == hop_length
        ), f"hop_length must be {prod(upsample_rates)}"

        self.conv_pre = PineConvNet(
            num_mels,
            upsample_initial_channel,
            pre_conv_kernel_size,
            stride=1,
        ).weight_norm()

        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)

        self.noise_convs = nn.ModuleList()
        self.ups = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                PineTransConvNet(
                    upsample_initial_channel // (2**i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k,
                    stride=u,
                ).weight_norm()
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            self.resblocks.append(
                ParallelBlock(ch, resblock_kernel_sizes, resblock_dilation_sizes)
            )

        self.activation_post = post_activation()
        self.conv_post = PineConvNet(
            ch, 1, post_conv_kernel_size, stride=1
        ).weight_norm()
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.silu(x, inplace=True)
            x = self.ups[i](x)

            if self.training and self.checkpointing:
                x = checkpoint(
                    self.resblocks[i],
                    x,
                    use_reentrant=False,
                )
            else:
                x = self.resblocks[i](x)

        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_parametrizations(self):
        for up in self.ups:
            remove_parametrizations(up, tensor_name="weight")
        for block in self.resblocks:
            block.remove_parametrizations()
        remove_parametrizations(self.conv_pre, tensor_name="weight")
        remove_parametrizations(self.conv_post, tensor_name="weight")


# DropPath copied from timm library
def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """  # noqa: E501

    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""  # noqa: E501

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """  # noqa: E501

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x


class ConvNeXtBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        mlp_ratio: float = 4.0,
        kernel_size: int = 7,
        dilation: int = 1,
    ):
        super().__init__()

        self.dwconv = PineConvNet(
            dim,
            dim,
            kernel_size=kernel_size,
            # padding=int(dilation * (kernel_size - 1) / 2),
            groups=dim,
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, int(mlp_ratio * dim)
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, apply_residual: bool = True):
        input = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
        x = self.drop_path(x)

        if apply_residual:
            x = input + x

        return x


class ConvNeXtEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        depths: list[int] = [3, 3, 9, 3],
        dims: list[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        kernel_size: int = 7,
    ):
        super().__init__()
        assert len(depths) == len(dims)

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            PineConvNet(
                input_channels,
                dims[0],
                kernel_size=7,
                # padding=3,
                # padding_mode="replicate",
                # padding_mode="zeros",
            ),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)

        for i in range(len(depths) - 1):
            mid_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=1),
            )
            self.downsample_layers.append(mid_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        kernel_size=kernel_size,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(len(self.downsample_layers)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        return self.norm(x)


class FireflyArchitecture(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        quantizer: nn.Module,
        spec_transform: nn.Module,
    ):
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.quantizer = quantizer
        self.spec_transform = spec_transform
        self.downsample_factor = math.prod(self.quantizer.downsample_factor)

    def forward(self, x: torch.Tensor, template=None, mask=None) -> torch.Tensor:
        if self.spec_transform is not None:
            x = self.spec_transform(x)

        x = self.backbone(x)
        if mask is not None:
            x = x * mask

        if self.quantizer is not None:
            vq_result = self.quantizer(x)
            x = vq_result.z

            if mask is not None:
                x = x * mask

        x = self.head(x, template=template)

        if x.ndim == 2:
            x = x[:, None, :]

        if self.vq is not None:
            return x, vq_result

        return x

    def encode(self, audios, audio_lengths):
        audios = audios.float()

        mels = self.spec_transform(audios)
        mel_lengths = audio_lengths // self.spec_transform.hop_length
        mel_masks = sequence_mask(mel_lengths, mels.shape[2])
        mel_masks_float_conv = mel_masks[:, None, :].float()
        mels = mels * mel_masks_float_conv

        # Encode
        encoded_features = self.backbone(mels) * mel_masks_float_conv
        feature_lengths = mel_lengths // self.downsample_factor

        return self.quantizer.encode(encoded_features), feature_lengths

    def decode(self, indices, feature_lengths) -> torch.Tensor:
        mel_masks = sequence_mask(
            feature_lengths * self.downsample_factor,
            indices.shape[2] * self.downsample_factor,
        )
        mel_masks_float_conv = mel_masks[:, None, :].float()
        audio_lengths = (
            feature_lengths * self.downsample_factor * self.spec_transform.hop_length
        )

        audio_masks = sequence_mask(
            audio_lengths,
            indices.shape[2] * self.downsample_factor * self.spec_transform.hop_length,
        )
        audio_masks_float_conv = audio_masks[:, None, :].float()

        z = self.quantizer.decode(indices) * mel_masks_float_conv
        x = self.head(z) * audio_masks_float_conv

        return x, audio_lengths

    def remove_parametrizations(self):
        if hasattr(self.backbone, "remove_parametrizations"):
            self.backbone.remove_parametrizations()

        if hasattr(self.head, "remove_parametrizations"):
            self.head.remove_parametrizations()

    @property
    def device(self):
        return next(self.parameters()).device


class LinearSpectrogram(nn.Module):
    def __init__(
        self,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        center=False,
        mode="pow2_sqrt",
    ):
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.mode = mode

        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, y: Tensor) -> Tensor:
        if y.ndim == 3:
            y = y.squeeze(1)

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                (self.win_length - self.hop_length) // 2,
                (self.win_length - self.hop_length + 1) // 2,
            ),
            mode="reflect",
        ).squeeze(1)

        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        spec = torch.view_as_real(spec)

        if self.mode == "pow2_sqrt":
            spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

        return spec


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate=44100,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_mels=128,
        center=False,
        f_min=0.0,
        f_max=None,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or float(sample_rate // 2)

        self.spectrogram = LinearSpectrogram(n_fft, win_length, hop_length, center)

        fb = torchaudio.functional.melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=self.f_min,
            f_max=self.f_max,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        self.register_buffer(
            "fb",
            fb,
            persistent=False,
        )

    def compress(self, x: Tensor) -> Tensor:
        return torch.log(torch.clamp(x, min=1e-5))

    def decompress(self, x: Tensor) -> Tensor:
        return torch.exp(x)

    def apply_mel_scale(self, x: Tensor) -> Tensor:
        return torch.matmul(x.transpose(-1, -2), self.fb).transpose(-1, -2)

    def forward(
        self, x: Tensor, return_linear: bool = False, sample_rate: int = None
    ) -> Tensor:
        if sample_rate is not None and sample_rate != self.sample_rate:
            x = F.resample(x, orig_freq=sample_rate, new_freq=self.sample_rate)

        linear = self.spectrogram(x)
        x = self.apply_mel_scale(linear)
        x = self.compress(x)

        if return_linear:
            return x, self.compress(linear)

        return x



@dataclass
class FSQResult:
    z: torch.Tensor
    codes: torch.Tensor
    latents: torch.Tensor


class DownsampleFiniteScalarQuantize(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        n_groups: int = 1,
        levels: tuple[int] = (8, 5, 5, 5),  # Approximate 2**10
        downsample_factor: tuple[int] = (2, 2),
        downsample_dims: tuple[int] | None = None,
    ):
        super().__init__()

        if downsample_dims is None:
            downsample_dims = [input_dim for _ in range(len(downsample_factor))]

        all_dims = (input_dim,) + tuple(downsample_dims)

        self.residual_fsq = GroupedResidualFSQ(
            dim=all_dims[-1],
            levels=levels,
            num_quantizers=n_codebooks,
            groups=n_groups,
        )

        self.downsample_factor = downsample_factor
        self.downsample_dims = downsample_dims

        self.downsample = nn.Sequential(
            *[
                nn.Sequential(
                    PineConvNet(
                        all_dims[idx],
                        all_dims[idx + 1],
                        kernel_size=factor,
                        stride=factor,
                    ),
                    ConvNeXtBlock(dim=all_dims[idx + 1]),
                )
                for idx, factor in enumerate(downsample_factor)
            ]
        )

        self.upsample = nn.Sequential(
            *[
                nn.Sequential(
                    PineTransConvNet(
                        all_dims[idx + 1],
                        all_dims[idx],
                        kernel_size=factor,
                        stride=factor,
                    ),
                    ConvNeXtBlock(dim=all_dims[idx]),
                )
                for idx, factor in reversed(list(enumerate(downsample_factor)))
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, z) -> FSQResult:
        original_shape = z.shape
        z = self.downsample(z)
        quantized, indices = self.residual_fsq(z.mT)
        result = FSQResult(
            z=quantized.mT,
            codes=indices.mT,
            latents=z,
        )
        result.z = self.upsample(result.z)

        # Pad or crop z to match original shape
        diff = original_shape[-1] - result.z.shape[-1]
        left = diff // 2
        right = diff - left

        if diff > 0:
            result.z = F.pad(result.z, (left, right))
        elif diff < 0:
            result.z = result.z[..., left:-right]

        return result

    def encode(self, z):
        z = self.downsample(z)
        _, indices = self.residual_fsq(z.mT)
        indices = rearrange(indices, "g b l r -> b (g r) l")
        return indices

    def decode(self, indices: torch.Tensor):
        indices = rearrange(indices, "b (g r) l -> g b l r", g=self.residual_fsq.groups)
        z_q = self.residual_fsq.get_output_from_indices(indices)
        z_q = self.upsample(z_q.mT)
        return z_q

from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Initialize whisper processor and model
whisperprocessor = WhisperProcessor.from_pretrained("openai/whisper-small")
whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
whisper.config.forced_decoder_ids = None

# Define the model loading function
def load_model(checkpoint_path, device="cuda"):
    model = FireflyArchitecture(
        backbone=ConvNeXtEncoder(
            input_channels=160,
            depths=[3, 3, 9, 3],
            dims=[128, 256, 384, 512],
            drop_path_rate=0.2,
            kernel_size=7,
        ),
        head=HiFiGANGenerator(
            hop_length=512,
            upsample_rates=[8, 8, 2, 2, 2],
            upsample_kernel_sizes=[16, 16, 4, 4, 4],
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            num_mels=512,
            upsample_initial_channel=512,
            pre_conv_kernel_size=13,
            post_conv_kernel_size=13,
        ),
        quantizer=DownsampleFiniteScalarQuantize(
            input_dim=512,
            n_groups=8,
            n_codebooks=1,
            levels=[8, 5, 5, 5],
            downsample_factor=[2, 2],
        ),
        spec_transform=LogMelSpectrogram(
            sample_rate=44100,
            n_mels=160,
            n_fft=2048,
            hop_length=512,
            win_length=2048,
        ),
    )
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    if any("generator" in k for k in state_dict):
        state_dict = {k.replace("generator.", ""): v for k, v in state_dict.items() if "generator." in k}
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    
    return model


huggingface_hub = huggingface_hub.HfApi()
path = huggingface_hub.hf_hub_download("Hazzzardous/audio_embed", "audioembed.pth")

# Load the model
model = load_model(path, device="cuda")

# Load audio files and process them
files = os.listdir("./")
files = [f for f in files if f.endswith(".wav")]

# choose using cmd choice menu

questions = [
    inquirer.List("file",
                  message="Choose a file to process",
                  choices=files,
              ),
]
answers = inquirer.prompt(questions)

f = answers["file"]

audio, sr = torchaudio.load(str(f))

# Resample to 16 kHz
resampled, *_ = torchaudio.transforms.Resample(sr, 16000)(audio)

# Decode using Whisper
txt = whisperprocessor.batch_decode(
    whisper.generate(
        whisperprocessor(resampled, sampling_rate=16000, return_tensors="pt").input_features
    ), skip_special_tokens=True
)[0]

# If stereo, convert to mono
if audio.shape[0] > 1:
    audio = audio.mean(0, keepdim=True)

# Resample to match model input
audio = torchaudio.functional.resample(audio, sr, model.spec_transform.sample_rate)

# Normalize audio to [-1, 1] range
audio = audio / audio.abs().max()

# Apply volume scaling (set the desired scaling factor here)
volume_scaling_factor = 0.75 # You can adjust this value
audio = audio * volume_scaling_factor    

# Process audio with the model
audios = audio[None].to("cuda")
audio_lengths = torch.tensor([audios.shape[2]], device="cuda", dtype=torch.long)

# VQ Encoder
indices = model.encode(audios, audio_lengths)[0][0]

# create archive
dictionary = {
    "indices": indices.cpu().detach().numpy(),
    "txt": txt,
}

# upload to huggingface

myname = huggingface_hub.whoami()["name"]
name = input("Enter the name of the new voice: ")
huggingface_hub.create_repo(myname + "/"+name)
torch.save(dictionary, "voice.npz")
huggingface_hub.upload_file(path_or_fileobj="./voice.npz", path_in_repo="voice.npz", repo_id=myname + "/"+name)
print("Voice uploaded successfully")
# Optionally, print out some diagnostics
print(f"Processed {f}")
print(f"Text: {txt}")
print(f"Max value after scaling: {audio.max()}")
print(f"Min value after scaling: {audio.min()}")
