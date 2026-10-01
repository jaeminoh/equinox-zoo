"""
FlashSlice JAX test.

The torch reference is the eager path of https://github.com/Shizheng-Wen/flashslice
(v0.3.0): `flashslice/layers/basic.py`, `flashslice/layers/physics_attention.py`
and `flashslice/models/transolver.py`. Copyright 2026 Shizheng Wen, Apache-2.0;
portions derive from Transolver (https://github.com/thuml/Transolver), Copyright
(c) 2024 THUML @ Tsinghua University, MIT (see the upstream NOTICE). Edits:
`use_fused_slice` and everything serving it (the Triton path, its checks and
fallback logging) are dropped, since the kernels need CUDA; the one einops
`rearrange` is spelled out as permute + reshape; the model docstring is
shortened. The forward math is unchanged.
"""

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


# -----------------------
# flashslice/layers/basic.py
# -----------------------
ACTIVATION = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(n_hidden, n_hidden), act())
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


# -----------------------
# flashslice/layers/physics_attention.py
# -----------------------
class Physics_Attention_Irregular_Mesh(nn.Module):
    ## for irregular meshes in 1D, 2D or 3D space
    # Ablation flags (defaults reproduce the original module exactly):
    #   untied_deslice: deslice uses its own projection/temperature instead of reusing slice_weights
    #   no_token_attention: skip attention among slice tokens (keeps to_v transform only)
    # Width decoupling (advisor's memory study):
    #   dim_head: full-resolution per-head width -> drives the heavy [B,N,heads*dim_head]
    #     activations (in_project_fx/x, slice_token, deslice, to_out). Smaller = less memory.
    #   slice_dim_head: width of the slice-token attention (on G tokens only, ~free in memory).
    #     None -> = dim_head (exact original). Lets a narrow full-res path keep a wide latent.
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        slice_num=64,
        shapelist=None,
        untied_deslice=False,
        no_token_attention=False,
        slice_dim_head=None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.slice_dim_head = slice_dim_head if slice_dim_head is not None else dim_head
        self.heads = heads
        self.scale = self.slice_dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.untied_deslice = untied_deslice
        self.no_token_attention = no_token_attention

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:  # noqa: E741
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        if untied_deslice:
            self.deslice_temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
            self.in_project_deslice = nn.Linear(dim_head, slice_num)
            torch.nn.init.orthogonal_(self.in_project_deslice.weight)
        # slice-token attention runs at slice_dim_head; project back to dim_head for deslice.
        self.to_q = nn.Linear(dim_head, self.slice_dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, self.slice_dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, self.slice_dim_head, bias=False)
        self.slice_down = (
            nn.Identity()
            if self.slice_dim_head == dim_head
            else nn.Linear(self.slice_dim_head, dim_head, bias=False)
        )
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, slice_weights_in=None):
        # B N C; slice_weights_in: externally provided slice weights (share-across-layers ablation)
        B, N, C = x.shape

        ### (1) Slice
        fx_mid = (
            self.in_project_fx(x)
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # B H N C
        if slice_weights_in is None or self.untied_deslice:
            x_mid = (
                self.in_project_x(x)
                .reshape(B, N, self.heads, self.dim_head)
                .permute(0, 2, 1, 3)
                .contiguous()
            )  # B H N C
        if slice_weights_in is None:
            slice_weights = self.softmax(
                self.in_project_slice(x_mid) / self.temperature
            )  # B H N G
        else:
            slice_weights = slice_weights_in
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (
            (slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head)
        )

        ### (2) Attention among slice tokens (at slice_dim_head width, on G tokens)
        if self.no_token_attention:
            out_slice_token = self.to_v(
                slice_token
            )  # B H G slice_dim_head, no inter-token mixing
        else:
            q_slice_token = self.to_q(slice_token)
            k_slice_token = self.to_k(slice_token)
            v_slice_token = self.to_v(slice_token)
            dots = (
                torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2))
                * self.scale
            )
            attn = self.softmax(dots)
            attn = self.dropout(attn)
            out_slice_token = torch.matmul(attn, v_slice_token)  # B H G slice_dim_head
        out_slice_token = self.slice_down(
            out_slice_token
        )  # B H G dim_head (Identity if equal)

        ### (3) Deslice
        if self.untied_deslice:
            deslice_weights = self.softmax(
                self.in_project_deslice(x_mid) / self.deslice_temperature
            )
        else:
            deslice_weights = slice_weights
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, deslice_weights)
        out_x = out_x.permute(0, 2, 1, 3).reshape(B, N, -1)  # 'b h n d -> b n (h d)'
        return self.to_out(out_x), slice_weights


# -----------------------
# flashslice/models/transolver.py
# -----------------------
class TransolverBlock(nn.Module):
    """One encoder block: physics-attention residual, then a pointwise MLP residual."""

    def __init__(
        self,
        num_heads,
        hidden_dim,
        dropout,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=32,
        untied_deslice=False,
        no_token_attention=False,
        mlp_only=False,
        dim_head=None,
        slice_dim_head=None,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.mlp_only = mlp_only
        self.ln_1 = nn.LayerNorm(hidden_dim)
        if not self.mlp_only:
            dh = dim_head if dim_head is not None else hidden_dim // num_heads
            self.Attn = Physics_Attention_Irregular_Mesh(
                hidden_dim,
                heads=num_heads,
                dim_head=dh,
                dropout=dropout,
                slice_num=slice_num,
                untied_deslice=untied_deslice,
                no_token_attention=no_token_attention,
                slice_dim_head=slice_dim_head,
            )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(
            hidden_dim,
            hidden_dim * mlp_ratio,
            hidden_dim,
            n_layers=0,
            res=False,
            act=act,
        )
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx, slice_weights_in=None):
        # Returns (fx, slice_weights). slice_weights is None under mlp_only, which
        # has no slice/deslice at all.
        slice_weights = None
        if not self.mlp_only:
            attn_out, slice_weights = self.Attn(
                self.ln_1(fx), slice_weights_in=slice_weights_in
            )
            fx = attn_out + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx)), slice_weights
        return fx, slice_weights


class TokenTransformerBlock(nn.Module):
    """Pre-LN transformer block over slice tokens; used only by ``slice_once``."""

    def __init__(self, num_heads, hidden_dim, dropout, act="gelu", mlp_ratio=4):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(
            hidden_dim,
            hidden_dim * mlp_ratio,
            hidden_dim,
            n_layers=0,
            res=False,
            act=act,
        )

    def forward(self, tokens):
        h = self.ln_1(tokens)
        tokens = self.attn(h, h, h, need_weights=False)[0] + tokens
        tokens = self.mlp(self.ln_2(tokens)) + tokens
        return tokens


class Transolver(nn.Module):
    """Transolver on unstructured point clouds."""

    def __init__(
        self,
        space_dim=3,
        fun_dim=1,
        out_dim=1,
        n_hidden=256,
        n_heads=8,
        n_layers=8,
        slice_num=32,
        mlp_ratio=2,
        dropout=0.0,
        act="gelu",
        dim_head=None,
        slice_dim_head=None,
        untie_slice_weights=False,
        share_slice_across_layers=False,
        slice_once=False,
        no_token_attention=False,
        mlp_only=False,
    ):
        super().__init__()
        self.n_hidden = n_hidden
        self.preprocess = MLP(
            fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act
        )

        self.untied_deslice = untie_slice_weights
        self.share_slice_across_layers = share_slice_across_layers
        self.slice_once = slice_once
        self.no_token_attention = no_token_attention
        self.mlp_only = mlp_only
        n_flags = sum(
            [
                untie_slice_weights,
                share_slice_across_layers,
                slice_once,
                no_token_attention,
                mlp_only,
            ]
        )
        if n_flags > 1:
            raise ValueError("At most one ablation flag may be enabled at a time.")

        if slice_once:
            # Perceiver limit: one slice, all depth in token space, one deslice.
            self.slice_proj = nn.Linear(n_hidden, slice_num)
            self.slice_temperature = nn.Parameter(torch.tensor(0.5))
            self.token_blocks = nn.ModuleList(
                [
                    TokenTransformerBlock(
                        num_heads=n_heads,
                        hidden_dim=n_hidden,
                        dropout=dropout,
                        act=act,
                        mlp_ratio=mlp_ratio,
                    )
                    for _ in range(n_layers)
                ]
            )
            self.out_norm = nn.LayerNorm(n_hidden)
            self.out_head = nn.Linear(n_hidden, out_dim)
        else:
            self.blocks = nn.ModuleList(
                [
                    TransolverBlock(
                        num_heads=n_heads,
                        hidden_dim=n_hidden,
                        dropout=dropout,
                        act=act,
                        mlp_ratio=mlp_ratio,
                        out_dim=out_dim,
                        slice_num=slice_num,
                        last_layer=(i == n_layers - 1),
                        untied_deslice=untie_slice_weights,
                        no_token_attention=no_token_attention,
                        mlp_only=mlp_only,
                        dim_head=dim_head,
                        slice_dim_head=slice_dim_head,
                    )
                    for i in range(n_layers)
                ]
            )

        self.placeholder = nn.Parameter(
            (1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float)
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _slice_once_forward(self, fx):
        slice_weights = torch.softmax(
            self.slice_proj(fx) / self.slice_temperature, dim=-1
        )
        slice_norm = slice_weights.sum(1)
        tokens = torch.einsum("bnc,bng->bgc", fx, slice_weights)
        tokens = tokens / (slice_norm + 1e-5).unsqueeze(-1)
        for block in self.token_blocks:
            tokens = block(tokens)
        out = torch.einsum("bgc,bng->bnc", tokens, slice_weights)
        out = out + fx  # point-level skip around the token stack
        return self.out_head(self.out_norm(out))

    def forward(self, x, fx=None):
        fx = self.preprocess(torch.cat((x, fx), dim=-1) if fx is not None else x)
        fx = fx + self.placeholder[None, None, :]

        if self.slice_once:
            return self._slice_once_forward(fx)

        shared_weights = None
        for block in self.blocks:
            fx, slice_weights = block(fx, slice_weights_in=shared_weights)
            if self.share_slice_across_layers and shared_weights is None:
                shared_weights = slice_weights
        return fx


# ---------------------------------------------------------------------------
# JAX side
# ---------------------------------------------------------------------------
import jax  # noqa: E402

# The parity tests compare against float64 PyTorch, so roundoff cannot hide a
# structural mismatch. Must be set before any array is created.
jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from zoo._flashslice import (  # noqa: E402
    PhysicsAttentionIrregularMesh,
    Transolver as JaxTransolver,
    fused_deslice,
    fused_slice,
    init_weights,
)

TOL = 1e-10


def assert_close(got, expected, tol=TOL):
    """Max abs difference within `tol`, relative to the reference's scale."""
    got, expected = np.asarray(got), np.asarray(expected)
    assert got.shape == expected.shape
    scale = max(1.0, float(np.abs(expected).max(initial=0.0)))
    assert np.abs(got - expected).max(initial=0.0) <= tol * scale


# ---------------------------------------------------------------------------
# The ops against a materialized reference
# ---------------------------------------------------------------------------
B, N, H, D, DV, G = 2, 37, 3, 8, 6, 5


def dense_weights(x, w, b, tau):
    w = jnp.broadcast_to(w, (x.shape[0], x.shape[2]) + w.shape[-2:])
    logits = jnp.einsum("bnhd,bhgd->bhng", x, w)
    if b is not None:
        logits = logits + jnp.broadcast_to(b, w.shape[:3])[:, :, None, :]
    return jax.nn.softmax(logits / tau[:, None, None], axis=-1)


def dense_slice(x, fx, w, b, tau, *, chunk_size):
    sw = dense_weights(x, w, b, tau)
    return jnp.einsum("bhng,bnhv->bhgv", sw, fx), sw.sum(axis=2)


def dense_deslice(x, w, b, tau, tokens, *, chunk_size):
    return jnp.einsum("bhng,bhgv->bnhv", dense_weights(x, w, b, tau), tokens)


def coupling(ops, order, leaves, mix, chunk_size):
    """One tied coupling round: slice first, its normalized tokens mixed and
    desliced; or deslice first on given tokens, then slice."""
    slice_op, deslice_op = ops
    x, fx, w, b, tau = (leaves[k] for k in ("x", "fx", "w", "b", "tau"))
    if order == "slice-first":
        z_num, s = slice_op(x, fx, w, b, tau, chunk_size=chunk_size)
        tokens = (z_num / (s + 1e-5)[..., None]) @ mix
        out = deslice_op(x, w, b, tau, tokens, chunk_size=chunk_size)
    else:
        out = deslice_op(x, w, b, tau, leaves["tokens"], chunk_size=chunk_size)
        z_num, s = slice_op(x, fx, w, b, tau, chunk_size=chunk_size)
    return out, z_num, s


FUSED = (fused_slice, fused_deslice)
DENSE = (dense_slice, dense_deslice)
WEIGHT_SHAPES = {"shared": (G, D), "per-head": (H, G, D), "per-sample": (B, H, G, D)}


def make_leaves(layout, with_bias, seed=0):
    keys = jr.split(jr.key(seed), 7)
    w_shape = WEIGHT_SHAPES[layout]
    return dict(
        x=jr.normal(keys[0], (B, N, H, D)),
        fx=jr.normal(keys[1], (B, N, H, DV)),
        w=jr.normal(keys[2], w_shape) / D**0.5,
        b=0.1 * jr.normal(keys[3], w_shape[:-1]) if with_bias else None,
        tau=0.7 + 0.1 * jr.normal(keys[4], (H,)),
        tokens=jr.normal(keys[5], (B, H, G, DV)),
    ), jr.normal(keys[6], (DV, DV)) / DV**0.5


@pytest.mark.parametrize("order", ["slice-first", "deslice-first"])
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("layout", list(WEIGHT_SHAPES))
def test_ops_match_dense(layout, with_bias, order):
    """Outputs and the custom VJP against autodiff of the materialized
    computation, with a ragged last tile (37 = 4 * 8 + 5 points)."""
    leaves, mix = make_leaves(layout, with_bias)
    k_out, k_z, k_s = jr.split(jr.key(1), 3)

    def loss(ops):
        def f(leaves):
            out, z_num, s = coupling(ops, order, leaves, mix, chunk_size=8)
            return (
                jnp.sum(out * jr.normal(k_out, out.shape))
                + jnp.sum(z_num * jr.normal(k_z, z_num.shape))
                + jnp.sum(s * jr.normal(k_s, s.shape))
            )

        return f

    for got, expected in zip(
        coupling(FUSED, order, leaves, mix, 8), coupling(DENSE, order, leaves, mix, 8)
    ):
        assert_close(got, expected)
    grads = jax.grad(loss(FUSED))(leaves), jax.grad(loss(DENSE))(leaves)
    jax.tree.map(assert_close, *grads)
    # Every leaf gets a gradient in its own shape, the broadcast ones included.
    assert jax.tree.map(jnp.shape, grads[0]) == jax.tree.map(jnp.shape, leaves)


@pytest.mark.parametrize("chunk_size", [1, 8, 37, 64])
def test_ops_tiling_is_invisible(chunk_size):
    """One-point tiles, a ragged tail, one exact tile, a tile larger than N."""
    leaves, mix = make_leaves("per-head", True, seed=2)

    def loss(ops, chunk):
        return lambda lv: sum(
            jnp.sum(jnp.sin(t)) for t in coupling(ops, "slice-first", lv, mix, chunk)
        )

    got = jax.value_and_grad(loss(FUSED, chunk_size))(leaves)
    expected = jax.value_and_grad(loss(DENSE, chunk_size))(leaves)
    jax.tree.map(assert_close, got, expected)


def _intermediate_shapes(jaxpr):
    """Shapes of every value `jaxpr` computes, inside loop bodies and other
    sub-jaxprs too."""
    for eqn in jaxpr.eqns:
        yield from (getattr(v.aval, "shape", None) for v in eqn.outvars)
        for param in eqn.params.values():
            for sub in param if isinstance(param, (tuple, list)) else (param,):
                sub = getattr(sub, "jaxpr", sub)  # ClosedJaxpr -> Jaxpr
                if hasattr(sub, "eqns"):
                    yield from _intermediate_shapes(sub)


def test_fused_never_materializes_slice_weights():
    """The point of FlashSlice: neither pass forms the (B, H, N, G) tensor,
    only (B, H, chunk_size, G) tiles of it."""
    leaves, mix = make_leaves("shared", True)
    chunk_size = 16

    def shapes(ops):
        def loss(lv):
            return sum(
                jnp.sum(t**2) for t in coupling(ops, "slice-first", lv, mix, chunk_size)
            )

        jaxpr = jax.make_jaxpr(jax.value_and_grad(loss))(leaves).jaxpr
        return set(_intermediate_shapes(jaxpr))

    fused, dense = shapes(FUSED), shapes(DENSE)
    assert (B, H, N, G) in dense  # the probe sees the tensor where it exists
    assert (B, H, N, G) not in fused
    assert {(B, H, chunk_size, G), (B, H, N % chunk_size, G)} <= fused


def test_ops_accumulate_16bit_inputs_in_fp32():
    """bf16 activations with fp32 parameters: pooled tokens come back fp32,
    the deslice output in the activations' dtype, and every gradient in its
    primal's dtype."""
    leaves, _ = make_leaves("shared", True)
    f32 = {k: v.astype(jnp.float32) for k, v in leaves.items()}
    lp = dict(f32, x=f32["x"].astype(jnp.bfloat16), fx=f32["fx"].astype(jnp.bfloat16))
    ops = lambda lv: coupling(FUSED, "deslice-first", lv, None, 8)  # noqa: E731

    out, z_num, s = ops(lp)
    assert (out.dtype, z_num.dtype, s.dtype) == (jnp.bfloat16, jnp.float32, jnp.float32)
    # Exact up to fp32 accumulation on the bf16-rounded inputs.
    upcast = dict(lp, x=lp["x"].astype(jnp.float32), fx=lp["fx"].astype(jnp.float32))
    assert_close(z_num, coupling(DENSE, "deslice-first", upcast, None, 8)[1], 1e-5)

    grads = jax.grad(lambda lv: sum(jnp.sum(t.astype(jnp.float32)) for t in ops(lv)))(
        lp
    )
    dtype = lambda a: a.dtype  # noqa: E731
    assert jax.tree.map(dtype, grads) == jax.tree.map(dtype, lp)


def test_ops_reject_bad_shapes():
    leaves, _ = make_leaves("shared", True)
    x, fx, w, b, tau = (leaves[k] for k in ("x", "fx", "w", "b", "tau"))
    with pytest.raises(ValueError, match="weight"):
        fused_slice(x, fx, w[:, :-1], b, tau)
    with pytest.raises(ValueError, match="bias"):
        fused_slice(x, fx, w, b[:-1], tau)
    with pytest.raises(ValueError, match="tau"):
        fused_slice(x, fx, w, b, tau[:-1])
    with pytest.raises(ValueError, match="chunk_size"):
        fused_slice(x, fx, w, b, tau, chunk_size=0)
    with pytest.raises(ValueError, match="tokens"):
        fused_deslice(x, w, b, tau, leaves["tokens"][:, :, :-1])


# ---------------------------------------------------------------------------
# Torch parameter names -> Equinox leaves
# ---------------------------------------------------------------------------
def at(get, *path):
    """Getter for the attribute (or list index) `path` below `get`."""

    def getter(m):
        m = get(m)
        for p in path:
            m = m[p] if isinstance(p, int) else getattr(m, p)
        return m

    return getter


def _linear(name, get, bias=True):
    names = ("weight", "bias") if bias else ("weight",)
    return [(f"{name}.{p}", at(get, p), None) for p in names]


def _mlp(name, get):
    # torch wraps the hidden layer in Sequential(Linear, act): index 0 is the Linear.
    return _linear(f"{name}.linear_pre.0", at(get, "linear_pre")) + _linear(
        f"{name}.linear_post", at(get, "linear_post")
    )


def _attention(name, get, attn):
    entries = [(f"{name}.temperature", at(get, "temperature"), None)]
    for proj in ("in_project_x", "in_project_fx", "in_project_slice"):
        entries += _linear(f"{name}.{proj}", at(get, proj))
    for proj in ("to_q", "to_k", "to_v"):
        entries += _linear(f"{name}.{proj}", at(get, proj), bias=False)
    if isinstance(attn.slice_down, eqx.nn.Linear):
        entries += _linear(f"{name}.slice_down", at(get, "slice_down"), bias=False)
    entries += _linear(f"{name}.to_out.0", at(get, "to_out"))
    if attn.untied_deslice:
        entries += [
            (f"{name}.deslice_temperature", at(get, "deslice_temperature"), None)
        ]
        entries += _linear(f"{name}.in_project_deslice", at(get, "in_project_deslice"))
    return entries


def _token_block(name, get, block):
    entries = _linear(f"{name}.ln_1", at(get, "ln_1"))
    entries += _linear(f"{name}.ln_2", at(get, "ln_2")) + _mlp(
        f"{name}.mlp", at(get, "mlp")
    )
    # torch fuses the query/key/value projections into one (3E, E) in-projection.
    E = block.attn.query_size
    for i, proj in enumerate(("query_proj", "key_proj", "value_proj")):
        rows = slice(i * E, (i + 1) * E)
        for p in ("weight", "bias"):
            entries.append((f"{name}.attn.in_proj_{p}", at(get, "attn", proj, p), rows))
    return entries + _linear(f"{name}.attn.out_proj", at(get, "attn", "output_proj"))


def param_map(model):
    """(torch parameter name, getter of the Equinox leaf, rows of the torch
    parameter that leaf holds) for every parameter of `model`."""
    model_ = lambda m: m  # noqa: E731
    entries = _mlp("preprocess", at(model_, "preprocess"))
    entries.append(("placeholder", at(model_, "placeholder"), None))
    if model.blocks is None:
        entries += _linear("slice_proj", at(model_, "slice_proj"))
        entries.append(("slice_temperature", at(model_, "slice_temperature"), None))
        entries += _linear("out_norm", at(model_, "out_norm"))
        entries += _linear("out_head", at(model_, "out_head"))
        for i, block in enumerate(model.token_blocks):
            entries += _token_block(
                f"token_blocks.{i}", at(model_, "token_blocks", i), block
            )
        return entries
    for i, block in enumerate(model.blocks):
        get, name = at(model_, "blocks", i), f"blocks.{i}"
        entries += _linear(f"{name}.ln_1", at(get, "ln_1"))
        entries += _linear(f"{name}.ln_2", at(get, "ln_2")) + _mlp(
            f"{name}.mlp", at(get, "mlp")
        )
        if block.attn is not None:
            entries += _attention(f"{name}.Attn", at(get, "attn"), block.attn)
        if block.mlp2 is not None:
            entries += _linear(f"{name}.ln_3", at(get, "ln_3"))
            entries += _linear(f"{name}.mlp2", at(get, "mlp2"))
    return entries


def t2j(x):
    return jnp.asarray(x.detach().numpy(), dtype=jnp.float64)


def copy_weights(jax_model, torch_model):
    """Overwrite every Equinox parameter with its PyTorch counterpart."""
    params = dict(torch_model.named_parameters())
    entries = param_map(jax_model)
    values = [
        t2j(params[name] if rows is None else params[name][rows])
        for name, _, rows in entries
    ]
    return eqx.tree_at(lambda m: [get(m) for _, get, _ in entries], jax_model, values)


# ---------------------------------------------------------------------------
# Parity with the PyTorch reference
# ---------------------------------------------------------------------------
SPACE_DIM, FUN_DIM, OUT_DIM = 3, 2, 2
NUM_LAYERS, HIDDEN_DIM, NUM_HEADS, NUM_SLICES, MLP_RATIO = 3, 32, 4, 6, 2
BATCH, NUM_POINTS, CHUNK = 2, 23, 8

# name -> (torch kwargs, Equinox kwargs)
CONFIGS = {
    "baseline": ({}, {}),
    "no_token_attention": ({"no_token_attention": True},) * 2,
    "untie_slice_weights": ({"untie_slice_weights": True},) * 2,
    "share_slice_across_layers": ({"share_slice_across_layers": True},) * 2,
    "slice_once": ({"slice_once": True},) * 2,
    "mlp_only": ({"mlp_only": True},) * 2,
    "head_dim": ({"dim_head": 6}, {"head_dim": 6}),
    "slice_head_dim": ({"slice_dim_head": 12}, {"slice_head_dim": 12}),
}
NOT_FUSABLE = {"share_slice_across_layers", "slice_once"}
CASES = [(name, False) for name in CONFIGS] + [
    (name, True) for name in CONFIGS if name not in NOT_FUSABLE
]


def make_pair(name, fused):
    """A float64 PyTorch reference and an Equinox model carrying its weights."""
    torch_kwargs, jax_kwargs = CONFIGS[name]
    torch.manual_seed(0)
    torch_model = Transolver(
        space_dim=SPACE_DIM,
        fun_dim=FUN_DIM,
        out_dim=OUT_DIM,
        n_hidden=HIDDEN_DIM,
        n_heads=NUM_HEADS,
        n_layers=NUM_LAYERS,
        slice_num=NUM_SLICES,
        mlp_ratio=MLP_RATIO,
        **torch_kwargs,
    )
    jax_model = JaxTransolver(
        space_dim=SPACE_DIM,
        fun_dim=FUN_DIM,
        out_dim=OUT_DIM,
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_slices=NUM_SLICES,
        mlp_ratio=MLP_RATIO,
        use_fused_slice=fused,
        chunk_size=CHUNK,
        key=jr.key(0),
        **jax_kwargs,
    )
    torch_model = torch_model.double().eval()
    return torch_model, copy_weights(jax_model, torch_model)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_param_map_covers_every_torch_parameter(name):
    torch_model, jax_model = make_pair(name, fused=False)
    names = [entry[0] for entry in param_map(jax_model)]
    assert set(names) == set(dict(torch_model.named_parameters()))


@pytest.mark.parametrize("name,fused", CASES)
def test_parity_forward_and_gradients(name, fused):
    """Output, and the gradient of a random projection of it with respect to
    the inputs and every parameter, against float64 PyTorch autograd."""
    torch_model, jax_model = make_pair(name, fused)
    torch.manual_seed(1)
    x = torch.randn(BATCH, NUM_POINTS, SPACE_DIM, dtype=torch.float64)
    fx = torch.randn(BATCH, NUM_POINTS, FUN_DIM, dtype=torch.float64)
    up = torch.randn(BATCH, NUM_POINTS, OUT_DIM, dtype=torch.float64)

    x.requires_grad_(True)
    fx.requires_grad_(True)
    expected = torch_model(x, fx)
    (expected * up).sum().backward()

    def loss(args):
        model, x_, fx_ = args
        out = model(x_, fx_, key=jr.key(0), inference=True)
        return jnp.sum(out * t2j(up)), out

    args = (jax_model, t2j(x), t2j(fx))
    grads, got = eqx.filter_grad(loss, has_aux=True)(args)

    assert_close(got, expected.detach().numpy())
    assert_close(grads[1], x.grad.numpy())
    assert_close(grads[2], fx.grad.numpy())
    params = dict(torch_model.named_parameters())
    for pname, get, rows in param_map(jax_model):
        # torch leaves .grad unset on parameters the forward does not reach
        # (ln_1 under mlp_only, the later layers' slice projections under
        # share_slice_across_layers); JAX returns zeros for them.
        expected_grad = params[pname].grad
        if expected_grad is None:
            expected_grad = torch.zeros_like(params[pname])
        if rows is not None:
            expected_grad = expected_grad[rows]
        assert_close(get(grads[0]), expected_grad.numpy())


def test_fused_switch_keeps_the_parameters():
    """`use_fused_slice` changes the implementation, not the model: the same
    key builds the same parameters, and both paths compute the same output."""
    kwargs = dict(
        space_dim=SPACE_DIM,
        fun_dim=FUN_DIM,
        num_layers=2,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_slices=NUM_SLICES,
        untie_slice_weights=True,
        chunk_size=CHUNK,
        key=jr.key(3),
    )
    eager = init_weights(JaxTransolver(**kwargs), key=jr.key(4))
    fused = init_weights(JaxTransolver(use_fused_slice=True, **kwargs), key=jr.key(4))
    # Leaf by leaf: the static flag itself makes the two treedefs differ.
    eager_leaves = jax.tree.leaves(eqx.filter(eager, eqx.is_array))
    fused_leaves = jax.tree.leaves(eqx.filter(fused, eqx.is_array))
    assert len(eager_leaves) == len(fused_leaves)
    for a, b in zip(eager_leaves, fused_leaves):
        np.testing.assert_array_equal(a, b)

    x = jr.normal(jr.key(5), (BATCH, NUM_POINTS, SPACE_DIM))
    fx = jr.normal(jr.key(6), (BATCH, NUM_POINTS, FUN_DIM))
    assert_close(
        fused(x, fx, key=jr.key(0), inference=True),
        eager(x, fx, key=jr.key(0), inference=True),
    )


# ---------------------------------------------------------------------------
# Standalone Equinox behaviour
# ---------------------------------------------------------------------------
def test_at_most_one_ablation_flag():
    with pytest.raises(ValueError, match="At most one"):
        JaxTransolver(no_token_attention=True, mlp_only=True, key=jr.key(0))


@pytest.mark.parametrize("flag", sorted(NOT_FUSABLE))
def test_fused_rejects_ablations_that_need_slice_weights(flag):
    with pytest.raises(ValueError, match="use_fused_slice"):
        JaxTransolver(use_fused_slice=True, key=jr.key(0), **{flag: True})


def test_fused_layer_rejects_external_slice_weights():
    attn = PhysicsAttentionIrregularMesh(
        HIDDEN_DIM,
        NUM_HEADS,
        HIDDEN_DIM // NUM_HEADS,
        NUM_SLICES,
        use_fused_slice=True,
        key=jr.key(0),
    )
    x = jr.normal(jr.key(1), (BATCH, NUM_POINTS, HIDDEN_DIM))
    weights = jnp.full((BATCH, NUM_HEADS, NUM_POINTS, NUM_SLICES), 1 / NUM_SLICES)
    with pytest.raises(ValueError, match="slice weights"):
        attn(x, weights, key=jr.key(2))


@pytest.mark.parametrize("flag", ["baseline", "slice_once"])
def test_init_weights_matches_reference_scheme(flag):
    model = JaxTransolver(
        hidden_dim=64, num_layers=2, key=jr.key(0), slice_once=(flag == "slice_once")
    )
    model = init_weights(model, key=jr.key(1))
    linears = [
        x
        for x in jax.tree.leaves(model, is_leaf=lambda y: isinstance(y, eqx.nn.Linear))
        if isinstance(x, eqx.nn.Linear)
    ]
    for linear in linears:
        if linear.bias is not None:
            assert not jnp.any(linear.bias)
    if flag == "baseline":
        weights = np.concatenate([np.ravel(lin.weight) for lin in linears])
        assert abs(weights.std() - 0.02) < 1e-3
        return
    attn = model.token_blocks[0].attn
    bound = (6 / (4 * 64)) ** 0.5  # xavier_uniform_ on the fused (3E, E) matrix
    for proj in (attn.query_proj, attn.key_proj, attn.value_proj):
        assert jnp.abs(proj.weight).max() <= bound
        assert jnp.abs(proj.weight).max() > 0.9 * bound
    assert abs(float(attn.output_proj.weight.std()) - 0.02) < 2e-3


def test_jit_training_step_with_dropout():
    """Dropout on, fused path, under `filter_jit`: finite gradients, and the
    output depends on the key only in training mode."""
    model = JaxTransolver(
        space_dim=SPACE_DIM,
        fun_dim=FUN_DIM,
        out_dim=OUT_DIM,
        num_layers=2,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_slices=NUM_SLICES,
        dropout=0.1,
        use_fused_slice=True,
        chunk_size=CHUNK,
        key=jr.key(0),
    )
    model = init_weights(model, key=jr.key(1))
    x = jr.normal(jr.key(2), (BATCH, NUM_POINTS, SPACE_DIM))
    fx = jr.normal(jr.key(3), (BATCH, NUM_POINTS, FUN_DIM))

    @eqx.filter_jit
    def step(m, key):
        return eqx.filter_value_and_grad(lambda m: jnp.mean(m(x, fx, key=key) ** 2))(m)

    loss, grads = step(model, jr.key(4))
    assert jnp.isfinite(loss)
    assert all(jnp.isfinite(g).all() for g in jax.tree.leaves(grads))

    forward = eqx.filter_jit(model)
    assert not jnp.allclose(
        forward(x, fx, key=jr.key(5)), forward(x, fx, key=jr.key(6))
    )
    np.testing.assert_array_equal(
        forward(x, fx, key=jr.key(5), inference=True),
        forward(x, fx, key=jr.key(6), inference=True),
    )
