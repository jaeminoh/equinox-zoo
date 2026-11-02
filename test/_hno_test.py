"""
Hyperbolic Neural Operator (HNO) JAX test.

The torch reference is the large-scale CFD model of
https://github.com/GuobaPei/Hyperbolic-Neural-Operator (commit c7a9e81):
`large_scale/airfrans/models/hno.py` in full, and from
`large_scale/shapenetcar/models/hno.py` the `LorentzManifold`, the
`HyperbolicSliceAttention` and the `HNO._get_grid_feat` that differ from it,
renamed with a `ShapeNetCar` prefix. Copyright (c) 2026 Jieyuan Pei, Zhuoxuan Li,
Wei Li, Haobo Zhang, Jiawei Jiang, and Jianwei Zheng, MIT. Edits: timm's
`trunc_normal_` (the same function) comes from `torch.nn.init`; the one einops
`rearrange` is spelled out as permute + reshape; unused imports are dropped and
the code is ruff-formatted; the `.float()` upcasts, meant to lift 16-bit inputs
to fp32, become `_at_least_fp32`, and the grids are built in `pos.dtype` instead
of fp32, so that float64 inputs stay float64 and the parity tests can run in
float64. The forward math is unchanged.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


def _at_least_fp32(t: torch.Tensor) -> torch.Tensor:
    """Stands in for the reference's `.float()`: lifts 16-bit to fp32, keeps fp64."""
    return t.to(torch.promote_types(t.dtype, torch.float32))


# -----------------------
# large_scale/airfrans/models/hno.py
# -----------------------
ACTIVATION = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}


class MLP(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        n_layers: int = 1,
        act: str = "gelu",
        res: bool = True,
    ):
        super().__init__()
        act_cls = ACTIVATION.get(act, None)
        if act_cls is None:
            raise NotImplementedError(f"Unknown activation: {act}")
        if callable(act_cls) and not isinstance(act_cls, type):
            # leaky_relu lambda
            act_layer = act_cls
        else:
            act_layer = act_cls

        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_layer())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(n_hidden, n_hidden), act_layer())
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_pre(x)
        for layer in self.linears:
            x = layer(x) + x if self.res else layer(x)
        return self.linear_post(x)


class LorentzManifold:
    """
    Lorentz ops (adapted from history/ela.py) for numerical stability.
    """

    ACOSH_EPS = 1e-4

    @staticmethod
    def to_lorentz(x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = _at_least_fp32(x)
        norm_sq = (x_f * x_f).sum(dim=-1, keepdim=True)
        t = torch.sqrt(1.0 + norm_sq)
        out = torch.cat([t, x_f], dim=-1)
        return out.to(orig_dtype)

    @staticmethod
    def pairwise_lorentz_distance_multihead(
        x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: (B,H,N,d+1), y: (B,H,M,d+1) -> (B,H,N,M)
        """
        if y is None:
            y = x
        orig_dtype = x.dtype
        x_f = _at_least_fp32(x)
        y_f = _at_least_fp32(y)
        inner = -torch.einsum(
            "bhni,bhmi->bhnm", x_f[..., :1], y_f[..., :1]
        ) + torch.einsum("bhni,bhmi->bhnm", x_f[..., 1:], y_f[..., 1:])
        dist = torch.acosh((-inner).clamp(min=1.0 + LorentzManifold.ACOSH_EPS))
        return dist.to(orig_dtype)


class HyperbolicSliceAttention(nn.Module):
    """
    Slice/deslice aggregation with Lorentz-distance attention among slice tokens.

    Input: x (B,N,C)
    Output: (B,N,C)
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 32,
        dropout: float = 0.0,
        slice_num: int = 64,
        hyp_dim: int = 16,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # slicing temperature
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)

        # hyperbolic Q/K projection
        self.hyp_dim = hyp_dim
        self.to_q_h = nn.Linear(dim_head, hyp_dim, bias=False)
        self.to_k_h = nn.Linear(dim_head, hyp_dim, bias=False)

        # value projection stays in dim_head
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

        # learnable temperature for hyperbolic token attention (per-head)
        self.temp_token = nn.Parameter(torch.ones([1, heads, 1, 1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        fx_mid = (
            self.in_project_fx(x)
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        x_mid = (
            self.in_project_x(x)
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / self.temperature
        )  # (B,H,N,G)
        slice_norm = slice_weights.sum(2)  # (B,H,G)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (
            (slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head)
        )

        # token attention in hyperbolic space
        q = self.to_q_h(slice_token)  # (B,H,G,hyp)
        k = self.to_k_h(slice_token)  # (B,H,G,hyp)
        q_l = LorentzManifold.to_lorentz(q)
        k_l = LorentzManifold.to_lorentz(k)
        dist = LorentzManifold.pairwise_lorentz_distance_multihead(
            q_l, k_l
        )  # (B,H,G,G)

        temp = self.temp_token.clamp(0.1, 3.0)
        attn = F.softmax(_at_least_fp32(-dist / temp), dim=-1).to(slice_token.dtype)
        attn = self.dropout(attn)

        v = self.to_v(slice_token)  # (B,H,G,D)
        out_slice = attn @ v  # (B,H,G,D)

        # deslice back
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out_x = out_x.permute(0, 2, 1, 3).reshape(B, N, -1)  # b h n d -> b n (h d)
        return self.to_out(out_x)


class HNOBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act: str = "gelu",
        mlp_ratio: float = 2.0,
        last_layer: bool = False,
        out_dim: int = 4,
        slice_num: int = 64,
        hyp_dim: int = 16,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = HyperbolicSliceAttention(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
            hyp_dim=hyp_dim,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(
            hidden_dim,
            int(hidden_dim * mlp_ratio),
            hidden_dim,
            n_layers=0,
            res=False,
            act=act,
        )
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx: torch.Tensor) -> torch.Tensor:
        fx = self.attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.proj(self.ln_3(fx))
        return fx


class HNO(nn.Module):
    """
    HNO for AirfRANS with an M-token interaction core.
    """

    def __init__(
        self,
        space_dim: int = 7,
        n_layers: int = 8,
        n_hidden: int = 256,
        dropout: float = 0.0,
        n_head: int = 8,
        act: str = "gelu",
        mlp_ratio: float = 2.0,
        fun_dim: int = 0,
        out_dim: int = 4,
        slice_num: int = 64,
        hyp_dim: int = 16,
        ref: int = 8,
        unified_pos: int = 1,
    ):
        super().__init__()
        self.__name__ = "HNO"
        self.ref = int(ref)
        self.unified_pos = bool(unified_pos)
        self.space_dim = int(space_dim)
        self.n_hidden = int(n_hidden)

        # preprocess
        if self.unified_pos:
            # append ref^2 distance features (2D grid) to x
            in_dim = fun_dim + space_dim + self.ref * self.ref
            self.preprocess = MLP(
                in_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act
            )
        else:
            in_dim = fun_dim + space_dim
            self.preprocess = MLP(
                in_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act
            )

        self.blocks = nn.ModuleList(
            [
                HNOBlock(
                    num_heads=n_head,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    out_dim=out_dim,
                    slice_num=slice_num,
                    hyp_dim=hyp_dim,
                    last_layer=(i == n_layers - 1),
                )
                for i in range(n_layers)
            ]
        )

        self.placeholder = nn.Parameter(
            (1.0 / n_hidden) * torch.rand(n_hidden, dtype=torch.float)
        )
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _get_grid_feat_2d(self, pos: torch.Tensor) -> torch.Tensor:
        """
        pos: (B,N,2) or (B,N,>=2) uses first 2 dims.
        returns: (B,N,ref^2) distances to a fixed 2D grid.
        """
        B, N, _ = pos.shape
        device = pos.device
        pos2 = pos[..., :2]
        gridx = torch.tensor(
            np.linspace(-2.0, 4.0, self.ref), dtype=pos.dtype, device=device
        )
        gridy = torch.tensor(
            np.linspace(-1.5, 1.5, self.ref), dtype=pos.dtype, device=device
        )
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat(B, 1, self.ref, 1)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat(B, self.ref, 1, 1)
        grid = torch.cat((gridx, gridy), dim=-1).reshape(B, self.ref * self.ref, 2)
        dist_feat = torch.sqrt(
            torch.sum((pos2[:, :, None, :] - grid[:, None, :, :]) ** 2, dim=-1)
        )
        return dist_feat.contiguous()

    def forward(self, data) -> torch.Tensor:
        # upstream uses batch_size=1 and wraps as (1,N,C)
        x = data.x[None, :, :]  # (1,N,C_in)
        if self.unified_pos:
            pos = data.pos[None, :, :]
            grid_feat = self._get_grid_feat_2d(pos)
            x = torch.cat((x, grid_feat), dim=-1)
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx)
        return fx[0]


# -----------------------
# large_scale/shapenetcar/models/hno.py (what differs)
# -----------------------
class ShapeNetCarLorentzManifold:
    """
    Minimal Lorentz model ops for hyperbolic attention.
    """

    ACOSH_EPS = 1e-4

    @staticmethod
    def to_lorentz(x: torch.Tensor) -> torch.Tensor:
        # x: (..., d) -> (..., d+1) with time-like coordinate
        orig_dtype = x.dtype
        x_f = _at_least_fp32(x)
        norm_sq = (x_f * x_f).sum(dim=-1, keepdim=True)
        t = torch.sqrt(1.0 + norm_sq)
        out = torch.cat([t, x_f], dim=-1)
        return out.to(orig_dtype)

    @staticmethod
    def pairwise_lorentz_distance_multihead(
        x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: (B,H,N,d+1), y: (B,H,M,d+1) -> dist: (B,H,N,M)
        """
        if y is None:
            y = x
        # Minkowski inner product: <x,y>_L = -t_x t_y + sum_i x_i y_i
        tx = x[..., :1]  # (B,H,N,1)
        xx = x[..., 1:]  # (B,H,N,d)
        ty = y[..., :1]  # (B,H,M,1)
        yy = y[..., 1:]  # (B,H,M,d)
        mink = -(tx @ ty.transpose(-1, -2)) + (xx @ yy.transpose(-1, -2))
        z = (-mink).clamp(min=1.0 + ShapeNetCarLorentzManifold.ACOSH_EPS)
        return torch.acosh(z)


class ShapeNetCarHyperbolicSliceAttention(nn.Module):
    """
    Slice -> hyperbolic attention among slice tokens -> deslice.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        slice_num: int = 64,
        hyp_dim: int = 16,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)

        # project slice tokens to low-dim hyperbolic space
        self.to_q = nn.Linear(dim_head, hyp_dim, bias=False)
        self.to_k = nn.Linear(dim_head, hyp_dim, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        B, N, C = x.shape

        fx_mid = (
            self.in_project_fx(x)
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # B H N D
        x_mid = (
            self.in_project_x(x)
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # B H N D

        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / self.temperature
        )  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (
            (slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head)
        )

        # hyperbolic attention on Lorentz manifold
        q = self.to_q(slice_token)  # B H G hyp
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)  # B H G D
        qL = ShapeNetCarLorentzManifold.to_lorentz(q)  # B H G hyp+1
        kL = ShapeNetCarLorentzManifold.to_lorentz(k)
        dist = ShapeNetCarLorentzManifold.pairwise_lorentz_distance_multihead(
            qL, kL
        )  # B H G G
        # temperature: (1,H,1,1) -> broadcast to (B,H,G,G)
        attn = self.softmax(-dist / self.temperature)
        attn = self.dropout(attn)
        out_slice = torch.matmul(attn, v)  # B H G D

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out_x = out_x.permute(0, 2, 1, 3).reshape(B, N, -1)  # b h n d -> b n (h d)
        return self.to_out(out_x)


class ShapeNetCarGridFeat:
    """`HNO._get_grid_feat` of the ShapeNet Car model, which reads only `self.ref`."""

    def __init__(self, ref: int):
        self.ref = int(ref)

    def _get_grid_feat(self, pos: torch.Tensor) -> torch.Tensor:
        """
        pos: (B, N, d), d in {2,3}
        returns: (B, N, ref^d)
        """
        B, N, d = pos.shape
        device = pos.device
        if d == 2:
            gridx = torch.linspace(-2.0, 4.0, self.ref, device=device, dtype=pos.dtype)
            gridy = torch.linspace(-1.5, 1.5, self.ref, device=device, dtype=pos.dtype)
            gridx = gridx.reshape(1, self.ref, 1, 1).repeat(B, 1, self.ref, 1)
            gridy = gridy.reshape(1, 1, self.ref, 1).repeat(B, self.ref, 1, 1)
            grid = torch.cat((gridx, gridy), dim=-1).reshape(B, self.ref**2, 2)
        elif d == 3:
            gridx = torch.linspace(-1.5, 1.5, self.ref, device=device, dtype=pos.dtype)
            gridy = torch.linspace(0.0, 2.0, self.ref, device=device, dtype=pos.dtype)
            gridz = torch.linspace(-4.0, 4.0, self.ref, device=device, dtype=pos.dtype)
            gridx = gridx.reshape(1, self.ref, 1, 1, 1).repeat(
                B, 1, self.ref, self.ref, 1
            )
            gridy = gridy.reshape(1, 1, self.ref, 1, 1).repeat(
                B, self.ref, 1, self.ref, 1
            )
            gridz = gridz.reshape(1, 1, 1, self.ref, 1).repeat(
                B, self.ref, self.ref, 1, 1
            )
            grid = torch.cat((gridx, gridy, gridz), dim=-1).reshape(B, self.ref**3, 3)
        else:
            raise ValueError(f"unified_pos expects pos dim 2 or 3, got {d}")

        dist_feat = torch.sqrt(
            torch.sum((pos[:, :, None, :] - grid[:, None, :, :]) ** 2, dim=-1)
        )
        return dist_feat.contiguous()


# ---------------------------------------------------------------------------
# JAX side
# ---------------------------------------------------------------------------
import types  # noqa: E402

import jax  # noqa: E402

# The parity tests compare against float64 PyTorch, so roundoff cannot hide a
# structural mismatch. Must be set before any array is created.
jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import pytest  # noqa: E402

from zoo import _hno as eqx_hno  # noqa: E402

SPACE_DIM, FUN_DIM, OUT_DIM = 2, 3, 4
NUM_LAYERS, NUM_HEADS, HEAD_DIM, NUM_SLICES, HYP_DIM = 3, 4, 8, 6, 5
HIDDEN_DIM = NUM_HEADS * HEAD_DIM
MLP_RATIO = 1.3  # int(32 * 1.3) = 41: exercises the reference's int() truncation
REF = 4
BATCH, NUM_POINTS = 2, 23

TOL = 1e-10

# JAX parameter path -> torch parameter name, for the parts that are named
# differently; torch wraps the Linear of `to_out` and `linear_pre` in a Sequential.
AIRFRANS_RENAMES = {
    "to_q": "to_q_h",
    "to_k": "to_k_h",
    "to_out": "to_out.0",
    "linear_pre": "linear_pre.0",
    "token_temperature": "temp_token",
}
SHAPENETCAR_RENAMES = {"to_out": "to_out.0"}


def make_jax_model(key, *, unified_pos=False, dropout=0.0, **kwargs):
    return eqx_hno.HNO(
        space_dim=SPACE_DIM,
        fun_dim=FUN_DIM,
        out_dim=OUT_DIM,
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_slices=NUM_SLICES,
        hyp_dim=HYP_DIM,
        mlp_ratio=MLP_RATIO,
        dropout=dropout,
        unified_pos=unified_pos,
        ref=REF,
        key=key,
        **kwargs,
    )


def make_jax_attention(key, **kwargs):
    return eqx_hno.HyperbolicSliceAttention(
        HIDDEN_DIM,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        num_slices=NUM_SLICES,
        hyp_dim=HYP_DIM,
        key=key,
        **kwargs,
    )


def make_inputs(key, batch=BATCH):
    kx, kf = jr.split(key)
    return (
        jr.normal(kx, (batch, NUM_POINTS, SPACE_DIM)),
        jr.normal(kf, (batch, NUM_POINTS, FUN_DIM)),
    )


def t2j(x):
    return jnp.asarray(x.detach().numpy(), dtype=jnp.float64)


def assert_close(got, expected, tol=TOL):
    got = np.asarray(got)
    expected = expected.detach().numpy() if torch.is_tensor(expected) else expected
    assert got.shape == expected.shape
    scale = max(1.0, np.abs(expected).max())
    assert np.abs(got - expected).max() <= tol * scale


# ---------------------------------------------------------------------------
# The Lorentz model
# ---------------------------------------------------------------------------
def test_lift_lands_on_the_hyperboloid():
    z = 2.0 * jr.normal(jr.key(0), (7, 5))
    u = eqx_hno.lorentz_lift(z)

    assert u.shape == (7, 6)
    minkowski = -(u[:, 0] ** 2) + jnp.sum(u[:, 1:] ** 2, axis=-1)
    assert np.allclose(np.asarray(minkowski), -1.0, atol=1e-12)
    assert (u[:, 0] >= 1).all()


def test_distance_is_the_geodesic_distance_in_1d():
    """On the 1D hyperboloid the lift is `(cosh t, sinh t)` with `t = asinh z`,
    so the geodesic distance is `|t_a - t_b|`, except on the clamped diagonal."""
    z = jnp.linspace(-3.0, 3.0, 7)[:, None]
    u = eqx_hno.lorentz_lift(z)
    dist = eqx_hno.lorentz_distance(u, u)

    t = jnp.arcsinh(z[:, 0])
    expected = jnp.abs(t[:, None] - t[None, :])
    off_diagonal = ~np.eye(7, dtype=bool)
    assert np.allclose(
        np.asarray(dist)[off_diagonal], np.asarray(expected)[off_diagonal]
    )
    assert np.allclose(np.asarray(jnp.diag(dist)), np.arccosh(1 + 1e-4))


def test_distance_is_symmetric_and_batched():
    ka, kb = jr.split(jr.key(0))
    u = eqx_hno.lorentz_lift(jr.normal(ka, (2, 3, 5, 4)))
    v = eqx_hno.lorentz_lift(jr.normal(kb, (2, 3, 6, 4)))

    d_uv = eqx_hno.lorentz_distance(u, v)
    d_vu = eqx_hno.lorentz_distance(v, u)

    assert d_uv.shape == (2, 3, 5, 6)
    assert np.allclose(np.asarray(d_uv), np.asarray(jnp.swapaxes(d_vu, -1, -2)))
    assert (d_uv >= np.arccosh(1 + 1e-4) - 1e-12).all()


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_lorentz_ops_compute_16bit_inputs_in_fp32(dtype):
    """The lift and distance keep the input dtype but compute in fp32, as the
    reference does to avoid NaNs at the arcosh: bit for bit, they are the fp32
    computation on the upcast inputs, rounded once at the end."""
    z = jr.normal(jr.key(0), (3, 9, 4)).astype(dtype)
    u = eqx_hno.lorentz_lift(z)
    dist = eqx_hno.lorentz_distance(u, u)

    assert u.dtype == dtype and dist.dtype == dtype
    assert jnp.isfinite(dist).all()
    u32 = u.astype(jnp.float32)
    lift32 = eqx_hno.lorentz_lift(z.astype(jnp.float32))
    assert (u == lift32.astype(dtype)).all()
    assert (dist == eqx_hno.lorentz_distance(u32, u32).astype(dtype)).all()


def test_parity_lorentz_ops():
    """Against both references' Lorentz ops, with repeated points so that the
    arcosh clamp is taken."""
    torch.manual_seed(0)
    z = torch.randn(2, 3, 5, 4, dtype=torch.float64)
    z = torch.cat([z, z[:, :, :2]], dim=2)  # repeated points: clamped distances

    expected_lift = LorentzManifold.to_lorentz(z)
    expected_dist = LorentzManifold.pairwise_lorentz_distance_multihead(expected_lift)
    expected_dist_snc = ShapeNetCarLorentzManifold.pairwise_lorentz_distance_multihead(
        ShapeNetCarLorentzManifold.to_lorentz(z)
    )
    u = eqx_hno.lorentz_lift(t2j(z))
    dist = eqx_hno.lorentz_distance(u, u)

    assert_close(u, expected_lift)
    assert_close(dist, expected_dist)
    assert_close(dist, expected_dist_snc)
    # The clamp is taken off the diagonal too, on the repeated points.
    clamped = expected_dist <= np.arccosh(1 + 1e-4) + 1e-12
    assert clamped.sum() > 2 * 3 * 7


@pytest.mark.parametrize("space_dim", [2, 3])
def test_parity_grid_features(space_dim):
    torch.manual_seed(0)
    pos = 2.0 * torch.randn(BATCH, NUM_POINTS, space_dim, dtype=torch.float64)

    got = eqx_hno._grid_features(t2j(pos), REF)

    assert got.shape == (BATCH, NUM_POINTS, REF**space_dim)
    assert_close(got, ShapeNetCarGridFeat(REF)._get_grid_feat(pos))
    if space_dim == 2:
        model = HNO(n_layers=1, n_hidden=8, n_head=2, slice_num=2, ref=REF)
        assert_close(got, model._get_grid_feat_2d(pos))


# ---------------------------------------------------------------------------
# Standalone JAX behaviour
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("unified_pos", [False, True])
def test_output_shape(unified_pos):
    key = jr.key(0)
    model = make_jax_model(key, unified_pos=unified_pos, dropout=0.1)
    x, fx = make_inputs(jr.fold_in(key, 1))

    out = model(x, fx, key=jr.fold_in(key, 2), inference=False)

    assert out.shape == (BATCH, NUM_POINTS, OUT_DIM)
    assert jnp.isfinite(out).all()


def test_fx_is_optional():
    key = jr.key(0)
    model = eqx_hno.HNO(
        space_dim=3, num_layers=2, hidden_dim=16, num_heads=2, num_slices=4, key=key
    )
    x = jr.normal(jr.fold_in(key, 1), (BATCH, NUM_POINTS, 3))

    out = model(x, key=jr.fold_in(key, 2), inference=True)

    assert out.shape == (BATCH, NUM_POINTS, 1)


def test_gradients_are_finite():
    key = jr.key(0)
    model = eqx_hno.init_weights(make_jax_model(key), key=jr.fold_in(key, 1))
    x, fx = make_inputs(jr.fold_in(key, 2))

    @eqx.filter_grad
    def loss(m):
        return jnp.mean(m(x, fx, key=jr.fold_in(key, 3), inference=True) ** 2)

    grads = eqx.filter(loss(model), eqx.is_inexact_array)
    leaves = jax.tree.leaves(grads)
    assert leaves and all(jnp.isfinite(g).all() for g in leaves)


def test_jit_training_step_with_dropout():
    key = jr.key(0)
    model = eqx_hno.init_weights(
        make_jax_model(key, unified_pos=True, dropout=0.1), key=jr.fold_in(key, 1)
    )
    x, fx = make_inputs(jr.fold_in(key, 2))
    y = jr.normal(jr.fold_in(key, 3), (BATCH, NUM_POINTS, OUT_DIM))

    @eqx.filter_jit
    def step(m, k):
        def loss(m):
            return jnp.mean((m(x, fx, key=k) - y) ** 2)

        value, grads = eqx.filter_value_and_grad(loss)(m)
        updates = jax.tree.map(lambda g: -1e-2 * g, grads)
        return eqx.apply_updates(m, updates), value

    losses = []
    for i in range(3):
        model, value = step(model, jr.fold_in(key, 10 + i))
        losses.append(float(value))
    assert np.isfinite(losses).all()


@pytest.mark.parametrize("unified_pos", [False, True])
def test_model_is_permutation_equivariant(unified_pos):
    key = jr.key(0)
    model = eqx_hno.init_weights(
        make_jax_model(key, unified_pos=unified_pos), key=jr.fold_in(key, 1)
    )
    x, fx = make_inputs(jr.fold_in(key, 2))
    perm = jr.permutation(jr.fold_in(key, 3), NUM_POINTS)

    out = model(x, fx, key=jr.fold_in(key, 4), inference=True)
    out_perm = model(x[:, perm], fx[:, perm], key=jr.fold_in(key, 4), inference=True)

    assert_close(out_perm, np.asarray(out[:, perm]))


def test_token_temperature_is_clamped():
    key = jr.key(0)
    attn = make_jax_attention(key)
    x = jr.normal(jr.fold_in(key, 1), (BATCH, NUM_POINTS, HIDDEN_DIM))

    def run(tau):
        a = eqx.tree_at(
            lambda m: m.token_temperature, attn, jnp.full((1, NUM_HEADS, 1, 1), tau)
        )
        return np.asarray(a(x, key=jr.fold_in(key, 2), inference=True))

    assert_close(run(10.0), run(3.0))
    assert_close(run(0.01), run(0.1))
    assert not np.allclose(run(1.0), run(2.0))


def test_share_temperature_reuses_the_slicing_temperature():
    key = jr.key(0)
    shared = make_jax_attention(key, share_temperature=True)
    separate = make_jax_attention(key)
    x = jr.normal(jr.fold_in(key, 1), (BATCH, NUM_POINTS, HIDDEN_DIM))

    assert shared.token_temperature is None
    # Same parameters, with the token temperature set to the slicing one (0.5,
    # inside the clamp range): the two must agree.
    separate = eqx.tree_at(
        lambda m: m.token_temperature, separate, separate.temperature
    )
    assert_close(
        shared(x, key=jr.fold_in(key, 2), inference=True),
        np.asarray(separate(x, key=jr.fold_in(key, 2), inference=True)),
    )


def test_unified_pos_needs_2d_or_3d_coordinates():
    with pytest.raises(ValueError, match="space_dim 2 or 3"):
        eqx_hno.HNO(space_dim=4, unified_pos=True, key=jr.key(0))


def _linears(model):
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)  # noqa: E731
    return [x for x in jax.tree.leaves(model, is_leaf=is_linear) if is_linear(x)]


def test_init_weights_matches_reference_scheme():
    """Every `Linear` weight, `in_project_slice` included, is N(0, 0.02^2) and
    every bias zero, as the reference's `initialize_weights` leaves them."""
    torch.manual_seed(0)
    torch_model = make_torch_model(unified_pos=True, perturb=False)
    torch_linears = [m for m in torch_model.modules() if isinstance(m, nn.Linear)]
    model = eqx_hno.init_weights(
        make_jax_model(jr.key(0), unified_pos=True), key=jr.key(1)
    )
    jax_linears = _linears(model)
    assert len(jax_linears) == len(torch_linears)

    def weight_std(weights):
        return np.concatenate([np.ravel(w) for w in weights]).std()

    torch_std = weight_std(m.weight.detach().numpy() for m in torch_linears)
    jax_std = weight_std(lin.weight for lin in jax_linears)
    assert abs(torch_std - 0.02) < 1e-3 and abs(jax_std - 0.02) < 1e-3
    for lin in jax_linears:
        assert lin.bias is None or not jnp.any(lin.bias)
    # The constructor's default init is replaced, `in_project_slice` included.
    fresh = make_jax_model(jr.key(0), unified_pos=True).blocks[0].attn
    assert not np.allclose(
        model.blocks[0].attn.in_project_slice.weight, fresh.in_project_slice.weight
    )


# ---------------------------------------------------------------------------
# Copying PyTorch weights into the Equinox model
# ---------------------------------------------------------------------------
def torch_name(path, renames):
    """`.blocks[0].attn.to_q.weight` -> `blocks.0.attn.to_q_h.weight`."""
    parts = []
    for k in path:
        if isinstance(k, jax.tree_util.GetAttrKey):
            parts.append(renames.get(k.name, k.name))
        else:
            parts.append(str(k.idx))
    return ".".join(parts)


def param_names(jax_model, renames):
    params = eqx.filter(jax_model, eqx.is_inexact_array)
    paths = jax.tree_util.tree_flatten_with_path(params)[0]
    return [torch_name(path, renames) for path, _ in paths]


def from_torch(jax_model, torch_module, renames, *, grad=False):
    """`jax_model` with every parameter (or, with `grad`, its gradient) taken
    from the torch parameter of the same name."""
    params, static = eqx.partition(jax_model, eqx.is_inexact_array)
    torch_params = dict(torch_module.named_parameters())

    def take(path, leaf):
        p = torch_params[torch_name(path, renames)]
        assert tuple(p.shape) == leaf.shape
        return t2j(p.grad if grad else p)

    return eqx.combine(jax.tree_util.tree_map_with_path(take, params), static)


def perturb_(torch_module):
    """Move every parameter off its structured init (zero biases, unit norms,
    equal temperatures), so that a mis-copied one cannot go unnoticed."""
    with torch.no_grad():
        for name, p in torch_module.named_parameters():
            noise = 0.1 * torch.randn_like(p)
            p.add_(noise.abs() if "temp" in name else noise)


def make_torch_model(unified_pos, perturb=True):
    model = HNO(
        space_dim=SPACE_DIM + FUN_DIM,
        n_layers=NUM_LAYERS,
        n_hidden=HIDDEN_DIM,
        dropout=0.0,
        n_head=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        fun_dim=0,
        out_dim=OUT_DIM,
        slice_num=NUM_SLICES,
        hyp_dim=HYP_DIM,
        ref=REF,
        unified_pos=int(unified_pos),
    )
    model = model.double().eval()
    if perturb:
        perturb_(model)
        with torch.no_grad():
            # Push one head's token temperature past each end of the clamp.
            model.blocks[0].attn.temp_token[0, 0] = 5.0
            model.blocks[0].attn.temp_token[0, 1] = 0.05
    return model


def torch_data(x, fx):
    """The reference reads `data.x` (positions first) and `data.pos`, batch 1."""
    return types.SimpleNamespace(x=torch.cat([x, fx], dim=-1), pos=x)


# ---------------------------------------------------------------------------
# Parity with the PyTorch references
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module", params=[False, True], ids=["plain", "unified_pos"])
def model_pair(request):
    """An AirfRANS reference model and an Equinox model carrying its weights."""
    torch.manual_seed(0)
    torch_model = make_torch_model(request.param)
    jax_model = make_jax_model(jr.key(0), unified_pos=request.param)
    return torch_model, from_torch(jax_model, torch_model, AIRFRANS_RENAMES)


def test_param_map_covers_every_torch_parameter(model_pair):
    torch_model, jax_model = model_pair
    names = param_names(jax_model, AIRFRANS_RENAMES)

    assert len(names) == len(set(names))
    assert set(names) == {n for n, _ in torch_model.named_parameters()}


def test_parity_forward_and_gradients(model_pair):
    torch_model, jax_model = model_pair
    torch.manual_seed(1)
    x = 1.5 * torch.randn(NUM_POINTS, SPACE_DIM, dtype=torch.float64)
    fx = torch.randn(NUM_POINTS, FUN_DIM, dtype=torch.float64)
    w = torch.randn(NUM_POINTS, OUT_DIM, dtype=torch.float64)

    torch_model.zero_grad()
    expected = torch_model(torch_data(x, fx))
    (expected * w).sum().backward()

    def loss(m):
        out = m(t2j(x)[None], t2j(fx)[None], key=jr.key(0), inference=True)[0]
        return jnp.sum(out * t2j(w)), out

    grads, got = eqx.filter_grad(loss, has_aux=True)(jax_model)
    assert_close(got, expected)

    expected_grads = from_torch(jax_model, torch_model, AIRFRANS_RENAMES, grad=True)
    got_leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))
    expected_leaves = jax.tree.leaves(eqx.filter(expected_grads, eqx.is_inexact_array))
    assert len(got_leaves) == len(expected_leaves)
    for g, e in zip(got_leaves, expected_leaves):
        assert_close(g, np.asarray(e))


def torch_attention_stages(attn, x):
    """The reference `HyperbolicSliceAttention.forward`, returning every stage."""
    B, N, C = x.shape
    H, D = attn.heads, attn.dim_head
    fx_mid = attn.in_project_fx(x).reshape(B, N, H, D).permute(0, 2, 1, 3)
    x_mid = attn.in_project_x(x).reshape(B, N, H, D).permute(0, 2, 1, 3)
    weights = attn.softmax(attn.in_project_slice(x_mid) / attn.temperature)
    tokens = torch.einsum("bhnc,bhng->bhgc", fx_mid, weights)
    tokens = tokens / (weights.sum(2) + 1e-5)[..., None]
    q_l = LorentzManifold.to_lorentz(attn.to_q_h(tokens))
    k_l = LorentzManifold.to_lorentz(attn.to_k_h(tokens))
    dist = LorentzManifold.pairwise_lorentz_distance_multihead(q_l, k_l)
    scores = F.softmax(-dist / attn.temp_token.clamp(0.1, 3.0), dim=-1)
    attended = scores @ attn.to_v(tokens)
    out = torch.einsum("bhgc,bhng->bhnc", attended, weights)
    out = attn.to_out(out.permute(0, 2, 1, 3).reshape(B, N, H * D))
    return weights, tokens, attended, out


@pytest.fixture(scope="module")
def attention_pair(model_pair):
    """The block-0 attention of each model (with the clamped temperatures), plus
    a shared input."""
    torch_model, jax_model = model_pair
    torch.manual_seed(2)
    x = torch.randn(BATCH, NUM_POINTS, HIDDEN_DIM, dtype=torch.float64)
    return torch_model.blocks[0].attn, jax_model.blocks[0].attn, x


def test_parity_attention_stages(attention_pair):
    """Pin each stage separately, so that a regression localizes itself."""
    torch_attn, jax_attn, x = attention_pair
    with torch.no_grad():
        weights_t, tokens_t, attended_t, out_t = torch_attention_stages(torch_attn, x)
        assert_close(out_t, torch_attn(x))  # the staging is the reference forward

    weights = jax_attn.slice_weights(t2j(x))
    assert weights.shape == (BATCH, NUM_HEADS, NUM_POINTS, NUM_SLICES)
    assert_close(weights, weights_t)

    tokens = jax_attn.slice_tokens(t2j(x), weights)
    assert tokens.shape == (BATCH, NUM_HEADS, NUM_SLICES, HEAD_DIM)
    assert_close(tokens, tokens_t)

    attended = jax_attn.attend(tokens, key=jr.key(0), inference=True)
    assert_close(attended, attended_t)

    out = jax_attn.deslice(attended, weights, key=jr.key(0), inference=True)
    assert_close(out, out_t)
    assert_close(jax_attn(t2j(x), key=jr.key(0), inference=True), out_t)


def test_parity_shapenetcar_attention():
    """`share_temperature=True` is the ShapeNet Car layer: one temperature for
    slicing and token attention, unclamped."""
    torch.manual_seed(0)
    torch_attn = ShapeNetCarHyperbolicSliceAttention(
        HIDDEN_DIM,
        heads=NUM_HEADS,
        dim_head=HEAD_DIM,
        slice_num=NUM_SLICES,
        hyp_dim=HYP_DIM,
    )
    torch_attn = torch_attn.double().eval()
    perturb_(torch_attn)
    jax_attn = from_torch(
        make_jax_attention(jr.key(0), share_temperature=True),
        torch_attn,
        SHAPENETCAR_RENAMES,
    )
    assert set(param_names(jax_attn, SHAPENETCAR_RENAMES)) == {
        n for n, _ in torch_attn.named_parameters()
    }
    x = torch.randn(BATCH, NUM_POINTS, HIDDEN_DIM, dtype=torch.float64)

    with torch.no_grad():
        expected = torch_attn(x)
    got = jax_attn(t2j(x), key=jr.key(0), inference=True)

    assert_close(got, expected)
