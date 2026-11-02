"""
JAX/Equinox port of the Hyperbolic Neural Operator (HNO) for large-scale CFD.

HNO is the companion code of Pei et al., "Hyperbolic Neural Operator" (ICML 2026),
https://github.com/GuobaPei/Hyperbolic-Neural-Operator. The reference ships one
kernel under several tokenizations; this ports the model it trains on AirfRANS
and ShapeNet Car (`large_scale/*/models/hno.py`): a Transolver whose attention
among slice tokens is a Gibbs kernel on a stabilized Lorentz distance instead of
a dot product. Per head, the `G` slice tokens are projected to `hyp_dim` queries
and keys, lifted onto the hyperboloid by `P(z) = (sqrt(1 + |z|^2), z)`, and
scored by

    softmax_k(-d(P(q), P(k)) / tau),   d(u, v) = arcosh(max(1 + eps, -<u, v>_M)),

with the Minkowski product `<u, v>_M = -u_0 v_0 + sum_i u_i v_i`.

The two reference models differ in the temperature of that kernel: the AirfRANS
model gives it its own per-head temperature clamped to [0.1, 3] (the paper's
choice, and the default here), the ShapeNet Car model reuses the slicing
temperature (`share_temperature=True`). Not ported: ShapeNet Car's optional
condition embedding (`cond_dim`, off in its run script) and the torch_geometric
`Data` interface; the `preprocess` that the reference builds lazily under
`unified_pos` is sized up front.

Arrays carry an explicit leading batch axis `(B, N, C)`, mirroring the PyTorch
reference in `test/_hno_test.py`, as `zoo/_transolver_3.py` does.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Key

from ._transolver_3 import MLP

# The reference's clamp of the arcosh argument, `1 + eps`, away from its
# singular derivative at 1.
_ACOSH_EPS = 1e-4
# The clamp of the token-attention temperature.
_TEMPERATURE_RANGE = (0.1, 3.0)
# Added to the slice mass before normalizing the tokens, as in the reference.
_SLICE_EPS = 1e-5
# Bounds of the `unified_pos` reference grid, per coordinate dimension.
_GRID_BOUNDS = {
    2: ((-2.0, 4.0), (-1.5, 1.5)),
    3: ((-1.5, 1.5), (0.0, 2.0), (-4.0, 4.0)),
}


def _apply(layer, x):
    """Apply a per-vector layer (`Linear`, `LayerNorm`, ...) over every leading axis."""
    for _ in range(x.ndim - 1):
        layer = jax.vmap(layer)
    return layer(x)


def _acc_dtype(*arrays):
    """At least fp32, as the reference's `.float()` casts; fp64 stays fp64."""
    return jnp.result_type(jnp.float32, *arrays)


# ---------------------------------------------------------------------------
# Lorentz model
# ---------------------------------------------------------------------------
def lorentz_lift(z: Float[Array, "... m"]) -> Float[Array, "... m+1"]:
    """Lift onto the hyperboloid `<u, u>_M = -1`: `z -> (sqrt(1 + |z|^2), z)`,
    with the time-like coordinate first."""
    zf = z.astype(_acc_dtype(z))
    time = jnp.sqrt(1 + jnp.sum(zf * zf, axis=-1, keepdims=True))
    return jnp.concatenate((time, zf), axis=-1).astype(z.dtype)


def lorentz_distance(
    u: Float[Array, "... N m+1"],
    v: Float[Array, "... M m+1"],
    eps: float = _ACOSH_EPS,
) -> Float[Array, "... N M"]:
    """
    Pairwise stabilized geodesic distance `arcosh(max(1 + eps, -<u, v>_M))`
    between points on the hyperboloid.

    Computed in at least fp32 and returned in the inputs' dtype, as in the
    reference. The clamp keeps the arcosh off its singular derivative at 1; it
    makes the distance of nearby points `arcosh(1 + eps)` rather than 0.
    """
    acc = _acc_dtype(u, v)
    uf, vf = u.astype(acc), v.astype(acc)
    time = jnp.einsum("...ni,...mi->...nm", uf[..., :1], vf[..., :1])
    space = jnp.einsum("...ni,...mi->...nm", uf[..., 1:], vf[..., 1:])
    dist = jnp.arccosh(jnp.maximum(time - space, 1 + eps))
    return dist.astype(jnp.result_type(u, v))


# ---------------------------------------------------------------------------
# HNO
# ---------------------------------------------------------------------------
class HyperbolicSliceAttention(eqx.Module):
    """Slice the points into tokens, attend among the tokens by Lorentz distance,
    deslice back to the points."""

    in_project_x: eqx.nn.Linear
    in_project_fx: eqx.nn.Linear
    in_project_slice: eqx.nn.Linear
    to_q: eqx.nn.Linear
    to_k: eqx.nn.Linear
    to_v: eqx.nn.Linear
    to_out: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    temperature: Array
    token_temperature: Array | None

    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    num_slices: int = eqx.field(static=True)
    hyp_dim: int = eqx.field(static=True)
    acosh_eps: float = eqx.field(static=True)

    def __init__(
        self,
        d_in: int,
        num_heads: int = 8,
        head_dim: int = 32,
        num_slices: int = 64,
        hyp_dim: int = 16,
        dropout: float = 0.0,
        share_temperature: bool = False,
        acosh_eps: float = _ACOSH_EPS,
        *,
        key: Key,
    ):
        """
        **Args:**
        - `d_in`: Input (and output) feature dimension.
        - `num_heads`: Number of heads.
        - `head_dim`: Per-head width of the slice tokens.
        - `num_slices`: Number of slices `G` per head.
        - `hyp_dim`: Dimension of the hyperbolic queries and keys; they live on
          the hyperboloid in `hyp_dim + 1` coordinates.
        - `dropout`: Dropout probability, on the token attention and the output.
        - `share_temperature`: Score the token attention with the slicing
          temperature, unclamped, as the reference's ShapeNet Car model does.
          By default it has its own per-head temperature, initialized to 1 and
          clamped to [0.1, 3], as in the AirfRANS model and the paper.
        - `acosh_eps`: Clamp of the arcosh argument at `1 + acosh_eps`.
        """
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_slices = num_slices
        self.hyp_dim = hyp_dim
        self.acosh_eps = acosh_eps

        keys = jr.split(key, 7)
        inner_dim = num_heads * head_dim
        self.in_project_x = eqx.nn.Linear(d_in, inner_dim, key=keys[0])
        self.in_project_fx = eqx.nn.Linear(d_in, inner_dim, key=keys[1])
        self.in_project_slice = eqx.nn.Linear(head_dim, num_slices, key=keys[2])
        self.to_q = eqx.nn.Linear(head_dim, hyp_dim, use_bias=False, key=keys[3])
        self.to_k = eqx.nn.Linear(head_dim, hyp_dim, use_bias=False, key=keys[4])
        self.to_v = eqx.nn.Linear(head_dim, head_dim, use_bias=False, key=keys[5])
        self.to_out = eqx.nn.Linear(inner_dim, d_in, key=keys[6])
        self.dropout = eqx.nn.Dropout(p=dropout)
        self.temperature = jnp.ones((1, num_heads, 1, 1)) * 0.5
        if share_temperature:
            self.token_temperature = None
        else:
            self.token_temperature = jnp.ones((1, num_heads, 1, 1))

    def _heads(self, linear, x):
        """Project `(B, N, C)` points to per-head features `(B, H, N, D)`."""
        B, N, _ = x.shape
        y = _apply(linear, x).reshape(B, N, self.num_heads, self.head_dim)
        return jnp.transpose(y, (0, 2, 1, 3))

    def slice_weights(
        self, x: Float[Array, "B N C"]
    ) -> Float[Array, "B H N {self.num_slices}"]:
        """Soft assignment of every point to the slices of each head."""
        logits = _apply(self.in_project_slice, self._heads(self.in_project_x, x))
        return jax.nn.softmax(logits / self.temperature, axis=-1)

    def slice_tokens(
        self,
        x: Float[Array, "B N C"],
        weights: Float[Array, "B H N G"],
    ) -> Float[Array, "B H G {self.head_dim}"]:
        """Pool the per-head point features into one token per slice."""
        fx_mid = self._heads(self.in_project_fx, x)
        tokens = jnp.einsum("bhnc,bhng->bhgc", fx_mid, weights)
        return tokens / (weights.sum(axis=2) + _SLICE_EPS)[..., None]

    def attend(
        self,
        tokens: Float[Array, "B H G D"],
        *,
        key: Key,
        inference: bool = False,
    ) -> Float[Array, "B H G D"]:
        """Gibbs-kernel attention among the slice tokens of each head."""
        q = lorentz_lift(_apply(self.to_q, tokens))
        k = lorentz_lift(_apply(self.to_k, tokens))
        dist = lorentz_distance(q, k, self.acosh_eps)  # (B, H, G, G)
        if self.token_temperature is None:
            tau = self.temperature
        else:
            tau = jnp.clip(self.token_temperature, *_TEMPERATURE_RANGE)
        logits = (-dist / tau).astype(_acc_dtype(dist))
        attn = jax.nn.softmax(logits, axis=-1).astype(tokens.dtype)
        attn = self.dropout(attn, key=key, inference=inference)
        return attn @ _apply(self.to_v, tokens)

    def deslice(
        self,
        tokens: Float[Array, "B H G D"],
        weights: Float[Array, "B H N G"],
        *,
        key: Key,
        inference: bool = False,
    ) -> Float[Array, "B N C"]:
        """Broadcast the tokens back to the points and project out."""
        B, H, N, _ = weights.shape
        out = jnp.einsum("bhgc,bhng->bnhc", tokens, weights)
        out = _apply(self.to_out, out.reshape(B, N, H * self.head_dim))
        return self.dropout(out, key=key, inference=inference)

    def __call__(
        self,
        x: Float[Array, "B N C"],
        *,
        key: Key,
        inference: bool = False,
    ) -> Float[Array, "B N C"]:
        attn_key, out_key = jr.split(key)
        weights = self.slice_weights(x)
        tokens = self.slice_tokens(x, weights)
        tokens = self.attend(tokens, key=attn_key, inference=inference)
        return self.deslice(tokens, weights, key=out_key, inference=inference)


class HNOBlock(eqx.Module):
    """Encoder block: a hyperbolic slice-attention residual, then a pointwise MLP
    residual; the last block also projects to the output channels."""

    ln_1: eqx.nn.LayerNorm
    attn: HyperbolicSliceAttention
    ln_2: eqx.nn.LayerNorm
    mlp: MLP
    ln_3: eqx.nn.LayerNorm | None
    proj: eqx.nn.Linear | None

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_slices: int = 64,
        hyp_dim: int = 16,
        dropout: float = 0.0,
        mlp_ratio: float = 2.0,
        act: str = "gelu",
        last_layer: bool = False,
        out_dim: int = 1,
        share_temperature: bool = False,
        acosh_eps: float = _ACOSH_EPS,
        *,
        key: Key,
    ):
        keys = jr.split(key, 3)
        self.ln_1 = eqx.nn.LayerNorm(hidden_dim)
        self.attn = HyperbolicSliceAttention(
            hidden_dim,
            num_heads=num_heads,
            head_dim=hidden_dim // num_heads,
            num_slices=num_slices,
            hyp_dim=hyp_dim,
            dropout=dropout,
            share_temperature=share_temperature,
            acosh_eps=acosh_eps,
            key=keys[0],
        )
        self.ln_2 = eqx.nn.LayerNorm(hidden_dim)
        self.mlp = MLP(
            hidden_dim,
            int(hidden_dim * mlp_ratio),
            hidden_dim,
            n_layers=0,
            act=act,
            res=False,
            key=keys[1],
        )
        if last_layer:
            self.ln_3 = eqx.nn.LayerNorm(hidden_dim)
            self.proj = eqx.nn.Linear(hidden_dim, out_dim, key=keys[2])
        else:
            self.ln_3 = None
            self.proj = None

    def __call__(
        self,
        fx: Float[Array, "B N C"],
        *,
        key: Key,
        inference: bool = False,
    ) -> Float[Array, "B N C"]:
        fx = self.attn(_apply(self.ln_1, fx), key=key, inference=inference) + fx
        fx = self.mlp(_apply(self.ln_2, fx)) + fx
        if self.proj is not None:
            fx = _apply(self.proj, _apply(self.ln_3, fx))
        return fx


def _grid_features(pos: Float[Array, "B N d"], ref: int) -> Float[Array, "B N ref**d"]:
    """Distances from every point to the `ref^d` nodes of the reference's fixed
    grid (the `unified_pos` features), in `ij` order."""
    d = pos.shape[-1]
    axes = [jnp.linspace(lo, hi, ref, dtype=pos.dtype) for lo, hi in _GRID_BOUNDS[d]]
    grid = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, d)
    return jnp.sqrt(jnp.sum((pos[:, :, None, :] - grid) ** 2, axis=-1))


class HNO(eqx.Module):
    """
    Hyperbolic Neural Operator on unstructured meshes (the reference's
    large-scale CFD model).

    The reference configurations, with the backbone defaults:

    - AirfRANS: `HNO(space_dim=2, fun_dim=5, out_dim=4, unified_pos=True)`,
      with `x` the node positions and `fx` the other five node features, so that
      `(x, fx)` is the reference's `data.x`.
    - ShapeNet Car: `HNO(space_dim=3, fun_dim=4, out_dim=4,
      share_temperature=True)`, with `x` the positions and `fx` the signed
      distance and normals.

    Call `init_weights` to initialize it as the reference does.
    """

    preprocess: MLP
    blocks: list
    placeholder: Array
    unified_pos: bool = eqx.field(static=True)
    ref: int = eqx.field(static=True)

    def __init__(
        self,
        space_dim: int = 2,
        fun_dim: int = 0,
        out_dim: int = 1,
        num_layers: int = 8,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_slices: int = 64,
        hyp_dim: int = 16,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        act: str = "gelu",
        unified_pos: bool = False,
        ref: int = 8,
        share_temperature: bool = False,
        acosh_eps: float = _ACOSH_EPS,
        *,
        key: Key,
    ):
        """
        **Args:**
        - `space_dim`: Coordinate dimension.
        - `fun_dim`: Number of per-point input features besides the coordinates.
        - `out_dim`: Number of predicted channels per point.
        - `num_layers`, `hidden_dim`, `num_heads`, `num_slices`, `mlp_ratio`,
          `dropout`, `act`: The backbone hyperparameters; `num_slices` is `G`,
          per head.
        - `hyp_dim`: Dimension of the hyperbolic queries and keys.
        - `unified_pos`: Append the distances from each point to a fixed
          `ref^space_dim` grid to the input features; needs `space_dim` 2 or 3.
        - `ref`: Grid nodes per axis under `unified_pos`.
        - `share_temperature`: Score the token attention with the slicing
          temperature (the ShapeNet Car model); see `HyperbolicSliceAttention`.
        - `acosh_eps`: Clamp of the arcosh argument at `1 + acosh_eps`.
        """
        if unified_pos and space_dim not in _GRID_BOUNDS:
            raise ValueError(
                f"unified_pos needs space_dim 2 or 3, got space_dim={space_dim}"
            )
        self.unified_pos = unified_pos
        self.ref = ref

        keys = jr.split(key, 2 + num_layers)
        in_dim = space_dim + fun_dim + (ref**space_dim if unified_pos else 0)
        self.preprocess = MLP(
            in_dim,
            hidden_dim * 2,
            hidden_dim,
            n_layers=0,
            act=act,
            res=False,
            key=keys[0],
        )
        self.blocks = [
            HNOBlock(
                hidden_dim,
                num_heads,
                num_slices=num_slices,
                hyp_dim=hyp_dim,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
                act=act,
                last_layer=(i == num_layers - 1),
                out_dim=out_dim,
                share_temperature=share_temperature,
                acosh_eps=acosh_eps,
                key=k,
            )
            for i, k in enumerate(keys[2:])
        ]
        self.placeholder = jr.uniform(keys[1], (hidden_dim,)) / hidden_dim

    def __call__(
        self,
        x: Float[Array, "B N space_dim"],
        fx: Float[Array, "B N fun_dim"] | None = None,
        *,
        key: Key,
        inference: bool = False,
    ) -> Float[Array, "B N out_dim"]:
        """
        **Args:**
        - `x`: Point coordinates.
        - `fx`: Per-point input features, or `None`.
        - `key`: Random key for dropout.
        - `inference`: Whether to disable dropout.

        **Returns:**
        The prediction at the input points.
        """
        h = x if fx is None else jnp.concatenate((x, fx), axis=-1)
        if self.unified_pos:
            h = jnp.concatenate((h, _grid_features(x, self.ref)), axis=-1)
        h = self.preprocess(h) + self.placeholder[None, None, :]
        for block, k in zip(self.blocks, jr.split(key, len(self.blocks))):
            h = block(h, key=k, inference=inference)
        return h


# ---------------------------------------------------------------------------
# Weight initialization (matches the PyTorch reference)
# ---------------------------------------------------------------------------
def _is_linear(x):
    return isinstance(x, eqx.nn.Linear)


def init_weights(model: eqx.Module, *, key: Key) -> eqx.Module:
    """Re-initialize `model` as the reference's `initialize_weights` does at
    construction: truncated-normal (std 0.02) weights and zero biases on every
    `Linear`, `in_project_slice` included (the reference's orthogonal init of
    it is overwritten by the same pass). LayerNorms keep their unit weights and
    zero biases."""

    def get_linears(m):
        return [x for x in jax.tree.leaves(m, is_leaf=_is_linear) if _is_linear(x)]

    linears = get_linears(model)
    keys = jr.split(key, len(linears))
    std = 0.02

    def reinit(linear, k):
        # torch's trunc_normal_ cuts at the absolute values +-2, i.e. +-100 std.
        shape, dtype = linear.weight.shape, linear.weight.dtype
        weight = jr.truncated_normal(k, -2.0 / std, 2.0 / std, shape, dtype) * std
        linear = eqx.tree_at(lambda lin: lin.weight, linear, weight)
        if linear.bias is not None:
            linear = eqx.tree_at(
                lambda lin: lin.bias, linear, jnp.zeros_like(linear.bias)
            )
        return linear

    return eqx.tree_at(
        get_linears, model, [reinit(lin, k) for lin, k in zip(linears, keys)]
    )
