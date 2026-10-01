"""
JAX/Equinox port of FlashSlice and the Transolver it accelerates.

FlashSlice is the companion code of Wen & Mishra, "Does Transolver really need a
Transformer?" (arXiv:2609.32525), https://github.com/Shizheng-Wen/flashslice.
Both of its parts are ported:

1. `fused_slice` / `fused_deslice`, the slice/deslice coupling of physics
   attention. With per-point slice weights `w = softmax_G((x_mid W^T + b) / tau)`,
   slice pools `z_num[g] = sum_n w[n, g] fx_mid[n]` and `s[g] = sum_n w[n, g]`,
   and deslice broadcasts tokens back, `out[n] = sum_g w[n, g] z'[g]`. The
   reference runs these as Triton kernels; here each is a `jax.custom_vjp` whose
   forward and backward stream over the points in tiles of `chunk_size`, forming
   `w` for one tile, using it and discarding it. Only the op's inputs are saved
   for the backward, which recomputes `w` tile by tile and applies the softmax
   Jacobian per point (FlashAttention's trade), so the `(B, H, N, G)` slice-weight
   tensor is never materialized in either pass.
2. The Transolver of the reference, with each ablation of the paper as one flag
   (at most one at a time) and `use_fused_slice` as an implementation switch: the
   fused and eager paths compute the same function from the same parameters.

What is specific to Triton and CUDA is not ported: the tuned tile tables, the
dot-precision modes (pass `precision` to the ops, or use
`jax.default_matmul_precision`), and the G-blocked kernel family with its saved
softmax statistics (`stats=` / `return_stats=`). That family exists because a
Triton tile holds at most 128 slots in registers; an XLA tile has no such limit,
so every tile here holds the whole slot axis, as the reference's single-tile
kernels do, and `chunk_size` bounds the `(B, H, chunk_size, G)` working set for
any `G`. The reference's eager fallback for head widths above 256 is unnecessary
for the same reason.

Arrays carry an explicit leading batch axis, mirroring the PyTorch reference in
`test/_flashslice_test.py`, as `zoo/_transolver_3.py` does.
"""

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jaxtyping import Array, Float, Key

from ._transolver_3 import MLP

# Added to the slice mass before normalizing the tokens, as in the reference.
_SLICE_EPS = 1e-5


# ---------------------------------------------------------------------------
# FlashSlice: slice / deslice streamed over the points
# ---------------------------------------------------------------------------
def _stream(body, carry, chunk_size, *arrays):
    """Fold `body(carry, start, *tiles)` over consecutive point tiles of `arrays`,
    each `(B, N, ...)`. Full tiles run in a `fori_loop`; a ragged tail runs once
    more on a static slice, so nothing is padded."""
    n = arrays[0].shape[1]
    size = max(1, min(chunk_size, n))
    num_full, tail = divmod(n, size)

    def step(i, carry):
        start = i * size
        tiles = [lax.dynamic_slice_in_dim(a, start, size, axis=1) for a in arrays]
        return body(carry, start, *tiles)

    carry = lax.fori_loop(0, num_full, step, carry)
    if tail:
        start = num_full * size
        carry = body(carry, start, *(a[:, start:] for a in arrays))
    return carry


def _put(buffer, tile, start):
    """Write a point tile into rows `start:start + n` of a `(B, N, ...)` buffer."""
    return lax.dynamic_update_slice_in_dim(
        buffer, tile.astype(buffer.dtype), start, axis=1
    )


def _acc_dtype(*arrays):
    """fp32 accumulation for 16-bit inputs, as in the reference; fp64 stays fp64."""
    return jnp.result_type(jnp.float32, *arrays)


def _cast_like(values, likes):
    return tuple(v.astype(like.dtype) for v, like in zip(values, likes))


def _slice_weights(x, weight, bias, tau, precision):
    """Logits `a = (x W^T + b) / tau` of a point tile (bias before the division,
    as in the eager module) and the slice weights `w = softmax_G(a)`, both
    `(B, H, n, G)`."""
    logits = jnp.einsum("bnhd,bhgd->bhng", x, weight, precision=precision)
    logits = (logits + bias[:, :, None, :]) / tau[:, None, None]
    return logits, jax.nn.softmax(logits, axis=-1)


def _slice_weights_bwd(x, params, logits, w, dw, precision):
    """Pull `dL/dw` of a point tile back through the softmax and the logits.
    Returns the tile's `dL/dx` and its share of `dL/d(weight, bias, tau)`."""
    weight, _, tau = params
    dl = w * (dw - jnp.sum(w * dw, axis=-1, keepdims=True))  # dL/da
    dlr = dl / tau[:, None, None]  # dL/d(x W^T + b)
    dx = jnp.einsum("bhng,bhgd->bnhd", dlr, weight, precision=precision)
    d_weight = jnp.einsum("bhng,bnhd->bhgd", dlr, x, precision=precision)
    # da/dtau = -a / tau. `dl` sums to zero over G at every point, so the sum
    # runs over G first and across points after, as in the reference kernels.
    d_tau = -jnp.sum(jnp.sum(dl * logits, axis=-1), axis=(0, 2)) / tau
    return dx, (d_weight, jnp.sum(dlr, axis=2), d_tau)


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6))
def _slice(x_mid, fx_mid, weight, bias, tau, chunk_size, precision):
    acc = _acc_dtype(x_mid, fx_mid, weight, bias, tau)
    params = tuple(p.astype(acc) for p in (weight, bias, tau))

    def body(carry, start, x, fx):
        z_num, s = carry
        _, w = _slice_weights(x.astype(acc), *params, precision)
        pooled = jnp.einsum("bhng,bnhv->bhgv", w, fx.astype(acc), precision=precision)
        return z_num + pooled, s + jnp.sum(w, axis=2)

    B, _, H, DV = fx_mid.shape
    G = weight.shape[2]
    init = (jnp.zeros((B, H, G, DV), acc), jnp.zeros((B, H, G), acc))
    return _stream(body, init, chunk_size, x_mid, fx_mid)


def _slice_fwd(x_mid, fx_mid, weight, bias, tau, chunk_size, precision):
    out = _slice(x_mid, fx_mid, weight, bias, tau, chunk_size, precision)
    # The inputs alone: the backward recomputes w rather than saving it.
    return out, (x_mid, fx_mid, weight, bias, tau)


def _slice_bwd(chunk_size, precision, residuals, cotangents):
    x_mid, fx_mid, weight, bias, tau = residuals
    acc = _acc_dtype(*residuals)
    params = tuple(p.astype(acc) for p in (weight, bias, tau))
    dz_num, ds = (c.astype(acc) for c in cotangents)

    def body(carry, start, x, fx):
        dx, dfx, d_params = carry
        x, fx = x.astype(acc), fx.astype(acc)
        logits, w = _slice_weights(x, *params, precision)
        dfx_tile = jnp.einsum("bhng,bhgv->bnhv", w, dz_num, precision=precision)
        dw = jnp.einsum("bnhv,bhgv->bhng", fx, dz_num, precision=precision)
        dx_tile, d_tile = _slice_weights_bwd(
            x, params, logits, w, dw + ds[:, :, None, :], precision
        )
        d_params = jax.tree.map(jnp.add, d_params, d_tile)
        return _put(dx, dx_tile, start), _put(dfx, dfx_tile, start), d_params

    init = (
        jnp.zeros_like(x_mid),
        jnp.zeros_like(fx_mid),
        jax.tree.map(jnp.zeros_like, params),
    )
    dx, dfx, d_params = _stream(body, init, chunk_size, x_mid, fx_mid)
    return (dx, dfx, *_cast_like(d_params, (weight, bias, tau)))


_slice.defvjp(_slice_fwd, _slice_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6))
def _deslice(x_mid, weight, bias, tau, tokens, chunk_size, precision):
    acc = _acc_dtype(x_mid, weight, bias, tau, tokens)
    params = tuple(p.astype(acc) for p in (weight, bias, tau))
    tokens = tokens.astype(acc)

    def body(out, start, x):
        _, w = _slice_weights(x.astype(acc), *params, precision)
        tile = jnp.einsum("bhng,bhgv->bnhv", w, tokens, precision=precision)
        return _put(out, tile, start)

    B, N, H, _ = x_mid.shape
    out = jnp.zeros((B, N, H, tokens.shape[3]), x_mid.dtype)
    return _stream(body, out, chunk_size, x_mid)


def _deslice_fwd(x_mid, weight, bias, tau, tokens, chunk_size, precision):
    out = _deslice(x_mid, weight, bias, tau, tokens, chunk_size, precision)
    return out, (x_mid, weight, bias, tau, tokens)


def _deslice_bwd(chunk_size, precision, residuals, d_out):
    x_mid, weight, bias, tau, tokens = residuals
    acc = _acc_dtype(*residuals)
    params = tuple(p.astype(acc) for p in (weight, bias, tau))
    tokens_acc = tokens.astype(acc)

    def body(carry, start, x, d_out):
        dx, d_tokens, d_params = carry
        x, d_out = x.astype(acc), d_out.astype(acc)
        logits, w = _slice_weights(x, *params, precision)
        d_tokens = d_tokens + jnp.einsum(
            "bhng,bnhv->bhgv", w, d_out, precision=precision
        )
        dw = jnp.einsum("bnhv,bhgv->bhng", d_out, tokens_acc, precision=precision)
        dx_tile, d_tile = _slice_weights_bwd(x, params, logits, w, dw, precision)
        d_params = jax.tree.map(jnp.add, d_params, d_tile)
        return _put(dx, dx_tile, start), d_tokens, d_params

    init = (
        jnp.zeros_like(x_mid),
        jnp.zeros_like(tokens_acc),
        jax.tree.map(jnp.zeros_like, params),
    )
    dx, d_tokens, d_params = _stream(body, init, chunk_size, x_mid, d_out)
    d_weight, d_bias, d_tau = _cast_like(d_params, (weight, bias, tau))
    return dx, d_weight, d_bias, d_tau, d_tokens.astype(tokens.dtype)


_deslice.defvjp(_deslice_fwd, _deslice_bwd)


def _broadcast_projection(x_mid, weight, bias, tau, chunk_size):
    """Validate the slice projection and broadcast it to `(B, H, G, D)` and
    `(B, H, G)`; autodiff sums the gradients back to the shapes given."""
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    B, _, H, D = x_mid.shape
    if weight.ndim not in (2, 3, 4) or weight.shape[-1] != D:
        raise ValueError(
            f"weight must be (G, D), (H, G, D) or (B, H, G, D) with D={D}, "
            f"got shape {weight.shape}"
        )
    G = weight.shape[-2]
    if bias is None:
        bias = jnp.zeros((G,), weight.dtype)
    elif bias.ndim not in (1, 2, 3) or bias.shape[-1] != G:
        raise ValueError(
            f"bias must be None, (G,), (H, G) or (B, H, G) with G={G}, "
            f"got shape {bias.shape}"
        )
    if tau.shape != (H,):
        raise ValueError(f"tau must have shape ({H},), got {tau.shape}")
    return jnp.broadcast_to(weight, (B, H, G, D)), jnp.broadcast_to(bias, (B, H, G))


def fused_slice(
    x_mid: Float[Array, "B N H D"],
    fx_mid: Float[Array, "B N H DV"],
    weight: Float[Array, "... G D"],
    bias: Float[Array, "... G"] | None,
    tau: Float[Array, " H"],
    *,
    chunk_size: int = 4096,
    precision=None,
) -> tuple[Float[Array, "B H G DV"], Float[Array, "B H G"]]:
    """
    Pool point features into `G` tokens per head without materializing the
    slice weights: `z_num[g] = sum_n w[n, g] fx_mid[n]`, `s[g] = sum_n w[n, g]`
    with `w = softmax_G((x_mid W^T + b) / tau)`.

    **Args:**
    - `x_mid`: Features the slice weights are computed from.
    - `fx_mid`: Features that are pooled; the value width `DV` may differ from `D`.
    - `weight`: Slice projection, shared `(G, D)`, per head `(H, G, D)` or per
      sample and head `(B, H, G, D)`; its gradient comes back in the same shape.
    - `bias`: `None`, `(G,)`, `(H, G)` or `(B, H, G)`.
    - `tau`: Per-head temperature.
    - `chunk_size`: Points per tile, bounding the `(B, H, chunk_size, G)` working
      set of either pass.
    - `precision`: Passed to every contraction, as in `jnp.einsum`.

    **Returns:**
    `z_num` and `s`, accumulated in at least fp32; normalize outside as
    `z_num / (s + eps)[..., None]`.
    """
    if fx_mid.shape[:3] != x_mid.shape[:3]:
        raise ValueError(
            f"fx_mid {fx_mid.shape} and x_mid {x_mid.shape} disagree on (B, N, H)"
        )
    weight, bias = _broadcast_projection(x_mid, weight, bias, tau, chunk_size)
    return _slice(x_mid, fx_mid, weight, bias, tau, chunk_size, precision)


def fused_deslice(
    x_mid: Float[Array, "B N H D"],
    weight: Float[Array, "... G D"],
    bias: Float[Array, "... G"] | None,
    tau: Float[Array, " H"],
    tokens: Float[Array, "B H G DV"],
    *,
    chunk_size: int = 4096,
    precision=None,
) -> Float[Array, "B N H DV"]:
    """
    Broadcast tokens back to the points without materializing the slice
    weights: `out[n] = sum_g w[n, g] tokens[g]`. `weight`, `bias`, `tau`,
    `chunk_size` and `precision` are as in `fused_slice`; a tied coupling passes
    the slice's own projection and temperature.

    **Returns:**
    `out` in `x_mid`'s dtype, in the layout an output projection expects after a
    reshape to `(B, N, H * DV)`.
    """
    B, _, H, _ = x_mid.shape
    G = weight.shape[-2]
    if tokens.shape[:3] != (B, H, G):
        raise ValueError(f"tokens must be ({B}, {H}, {G}, DV), got {tokens.shape}")
    weight, bias = _broadcast_projection(x_mid, weight, bias, tau, chunk_size)
    return _deslice(x_mid, weight, bias, tau, tokens, chunk_size, precision)


# ---------------------------------------------------------------------------
# Transolver
# ---------------------------------------------------------------------------
def _apply(layer, x):
    """Apply a per-vector layer (`Linear`, `LayerNorm`, ...) over every leading axis."""
    for _ in range(x.ndim - 1):
        layer = jax.vmap(layer)
    return layer(x)


class PhysicsAttentionIrregularMesh(eqx.Module):
    """Physics attention for irregular meshes, with the paper's ablation flags.

    The defaults reproduce the original Transolver layer. `use_fused_slice`
    routes slice/deslice through `fused_slice` / `fused_deslice`; the
    parameters and the function are the same either way.
    """

    in_project_x: eqx.nn.Linear
    in_project_fx: eqx.nn.Linear
    in_project_slice: eqx.nn.Linear
    in_project_deslice: eqx.nn.Linear | None
    to_q: eqx.nn.Linear
    to_k: eqx.nn.Linear
    to_v: eqx.nn.Linear
    slice_down: eqx.nn.Linear | eqx.nn.Identity
    to_out: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    temperature: Array
    deslice_temperature: Array | None

    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    slice_head_dim: int = eqx.field(static=True)
    untied_deslice: bool = eqx.field(static=True)
    no_token_attention: bool = eqx.field(static=True)
    use_fused_slice: bool = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)

    def __init__(
        self,
        d_in: int,
        num_heads: int = 8,
        head_dim: int = 64,
        num_slices: int = 64,
        dropout: float = 0.0,
        untied_deslice: bool = False,
        no_token_attention: bool = False,
        slice_head_dim: int | None = None,
        use_fused_slice: bool = False,
        chunk_size: int = 4096,
        *,
        key: Key,
    ):
        """
        **Args:**
        - `d_in`: Input (and output) feature dimension.
        - `num_heads`: Number of heads.
        - `head_dim`: Per-head width of the full-resolution path.
        - `num_slices`: Number of slices `G` per head.
        - `dropout`: Dropout probability, on the token attention and the output.
        - `untied_deslice`: Deslice with its own projection and temperature
          instead of reusing the slice weights.
        - `no_token_attention`: Replace the attention among slice tokens by the
          per-token linear map `to_v`; tokens no longer interact.
        - `slice_head_dim`: Width of the token attention, which acts on `G`
          tokens only. Defaults to `head_dim`, the original layer.
        - `use_fused_slice`: Run slice/deslice through the streamed ops.
        - `chunk_size`: Points per tile of the streamed ops.
        """
        slice_head_dim = head_dim if slice_head_dim is None else slice_head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.slice_head_dim = slice_head_dim
        self.untied_deslice = untied_deslice
        self.no_token_attention = no_token_attention
        self.use_fused_slice = use_fused_slice
        self.chunk_size = chunk_size

        keys = jr.split(key, 9)
        inner_dim = num_heads * head_dim
        self.in_project_x = eqx.nn.Linear(d_in, inner_dim, key=keys[0])
        self.in_project_fx = eqx.nn.Linear(d_in, inner_dim, key=keys[1])
        self.in_project_slice = eqx.nn.Linear(head_dim, num_slices, key=keys[2])
        self.to_q = eqx.nn.Linear(head_dim, slice_head_dim, use_bias=False, key=keys[3])
        self.to_k = eqx.nn.Linear(head_dim, slice_head_dim, use_bias=False, key=keys[4])
        self.to_v = eqx.nn.Linear(head_dim, slice_head_dim, use_bias=False, key=keys[5])
        if slice_head_dim == head_dim:
            self.slice_down = eqx.nn.Identity()
        else:
            self.slice_down = eqx.nn.Linear(
                slice_head_dim, head_dim, use_bias=False, key=keys[6]
            )
        self.to_out = eqx.nn.Linear(inner_dim, d_in, key=keys[7])
        self.dropout = eqx.nn.Dropout(p=dropout)
        self.temperature = jnp.ones((1, num_heads, 1, 1)) * 0.5

        if untied_deslice:
            self.in_project_deslice = eqx.nn.Linear(head_dim, num_slices, key=keys[8])
            self.deslice_temperature = jnp.ones((1, num_heads, 1, 1)) * 0.5
        else:
            self.in_project_deslice = None
            self.deslice_temperature = None

    def _eager_weights(self, project, temperature, x_mid):
        """Materialized slice weights `(B, H, N, G)` from `x_mid` `(B, N, H, D)`."""
        logits = jnp.transpose(_apply(project, x_mid), (0, 2, 1, 3))
        return jax.nn.softmax(logits / temperature, axis=-1)

    def _mix(self, tokens, *, key, inference):
        """Attention among the slice tokens, or `to_v` alone without it."""
        if self.no_token_attention:
            out = _apply(self.to_v, tokens)
        else:
            q, k, v = (_apply(lin, tokens) for lin in (self.to_q, self.to_k, self.to_v))
            dots = jnp.einsum("bhgd,bhed->bhge", q, k) * self.slice_head_dim**-0.5
            attn = self.dropout(
                jax.nn.softmax(dots, axis=-1), key=key, inference=inference
            )
            out = jnp.einsum("bhge,bhed->bhgd", attn, v)
        return _apply(self.slice_down, out)

    def _project_out(self, out_x, *, key, inference):
        return self.dropout(_apply(self.to_out, out_x), key=key, inference=inference)

    def _fused(self, x, *, key, inference):
        B, N, _ = x.shape
        H, D = self.num_heads, self.head_dim
        mix_key, out_key = jr.split(key)

        fx_mid = _apply(self.in_project_fx, x).reshape(B, N, H, D)
        x_mid = _apply(self.in_project_x, x).reshape(B, N, H, D)
        tau = self.temperature.reshape(H)
        slice_proj = (self.in_project_slice.weight, self.in_project_slice.bias, tau)
        z_num, s = fused_slice(x_mid, fx_mid, *slice_proj, chunk_size=self.chunk_size)
        tokens = self._mix(
            z_num / (s + _SLICE_EPS)[..., None], key=mix_key, inference=inference
        )

        if self.untied_deslice:
            assert self.in_project_deslice is not None
            assert self.deslice_temperature is not None
            deslice_proj = (
                self.in_project_deslice.weight,
                self.in_project_deslice.bias,
                self.deslice_temperature.reshape(H),
            )
        else:
            deslice_proj = slice_proj
        out_x = fused_deslice(x_mid, *deslice_proj, tokens, chunk_size=self.chunk_size)
        return self._project_out(
            out_x.reshape(B, N, H * D), key=out_key, inference=inference
        )

    def __call__(
        self,
        x: Float[Array, "B N C"],
        slice_weights_in: Float[Array, "B H N G"] | None = None,
        *,
        key: Key,
        inference: bool = False,
    ) -> tuple[Float[Array, "B N C"], Float[Array, "B H N G"] | None]:
        """
        **Args:**
        - `x`: Point features.
        - `slice_weights_in`: Slice weights to use instead of computing them
          (the `share_slice_across_layers` ablation); eager path only.
        - `key`: Random key for dropout.
        - `inference`: Whether to disable dropout.

        **Returns:**
        The layer output and the slice weights it used, which the fused path
        never forms and returns as `None`.
        """
        if self.use_fused_slice:
            if slice_weights_in is not None:
                raise ValueError(
                    "use_fused_slice cannot consume externally provided slice weights."
                )
            return self._fused(x, key=key, inference=inference), None

        B, N, _ = x.shape
        H, D = self.num_heads, self.head_dim
        mix_key, out_key = jr.split(key)

        fx_mid = _apply(self.in_project_fx, x).reshape(B, N, H, D)
        if slice_weights_in is None or self.untied_deslice:
            x_mid = _apply(self.in_project_x, x).reshape(B, N, H, D)
        if slice_weights_in is None:
            slice_weights = self._eager_weights(
                self.in_project_slice, self.temperature, x_mid
            )
        else:
            slice_weights = slice_weights_in
        slice_norm = slice_weights.sum(axis=2)
        tokens = jnp.einsum("bnhc,bhng->bhgc", fx_mid, slice_weights)
        tokens = tokens / (slice_norm + _SLICE_EPS)[..., None]
        tokens = self._mix(tokens, key=mix_key, inference=inference)

        if self.untied_deslice:
            deslice_weights = self._eager_weights(
                self.in_project_deslice, self.deslice_temperature, x_mid
            )
        else:
            deslice_weights = slice_weights
        out_x = jnp.einsum("bhgc,bhng->bnhc", tokens, deslice_weights)
        out = self._project_out(
            out_x.reshape(B, N, H * D), key=out_key, inference=inference
        )
        return out, slice_weights


class TransolverBlock(eqx.Module):
    """Encoder block: a physics-attention residual, then a pointwise MLP residual."""

    ln_1: eqx.nn.LayerNorm
    attn: PhysicsAttentionIrregularMesh | None
    ln_2: eqx.nn.LayerNorm
    mlp: MLP
    ln_3: eqx.nn.LayerNorm | None
    mlp2: eqx.nn.Linear | None

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_slices: int = 32,
        dropout: float = 0.0,
        mlp_ratio: int = 4,
        act: str = "gelu",
        last_layer: bool = False,
        out_dim: int = 1,
        head_dim: int | None = None,
        slice_head_dim: int | None = None,
        untied_deslice: bool = False,
        no_token_attention: bool = False,
        mlp_only: bool = False,
        use_fused_slice: bool = False,
        chunk_size: int = 4096,
        *,
        key: Key,
    ):
        keys = jr.split(key, 3)
        # Built under `mlp_only` too, as in the reference, though it is then unused.
        self.ln_1 = eqx.nn.LayerNorm(hidden_dim)
        if mlp_only:
            self.attn = None
        else:
            self.attn = PhysicsAttentionIrregularMesh(
                d_in=hidden_dim,
                num_heads=num_heads,
                head_dim=hidden_dim // num_heads if head_dim is None else head_dim,
                num_slices=num_slices,
                dropout=dropout,
                untied_deslice=untied_deslice,
                no_token_attention=no_token_attention,
                slice_head_dim=slice_head_dim,
                use_fused_slice=use_fused_slice,
                chunk_size=chunk_size,
                key=keys[0],
            )
        self.ln_2 = eqx.nn.LayerNorm(hidden_dim)
        self.mlp = MLP(
            hidden_dim,
            hidden_dim * mlp_ratio,
            hidden_dim,
            n_layers=0,
            act=act,
            res=False,
            key=keys[1],
        )
        if last_layer:
            self.ln_3 = eqx.nn.LayerNorm(hidden_dim)
            self.mlp2 = eqx.nn.Linear(hidden_dim, out_dim, key=keys[2])
        else:
            self.ln_3 = None
            self.mlp2 = None

    def __call__(
        self,
        fx: Float[Array, "B N C"],
        slice_weights_in: Float[Array, "B H N G"] | None = None,
        *,
        key: Key,
        inference: bool = False,
    ) -> tuple[Float[Array, "B N C"], Float[Array, "B H N G"] | None]:
        """Returns the block output and the slice weights the attention used:
        `None` under `mlp_only`, which has no attention, and on the fused path."""
        slice_weights = None
        if self.attn is not None:
            attn_out, slice_weights = self.attn(
                _apply(self.ln_1, fx), slice_weights_in, key=key, inference=inference
            )
            fx = attn_out + fx
        fx = self.mlp(_apply(self.ln_2, fx)) + fx
        if self.mlp2 is not None:
            fx = _apply(self.mlp2, _apply(self.ln_3, fx))
        return fx, slice_weights


class TokenTransformerBlock(eqx.Module):
    """Pre-LN transformer block over slice tokens; used only by `slice_once`."""

    ln_1: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    ln_2: eqx.nn.LayerNorm
    mlp: MLP

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        mlp_ratio: int = 4,
        act: str = "gelu",
        *,
        key: Key,
    ):
        keys = jr.split(key, 2)
        self.ln_1 = eqx.nn.LayerNorm(hidden_dim)
        # Biases throughout, as torch's nn.MultiheadAttention.
        self.attn = eqx.nn.MultiheadAttention(
            num_heads,
            hidden_dim,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            dropout_p=dropout,
            key=keys[0],
        )
        self.ln_2 = eqx.nn.LayerNorm(hidden_dim)
        self.mlp = MLP(
            hidden_dim,
            hidden_dim * mlp_ratio,
            hidden_dim,
            n_layers=0,
            act=act,
            res=False,
            key=keys[1],
        )

    def __call__(
        self, tokens: Float[Array, "B G C"], *, key: Key, inference: bool = False
    ) -> Float[Array, "B G C"]:
        def attend(t, k):
            return self.attn(t, t, t, key=k, inference=inference)

        h = _apply(self.ln_1, tokens)
        tokens = jax.vmap(attend)(h, jr.split(key, tokens.shape[0])) + tokens
        return self.mlp(_apply(self.ln_2, tokens)) + tokens


class Transolver(eqx.Module):
    """
    Transolver on unstructured point clouds, with the ablations of the paper.

    For the PyTorch implementation see https://github.com/Shizheng-Wen/flashslice;
    the defaults reproduce the original Transolver. Each ablation flag changes
    exactly one thing, and at most one may be set:

    - `no_token_attention`: the attention among slice tokens becomes a per-token
      linear map; tokens stop interacting.
    - `untie_slice_weights`: deslice gets its own projection and temperature.
    - `share_slice_across_layers`: slice weights are computed once, in the first
      layer, and reused by every layer.
    - `slice_once`: slice once, run a transformer on the `G` tokens, deslice once
      (the Perceiver limit; no point stream).
    - `mlp_only`: the attention sublayer is removed entirely.

    `use_fused_slice` is orthogonal to them: an implementation switch that
    leaves the function and the parameters unchanged. It is refused with
    `share_slice_across_layers` and `slice_once`, which need the slice weights
    the fused ops never form.
    """

    preprocess: MLP
    blocks: list | None
    slice_proj: eqx.nn.Linear | None
    slice_temperature: Array | None
    token_blocks: list | None
    out_norm: eqx.nn.LayerNorm | None
    out_head: eqx.nn.Linear | None
    placeholder: Array
    share_slice_across_layers: bool = eqx.field(static=True)

    def __init__(
        self,
        space_dim: int = 3,
        fun_dim: int = 1,
        out_dim: int = 1,
        num_layers: int = 8,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_slices: int = 32,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
        act: str = "gelu",
        head_dim: int | None = None,
        slice_head_dim: int | None = None,
        untie_slice_weights: bool = False,
        share_slice_across_layers: bool = False,
        slice_once: bool = False,
        no_token_attention: bool = False,
        mlp_only: bool = False,
        use_fused_slice: bool = False,
        chunk_size: int = 4096,
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
        - `head_dim`: Per-head width of the full-resolution path; defaults to
          `hidden_dim // num_heads`.
        - `slice_head_dim`: Width of the token attention; defaults to `head_dim`.
        - `untie_slice_weights`, `share_slice_across_layers`, `slice_once`,
          `no_token_attention`, `mlp_only`: The ablation flags, see above.
        - `use_fused_slice`: Run slice/deslice through the streamed ops.
        - `chunk_size`: Points per tile of the streamed ops.
        """
        flags = (
            untie_slice_weights,
            share_slice_across_layers,
            slice_once,
            no_token_attention,
            mlp_only,
        )
        if sum(flags) > 1:
            raise ValueError("At most one ablation flag may be enabled at a time.")
        if use_fused_slice and (share_slice_across_layers or slice_once):
            raise ValueError(
                "use_fused_slice is incompatible with share_slice_across_layers and "
                "slice_once: both need the slice weights the fused ops never form."
            )
        self.share_slice_across_layers = share_slice_across_layers

        keys = jr.split(key, 4 + num_layers)
        self.preprocess = MLP(
            fun_dim + space_dim,
            hidden_dim * 2,
            hidden_dim,
            n_layers=0,
            act=act,
            res=False,
            key=keys[0],
        )
        if slice_once:
            self.blocks = None
            self.slice_proj = eqx.nn.Linear(hidden_dim, num_slices, key=keys[1])
            self.slice_temperature = jnp.array(0.5)
            self.token_blocks = [
                TokenTransformerBlock(
                    hidden_dim,
                    num_heads,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                    act=act,
                    key=k,
                )
                for k in keys[4:]
            ]
            self.out_norm = eqx.nn.LayerNorm(hidden_dim)
            self.out_head = eqx.nn.Linear(hidden_dim, out_dim, key=keys[2])
        else:
            self.blocks = [
                TransolverBlock(
                    hidden_dim,
                    num_heads,
                    num_slices=num_slices,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                    act=act,
                    last_layer=(i == num_layers - 1),
                    out_dim=out_dim,
                    head_dim=head_dim,
                    slice_head_dim=slice_head_dim,
                    untied_deslice=untie_slice_weights,
                    no_token_attention=no_token_attention,
                    mlp_only=mlp_only,
                    use_fused_slice=use_fused_slice,
                    chunk_size=chunk_size,
                    key=k,
                )
                for i, k in enumerate(keys[4:])
            ]
            self.slice_proj = None
            self.slice_temperature = None
            self.token_blocks = None
            self.out_norm = None
            self.out_head = None
        self.placeholder = jr.uniform(keys[3], (hidden_dim,)) / hidden_dim

    def _slice_once(self, fx, *, key, inference):
        assert self.slice_proj is not None and self.token_blocks is not None
        w = jax.nn.softmax(
            _apply(self.slice_proj, fx) / self.slice_temperature, axis=-1
        )  # (B, N, G)
        tokens = jnp.einsum("bnc,bng->bgc", fx, w)
        tokens = tokens / (w.sum(axis=1) + _SLICE_EPS)[..., None]
        for block, k in zip(self.token_blocks, jr.split(key, len(self.token_blocks))):
            tokens = block(tokens, key=k, inference=inference)
        out = jnp.einsum("bgc,bng->bnc", tokens, w) + fx  # point-level skip
        return _apply(self.out_head, _apply(self.out_norm, out))

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
        h = self.preprocess(h) + self.placeholder[None, None, :]
        if self.blocks is None:
            return self._slice_once(h, key=key, inference=inference)

        shared = None
        for block, k in zip(self.blocks, jr.split(key, len(self.blocks))):
            h, slice_weights = block(h, shared, key=k, inference=inference)
            if self.share_slice_across_layers and shared is None:
                shared = slice_weights
        return h


# ---------------------------------------------------------------------------
# Weight initialization (matches the PyTorch reference)
# ---------------------------------------------------------------------------
def _is_linear(x):
    return isinstance(x, eqx.nn.Linear)


def _is_mha(x):
    return isinstance(x, eqx.nn.MultiheadAttention)


def init_weights(model: eqx.Module, *, key: Key) -> eqx.Module:
    """
    Re-initialize `model` as the PyTorch reference does at construction:
    truncated-normal (std 0.02) weights and zero biases on every `Linear`,
    `in_project_slice` included (the reference's orthogonal init of it is
    overwritten by the same pass). The query/key/value projections of the
    `slice_once` token transformer instead take `nn.MultiheadAttention`'s own
    init, which that pass does not reach: Xavier-uniform over the fused
    `(3E, E)` in-projection and zero biases.
    """

    def get_linears(m):
        return [x for x in jax.tree.leaves(m, is_leaf=_is_linear) if _is_linear(x)]

    def get_in_projections(m):
        mhas = [x for x in jax.tree.leaves(m, is_leaf=_is_mha) if _is_mha(x)]
        return [
            proj.weight
            for mha in mhas
            for proj in (mha.query_proj, mha.key_proj, mha.value_proj)
        ]

    linears = get_linears(model)
    keys = jr.split(key, len(linears) + 1)
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

    model = eqx.tree_at(
        get_linears, model, [reinit(lin, k) for lin, k in zip(linears, keys)]
    )

    in_projections = get_in_projections(model)
    if in_projections:
        ks = jr.split(keys[-1], len(in_projections))

        def xavier(w, k):
            e_out, e_in = w.shape  # one third of the fused (3E, E) matrix
            bound = (6.0 / (e_in + 3 * e_out)) ** 0.5
            return jr.uniform(k, w.shape, w.dtype, -bound, bound)

        model = eqx.tree_at(
            get_in_projections,
            model,
            [xavier(w, k) for w, k in zip(in_projections, ks)],
        )
    return model
