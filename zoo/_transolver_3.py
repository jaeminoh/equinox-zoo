"""
JAX/Equinox port of the fused ("associativity-optimized") 3D Transolver.

Compared to `zoo/transolver.py`, the physics attention here avoids materializing
per-point head features:

1. `in_project` (the `x_mid` half) is fused with `in_project_slice` into a single
   `C -> G` matmul per head, so slice weights come from one projection;
2. raw features are aggregated into the slice domain first, then projected there
   (`O(G*C*D)` instead of `O(N*C*D)`);
3. deslicing projects slice tokens through `to_out_linear` *before* scattering
   back to points, saving `O(N*H*D*C)`.

Arrays carry an explicit leading batch axis `(B, N, C)`, mirroring the PyTorch
reference in `test/_transolver_3_test.py`. Only the dense path is ported; the
chunked-mesh path (`chunk_stats` / `forward_chunks`) and the `unified_pos` / `ref`
options of the reference are omitted.
"""

import functools

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Array, Float, Key


_ACTIVATION = {
    # torch's nn.GELU is the exact erf form; jax defaults to the tanh approximation.
    "gelu": functools.partial(jax.nn.gelu, approximate=False),
    "relu": jax.nn.relu,
    "silu": jax.nn.silu,
}


class PhysicsAttentionIrregularMesh(eqx.Module):
    """Fused physics attention for irregular meshes."""

    in_project: eqx.nn.Linear
    in_project_slice: eqx.nn.Linear
    to_out_linear: eqx.nn.Linear
    to_q: eqx.nn.Linear
    to_k: eqx.nn.Linear
    to_v: eqx.nn.Linear
    attn_dropout: eqx.nn.Dropout
    out_dropout: eqx.nn.Dropout
    temperature: Array

    d_in: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    num_slices: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __init__(
        self,
        d_in: int,
        num_heads: int = 8,
        head_dim: int = 64,
        num_slices: int = 64,
        dropout: float = 0.0,
        eps: float = 1e-5,
        *,
        key: Key,
    ):
        """
        **Args:**
        - `d_in`: Input (and output) feature dimension.
        - `num_heads`: Number of attention heads.
        - `head_dim`: Dimension of each attention head.
        - `num_slices`: Number of slices.
        - `dropout`: Dropout probability.
        - `eps`: Numerical floor for the slice normalization.
        """
        self.d_in = d_in
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_slices = num_slices
        self.eps = eps

        keys = jr.split(key, 6)
        inner_dim = num_heads * head_dim
        self.in_project = eqx.nn.Linear(d_in, 2 * inner_dim, key=keys[0])
        self.in_project_slice = eqx.nn.Linear(head_dim, num_slices, key=keys[1])
        self.to_out_linear = eqx.nn.Linear(inner_dim, d_in, key=keys[2])
        self.to_q = eqx.nn.Linear(head_dim, head_dim, use_bias=False, key=keys[3])
        self.to_k = eqx.nn.Linear(head_dim, head_dim, use_bias=False, key=keys[4])
        self.to_v = eqx.nn.Linear(head_dim, head_dim, use_bias=False, key=keys[5])

        self.attn_dropout = eqx.nn.Dropout(p=dropout)
        self.out_dropout = eqx.nn.Dropout(p=dropout)
        self.temperature = jnp.ones((1, num_heads, 1, 1)) * 0.5

    # -- fused projections -------------------------------------------------

    def _fused_weight_slice(
        self,
    ) -> tuple[Float[Array, "H G C"], Float[Array, "H G"]]:
        """Fuse the `x_mid` half of `in_project` with `in_project_slice`: C -> G."""
        H, D = self.num_heads, self.head_dim
        w_in = self.in_project.weight[H * D :].reshape(H, D, self.d_in)
        b_in = self.in_project.bias[H * D :].reshape(H, D)
        w_slice = self.in_project_slice.weight  # (G, D)

        fused_w = jnp.einsum("gd,hdc->hgc", w_slice, w_in)
        fused_b = jnp.einsum("gd,hd->hg", w_slice, b_in) + self.in_project_slice.bias
        return fused_w, fused_b

    def _fx_weight(self) -> tuple[Float[Array, "H D C"], Float[Array, "H D"]]:
        H, D = self.num_heads, self.head_dim
        w_fx = self.in_project.weight[: H * D].reshape(H, D, self.d_in)
        b_fx = self.in_project.bias[: H * D].reshape(H, D)
        return w_fx, b_fx

    # -- stages ------------------------------------------------------------

    def slice_weights(
        self, x: Float[Array, "B N C"]
    ) -> Float[Array, "B H N {self.num_slices}"]:
        """Per-point slice assignment weights via a single C -> G matmul."""
        fused_w, fused_b = self._fused_weight_slice()
        logits = jnp.einsum("bnc,hgc->bhng", x, fused_w) + fused_b[None, :, None, :]
        return jax.nn.softmax(logits / self.temperature, axis=-1)

    def slice_tokens(
        self,
        x: Float[Array, "B N C"],
        weights: Float[Array, "B H N {self.num_slices}"],
    ) -> Float[Array, "B H {self.num_slices} {self.head_dim}"]:
        """Aggregate raw features into the slice domain, then project there."""
        raw_states = jnp.einsum("bnc,bhng->bhgc", x, weights)
        norm = weights.sum(axis=2) + self.eps  # (B, H, G)
        raw_states = raw_states / norm[..., None]

        w_fx, b_fx = self._fx_weight()
        return jnp.einsum("bhgc,hdc->bhgd", raw_states, w_fx) + b_fx[None, :, None, :]

    def slice_attend(
        self,
        tokens: Float[Array, "B H G {self.head_dim}"],
        *,
        key: Key,
        inference: bool = False,
    ) -> Float[Array, "B H G {self.head_dim}"]:
        """Scaled dot-product self-attention over the slice tokens."""

        def proj(lin, t):
            return jax.vmap(jax.vmap(jax.vmap(lin)))(t)

        q = proj(self.to_q, tokens)
        k = proj(self.to_k, tokens)
        v = proj(self.to_v, tokens)

        logits = jnp.einsum("bhgd,bhed->bhge", q, k) / jnp.sqrt(self.head_dim)
        attn = jax.nn.softmax(logits, axis=-1)
        attn = self.attn_dropout(attn, key=key, inference=inference)
        return jnp.einsum("bhge,bhed->bhgd", attn, v)

    def deslice_to_out(
        self,
        out_slice_token: Float[Array, "B H G {self.head_dim}"],
        weights: Float[Array, "B H N {self.num_slices}"],
        *,
        key: Key,
        inference: bool = False,
    ) -> Float[Array, "B N C"]:
        """Project in the slice domain first, then scatter back to points."""
        H, D = self.num_heads, self.head_dim
        w_out = jnp.transpose(
            self.to_out_linear.weight.reshape(-1, H, D), (1, 2, 0)
        )  # (H, D, C)

        projected = jnp.einsum("bhgd,hdc->bhgc", out_slice_token, w_out)
        out = jnp.einsum("bhng,bhgc->bnc", weights, projected)
        out = out + self.to_out_linear.bias
        return self.out_dropout(out, key=key, inference=inference)

    def __call__(
        self,
        x: Float[Array, "B N C"],
        *,
        key: Key,
        inference: bool = False,
    ) -> Float[Array, "B N C"]:
        attn_key, out_key = jr.split(key, 2)
        weights = self.slice_weights(x)
        tokens = self.slice_tokens(x, weights)
        out_tokens = self.slice_attend(tokens, key=attn_key, inference=inference)
        return self.deslice_to_out(
            out_tokens, weights, key=out_key, inference=inference
        )


class MLP(eqx.Module):
    """Multi-layer perceptron with optional ResNet-style connections."""

    linear_pre: eqx.nn.Linear
    linear_post: eqx.nn.Linear
    linears: list
    act: str = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    res: bool = eqx.field(static=True)

    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        n_layers: int = 1,
        act: str = "gelu",
        res: bool = True,
        *,
        key: Key,
    ):
        if act not in _ACTIVATION:
            raise NotImplementedError(f"Activation {act} not implemented")
        self.act = act
        self.n_layers = n_layers
        self.res = res

        keys = jr.split(key, 2 + n_layers)
        self.linear_pre = eqx.nn.Linear(n_input, n_hidden, key=keys[0])
        self.linear_post = eqx.nn.Linear(n_hidden, n_output, key=keys[1])
        self.linears = [
            eqx.nn.Linear(n_hidden, n_hidden, key=keys[2 + i]) for i in range(n_layers)
        ]

    def __call__(self, x: Float[Array, "B N n_input"]) -> Float[Array, "B N n_output"]:
        act = _ACTIVATION[self.act]

        def apply(lin, t):
            return jax.vmap(jax.vmap(lin))(t)

        x = act(apply(self.linear_pre, x))
        for linear in self.linears:
            y = act(apply(linear, x))
            x = y + x if self.res else y
        return apply(self.linear_post, x)


class TransolverBlock(eqx.Module):
    """Transolver encoder block: fused physics attention + MLP."""

    ln_1: eqx.nn.LayerNorm
    attn: PhysicsAttentionIrregularMesh
    ln_2: eqx.nn.LayerNorm
    mlp: MLP
    ln_3: eqx.nn.LayerNorm | None
    mlp2: eqx.nn.Linear | None
    last_layer: bool = eqx.field(static=True)

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_slices: int,
        dropout: float = 0.0,
        mlp_ratio: int = 4,
        act: str = "gelu",
        last_layer: bool = False,
        out_dim: int = 1,
        eps: float = 1e-5,
        *,
        key: Key,
    ):
        self.last_layer = last_layer
        keys = jr.split(key, 3)

        self.ln_1 = eqx.nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttentionIrregularMesh(
            d_in=hidden_dim,
            num_heads=num_heads,
            head_dim=hidden_dim // num_heads,
            num_slices=num_slices,
            dropout=dropout,
            eps=eps,
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
        fx: Float[Array, "B N D"],
        *,
        key: Key,
        inference: bool = False,
    ) -> Float[Array, "B N D"]:
        def norm(ln, t):
            return jax.vmap(jax.vmap(ln))(t)

        fx = self.attn(norm(self.ln_1, fx), key=key, inference=inference) + fx
        fx = self.mlp(norm(self.ln_2, fx)) + fx
        if self.last_layer:
            assert self.ln_3 is not None and self.mlp2 is not None
            fx = jax.vmap(jax.vmap(self.mlp2))(norm(self.ln_3, fx))
        return fx


class Transolver(eqx.Module):
    """
    Fused 3D Transolver for irregular meshes.

    For the original PyTorch implementation see https://github.com/thuml/Transolver;
    this module ports the fused variant in `test/_transolver_3_test.py`. The
    `unified_pos` / `ref` options of the reference are not ported (the reference
    only uses them to size `preprocess` and ships no feature builder for them).
    """

    preprocess: MLP
    blocks: list
    placeholder: Array

    def __init__(
        self,
        space_dim: int = 1,
        fun_dim: int = 1,
        out_dim: int = 1,
        num_attn_layers: int = 5,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_slices: int = 32,
        dropout: float = 0.0,
        mlp_ratio: int = 1,
        act: str = "gelu",
        eps: float = 1e-5,
        *,
        key: Key,
    ):
        keys = jr.split(key, 2 + num_attn_layers)

        self.preprocess = MLP(
            fun_dim + space_dim,
            hidden_dim * 2,
            hidden_dim,
            n_layers=0,
            act=act,
            res=False,
            key=keys[0],
        )
        self.blocks = [
            TransolverBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_slices=num_slices,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
                act=act,
                last_layer=(i == num_attn_layers - 1),
                out_dim=out_dim,
                eps=eps,
                key=keys[1 + i],
            )
            for i in range(num_attn_layers)
        ]
        self.placeholder = jr.uniform(keys[-1], (hidden_dim,)) / hidden_dim

    def __call__(
        self,
        x: Float[Array, "B N {fun_dim + space_dim}"],
        *,
        key: Key,
        inference: bool = False,
    ) -> Float[Array, "B N out_dim"]:
        """
        **Args:**
        - `x`: Input of shape `(B, N, fun_dim + space_dim)`, function values first,
          spatial coordinates last.
        - `key`: Random key for dropout.
        - `inference`: Whether to disable dropout.
        """
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        keys = jr.split(key, len(self.blocks))
        for block, block_key in zip(self.blocks, keys):
            fx = block(fx, key=block_key, inference=inference)
        return fx


# -----------------------
# Weight initialization (matches the PyTorch reference)
# -----------------------
def _is_linear(x):
    return isinstance(x, eqx.nn.Linear)


def init_weights(model: eqx.Module, *, key: Key) -> eqx.Module:
    """Truncated-normal(std=0.02) weights and zero biases on every `Linear`,
    plus orthogonal initialization for each `in_project_slice`."""

    def get_linears(m):
        return [x for x in jax.tree.leaves(m, is_leaf=_is_linear) if _is_linear(x)]

    linears = get_linears(model)
    keys = jr.split(key, len(linears) + 1)

    def _reinit(linear, k):
        # torch's trunc_normal_ truncates at +-2 std by default
        weight = jr.truncated_normal(k, -2.0, 2.0, linear.weight.shape) * 0.02
        linear = eqx.tree_at(lambda lin: lin.weight, linear, weight)
        if linear.bias is not None:
            linear = eqx.tree_at(
                lambda lin: lin.bias, linear, jnp.zeros_like(linear.bias)
            )
        return linear

    model = eqx.tree_at(
        get_linears, model, [_reinit(lin, k) for lin, k in zip(linears, keys)]
    )

    # Orthogonal init for the slice projections.
    def get_slice_weights(m):
        return [
            x.in_project_slice.weight
            for x in jax.tree.leaves(
                m, is_leaf=lambda y: isinstance(y, PhysicsAttentionIrregularMesh)
            )
            if isinstance(x, PhysicsAttentionIrregularMesh)
        ]

    slice_weights = get_slice_weights(model)
    if slice_weights:
        ortho = jax.nn.initializers.orthogonal()
        ks = jr.split(keys[-1], len(slice_weights))
        model = eqx.tree_at(
            get_slice_weights,
            model,
            [ortho(k, w.shape) for w, k in zip(slice_weights, ks)],
        )
    return model


if __name__ == "__main__":
    import time

    key = jr.key(0)
    mkey, xkey, ckey = jr.split(key, 3)

    model = Transolver(
        space_dim=3,
        fun_dim=1,
        out_dim=1,
        num_attn_layers=3,
        hidden_dim=128,
        num_heads=8,
        num_slices=32,
        mlp_ratio=1,
        key=mkey,
    )
    model = init_weights(model, key=ckey)

    x = jr.normal(xkey, (2, 1024, 4))
    fwd = eqx.filter_jit(model)
    out = fwd(x, key=key, inference=True)  # warmup
    tic = time.time()
    for _ in range(5):
        out = fwd(x, key=key, inference=True).block_until_ready()
    toc = time.time()
    print(out.shape, jnp.isfinite(out).all(), toc - tic)

    # gradients flow
    @eqx.filter_grad
    def loss(m, x):
        return jnp.mean(m(x, key=key, inference=True) ** 2)

    grads = loss(model, x)
    leaves = [g for g in jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))]
    print("grad ok:", all(jnp.isfinite(g).all() for g in leaves))
