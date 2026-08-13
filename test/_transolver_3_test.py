"""
Transolver-3 JAX test.

The torch implementation is copied from
https://github.com/thuml/Transolver-3/blob/main/models/Transolver_chunk_opt_matrix_mul.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


ACTIVATION = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}
trunc_normal_ = nn.init.trunc_normal_


def _as_feature_tensor(chunk):
    if isinstance(chunk, (tuple, list)):
        return chunk[0]
    return chunk


# -----------------------
# Physics Attention
# -----------------------
class Physics_Attention_Irregular_Mesh(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        super().__init__()

        self.heads = heads
        self.dim_head = dim_head
        self.slice_num = slice_num

        self.in_project = nn.Linear(dim, 2 * heads * dim_head)  # produces fx and x
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_out_linear = nn.Linear(heads * dim_head, dim)

        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

    def forward(self, x):
        B, N, C = x.shape
        H, D, G = self.heads, self.dim_head, self.slice_num  # noqa

        # --- Stage 1: Slice - generate weights and aggregate into slice tokens ---
        f_w, f_b = self._get_fused_weight_slice()
        slice_weights = self.chunk_weights(x, f_w, f_b)  # (B, H, N, G)

        slice_norm = slice_weights.sum(dim=2, keepdim=True) + 1e-5  # (B, H, 1, G)

        # Aggregate raw features into slice domain: (B, N, C) x (B, H, N, G) -> (B, H, G, C)
        raw_states = torch.einsum("bnc, bhng -> bhgc", x, slice_weights)
        raw_states = raw_states / slice_norm.transpose(-1, -2)

        # Project in slice domain O(G*C*D) instead of O(N*C*D)
        w_fx = self.in_project.weight[: H * D].view(H, D, C)
        b_fx = self.in_project.bias[: H * D].view(H, D)
        slice_token = torch.einsum("bhgc, hdc -> bhgd", raw_states, w_fx) + b_fx.view(
            1, H, 1, D
        )

        # --- Stage 2: Attention over slice tokens (complexity depends on G, not N) ---
        out_slice_token = self.slice_attend(slice_token)  # (B, H, G, D)

        # --- Stage 3: Deslice back to point space ---
        return self.chunk_deslice_to_out(x, out_slice_token, slice_weights)

    def chunk_stats(self, x: torch.Tensor):
        B, N, C = x.shape
        H, D, G = self.heads, self.dim_head, self.slice_num  # noqa

        # 1. Get fused weights for slice assignment
        f_w, f_b = self._get_fused_weight_slice()
        slice_weights = self.chunk_weights(x, f_w, f_b)  # (B, H, N, G)

        # 2. Compute density in slice domain: O(NG) -> O(G)
        den = slice_weights.sum(dim=2)  # (B, H, G)

        # 3. Aggregate raw features into slice domain first
        # x: (B, N, C), slice_weights: (B, H, N, G) -> x_agg: (B, H, G, C)
        x_agg = torch.einsum("bnc, bhng -> bhgc", x, slice_weights)

        # 4. Project in slice domain (G tokens instead of N)
        w_fx = self.in_project.weight[: H * D].view(H, D, C)
        b_fx = self.in_project.bias[: H * D].view(H, D)

        # (B, H, G, C) @ (H, C, D) -> (B, H, G, D)
        num = torch.einsum("bhgc, hdc -> bhgd", x_agg, w_fx)

        # Account for the bias term accumulated over all N points
        num = num + den.unsqueeze(-1) * b_fx.unsqueeze(1)

        return num, den

    def slice_attend(self, slice_token: torch.Tensor):
        """slice_token: (B,H,G,D) -> out_slice_token: (B,H,G,D)"""
        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        out_slice_token = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout.p if self.training else 0.0, is_causal=False
        )
        return out_slice_token

    def _get_fused_weight_slice(self):
        """Fuse in_project (x_mid part) with in_project_slice to project C -> G in one matmul."""
        # Extract the x_mid portion of in_project: (H*D, C) -> (H, D, C)
        w_in = self.in_project.weight[self.heads * self.dim_head :]
        w_in = w_in.view(self.heads, self.dim_head, -1)

        w_slice = self.in_project_slice.weight  # (G, D)

        # Fuse: (G, D) @ (H, D, C) -> (H, G, C)
        fused_w = torch.matmul(w_slice, w_in)

        # Fuse bias
        b_in = self.in_project.bias[self.heads * self.dim_head :].view(
            self.heads, self.dim_head
        )
        fused_b = torch.matmul(w_slice, b_in.unsqueeze(-1)).squeeze(-1)  # (H, G)
        fused_b = fused_b + self.in_project_slice.bias

        return fused_w, fused_b

    def chunk_weights(self, x: torch.Tensor, fused_w=None, fused_b=None):
        """Compute per-point slice assignment weights via a single C -> G matmul."""
        if fused_w is None:
            fused_w, fused_b = self._get_fused_weight_slice()

        # x: (B, N, C), fused_w: (H, G, C) -> logits: (B, H, N, G)
        logits = torch.einsum("bnc, hgc -> bhng", x, fused_w) + fused_b.view(
            1, self.heads, 1, self.slice_num
        )
        return F.softmax(logits / self.temperature, dim=-1)

    def chunk_deslice_to_out(
        self, x: torch.Tensor, out_slice_token: torch.Tensor, slice_weights=None
    ):
        """
        Deslice back to point space using associativity of linear operators:
        instead of deslice then project (O(N*HD*C)),
        project in slice domain first then deslice (O(G*HD*C) + O(N*G*C)).
        """
        B, N, C = x.shape
        H, G, D = self.heads, self.slice_num, self.dim_head  # noqa

        if slice_weights is None:
            slice_weights = self.chunk_weights(x)  # (B, H, N, G)

        w_out = self.to_out_linear.weight.view(-1, H, D).permute(1, 2, 0)  # (H, D, C)

        # Project in slice domain: (B, H, G, D) @ (H, D, C) -> (B, H, G, C)
        projected_slices = torch.einsum("bhgd, hdc -> bhgc", out_slice_token, w_out)

        # Aggregate back to point space, summing over heads:
        # (B, H, N, G) @ (B, H, G, C) -> (B, N, C)
        out_x = torch.einsum("bhng, bhgc -> bnc", slice_weights, projected_slices)

        return self.dropout(out_x + self.to_out_linear.bias)


# -----------------------
# MLP
# -----------------------
class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True):
        super().__init__()
        if act in ACTIVATION:
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
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
            x = self.linears[i](x) + x if self.res else self.linears[i](x)
        x = self.linear_post(x)
        return x


# -----------------------
# Transolver block
# -----------------------
class Transolver_block(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=32,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Irregular_Mesh(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
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

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx

    def forward_chunks(self, fx_list, eps=1e-5, use_checkpoint=True):
        """
        fx_list: list of tensors [(B,n1,C), (B,n2,C), ...]
        Returns list with same chunking, updated for the next layer.
        """
        # Pass 1: accumulate global num/den from all chunks
        global_num, global_den = None, None
        for fxk in fx_list:
            uk = self.ln_1(fxk)
            if use_checkpoint:
                num_k, den_k = checkpoint(
                    self.Attn.chunk_stats,
                    uk,
                    preserve_rng_state=True,
                    use_reentrant=False,
                )
            else:
                num_k, den_k = self.Attn.chunk_stats(uk)

            global_num = num_k if global_num is None else (global_num + num_k)
            global_den = den_k if global_den is None else (global_den + den_k)

        slice_token = global_num / (global_den[..., None] + eps)  # (B,H,G,D)

        # Global slice attention (once for all chunks)
        out_slice = self.Attn.slice_attend(slice_token)  # (B,H,G,D)

        # Pass 2: per-chunk deslice + residual + MLP
        out_list = []
        for fxk in fx_list:

            def chunk_compute(f_k, o_slice):
                uk = self.ln_1(f_k)
                a_out = self.Attn.chunk_deslice_to_out(uk, o_slice)
                res_fx = a_out + f_k
                mlp_out = self.mlp(self.ln_2(res_fx)) + res_fx
                if self.last_layer:
                    mlp_out = self.mlp2(self.ln_3(mlp_out))
                return mlp_out

            if use_checkpoint:
                fxk2 = checkpoint(chunk_compute, fxk, out_slice, use_reentrant=False)
            else:
                fxk2 = chunk_compute(fxk, out_slice)

            out_list.append(fxk2)
        return out_list


# -----------------------
# Model
# -----------------------
class Model(nn.Module):
    def __init__(
        self,
        space_dim=1,
        n_layers=5,
        n_hidden=256,
        dropout=0,
        n_head=8,
        act="gelu",
        mlp_ratio=1,
        fun_dim=1,
        out_dim=1,
        slice_num=32,
        ref=8,
        unified_pos=False,
    ):
        super().__init__()
        self.__name__ = "UniPDE_3D"
        self.ref = ref
        self.unified_pos = unified_pos

        if self.unified_pos:
            self.preprocess = MLP(
                fun_dim + self.ref**3,
                n_hidden * 2,
                n_hidden,
                n_layers=0,
                res=False,
                act=act,
            )
        else:
            self.preprocess = MLP(
                fun_dim + space_dim,
                n_hidden * 2,
                n_hidden,
                n_layers=0,
                res=False,
                act=act,
            )

        self.blocks = nn.ModuleList(
            [
                Transolver_block(
                    num_heads=n_head,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    out_dim=out_dim,
                    slice_num=slice_num,
                    last_layer=(i == n_layers - 1),
                )
                for i in range(n_layers)
            ]
        )

        self.initialize_weights()
        self.placeholder = nn.Parameter(
            (1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float)
        )

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

    def forward(self, data, use_checkpoint=True, input_list=True):
        """
        data: list of feature tensors when input_list=True, or a single feature tensor otherwise.
        Returns a list of per-chunk outputs when input_list=True.
        """
        eps = 1e-5
        if input_list:
            fx_list = []
            for chunk in data:
                xk = _as_feature_tensor(chunk)
                fxk = self.preprocess(xk)
                fxk = fxk + self.placeholder[None, None, :]
                fx_list.append(fxk)

            for block in self.blocks:
                fx_list = block.forward_chunks(
                    fx_list, eps=eps, use_checkpoint=use_checkpoint
                )
            return fx_list
        else:
            x = _as_feature_tensor(data)
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]
            for block in self.blocks:
                fx = block(fx)
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

from zoo._transolver_3 import (  # noqa: E402
    PhysicsAttentionIrregularMesh,
    Transolver,
    init_weights,
)

SPACE_DIM, FUN_DIM, OUT_DIM = 3, 1, 2
NUM_LAYERS, NUM_HEADS, HEAD_DIM, NUM_SLICES = 3, 4, 16, 8
HIDDEN_DIM = NUM_HEADS * HEAD_DIM
BATCH, NUM_POINTS = 2, 11
IN_DIM = SPACE_DIM + FUN_DIM

TOL = 1e-10


def make_jax_model(key):
    return Transolver(
        space_dim=SPACE_DIM,
        fun_dim=FUN_DIM,
        out_dim=OUT_DIM,
        num_attn_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_slices=NUM_SLICES,
        mlp_ratio=1,
        key=key,
    )


def t2j(x):
    return jnp.asarray(x.detach().numpy(), dtype=jnp.float64)


def max_abs_diff(jax_array, torch_tensor):
    return np.abs(np.asarray(jax_array) - torch_tensor.numpy()).max()


# ---------------------------------------------------------------------------
# Standalone JAX behaviour
# ---------------------------------------------------------------------------
def test_output_shape():
    key = jr.key(0)
    model = make_jax_model(key)
    x = jr.normal(jr.fold_in(key, 1), (BATCH, NUM_POINTS, IN_DIM))

    out = model(x, key=jr.fold_in(key, 2), inference=False)

    assert out.shape == (BATCH, NUM_POINTS, OUT_DIM)


def test_init_weights_preserves_structure():
    key = jr.key(0)
    model = init_weights(make_jax_model(key), key=jr.fold_in(key, 1))
    x = jr.normal(jr.fold_in(key, 2), (BATCH, NUM_POINTS, IN_DIM))

    out = model(x, key=jr.fold_in(key, 3), inference=True)

    assert out.shape == (BATCH, NUM_POINTS, OUT_DIM)
    assert jnp.isfinite(out).all()
    # Orthogonal slice projections: W W^T = I with W of shape (G, D), G <= D.
    for block in model.blocks:
        w = block.attn.in_project_slice.weight
        assert np.allclose(w @ w.T, np.eye(NUM_SLICES), atol=1e-5)


def test_gradients_are_finite():
    key = jr.key(0)
    model = make_jax_model(key)
    x = jr.normal(jr.fold_in(key, 1), (BATCH, NUM_POINTS, IN_DIM))

    @eqx.filter_grad
    def loss(m):
        return jnp.mean(m(x, key=jr.fold_in(key, 2), inference=True) ** 2)

    grads = eqx.filter(loss(model), eqx.is_inexact_array)
    assert all(jnp.isfinite(g).all() for g in jax.tree.leaves(grads))


def test_attention_is_permutation_equivariant():
    key = jr.key(0)
    attn = PhysicsAttentionIrregularMesh(
        d_in=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        num_slices=NUM_SLICES,
        key=key,
    )
    x = jr.normal(jr.fold_in(key, 1), (BATCH, NUM_POINTS, HIDDEN_DIM))
    perm = jr.permutation(jr.fold_in(key, 2), NUM_POINTS)

    out = attn(x, key=jr.fold_in(key, 3), inference=True)
    out_perm = attn(x[:, perm], key=jr.fold_in(key, 3), inference=True)

    assert np.allclose(np.asarray(out_perm), np.asarray(out[:, perm]), atol=TOL)


# ---------------------------------------------------------------------------
# Copying PyTorch weights into the Equinox model
# ---------------------------------------------------------------------------
def copy_linear(jax_linear, torch_linear):
    jax_linear = eqx.tree_at(
        lambda _lin: _lin.weight, jax_linear, t2j(torch_linear.weight)
    )
    if jax_linear.bias is not None:
        jax_linear = eqx.tree_at(
            lambda _lin: _lin.bias, jax_linear, t2j(torch_linear.bias)
        )
    return jax_linear


def copy_layer_norm(jax_ln, torch_ln):
    return eqx.tree_at(
        lambda ln: (ln.weight, ln.bias),
        jax_ln,
        (t2j(torch_ln.weight), t2j(torch_ln.bias)),
    )


def copy_mlp(jax_mlp, torch_mlp):
    # torch wraps each hidden layer in Sequential(Linear, act); index 0 is the Linear.
    return eqx.tree_at(
        lambda m: (m.linear_pre, m.linear_post),
        jax_mlp,
        (
            copy_linear(jax_mlp.linear_pre, torch_mlp.linear_pre[0]),
            copy_linear(jax_mlp.linear_post, torch_mlp.linear_post),
        ),
    )


def copy_attention(jax_attn, torch_attn):
    names = ["in_project", "in_project_slice", "to_out_linear", "to_q", "to_k", "to_v"]
    jax_attn = eqx.tree_at(
        lambda a: [getattr(a, n) for n in names],
        jax_attn,
        [copy_linear(getattr(jax_attn, n), getattr(torch_attn, n)) for n in names],
    )
    return eqx.tree_at(lambda a: a.temperature, jax_attn, t2j(torch_attn.temperature))


def copy_block(jax_block, torch_block):
    jax_block = eqx.tree_at(
        lambda b: (b.ln_1, b.ln_2, b.attn, b.mlp),
        jax_block,
        (
            copy_layer_norm(jax_block.ln_1, torch_block.ln_1),
            copy_layer_norm(jax_block.ln_2, torch_block.ln_2),
            copy_attention(jax_block.attn, torch_block.Attn),
            copy_mlp(jax_block.mlp, torch_block.mlp),
        ),
    )
    if torch_block.last_layer:
        jax_block = eqx.tree_at(
            lambda b: (b.ln_3, b.mlp2),
            jax_block,
            (
                copy_layer_norm(jax_block.ln_3, torch_block.ln_3),
                copy_linear(jax_block.mlp2, torch_block.mlp2),
            ),
        )
    return jax_block


def copy_weights(jax_model, torch_model):
    """Overwrite every JAX parameter with its PyTorch counterpart."""
    return eqx.tree_at(
        lambda m: (m.preprocess, m.placeholder, *m.blocks),
        jax_model,
        (
            copy_mlp(jax_model.preprocess, torch_model.preprocess),
            t2j(torch_model.placeholder),
            *(
                copy_block(jb, tb)
                for jb, tb in zip(jax_model.blocks, torch_model.blocks)
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Parity with the PyTorch reference
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def model_pair():
    """A PyTorch reference model and an Equinox model carrying identical weights."""
    torch.manual_seed(0)
    torch_model = (
        Model(
            space_dim=SPACE_DIM,
            n_layers=NUM_LAYERS,
            n_hidden=HIDDEN_DIM,
            dropout=0.0,
            n_head=NUM_HEADS,
            mlp_ratio=1,
            fun_dim=FUN_DIM,
            out_dim=OUT_DIM,
            slice_num=NUM_SLICES,
        )
        .double()
        .eval()
    )
    return torch_model, copy_weights(make_jax_model(jr.key(0)), torch_model)


@pytest.fixture
def attention_pair(model_pair):
    """The block-0 physics attention from each model, plus a shared input."""
    torch_model, jax_model = model_pair
    torch.manual_seed(1)
    x = torch.randn(BATCH, NUM_POINTS, HIDDEN_DIM, dtype=torch.float64)
    return torch_model.blocks[0].Attn, jax_model.blocks[0].attn, x


def test_parity_full_model(model_pair):
    torch_model, jax_model = model_pair
    x = torch.randn(BATCH, NUM_POINTS, IN_DIM, dtype=torch.float64)

    with torch.no_grad():
        expected = torch_model(x, input_list=False)
    got = jax_model(t2j(x), key=jr.key(0), inference=True)

    assert got.shape == expected.shape
    assert max_abs_diff(got, expected) < TOL


def test_slice_weights_are_a_simplex(attention_pair):
    """Slice assignments are a softmax over G, so each point's row sums to one."""
    _, jax_attn, x = attention_pair
    weights = jax_attn.slice_weights(t2j(x))

    assert weights.shape == (BATCH, NUM_HEADS, NUM_POINTS, NUM_SLICES)
    assert (weights >= 0).all()
    assert np.allclose(np.asarray(weights.sum(axis=-1)), 1.0)


def test_parity_attention(attention_pair):
    torch_attn, jax_attn, x = attention_pair

    with torch.no_grad():
        expected = torch_attn(x)
    got = jax_attn(t2j(x), key=jr.key(0), inference=True)

    assert max_abs_diff(got, expected) < TOL


def _torch_slice_tokens(torch_attn, x, slice_weights):
    """Stage 2 of the reference `forward`, which it does not expose as a method."""
    H, D, C = torch_attn.heads, torch_attn.dim_head, HIDDEN_DIM
    norm = slice_weights.sum(dim=2, keepdim=True) + 1e-5
    raw = torch.einsum("bnc,bhng->bhgc", x, slice_weights) / norm.transpose(-1, -2)
    w_fx = torch_attn.in_project.weight[: H * D].view(H, D, C)
    b_fx = torch_attn.in_project.bias[: H * D].view(H, D)
    return torch.einsum("bhgc,hdc->bhgd", raw, w_fx) + b_fx.view(1, H, 1, D)


def test_parity_attention_stages(attention_pair):
    """Pin each fused stage separately, so a regression localizes itself."""
    torch_attn, jax_attn, x = attention_pair

    with torch.no_grad():
        weights_t = torch_attn.chunk_weights(x)
        tokens_t = _torch_slice_tokens(torch_attn, x, weights_t)
        attended_t = torch_attn.slice_attend(tokens_t)
        out_t = torch_attn.chunk_deslice_to_out(x, attended_t, weights_t)

    weights_j = jax_attn.slice_weights(t2j(x))
    assert max_abs_diff(weights_j, weights_t) < TOL

    tokens_j = jax_attn.slice_tokens(t2j(x), weights_j)
    assert tokens_j.shape == (BATCH, NUM_HEADS, NUM_SLICES, HEAD_DIM)
    assert max_abs_diff(tokens_j, tokens_t) < TOL

    attended_j = jax_attn.slice_attend(tokens_j, key=jr.key(0), inference=True)
    assert max_abs_diff(attended_j, attended_t) < TOL

    out_j = jax_attn.deslice_to_out(
        attended_j, weights_j, key=jr.key(0), inference=True
    )
    assert max_abs_diff(out_j, out_t) < TOL
