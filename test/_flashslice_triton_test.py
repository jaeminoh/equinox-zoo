"""
FlashSlice's Triton kernels through jax-triton (`backend="triton"`).

The numerical tests need a CUDA GPU and are skipped elsewhere; they hold the
Triton path to the pure-JAX one, which `test/_flashslice_test.py` holds to the
PyTorch reference in float64. The tracing and validation tests run anywhere
jax-triton is installed.
"""

import pytest

pytest.importorskip("jax_triton")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402

from zoo._flashslice import (  # noqa: E402
    PhysicsAttentionIrregularMesh,
    Transolver,
    fused_deslice,
    fused_slice,
    init_weights,
)

gpu = pytest.mark.skipif(
    jax.default_backend() != "gpu", reason="the Triton kernels need a CUDA GPU"
)

# Ragged on purpose: N is not a multiple of any tile size.
B, N, H, D = 2, 1000, 4, 32
F32 = jnp.float32


def rel_err(got, expected):
    got, expected = np.asarray(got, np.float64), np.asarray(expected, np.float64)
    return np.abs(got - expected).max() / max(np.abs(expected).max(), 1e-30)


def make_leaves(G, layout, with_bias, dtype=F32, seed=0):
    keys = jr.split(jr.key(seed), 6)
    w_shape = {"shared": (G, D), "per-head": (H, G, D), "per-sample": (B, H, G, D)}
    w_shape = w_shape[layout]
    return dict(
        x=jr.normal(keys[0], (B, N, H, D), F32).astype(dtype),
        fx=jr.normal(keys[1], (B, N, H, D), F32).astype(dtype),
        w=jr.normal(keys[2], w_shape, F32) / D**0.5,
        b=0.1 * jr.normal(keys[3], w_shape[:-1], F32) if with_bias else None,
        tau=0.7 + 0.1 * jr.normal(keys[4], (H,), F32),
        tokens=jr.normal(keys[5], (B, H, G, D), F32),
    )


def coupling(backend, order, leaves, **options):
    """A tied coupling round in either call order, returning a scalar loss
    that weighs every output, plus the outputs."""
    x, fx, w, b, tau = (leaves[k] for k in ("x", "fx", "w", "b", "tau"))
    ops = dict(backend=backend, **options)
    if order == "slice-first":
        z_num, s = fused_slice(x, fx, w, b, tau, **ops)
        out = fused_deslice(x, w, b, tau, z_num / (s + 1e-5)[..., None], **ops)
    else:
        out = fused_deslice(x, w, b, tau, leaves["tokens"], **ops)
        z_num, s = fused_slice(x, fx, w, b, tau, **ops)
    out = out.astype(F32)
    loss = jnp.sum(jnp.sin(out)) + jnp.sum(jnp.cos(z_num)) + jnp.sum(s**2) / N
    return loss, (out, z_num, s)


def run(backend, order, leaves, **options):
    f = jax.jit(
        jax.value_and_grad(
            lambda lv: coupling(backend, order, lv, **options), has_aux=True
        )
    )
    (_, outs), grads = f(leaves)
    return outs, grads


def reference(order, leaves):
    """The pure-JAX path at exact fp32 dots."""
    return run("jax", order, leaves, precision="highest")


# ---------------------------------------------------------------------------
# Anywhere: tracing and validation
# ---------------------------------------------------------------------------
def test_triton_path_traces_with_parameter_shaped_gradients():
    leaves = make_leaves(32, "per-head", True)
    out_avals = jax.eval_shape(
        jax.value_and_grad(lambda lv: coupling("triton", "slice-first", lv)[0]),
        leaves,
    )
    assert jax.tree.map(lambda a: (a.shape, a.dtype), out_avals[1]) == jax.tree.map(
        lambda a: (a.shape, a.dtype), leaves
    )


@pytest.mark.parametrize("D_, G", [(32, 48), (32, 256), (8, 32), (256, 32)])
def test_triton_path_rejects_unsupported_dims(D_, G):
    x = jnp.zeros((1, 8, 2, D_), F32)
    w = jnp.zeros((G, D_), F32)
    with pytest.raises(ValueError, match="backend='jax'"):
        fused_slice(x, x, w, None, jnp.ones(2, F32), backend="triton")
    with pytest.raises(ValueError, match="backend='jax'"):
        PhysicsAttentionIrregularMesh(
            2 * D_,
            2,
            D_,
            G,
            use_fused_slice=True,
            backend="triton",
            key=jr.key(0),
        )


def test_triton_path_rejects_unequal_value_width_and_other_dtypes():
    x = jnp.zeros((1, 8, 2, 32), F32)
    w = jnp.zeros((32, 32), F32)
    with pytest.raises(ValueError, match="value width"):
        fused_slice(x, x[..., :16], w, None, jnp.ones(2, F32), backend="triton")
    # The kernels take 16- and 32-bit floats (fp64 is refused the same way, when
    # x64 is on; an integer dtype exists either way).
    xi = x.astype(jnp.int32)
    with pytest.raises(ValueError, match="float32"):
        fused_slice(xi, xi, w, None, jnp.ones(2, F32), backend="triton")
    with pytest.raises(ValueError, match="backend"):
        fused_slice(x, x, w, None, jnp.ones(2, F32), backend="cuda")


# ---------------------------------------------------------------------------
# GPU: numerics
# ---------------------------------------------------------------------------
@gpu
@pytest.mark.parametrize("order", ["slice-first", "deslice-first"])
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("layout", ["shared", "per-head", "per-sample"])
@pytest.mark.parametrize("G", [16, 32, 128])
def test_ops_match_pure_jax(G, layout, with_bias, order):
    """Outputs and every gradient, temperature included, at exact fp32 dots."""
    leaves = make_leaves(G, layout, with_bias)
    got, expected = run("triton", order, leaves), reference(order, leaves)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(expected)):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert rel_err(a, b) < 1e-4


@gpu
def test_triton_path_is_bitwise_deterministic():
    """No atomics in the kernels: two runs agree to the bit."""
    leaves = make_leaves(32, "shared", True)
    first, second = (
        run("triton", "slice-first", leaves),
        run("triton", "slice-first", leaves),
    )
    for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@gpu
@pytest.mark.parametrize(
    "dot,dtype,tol",
    [
        ("tf32x3", F32, 1e-4),
        ("tf32", F32, 2e-2),
        ("bf16v", jnp.bfloat16, 5e-2),
        ("bf16", jnp.bfloat16, 5e-2),
    ],
)
def test_dot_modes_stay_close(dot, dtype, tol):
    """The faster dot modes, against exact fp32 on the same (rounded) inputs."""
    leaves = make_leaves(64, "shared", True, dtype=dtype)
    got = run("triton", "slice-first", leaves, dot=dot)
    exact = {k: v if v is None else v.astype(F32) for k, v in leaves.items()}
    expected = reference("slice-first", exact)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(expected)):
        assert rel_err(a.astype(F32), b) < tol


def _model(backend, flag):
    model = Transolver(
        space_dim=3,
        fun_dim=1,
        out_dim=2,
        num_layers=3,
        hidden_dim=128,
        num_heads=4,
        num_slices=32,
        use_fused_slice=True,
        backend=backend,
        key=jr.key(0),
        **({flag: True} if flag else {}),
    )
    model = init_weights(model, key=jr.key(1))
    # fp32 parameters even if a test module enabled x64 in this process.
    return jax.tree.map(
        lambda a: a.astype(F32) if eqx.is_inexact_array(a) else a, model
    )


@gpu
@pytest.mark.parametrize("flag", [None, "untie_slice_weights", "no_token_attention"])
def test_model_matches_pure_jax(flag):
    x = jr.normal(jr.key(2), (B, N, 3), F32)
    fx = jr.normal(jr.key(3), (B, N, 1), F32)

    def value_and_grads(backend):
        @eqx.filter_jit
        @eqx.filter_value_and_grad
        def loss(m):
            return jnp.mean(m(x, fx, key=jr.key(4), inference=True) ** 2)

        with jax.default_matmul_precision("highest"):
            return loss(_model(backend, flag))

    got, expected = value_and_grads("triton"), value_and_grads("jax")
    assert rel_err(got[0], expected[0]) < 1e-4
    for a, b in zip(jax.tree.leaves(got[1]), jax.tree.leaves(expected[1])):
        assert rel_err(a, b) < 1e-3
