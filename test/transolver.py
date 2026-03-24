import jax.random as jr

from zoo._attention import PhysicsAttention
from zoo.transolver import Transolver


def test_physics_attention_output_shape():
    key = jr.key(43321)
    model = PhysicsAttention(num_heads=8, head_dim=64, num_slices=64, key=key)
    x = jr.normal(jr.fold_in(key, 1), (4, model.hidden_dim))

    out = model(x, key=jr.fold_in(key, 2), inference=False)

    assert out.shape == (4, 512)


def test_transolver_output_shape():
    key = jr.key(0)
    model = Transolver(
        space_dim=1,
        fun_dim=1,
        out_dim=1,
        num_attn_layers=3,
        hidden_dim=128,
        num_heads=8,
        head_dim=16,
        num_slices=32,
        dropout=0.1,
        mlp_ratio=4,
        key=key,
    )
    x = jr.normal(jr.fold_in(key, 1), (4, 2))

    out = model(x, key=jr.fold_in(key, 2), inference=False)

    assert out.shape == (4, 1)
