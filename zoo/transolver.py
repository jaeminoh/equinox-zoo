import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Array, Float, Key

from ._attention import PhysicsAttention


class MLP(eqx.Module):
    """Multi-layer perceptron with ResNet-style connections."""

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
        self.n_layers = n_layers
        self.res = res
        self.act = act

        if act not in ["gelu", "relu", "tanh"]:
            raise NotImplementedError(f"Activation {act} not implemented")

        keys = jr.split(key, 2 + n_layers)
        self.linear_pre = eqx.nn.Linear(n_input, n_hidden, key=keys[0])
        self.linear_post = eqx.nn.Linear(n_hidden, n_output, key=keys[1])
        self.linears = [
            eqx.nn.Linear(n_hidden, n_hidden, key=keys[2 + i]) for i in range(n_layers)
        ]

    def _apply_activation(self, x):
        if self.act == "gelu":
            return jax.nn.gelu(x)
        elif self.act == "relu":
            return jax.nn.relu(x)
        elif self.act == "tanh":
            return jnp.tanh(x)
        else:
            return x

    def __call__(self, x: Float[Array, "N n_input"]) -> Float[Array, "N n_output"]:
        x = self._apply_activation(jax.vmap(self.linear_pre)(x))
        for i in range(self.n_layers):
            if self.res:
                x = self._apply_activation(jax.vmap(self.linears[i])(x)) + x
            else:
                x = self._apply_activation(jax.vmap(self.linears[i])(x))
        return jax.vmap(self.linear_post)(x)


class TransolverBlock(eqx.Module):
    """Transolver encoder block with attention and MLP."""

    ln_1: eqx.nn.LayerNorm
    attn: PhysicsAttention
    ln_2: eqx.nn.LayerNorm
    mlp: MLP
    ln_3: eqx.nn.LayerNorm | None
    mlp2: eqx.nn.Linear | None
    last_layer: bool = eqx.field(static=True)

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
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
        keys = jr.split(key, 4 if last_layer else 3)

        self.ln_1 = eqx.nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            num_heads=num_heads,
            head_dim=head_dim,
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
        x: Float[Array, "N D"],
        *,
        key: Key,
        inference: bool = False,
    ) -> Float[Array, "N D"]:
        # Attention block
        x = self.attn(jax.vmap(self.ln_1)(x), key=key, inference=inference) + x

        # MLP block
        x = self.mlp(jax.vmap(self.ln_2)(x)) + x

        # Optional output projection
        if self.last_layer:
            assert self.ln_3 is not None and self.mlp2 is not None
            x = jax.vmap(self.mlp2)(jax.vmap(self.ln_3)(x))

        return x


class Transolver(eqx.Module):
    """
    Transolver model for learning on irregular meshes (PDEs).
    For the original PyTorch implementation, see https://github.com/thuml/Transolver.
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
        head_dim: int = 64,
        num_slices: int = 64,
        dropout: float = 0.0,
        mlp_ratio: int = 4,
        act: str = "gelu",
        eps: float = 1e-5,
        *,
        key: Key,
    ):
        keys = jr.split(key, 2 + num_attn_layers)

        # Preprocessing: embed (function_values, spatial_coords) → hidden_dim
        self.preprocess = MLP(
            fun_dim + space_dim,
            hidden_dim * 2,
            hidden_dim,
            n_layers=0,
            act=act,
            res=False,
            key=keys[0],
        )

        # Transolver blocks
        self.blocks = [
            TransolverBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=head_dim,
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

        # Placeholder for initialization
        self.placeholder = jr.normal(keys[-1], (hidden_dim,)) / jnp.sqrt(hidden_dim)

    def __call__(
        self,
        x: Float[Array, "N {fun_dim + space_dim}"],
        *,
        key: Key,
        inference: bool = False,
    ) -> Float[Array, "N out_dim"]:
        """
        Args:
            x: Input of shape (N, fun_dim + space_dim), where first fun_dim are function values
               and last space_dim are spatial coordinates.
            key: Random key for dropout.
            inference: Whether in inference mode (no dropout).

        Returns:
            Output of shape (N, out_dim).
        """
        # Preprocess
        fx = self.preprocess(x)
        fx = fx + self.placeholder[None, :]

        # Apply blocks
        keys = jr.split(key, len(self.blocks))
        for block, block_key in zip(self.blocks, keys):
            fx = block(fx, key=block_key, inference=inference)

        return fx
