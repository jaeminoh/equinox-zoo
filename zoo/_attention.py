import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Array, Float, Key


class PhysicsAttention(eqx.Module):
    r"""
    Implements the physics attention mechanism used in Transolver (Wu et al., https://arxiv.org/abs/2402.02366).
    """

    x_proj: eqx.nn.Linear
    fx_proj: eqx.nn.Linear
    slice_proj: eqx.nn.Linear
    slice_attention: eqx.nn.MultiheadAttention
    out_proj: eqx.nn.Linear
    out_dropout: eqx.nn.Dropout
    temperature: Array

    eps: float = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    num_slices: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)

    def __init__(
        self,
        num_heads: int = 8,
        head_dim: int = 64,
        num_slices: int = 64,
        dropout: float = 0.0,
        eps: float = 1e-5,
        *,
        key: Key,
    ):
        self.hidden_dim = head_dim * num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_slices = num_slices
        self.eps = eps

        keys = jr.split(key, 5)
        self.x_proj = eqx.nn.Linear(self.hidden_dim, self.hidden_dim, key=keys[0])
        self.fx_proj = eqx.nn.Linear(self.hidden_dim, self.hidden_dim, key=keys[1])
        self.slice_proj = eqx.nn.Linear(self.head_dim, self.num_slices, key=keys[2])
        self.slice_attention = eqx.nn.MultiheadAttention(
            num_heads=1,
            query_size=self.head_dim,
            dropout_p=dropout,
            key=keys[3],
        )
        self.out_proj = eqx.nn.Linear(self.hidden_dim, self.hidden_dim, key=keys[4])
        self.out_dropout = eqx.nn.Dropout(p=dropout)
        self.temperature = jnp.ones((num_heads, 1, 1)) * 0.5

    def _slice_weights(
        self, x_mid: Float[Array, "H N {self.head_dim}"]
    ) -> Float[Array, "H N {self.num_slices}"]:
        logits = jax.vmap(jax.vmap(self.slice_proj))(x_mid)
        return jax.nn.softmax(logits * self.temperature, axis=-1)

    def _slice_tokens(
        self,
        fx_mid: Float[Array, "H N {self.head_dim}"],
        weights: Float[Array, "H N {self.num_slices}"],
    ):
        norm = weights.sum(axis=1)
        tokens = jnp.einsum("hnd,hng->hgd", fx_mid, weights)
        return tokens / (norm[..., None] + self.eps)

    def _attn_slices(
        self,
        tokens: Float[Array, "H {self.num_slices} {self.head_dim}"],
        *,
        key: Key,
        inference: bool,
    ) -> Float[Array, "H {self.num_slices} {self.head_dim}"]:
        keys = jr.split(key, self.num_heads)

        def _attn_single(head_tokens, head_key):
            return self.slice_attention(
                head_tokens,
                head_tokens,
                head_tokens,
                key=head_key,
                inference=inference,
            )

        return jax.vmap(_attn_single)(tokens, keys)

    def slices_and_weights(
        self, x: Float[Array, "N {self.hidden_dim}"]
    ) -> tuple[
        Float[Array, "H G {self.head_dim}"], Float[Array, "H N {self.num_slices}"]
    ]:
        x_proj = jax.vmap(self.x_proj)(x)
        fx_proj = jax.vmap(self.fx_proj)(x)
        x_mid = jnp.transpose(
            x_proj.reshape(x_proj.shape[0], self.num_heads, self.head_dim),
            (1, 0, 2),
        )
        fx_mid = jnp.transpose(
            fx_proj.reshape(fx_proj.shape[0], self.num_heads, self.head_dim),
            (1, 0, 2),
        )
        weights = self._slice_weights(x_mid)
        slice_tokens = self._slice_tokens(fx_mid, weights)
        return slice_tokens, weights

    def deslice(
        self,
        x: Float[Array, "H G D"],
        weights: Float[Array, "H N G"],
        *,
        key: Key,
        inference: bool = False,
    ) -> Float[Array, "N {self.hidden_dim}"]:
        out = jnp.einsum("hgd,hng->hnd", x, weights)
        out = jnp.transpose(out, (1, 0, 2)).reshape(out.shape[1], self.hidden_dim)
        out = jax.vmap(self.out_proj)(out)
        out = self.out_dropout(out, key=key, inference=inference)
        return out

    def __call__(
        self,
        x: Float[Array, "N {self.hidden_dim}"],
        *,
        key: Key,
        inference: bool = False,
    ) -> Float[Array, "N {self.hidden_dim}"]:
        slice_tokens, weights = self.slices_and_weights(x)
        attn_key, out_key = jr.split(key, 2)
        out_slice_tokens = self._attn_slices(
            slice_tokens, key=attn_key, inference=inference
        )
        return self.deslice(out_slice_tokens, weights, key=out_key, inference=inference)


class EideticPhysicsAttention(eqx.Module):
    r"""
    Eidetic Physics Attention module. Lighter than PhysicsAttention while maintaining similar performance.
    """

    x_proj: eqx.nn.Linear
    slice_proj: eqx.nn.Linear
    slice_attention: eqx.nn.MultiheadAttention
    out_proj: eqx.nn.Linear
    out_dropout: eqx.nn.Dropout
    proj_temperature: eqx.nn.Sequential
    proj_temperature_bias: Array

    hidden_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    num_slices: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __init__(
        self,
        num_heads: int = 8,
        head_dim: int = 64,
        num_slices: int = 64,
        dropout: float = 0.0,
        eps: float = 1e-5,
        *,
        key: Key,
    ):
        self.hidden_dim = head_dim * num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_slices = num_slices
        self.eps = eps

        keys = jr.split(key, 6)
        self.x_proj = eqx.nn.Linear(self.hidden_dim, self.hidden_dim, key=keys[0])
        self.slice_proj = eqx.nn.Linear(self.head_dim, self.num_slices, key=keys[1])
        self.slice_attention = eqx.nn.MultiheadAttention(
            num_heads=1,
            query_size=self.head_dim,
            dropout_p=dropout,
            key=keys[2],
        )
        self.out_proj = eqx.nn.Linear(self.hidden_dim, self.hidden_dim, key=keys[3])
        self.out_dropout = eqx.nn.Dropout(p=dropout)
        self.proj_temperature = eqx.nn.Sequential(
            [
                eqx.nn.Linear(self.head_dim, self.num_slices, key=keys[4]),
                eqx.nn.Lambda(jax.nn.gelu),
                eqx.nn.Linear(self.num_slices, 1, key=keys[5]),
                eqx.nn.Lambda(jax.nn.gelu),
            ]
        )
        self.proj_temperature_bias = jnp.ones((num_heads, 1, 1)) * 0.5

    @staticmethod
    def gumbel_softmax(logits, temperature, *, key: Key):
        _uniform = jr.uniform(key, logits.shape)
        gumbel_noise = -jnp.log(-jnp.log(_uniform + 1e-8) + 1e-8)
        y = logits + gumbel_noise
        return jax.nn.softmax(y / temperature, axis=-1)

    def _slice_to_weights(
        self, x_mid: Float[Array, "H N {self.head_dim}"], *, key: Key
    ) -> Float[Array, "H N {self.num_slices}"]:
        temperature = (
            jax.vmap(jax.vmap(self.proj_temperature))(x_mid)
            + self.proj_temperature_bias
        )
        temperature = jnp.clip(temperature, a_min=0.01)
        logits = jax.vmap(jax.vmap(self.slice_proj))(x_mid)
        return self.gumbel_softmax(logits, temperature, key=key)

    def slices_and_weights(
        self, x: Float[Array, "N {self.hidden_dim}"], *, key: Key
    ) -> tuple[
        Float[Array, "H {self.num_slices} {self.head_dim}"],
        Float[Array, "H N {self.num_slices}"],
    ]:
        x_proj = jax.vmap(self.x_proj)(x)
        x_mid = jnp.transpose(
            x_proj.reshape(x_proj.shape[0], self.num_heads, self.head_dim),
            (1, 0, 2),
        )  # (H N D)
        weights = self._slice_to_weights(x_mid, key=key)
        slice_tokens = self._slice_tokens(x_mid, weights)
        return slice_tokens, weights

    def _slice_tokens(
        self,
        x_mid: Float[Array, "H N {self.head_dim}"],
        weights: Float[Array, "H N {self.num_slices}"],
    ) -> Float[Array, "H {self.num_slices} {self.head_dim}"]:
        norm = weights.sum(axis=1)
        tokens = jnp.einsum("hnd,hng->hgd", x_mid, weights)
        return tokens / (norm[..., None] + self.eps)

    def _attn_slices(
        self,
        tokens: Float[Array, "H {self.num_slices} {self.head_dim}"],
        *,
        key: Key,
        inference: bool,
    ) -> Float[Array, "H {self.num_slices} {self.head_dim}"]:
        keys = jr.split(key, self.num_heads)

        def _attn_single(head_tokens, head_key):
            return self.slice_attention(
                head_tokens,
                head_tokens,
                head_tokens,
                key=head_key,
                inference=inference,
            )

        return jax.vmap(_attn_single)(tokens, keys)

    def deslice(
        self,
        x: Float[Array, "H G D"],
        weights: Float[Array, "H N G"],
        *,
        key: Key,
        inference: bool = False,
    ) -> Float[Array, "N {self.hidden_dim}"]:
        out = jnp.einsum("hgd,hng->hnd", x, weights)
        out = jnp.transpose(out, (1, 0, 2)).reshape(out.shape[1], self.hidden_dim)
        out = jax.vmap(self.out_proj)(out)
        out = self.out_dropout(out, key=key, inference=inference)
        return out

    def __call__(self, x, *, key, inference=False):
        weight_key, attn_key, out_key = jr.split(key, 3)
        slice_tokens, slice_weights = self.slices_and_weights(x, key=weight_key)
        out_slice_tokens = self._attn_slices(
            slice_tokens, key=attn_key, inference=inference
        )
        return self.deslice(
            out_slice_tokens, slice_weights, key=out_key, inference=inference
        )


if __name__ == "__main__":
    import time

    key = jr.key(0)

    # eidetic physics attention
    attn = EideticPhysicsAttention(key=key)
    x = jr.normal(key, (16, 512))
    attn = eqx.filter_jit(attn)
    out = attn(x, key=key)  # warmup
    tic = time.time()
    for _ in range(5):
        out = attn(x, key=key).block_until_ready()
    toc = time.time()
    print(out.shape, toc - tic)  # (16, 512), 0.00747

    # physics attention
    attn = PhysicsAttention(key=key)
    eqx.filter_jit(attn)
    out = attn(x, key=key)  # warmup
    tic = time.time()
    for _ in range(5):
        out = attn(x, key=key).block_until_ready()
    toc = time.time()
    print(out.shape, toc - tic)  # (16, 512), 0.0537
