import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from typing import Union
from jaxtyping import PRNGKeyArray, Float, ArrayLike, jaxtyped
from beartype import beartype


class RationalActivation(eqx.Module):
    """
    Comes from https://github.com/NBoulle/RationalNets/blob/master/src/PyTorch%20implementation/rational.py
    """

    P: jnp.array
    Q: jnp.array

    def __init__(self):
        self.P = jnp.array([1.1915, 1.5957, 0.5, 0.0218])
        self.Q = jnp.array([2.383, 0.0, 1.0])

    def __call__(self, x):
        x = jnp.power(x, jnp.arange(3, -1, -1))
        return (self.P @ x) / (self.Q @ x[1:])


class MultiLayerPerceptron(eqx.Module):
    layers: list

    def __init__(
        self,
        *,
        d_in: Union[str, int] = 2,
        width: int = 32,
        depth: int = 4,
        d_out: Union[str, int] = "scalar",
        key: PRNGKeyArray = jr.key(4321),
    ):
        layers = [d_in] + [width] * (depth - 1) + [d_out]
        keys = jr.split(key, depth)
        self.layers = [
            eqx.nn.Linear(_in, _out, key=_k)
            for _in, _out, _k in zip(layers[:-1], layers[1:], keys)
        ]

    def __call__(self, *inputs):
        x = jnp.stack(inputs)
        for layer in self.layers[:-1]:
            x = jnp.tanh(layer(x))
        return self.layers[-1](x)


class Siren(MultiLayerPerceptron):
    w0: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        d_in: Union[str, int] = 2,
        width: int = 32,
        depth: int = 4,
        d_out: Union[str, int] = "scalar",
        key: PRNGKeyArray = jr.key(4321),
        w0: float = 10.0,
    ):
        super().__init__(d_in=d_in, width=width, depth=depth, d_out=d_out, key=key)
        self.w0 = w0
        self = convert_mlp_to_siren(self, key)

    @jaxtyped(typechecker=beartype)
    def __call__(self, x: Float[ArrayLike, "..."]) -> Float[ArrayLike, "..."]:
        x = jnp.atleast_1d(x)
        for layer in self.layers[:-1]:
            x = jnp.sin(self.w0 * layer(x))
        return self.layers[-1](x)


def _siren_init(mlp: MultiLayerPerceptron, key: PRNGKeyArray = jr.key(4123)):
    def init_weight(layer: eqx.nn.Linear, is_first: bool, key: PRNGKeyArray):
        assert isinstance(layer, eqx.nn.Linear)
        d_out, d_in = layer.weight.shape
        if is_first:
            scale = 1 / d_in
        else:
            scale = jnp.sqrt(6 / d_in) / mlp.w0
        W = jr.uniform(key, (d_out, d_in), minval=-1, maxval=1) * scale
        return W

    def init_bias(layer: eqx.nn.Linear, key: PRNGKeyArray):
        assert isinstance(layer, eqx.nn.Linear)
        d_out, d_in = layer.weight.shape
        scale = jnp.sqrt(1 / d_in)
        b = jr.uniform(key, (d_out,), minval=-1, maxval=1) * scale
        return b

    num_layers = len(mlp.layers)
    is_first = [True] + [False for _ in range(num_layers - 1)]
    keys = jr.split(key, num_layers)

    def get_weights(mlp: MultiLayerPerceptron):
        def is_linear(x):
            return isinstance(x, eqx.nn.Linear)

        params = [
            x.weight for x in jtu.tree_leaves(mlp, is_leaf=is_linear) if is_linear(x)
        ]
        return params

    def get_biases(mlp: MultiLayerPerceptron):
        def is_linear(x):
            return isinstance(x, eqx.nn.Linear)

        params = [
            x.bias for x in jtu.tree_leaves(mlp, is_leaf=is_linear) if is_linear(x)
        ]
        return params

    new_weight = list(map(init_weight, mlp.layers, is_first, keys))
    new_bias = list(map(init_bias, mlp.layers, keys))

    mlp = eqx.tree_at(get_weights, mlp, new_weight)
    mlp = eqx.tree_at(get_biases, mlp, new_bias)

    return mlp


def convert_mlp_to_siren(net: eqx.Module, key=jr.key(4321)):
    def is_mlp(mlp: MultiLayerPerceptron):
        return isinstance(mlp, MultiLayerPerceptron)

    def get_mlps(net: eqx.Module):
        return [x for x in jtu.tree_leaves(net, is_mlp) if is_mlp(x)]

    mlps = get_mlps(net)
    num_mlps = len(mlps)
    keys = jr.split(key, num_mlps)
    new_mlps = list(map(_siren_init, mlps, keys))

    net = eqx.tree_at(get_mlps, net, new_mlps)
    return net
