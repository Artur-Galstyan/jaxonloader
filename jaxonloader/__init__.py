from jaxonloader._datasets import *  # noqa
from jaxonloader.dataloader import JaxonDataLoader, make  # noqa
import equinox as eqx
from jaxtyping import Array
from collections.abc import Callable

Index = eqx.nn.State
JITJaxonDataLoader = Callable[[eqx.nn.State], tuple[Array, eqx.nn.State, bool]]
