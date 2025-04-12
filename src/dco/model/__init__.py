from jax import config

config.update("jax_platforms", "cpu")

from .model import Model
