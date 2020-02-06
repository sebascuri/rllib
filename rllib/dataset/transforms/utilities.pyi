"""Utilities for the transformers."""

from numpy import ndarray
from torch import Tensor
import numpy as np
import torch.__spec__ as torch_mod
from typing import Union


Array = Union[ndarray, Tensor]
Module = Union[np, torch_mod]

def get_backend(array: Array) -> Module: ...


def update_mean(old_mean: Array, old_count: int,
                new_mean: Array, new_count: int) -> Array: ...


def update_var(old_mean: Array, old_var: Array, old_count: int,
               new_mean: Array, new_var: Array, new_count: int,
               biased: bool = True) -> Array: ...


def normalize(array: Array, mean: Array, variance: Array,
              preserve_origin: bool = False) -> Array: ...


def denormalize(array: Array, mean: Array, variance: Array,
                preserve_origin: bool = False) -> Array: ...
