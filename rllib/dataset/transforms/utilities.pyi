"""Utilities for the transformers."""

from rllib.dataset.datatypes import Array

def update_mean(old_mean: Array, old_count: int,
                new_mean: Array, new_count: int) -> Array: ...


def update_var(old_mean: Array, old_var: Array, old_count: int,
               new_mean: Array, new_var: Array, new_count: int,
               biased: bool = True) -> Array: ...


def normalize(array: Array, mean: Array, variance: Array,
              preserve_origin: bool = False) -> Array: ...


def denormalize(array: Array, mean: Array, variance: Array,
                preserve_origin: bool = False) -> Array: ...
