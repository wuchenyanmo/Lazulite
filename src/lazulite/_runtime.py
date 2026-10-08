"""Small runtime boundaries shared by lazily loaded acoustic backends."""
from contextlib import contextmanager
import random


@contextmanager
def preserve_loading_random_state(torch):
    """CPU model construction must not consume the caller's random streams.

    These loaders construct on CPU and move fixed weights to the requested device;
    they never initialize random parameters directly on CUDA.
    """
    state = random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        random.setstate(state)
