"""Small runtime boundaries shared by lazily loaded acoustic backends."""
from contextlib import contextmanager
import random


@contextmanager
def preserve_random_state(torch):
    """Model construction and eval-time layer-drop draws must not consume caller RNG.

    These loaders construct on CPU and move fixed weights to the requested device;
    they never initialize random parameters directly on CUDA.
    """
    state = random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        random.setstate(state)
