from src.add import add

import numpy as np


def test_add():
    assert add(np.array([1.0])) == 101
    assert add(1) == 101
