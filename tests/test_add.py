from src.add import add

import numpy as np


def test_add():
    assert add(np.array([1.0])) == 101
    assert add(np.array([1]), np.array([100])) == 101
    assert add("fourth", "Brain") == "fourthBrain"
    assert add(1) == 101
