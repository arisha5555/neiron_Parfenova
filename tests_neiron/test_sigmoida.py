from __future__ import division

from parfenova_network import sigmoid, proizvodnaya_sigmoid
import pytest

def test_sigmoida_good():
    assert sigmoid(0) == 0.5

@pytest.mark.parametrize('a, result', [(0, 0), (1, 0)])
def test_proizvodnaya_sigmoida_good(a, result):
    assert proizvodnaya_sigmoid(a) == result

def test_type_error():
    with pytest.raises(TypeError):
         division('0')

