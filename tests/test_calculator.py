import pytest
from novexa.calculator import calculate
def test_arithmetic(): assert calculate("2+3*4")["result"] == 14
def test_caret_power(): assert calculate("5^2")["result"] == 25
def test_degrees(): assert abs(calculate("sin(30)","DEG")["result"]-.5)<1e-8
def test_radians(): assert abs(calculate("sin(pi/2)","RAD")["result"]-1)<1e-8
def test_bad_expression():
    with pytest.raises(ValueError): calculate("1/0")
def test_unsafe():
    with pytest.raises(ValueError): calculate("__import__('os')")
