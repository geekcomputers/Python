import pytest

from is_number import check_number

def test_check_number_float():
    assert check_number(3.14) == '3.14 is a number.'
    assert check_number(1e-5) == '1e-05 is a number.'

def test_check_number_negative_float():
    assert check_number(-3.14) == '-3.14 is a number.'
    assert check_number(-1e-5) == '-1e-05 is a number.'

def test_check_number_boolean():
    assert check_number(True) == 'True is a number.'
    assert check_number(False) == 'False is a number.'

def test_check_number_list():
    assert check_number([1, 2, 3]) == '[1, 2, 3] is not a number.'
    assert check_number([]) == '[] is not a number.'

def test_check_number_dict():
    assert check_number({'key': 'value'}) == "{'key': 'value'} is not a number."
    assert check_number({}) == "{} is not a number."
