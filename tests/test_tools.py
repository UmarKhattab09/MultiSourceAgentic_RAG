from tools import calculator, get_datetime


def test_calculator_basic_math():
    assert calculator("(2+3)*4") == "20"
    assert calculator("2^10") == "1024"
    assert calculator("1,200 * 0.15") == "180.0"


def test_calculator_rejects_code():
    assert calculator("__import__('os').system('dir')").startswith("Error")


def test_get_datetime_has_year():
    assert get_datetime()[-19:-15].isdigit()
