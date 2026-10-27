import json

import pytest

from agent import parse_route


def test_parse_route_accepts_fenced_json():
    assert parse_route('```json\n{"tool": "calculator", "input": "2+2"}\n```') == {"tool": "calculator", "input": "2+2"}


@pytest.mark.parametrize("bad", ["not json", '{"tool": "hack"}', "[1]", '{"input": "x"}'])
def test_parse_route_rejects_bad_replies(bad):
    with pytest.raises((ValueError, json.JSONDecodeError)):
        parse_route(bad)


def test_parse_route_mcp_tools_take_an_object():
    allowed, structured = ["get_leave_balance", "none"], {"get_leave_balance": {}}
    assert parse_route('{"tool": "get_leave_balance", "input": {}}', allowed, structured)["input"] == {}
    with pytest.raises(ValueError):
        parse_route('{"tool": "get_leave_balance", "input": "E007"}', allowed, structured)
