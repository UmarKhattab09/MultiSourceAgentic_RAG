import json

import pytest

from agent import parse_route


def test_parse_route_accepts_fenced_json():
    assert parse_route('```json\n{"tool": "calculator", "input": "2+2"}\n```') == {"tool": "calculator", "input": "2+2"}


@pytest.mark.parametrize("bad", ["not json", '{"tool": "hack"}', "[1]", '{"input": "x"}'])
def test_parse_route_rejects_bad_replies(bad):
    with pytest.raises((ValueError, json.JSONDecodeError)):
        parse_route(bad)
