import pytest

import hr_server


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(hr_server, "DB_PATH", tmp_path / "hr.db")


def test_balance_is_seeded():
    assert hr_server.leave_balance("E001") == {"employee": "Ava Chen", "balances": {"vacation": 15, "sick": 8, "personal": 3}}


def test_request_counts_weekdays_and_deducts():
    result = hr_server.leave_request("E001", "vacation", "2030-11-08", "2030-11-11", "trip")  # Fri..Mon
    assert result["days"] == 2 and result["remaining"] == 13


def test_request_over_balance_is_refused():
    with pytest.raises(ValueError, match="Not enough"):
        hr_server.leave_request("E006", "personal", "2030-11-11", "2030-11-15")


def test_other_employees_requests_are_hidden():
    with pytest.raises(ValueError, match="No leave request"):
        hr_server.request_status("E001", 3)  # request 3 belongs to E005


def test_no_signed_in_employee():
    with pytest.raises(ValueError, match="No employee"):
        hr_server.leave_balance(None)


def test_past_dates_are_refused():
    with pytest.raises(ValueError, match="in the past"):
        hr_server.leave_request("E001", "vacation", "2023-12-25", "2023-12-25")


def test_login():
    assert hr_server.login("e007", hr_server.DEMO_PASSWORD)["is_hr"]
    assert not hr_server.login("E001", hr_server.DEMO_PASSWORD)["is_hr"]
    assert hr_server.login("E001", "wrong") is None
    assert hr_server.login("E999", hr_server.DEMO_PASSWORD) is None


def test_hr_review():
    req = hr_server.leave_request("E001", "vacation", "2030-11-11", "2030-11-12")["request_id"]
    with pytest.raises(PermissionError):
        hr_server.pending_requests("E001")  # not HR
    assert req in [r["id"] for r in hr_server.pending_requests("E007")]
    hr_server.decide_request("E007", req, approve=False)
    assert hr_server.leave_balance("E001")["balances"]["vacation"] == 15  # rejection gives the days back
    assert hr_server.my_requests("E001")[0]["status"] == "rejected"
    with pytest.raises(ValueError, match="No pending"):
        hr_server.decide_request("E007", req, approve=True)


def test_hr_cannot_review_own_request():
    req = hr_server.leave_request("E007", "sick", "2030-11-11", "2030-11-11")["request_id"]
    with pytest.raises(ValueError, match="own"):
        hr_server.decide_request("E007", req, approve=True)
