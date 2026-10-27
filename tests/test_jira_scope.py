import json

import pytest

from agent import clean_answer
from mcp_tools import owns_issue, scope_jira


def test_search_is_limited_to_the_employee():
    jql = scope_jira("jira_search", {"jql": "status = 'To Do' order by updated DESC"}, "E007")["jql"]
    assert jql == "labels = \"zava-e007\" AND (status = 'To Do') ORDER BY updated DESC"


def test_empty_search_lists_newest_first():
    assert scope_jira("jira_search", {"jql": "ORDER BY created DESC"}, "E007")["jql"] == 'labels = "zava-e007" ORDER BY created DESC'
    assert scope_jira("jira_search", {}, "E007")["jql"] == 'labels = "zava-e007" ORDER BY created DESC'


@pytest.mark.parametrize("jql", ["x = 1) OR (project = KAN", 'summary ~ "(" ) OR project = KAN OR (x = ")"', 'summary ~ "open'])
def test_search_cannot_escape_the_filter(jql):
    with pytest.raises(ValueError):
        scope_jira("jira_search", {"jql": jql}, "E007")


def test_create_is_tagged():
    args = scope_jira("jira_create_issue", {"summary": "x", "description": "Wi-Fi down",
                                            "additional_fields": json.dumps({"priority": {"name": "High"}, "labels": ["net"]})}, "E007")
    assert json.loads(args["additional_fields"]) == {"priority": {"name": "High"}, "labels": ["net", "zava-e007"]}
    assert args["description"].startswith("Reported by Zava employee E007.") and "Wi-Fi down" in args["description"]


def test_no_employee_no_jira():
    with pytest.raises(ValueError, match="no employee"):
        scope_jira("jira_search", {"jql": ""}, None)


def test_get_issue_ownership():
    assert owns_issue(json.dumps({"key": "KAN-7", "labels": ["zava-e007"]}), "E007")
    assert not owns_issue(json.dumps({"key": "KAN-7", "labels": ["zava-e007"]}), "E001")
    assert not owns_issue("Error: boom", "E007")


def test_clean_answer():
    raw = '<tool_code>jira_search(jql="x")</tool_code> <untrusted_data> {"issues": []} </untrusted_data> You have no tickets yet.'
    assert clean_answer(raw) == '{"issues": []}  You have no tickets yet.'


def test_create_ignores_blank_model_args(monkeypatch):
    monkeypatch.setenv("JIRA_PROJECT", "KAN")
    args = scope_jira("jira_create_issue", {"project_key": "", "summary": "x", "assignee": None, "components": [],
                                            "additional_fields": {"priority": {"name": "High"}}}, "E007")
    assert args["project_key"] == "KAN" and "assignee" not in args and "components" not in args
    assert json.loads(args["additional_fields"]) == {"priority": {"name": "High"}, "labels": ["zava-e007"]}


def test_create_fixes_type_drops_assignee_and_is_idempotent():
    args = scope_jira("jira_create_issue", {"summary": "Network is slow", "issue_type": "Bug", "assignee": "currentUser()"}, "E007")
    assert args["issue_type"] == "Task" and "assignee" not in args
    assert scope_jira("jira_create_issue", args, "E007") == args  # scoping twice changes nothing
