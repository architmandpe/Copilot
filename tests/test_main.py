import datetime as dt
from parser import parse_task, new_conversation

def test_parse_extracts_a_title():
    draft = parse_task("email the design team about the logo")
    assert draft.title
    assert draft.priority in {"low", "normal", "high"}

def test_next_week_maybe_is_low_priority_no_date():
    draft = parse_task("sometime next week maybe")
    assert draft.due_date is None
    assert draft.priority == "low"

def test_asap_is_high_priority_no_date():
    draft = parse_task("ASAP!!")
    assert draft.due_date is None
    assert draft.priority == "high"

def test_tomorrow_is_pinned_exactly():
    draft = parse_task("finish this tomorrow")
    tomorrow = (dt.date.today() + dt.timedelta(days=1)).isoformat()
    assert draft.due_date == tomorrow

def test_eod_friday_returns_a_valid_future_date():
    draft = parse_task("eod friday")
    today = dt.date.today().isoformat()
    assert draft.due_date is not None
    assert draft.due_date >= today

def test_two_tasks_are_merged_not_dropped():
    draft = parse_task("call mom and also finish the report by friday")
    assert "mom" in draft.title.lower()
    assert "report" in draft.title.lower()
    today = dt.date.today().isoformat()
    assert draft.due_date is not None
    assert draft.due_date >= today

def test_explicit_date_is_pinned_exactly():
    draft = parse_task("submit the report by 2026-08-01")
    assert draft.due_date == "2026-08-01"

def test_conversational_chain_resolves_it_across_turns():
    say = new_conversation()
    say("add a task to email the design team")
    reply = say("actually make it high priority")
    assert "high" in reply.lower()
