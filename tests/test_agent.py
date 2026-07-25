from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END
from agent import graph, route

USER_ID = 11


def tool_calls_made(result: dict) -> list[str]:
    names = []
    for msg in result["messages"]:
        for tc in getattr(msg, "tool_calls", None) or []:
            names.append(tc["name"])
    return names


def test_create_request_calls_create_task_tool():
    config = {"configurable": {"thread_id": "test-behavior-create"}}
    result = graph.invoke(
        {"messages": [HumanMessage("make a task to test behavior assertion")], "user_id": USER_ID, "tool_call_count": 0},
        config=config,
    )
    assert "create_task" in tool_calls_made(result)


def test_question_goes_through_a_lookup_not_fabrication():
    config = {"configurable": {"thread_id": "test-behavior-search-3"}}
    result = graph.invoke(
        {"messages": [HumanMessage("what do I need to write?")], "user_id": USER_ID, "tool_call_count": 0},
        config=config,
    )
    calls = tool_calls_made(result)
    assert "search_tasks" in calls or "list_my_tasks" in calls


def test_delete_request_routes_to_confirmation_not_tools():
    message = AIMessage(content="", tool_calls=[{"name": "delete_task", "args": {"task_id": 1}, "id": "x"}])
    state = {"messages": [message], "user_id": USER_ID, "tool_call_count": 0}
    assert route(state) == "confirm_delete"


def test_no_tool_calls_routes_to_end():
    message = AIMessage(content="done")
    state = {"messages": [message], "user_id": USER_ID, "tool_call_count": 0}
    assert route(state) == END


def test_loop_guard_forces_end_after_max_tool_calls():
    message = AIMessage(content="", tool_calls=[{"name": "list_my_tasks", "args": {}, "id": "x"}])
    state = {"messages": [message], "user_id": USER_ID, "tool_call_count": 5}
    assert route(state) == END
