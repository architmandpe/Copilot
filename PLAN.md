# Agent — Plan

## State

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # conversation, appended each hop
    user_id: int                               # set once from the authenticated session, never model-supplied
    tool_call_count: int                       # incremented each hop through tools, caps runaway loops
```

`user_id` is placed into the initial state by the FastAPI endpoint (from `get_current_user`), before the graph ever runs. The model never sees it — `agent_node` only ever passes `state["messages"]` to the model, not the full state dict.

## Tools

Built per-request via a factory that closes over `user_id`, so `user_id` is never a parameter the model can see or fill in:

```python
def make_tools_for_user(user_id: int) -> list:
    @tool
    def create_task(title: str) -> str:
        """Create a task for the user. Returns a confirmation."""
        ...

    @tool
    def list_my_tasks() -> str:
        """List the user's current tasks."""
        ...

    @tool
    def search_tasks(query: str) -> str:
        """Find the user's tasks relevant to a query (uses M3 RAG retrieval)."""
        ...

    @tool
    def delete_task(task_id: int) -> str:
        """Delete a task. Requires human confirmation before it actually runs."""
        ...

    return [create_task, list_my_tasks, search_tasks, delete_task]
```

- `create_task`, `list_my_tasks`, `search_tasks` — call real Phase 1 repositories/services, scoped to `user_id`.
- `delete_task` — destructive, routed through a confirmation step before it ever executes (see below).

## Nodes

- **agent** — calls the model (tools bound) with `state["messages"]`. Returns the model's reply as a state update.
- **tools** — runs whatever non-destructive tool the model requested (`create_task`, `list_my_tasks`, `search_tasks`). Increments `tool_call_count`.
- **confirm_delete** — reached only when the model requests `delete_task`. Pauses the graph (interrupt + checkpoint), returns a confirmation question to the user. On a later "yes" request, resumes and actually runs the delete.

## Edges

```
                         tool_calls: delete_task
                    ┌─────────────────────────────► confirm_delete ───┐
                    │                                                  │ user confirms
     ┌─────────┐    │  tool_calls: other tool     ┌────────┐          │
     │  agent  │ ───┤────────────────────────────▶│ tools  │◀─────────┘
     │ (model) │ ◀──┴─────────────────────────────│ (run)  │
     └─────────┘         loop back                └────────┘
          │ no tool_calls, or tool_call_count >= 5
          ▼
         END → final answer
```

- `agent → tools` / `agent → confirm_delete`: conditional edge (`route()`), decided by whether the model's last message has `tool_calls`, and which tool it named.
- `tools → agent`: fixed edge, always loops back so the model can turn the tool result into a reply and decide on the next step.
- `confirm_delete → tools` (or directly executes the delete): only after the user confirms; a rejected confirmation routes straight back to `agent` with a cancellation message, no delete runs.
- `agent → END`: when there's no `tool_calls` in the model's last message, or `tool_call_count` has hit the cap.

## Loop guard

`tool_call_count` lives in `state`, since state is the only thing that persists across repeated node calls — no Python object (not "the agent," not "the tools") retains memory between hops. Incremented once per pass through `tools`. Capped at **5**: `route()` checks this before allowing another tool call and forces `END` (with an explanatory message) if exceeded, regardless of what the model requests next.

## Confirmation (human-in-the-loop) flow

1. Model requests `delete_task` → routed to `confirm_delete` instead of `tools`.
2. `confirm_delete` triggers an interrupt: graph execution stops, state is checkpointed (persisted, not kept in a Python variable — the confirming request may hit a different process or arrive much later), and a confirmation question is returned to the user in the HTTP response.
3. A separate, later `POST /assistant/chat` request carries the user's yes/no. The graph resumes from the checkpointed state.
4. On "yes": the real `delete_task` tool runs, result flows back to `agent`. On "no": routes back to `agent` with a cancellation, no delete happens.
