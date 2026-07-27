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
    def create_task(title, priority=None, due_date=None, recurrence=None) -> str: ...
    @tool
    def create_multiple_tasks(titles: list[str]) -> str: ...
    @tool
    def list_my_tasks() -> str: ...                       # includes status/priority/due date/recurrence
    @tool
    def search_tasks(query: str) -> str: ...               # semantic retrieval
    @tool
    def update_task(task_id, title=None, status=None, priority=None, due_date=None, recurrence=None) -> str: ...
    @tool
    def update_multiple_tasks(task_ids, title=None, status=None, priority=None, due_date=None, recurrence=None) -> str: ...
    @tool
    def delete_task(task_id, title) -> str: ...            # title required, for display only
    @tool
    def delete_multiple_tasks(task_ids, titles) -> str: ...

    return [create_task, create_multiple_tasks, list_my_tasks, search_tasks,
            update_task, update_multiple_tasks, delete_task, delete_multiple_tasks]
```

- All 8 tools call task-tracker's `/internal/tasks/*` endpoints, scoped to `user_id`.
- `create_task`/`create_multiple_tasks`/`update_task`/`update_multiple_tasks` also call `upsert_task()` (`rag.py`) after a successful call, so the RAG vector store stays in sync — without this, new/edited tasks would be invisible to `search_tasks` and the UI search bar.
- `delete_task`/`delete_multiple_tasks` call `remove_task()` after a successful delete, for the same reason in reverse (otherwise deleted tasks keep showing up in search results).
- `delete_task`/`delete_multiple_tasks` — destructive, routed through a confirmation step before they ever execute (see below). Both require the model to also supply the task's title(s) as an argument — used only to build a human-readable confirmation question and result message, never sent to task-tracker. This is what keeps raw database ids out of anything the user sees.
- **Task decomposition has no dedicated tool.** When asked to break a goal into subtasks, the model is instructed (system prompt) to propose a plain-text numbered list and wait — no tool call at all for the proposal step. The user's next message ("just 1 and 3") is then interpreted normally, triggering `create_task`/`create_multiple_tasks` for only the approved titles.
- **Recurring tasks have no dedicated tool either.** `recurrence` is just a field on `create_task`/`update_task`. The actual rollover — creating the next occurrence when a recurring task is marked `done` — happens in `TaskRepository.update()` on the task-tracker side, not in the agent at all, so it fires correctly regardless of which caller (agent, or a future direct API caller) completes the task.

## Nodes

- **agent** — calls the model (tools bound, system prompt rebuilt fresh each call to inject today's date) with `state["messages"]`. Returns the model's reply as a state update.
- **tools** — runs whatever non-destructive tool the model requested. Increments `tool_call_count`. Also logs an audit entry (`log_agent_action`) for every non-read-only tool call, using the tool's own return string as the summary — a generic hook here rather than instrumenting each tool individually.
- **confirm_delete** — reached when the model requests `delete_task` OR `delete_multiple_tasks` (`DELETE_TOOL_NAMES`). Pauses the graph (interrupt + checkpoint), returns a confirmation question (built from the title(s) the model supplied) to the user. On a later "yes" request, resumes, actually runs the delete, and logs the audit entry.

## Edges

```
                         tool_calls: delete_task / delete_multiple_tasks
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
- `tools → agent`: fixed edge, always loops back so the model can turn the tool result into a reply and decide on the next step — this is also what lets it chain, e.g., confirming several deletes across turns without a fresh user message each time.
- `confirm_delete → agent`: only after the user confirms; a rejected confirmation routes straight back to `agent` with a cancellation message, no delete runs.
- `agent → END`: when there's no `tool_calls` in the model's last message, or `tool_call_count` has hit the cap.

## Loop guard

`tool_call_count` lives in `state`, since state is the only thing that persists across repeated node calls — no Python object (not "the agent," not "the tools") retains memory between hops. Incremented once per pass through `tools` or `confirm_delete`. Capped at **5**: `route()` checks this before allowing another tool call and forces `END` (with an explanatory message) if exceeded, regardless of what the model requests next. Bulk tools (`create_multiple_tasks`, `update_multiple_tasks`, `delete_multiple_tasks`) exist partly to keep large batch operations under this cap — one bulk call costs the same 1 toward the count as a single-item call.

## Confirmation (human-in-the-loop) flow

1. Model requests `delete_task` or `delete_multiple_tasks` → routed to `confirm_delete` instead of `tools`.
2. `confirm_delete` triggers an interrupt: graph execution stops, state is checkpointed (persisted, not kept in a Python variable — the confirming request may hit a different process or arrive much later), and a confirmation question naming the task(s) by title is returned to the user in the HTTP response (or as a `[CONFIRM_REQUIRED]` SSE frame when streaming).
3. A separate, later `POST /assistant/chat` (or `/stream`) request carries the user's yes/no. The graph resumes from the checkpointed state.
4. On "yes": the real delete tool runs, its RAG embedding is removed, the action is audit-logged, result flows back to `agent`. On "no": routes back to `agent` with a cancellation, nothing is deleted or logged.
