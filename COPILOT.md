# Copilot service

LLM features (task parsing, semantic search, grounded Q&A, and a full conversational agent) on top of the task tracker's data. This service never touches the task tracker's database directly — everything goes through the task tracker's HTTP API, scoped to a specific user.

## Architecture

Two independent services, each with its own Postgres, plus a React frontend served by task-tracker:

```
Browser ──cookie──▶ task-tracker (FastAPI, :8000, Postgres :5432)
   (React UI)             │
                          │ X-Internal-Secret header
                          ▼
                     copilot (FastAPI, :8001, Postgres+pgvector :5433)
```

- **task-tracker** owns users/tasks and cookie-based auth, and serves the React frontend (`frontend/`, built with Vite). Its `app/routers/assistant.py` is the only thing that talks to copilot — it forwards the authenticated user's id, never a token or cookie.
- **copilot** (this service) has no concept of login. Every endpoint is gated by a shared `X-Internal-Secret` header, checked in `verify_internal_secret`. It calls task-tracker's `/internal/tasks/*` and `/internal/audit/*` endpoints (same secret, reverse direction) to read/create/update/delete tasks and log actions on the user's behalf.
- copilot has its own Postgres (pgvector) purely for the RAG vector store — a separate instance, not shared with task-tracker's DB. Task embeddings are kept in sync automatically: created/updated whenever a task is created or updated via the agent, removed when a task is deleted (`upsert_task`/`remove_task` in `rag.py`).
- task-tracker runs in Docker; copilot runs natively on the host. task-tracker reaches copilot via `COPILOT_URL=http://host.docker.internal:8001` (not `localhost`, which inside the container means the container itself).

**Providers:** chat generation uses Groq (`openai/gpt-oss-20b` via `langchain-groq`). Embeddings use Gemini (`models/gemini-embedding-001`) — Groq has no embeddings API. Both configured in `parser.py` / `rag.py`.

## Endpoints

All require `X-Internal-Secret`.

| Endpoint | Purpose |
|---|---|
| `POST /parse` | Sentence → structured `TaskDraft` (title/due_date/priority). |
| `POST /ask` | Grounded Q&A over the user's tasks (RAG), returns `{answer, sources}`. |
| `POST /search` | Retrieval-only semantic search (no LLM call) — returns `[{task_id, snippet}, ...]`. Backs the UI search bar. |
| `POST /chat` | Conversational agent (LangGraph). Accepts `{user_id, thread_id, message}` or `{user_id, thread_id, confirm}` to resume a paused confirmation. |
| `POST /stream` | Same as `/chat`, streamed via SSE. Emits a `[CONFIRM_REQUIRED] <question>` frame if the graph pauses for a delete confirmation. |

task-tracker exposes the user-facing equivalents behind real cookie auth: `POST /assistant/{parse,ask,search,chat,stream}`, plus `GET /assistant/audit` (reads the agent action log directly from task-tracker's own DB, no copilot round-trip needed).

## The agent

`agent.py` — a LangGraph graph with **9 tools**, built per-request via a closure factory (`make_tools_for_user(user_id)`) so `user_id` is baked in and never a parameter the model can see or supply itself. Full design in `PLAN.md`.

- **Read-only:** `list_my_tasks` (title/status/priority/due date/recurrence for every task — also how the agent answers "what's overdue"/"what's due today", by comparing to today's date), `search_tasks` (semantic retrieval).
- **Create:** `create_task` (title, priority, due_date, recurrence — all optional except title), `create_multiple_tasks` (bulk, one call).
- **Update:** `update_task`, `update_multiple_tasks` (bulk) — any of title/status/priority/due_date/recurrence. Setting a recurring task's status to `done` auto-creates the next occurrence (in `TaskRepository.update()` on the task-tracker side, not the agent — fires regardless of which caller completes the task).
- **Delete:** `delete_task`, `delete_multiple_tasks` (bulk) — both require human confirmation (see below), and both require the model to also pass the task's title(s) (for display only, never sent to the server) so confirmations and replies never show raw database ids.
- **Task decomposition** isn't a separate tool or graph node — it's purely a system-prompt instruction: when asked to break a goal into subtasks, the model proposes a plain-text numbered list without calling any tool, then only creates the ones the user approves in their next message.
- The system prompt (`build_system_prompt()`) is rebuilt fresh on every call (not a static constant) so it can inject today's date/weekday — needed for the model to resolve relative dates ("tomorrow", "next Friday") and reason about overdue tasks correctly.
- The model is explicitly told never to read the bracketed ids in `list_my_tasks`/`search_tasks` output back to the user — those are for its own tool-call use only; it refers to tasks by title in replies.

**Human-in-the-loop:** either delete tool pauses via `interrupt()`, checkpointed via LangGraph's `MemorySaver`, resumed by a later `/chat` or `/stream` call with `confirm: true/false`. The confirmation question names the task(s) by title, not id.

**Loop guard:** `tool_call_count` in state, capped at 5.

**Agent action audit log:** every non-read-only tool call is logged (task-tracker's `agent_actions` table, written via `POST /internal/audit/{user_id}`) using the tool's own return string as the summary. The hook lives in `tools_node`/`confirm_delete_node`, not in each individual tool. Visible to the user via `GET /assistant/audit` and the frontend's Activity panel.

## Guardrails

- **Max input length** (500 chars) and **rate limiting** (10 requests/60s, sliding window) — enforced in code (`enforce_message_guardrails()` in task-tracker's `assistant.py`), not prompt-based. These run before the model is ever called, so no prompt injection can bypass them.
- **Off-topic refusal** and **system-prompt-leak refusal** — enforced via the agent's system prompt. This is best-effort, not a hard boundary — treat it as reducing the attack surface, not eliminating it.
- **Per-user isolation** — RAG retrieval is filtered by `user_id` metadata (not semantic similarity), and each tool closure is scoped to one `user_id` baked in at creation. Tested in `tests/test_rag_isolation.py` and `tests/test_agent.py`.

## Observability

Every `/chat` turn logs one JSON line to stdout via `log_turn()` (`observability.py`): timestamp, `user_id`, `thread_id`, tools called, total tokens (aggregated across every LLM call in that turn via `UsageMetadataCallbackHandler`), latency, and outcome (`done` / `confirm_required` / `error`). Separately, the agent action audit log (above) is the user-facing equivalent — "what did the agent actually do," not "what happened technically."

## Resilience

The shared `ChatGroq` model (`parser.py`) is configured with `timeout=10s, max_retries=2`. If the LLM is still unreachable after retries, `/chat` and `/stream` catch the failure and return/stream a clean message ("I'm having trouble reaching the assistant right now...") instead of a stack trace or dropped connection. task-tracker's `/assistant/stream` relay also has its own timeout (`httpx.Timeout(60.0, connect=10.0)`) and error handling, since it's a separate hop that can independently hang or drop.

## Running it

```bash
# task-tracker (Docker, :8000, Postgres :5432)
cd task_tracker/task-tracker
docker compose up -d --build

# copilot (native, :8001; its own Postgres+pgvector on :5433 via Docker)
cd copilot
docker compose up -d
uv run uvicorn main:app --reload --port 8001

# frontend (native, :5173, proxies /auth /tasks /assistant to task-tracker for cookie auth)
cd task_tracker/task-tracker/frontend
npm install
npm run dev
```

Requires `.env` in both `task-tracker` and `copilot` (see `.env.example`): `JWT_SECRET`, `DATABASE_URL`, `INTERNAL_SECRET` (shared, must match on both sides), `GROQ_API_KEY`, `GOOGLE_API_KEY`.

## Eval

```bash
cd copilot
uv run python eval.py
```

`eval.py` runs 10 fixed cases (Q&A/lookup, task creation, a regression check against a real past bug, delete-confirmation, and 2 refusal/guardrail cases) and asserts on structure/behavior, never exact prose. Current score: 10/10. **Not yet updated to cover the newer tools** (update, bulk operations, decomposition, recurring tasks) — a real coverage gap, since those shipped without regression cases. Run this after any prompt or agent change — it's how you catch regressions instead of guessing.

## Known limits

- `MemorySaver` checkpointing is in-memory only. A paused delete confirmation is lost if the process restarts before it's resumed. A real deployment needs a durable checkpointer (e.g. Postgres-backed).
- Off-topic and prompt-leak refusals are prompt-enforced, not code-enforced — a sufficiently creative prompt injection could still bypass them. Only length/rate-limit guardrails are hard boundaries.
- One conversation thread per user (`thread_id = user_id`) — no support for multiple concurrent conversations per user.
- The eval suite and RAG retrieval tests run against a small, evolving toy dataset (the same account used for manual testing) — `tests/test_rag_eval.py`'s hardcoded expected task id can drift out of date if that task gets deleted during manual testing; it currently does and needs a fixture refresh.
- Task ids shown to the user are the raw database primary key (global across all users), not a per-user sequential number — a minor cosmetic quirk, not a bug (the agent never surfaces these to the user directly, only in its own tool-call reasoning).

## Cost notes

Both providers are currently on free tiers (Groq for chat, Gemini for embeddings), with no cost ceiling or budget alarm in place. Token usage is logged per call (`parse_task` logs prompt/completion tokens directly; `/chat` logs total tokens per turn via observability). If usage grows, add a per-user or global token budget check before the model call, using the same aggregation already wired up in `observability.py`.
