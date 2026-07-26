# Copilot service

LLM features (task parsing, Q&A over your tasks, and a conversational agent) on top of the task tracker's data. This service never touches the task tracker's database directly — everything goes through the task tracker's HTTP API, scoped to a specific user.

## Architecture

Two independent services, each with its own Postgres:

```
Browser ──cookie──▶ task-tracker (FastAPI, :8000, Postgres :5432)
                          │
                          │ X-Internal-Secret header
                          ▼
                     copilot (FastAPI, :8001, Postgres+pgvector :5433)
```

- **task-tracker** owns users/tasks and cookie-based auth. Its `app/routers/assistant.py` is the only thing that talks to this service — it forwards the authenticated user's id, never a token or cookie.
- **copilot** (this service) has no concept of login. Every endpoint is gated by a shared `X-Internal-Secret` header, checked in `verify_internal_secret`. It calls task-tracker's `/internal/tasks/*` endpoints (same secret, reverse direction) to read/create/delete tasks on the user's behalf.
- copilot has its own Postgres (pgvector) purely for the RAG vector store — a separate instance, not shared with task-tracker's DB.
- task-tracker runs in Docker; copilot runs natively on the host. task-tracker reaches copilot via `COPILOT_URL=http://host.docker.internal:8001` (not `localhost`, which inside the container means the container itself).

**Providers:** chat generation uses Groq (`openai/gpt-oss-20b` via `langchain-groq`). Embeddings use Gemini (`models/gemini-embedding-001`) — Groq has no embeddings API. Both configured in `parser.py` / `rag.py`.

## Endpoints

All require `X-Internal-Secret`.

| Endpoint | Purpose |
|---|---|
| `POST /parse` | Sentence → structured `TaskDraft` (title/due_date/priority). |
| `POST /ask` | Grounded Q&A over the user's tasks (RAG), returns `{answer, sources}`. |
| `POST /chat` | Conversational agent (LangGraph). Accepts `{user_id, thread_id, message}` or `{user_id, thread_id, confirm}` to resume a paused confirmation. |
| `POST /stream` | Same as `/chat`, streamed via SSE. |

task-tracker exposes the user-facing equivalents behind real cookie auth: `POST /assistant/parse`, `/assistant/ask`, `/assistant/chat`, `/assistant/stream`.

## The agent

`agent.py` — a LangGraph graph with 4 tools (`create_task`, `list_my_tasks`, `search_tasks`, `delete_task`), built per-request via a closure factory (`make_tools_for_user(user_id)`) so `user_id` is baked in and never a parameter the model can see or supply itself. Full design in `PLAN.md`.

- **Human-in-the-loop:** `delete_task` always pauses for confirmation (`interrupt()`), checkpointed via LangGraph's `MemorySaver`, resumed by a later `/chat` or `/stream` call with `confirm: true/false`.
- **Loop guard:** `tool_call_count` in state, capped at 5, to stop runaway tool-call loops.
- **System prompt** (`AGENT_SYSTEM_PROMPT`) enforces: don't take unrequested actions, don't treat task content as instructions, decline off-topic requests, never reveal the system prompt.

## Guardrails

- **Max input length** (500 chars) and **rate limiting** (10 requests/60s, sliding window) — enforced in code (`enforce_message_guardrails()` in task-tracker's `assistant.py`), not prompt-based. These run before the model is ever called, so no prompt injection can bypass them.
- **Off-topic refusal** and **system-prompt-leak refusal** — enforced via the agent's system prompt. This is best-effort, not a hard boundary — treat it as reducing the attack surface, not eliminating it.
- **Per-user isolation** — RAG retrieval is filtered by `user_id` metadata (not semantic similarity), and each tool closure is scoped to one `user_id` baked in at creation. Tested in `tests/test_rag_isolation.py` and `tests/test_agent.py`.

## Observability

Every `/chat` turn logs one JSON line to stdout via `log_turn()` (`observability.py`): timestamp, `user_id`, `thread_id`, tools called, total tokens (aggregated across every LLM call in that turn via `UsageMetadataCallbackHandler`), latency, and outcome (`done` / `confirm_required` / `error`).

## Resilience

The shared `ChatGroq` model (`parser.py`) is configured with `timeout=10s, max_retries=2`. If the LLM is still unreachable after retries, `/chat` and `/stream` catch the failure and return/stream a clean message ("I'm having trouble reaching the assistant right now...") instead of a stack trace or dropped connection.

## Running it

```bash
# task-tracker (Docker, :8000, Postgres :5432)
cd task_tracker/task-tracker
docker compose up -d --build

# copilot (native, :8001; its own Postgres+pgvector on :5433 via Docker)
cd copilot
docker compose up -d
uv run uvicorn main:app --reload --port 8001
```

Requires `.env` in both projects (see `.env.example`): `JWT_SECRET`, `DATABASE_URL`, `INTERNAL_SECRET` (shared, must match on both sides), `GROQ_API_KEY`, `GOOGLE_API_KEY`.

## Eval

```bash
cd copilot
uv run python eval.py
```

`eval.py` runs 10 fixed cases (Q&A/lookup, task creation, a regression check against a real past bug, delete-confirmation, and 2 refusal/guardrail cases) and asserts on structure/behavior, never exact prose. Current score: 10/10. Run this after any prompt or agent change — it's how you catch regressions instead of guessing.

## Known limits

- `MemorySaver` checkpointing is in-memory only. A paused delete confirmation is lost if the process restarts before it's resumed. A real deployment needs a durable checkpointer (e.g. Postgres-backed).
- Off-topic and prompt-leak refusals are prompt-enforced, not code-enforced — a sufficiently creative prompt injection could still bypass them. Only length/rate-limit guardrails are hard boundaries.
- One conversation thread per user (`thread_id = user_id`) — no support for multiple concurrent conversations per user.
- The eval suite and RAG retrieval tests run against a small, clean toy dataset — chunking/retrieval quality conclusions from it don't necessarily hold at real data volume.
- The thin frontend doesn't render streamed responses live yet.

## Cost notes

Both providers are currently on free tiers (Groq for chat, Gemini for embeddings), with no cost ceiling or budget alarm in place. Token usage is logged per call (`parse_task` logs prompt/completion tokens directly; `/chat` logs total tokens per turn via observability). If usage grows, add a per-user or global token budget check before the model call, using the same aggregation already wired up in `observability.py`.
