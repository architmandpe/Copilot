# Copilot

The AI assistant behind [task-tracker](https://github.com/architmandpe/task-tracker) — task parsing, semantic search, grounded Q&A, and a full conversational agent (LangGraph) that can create, update, delete, and search a user's tasks on request.

Not a standalone app: every endpoint requires a shared `X-Internal-Secret` header and is only ever called by task-tracker's backend, never directly by a browser. It is deployed (alongside task-tracker, on Render), but there's nothing to visit — no UI, and no unauthenticated route.

## Architecture

```
task-tracker (FastAPI, Postgres) ──X-Internal-Secret──▶ copilot (this repo)
                                                              │
                                                    pgvector Postgres (own instance)
```

copilot never touches task-tracker's database directly — it reads and writes tasks through task-tracker's internal HTTP API, scoped to one `user_id` per request. Its own Postgres (pgvector) exists only to store task embeddings for semantic search / RAG, kept in sync whenever a task is created, updated, or deleted.

## The agent

`agent.py` is a LangGraph graph with 8 tools: read (`list_my_tasks`, `search_tasks`), create (single/bulk), update (single/bulk), and delete (single/bulk — both require human confirmation via LangGraph's interrupt/resume). Task decomposition ("break this into subtasks") is prompt-driven, not a separate tool. Full design notes, guardrails, and known limits are in [COPILOT.md](COPILOT.md).

## Tech stack

FastAPI, LangGraph + LangChain, Groq (chat — `openai/gpt-oss-20b`), Gemini (embeddings), pgvector

## Local development

See task-tracker's [RUNBOOK.md](../task-tracker/RUNBOOK.md) — both services are set up together, cloned as sibling repos.

## Testing

```
uv run pytest          # 17 tests
uv run python eval.py  # 14-case agent behavior eval — asserts on structure/behavior, not exact prose
```
