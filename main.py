import json
import os
import time
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from parser import parse_task, TaskDraft
from rag import ask as rag_ask, store
from agent import graph
from observability import log_turn

app = FastAPI(title="Copilot")

class ParseRequest(BaseModel):
    sentence: str

def verify_internal_secret(x_internal_secret: str = Header(...)) -> None:
    if x_internal_secret != os.environ["INTERNAL_SECRET"]:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid internal secret")


@app.post("/parse", response_model=TaskDraft, dependencies=[Depends(verify_internal_secret)])
def parse(body: ParseRequest) -> TaskDraft:
    return parse_task(body.sentence)


class AskRequest(BaseModel):
    user_id: int
    question: str

@app.post("/ask", dependencies=[Depends(verify_internal_secret)])
def ask(body: AskRequest) -> dict:
    return rag_ask(body.user_id, body.question)


class SearchRequest(BaseModel):
    user_id: int
    query: str

@app.post("/search", dependencies=[Depends(verify_internal_secret)])
def search(body: SearchRequest) -> list[dict]:
    hits = store.similarity_search(body.query, k=5, filter={"user_id": body.user_id})
    return [{"task_id": d.metadata["task_id"], "snippet": d.page_content} for d in hits]


class ChatRequest(BaseModel):
    user_id: int
    thread_id: str
    message: str | None = None
    confirm: bool | None = None

@app.post("/chat", dependencies=[Depends(verify_internal_secret)])
def chat(body: ChatRequest) -> dict:
    config = {"configurable": {"thread_id": body.thread_id}}
    usage_handler = UsageMetadataCallbackHandler()
    config["callbacks"] = [usage_handler]

    prior_state = graph.get_state(config).values
    messages_before = len(prior_state.get("messages", []))

    start = time.perf_counter()
    outcome = "error"
    tools_called: list[str] = []
    try:
        if body.confirm is not None:
            result = graph.invoke(Command(resume=body.confirm), config=config)
        else:
            result = graph.invoke(
                {"messages": [HumanMessage(body.message)], "user_id": body.user_id, "tool_call_count": 0},
                config=config,
            )

        new_messages = result["messages"][messages_before:]
        tools_called.extend(
            call["name"] for m in new_messages for call in (getattr(m, "tool_calls", None) or [])
        )

        interrupt = result.get("__interrupt__")
        if interrupt:
            outcome = "confirm_required"
            return {"status": "confirm_required", "question": interrupt[0].value["question"]}
        outcome = "done"
        return {"status": "done", "answer": result["messages"][-1].content}
    except Exception:
        outcome = "error"
        return {"status": "error", "answer": "I'm having trouble reaching the assistant right now. Please try again in a moment."}
    finally:
        total_tokens = sum(u["total_tokens"] for u in usage_handler.usage_metadata.values())
        log_turn(
            user_id=body.user_id,
            thread_id=body.thread_id,
            tools_called=tools_called,
            total_tokens=total_tokens,
            latency_s=time.perf_counter() - start,
            outcome=outcome,
        )


@app.post("/stream", dependencies=[Depends(verify_internal_secret)])
def stream(body: ChatRequest) -> StreamingResponse:
    config = {"configurable": {"thread_id": body.thread_id}}
    if body.confirm is not None:
        stream_input = Command(resume=body.confirm)
    else:
        stream_input = {"messages": [HumanMessage(body.message)], "user_id": body.user_id, "tool_call_count": 0}

    def event_stream():
        # Each frame's payload is JSON-encoded (not sent as raw text) so a chunk
        # containing a literal newline - e.g. a numbered list mid-stream - can never
        # collide with the blank-line "\n\n" that terminates an SSE frame.
        try:
            for chunk, _metadata in graph.stream(stream_input, config=config, stream_mode="messages"):
                if isinstance(chunk.content, str) and chunk.content:
                    yield f"data: {json.dumps(chunk.content)}\n\n"
            state = graph.get_state(config)
            if state.interrupts:
                question = state.interrupts[0].value["question"]
                yield f"data: {json.dumps('[CONFIRM_REQUIRED] ' + question)}\n\n"
        except Exception:
            msg = "I'm having trouble reaching the assistant right now. Please try again in a moment."
            yield f"data: {json.dumps(msg)}\n\n"
        yield f"data: {json.dumps('[DONE]')}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
