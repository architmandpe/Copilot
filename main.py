import os
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from parser import parse_task, TaskDraft
from rag import ask as rag_ask
from agent import graph

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


class ChatRequest(BaseModel):
    user_id: int
    thread_id: str
    message: str | None = None
    confirm: bool | None = None

@app.post("/chat", dependencies=[Depends(verify_internal_secret)])
def chat(body: ChatRequest) -> dict:
    config = {"configurable": {"thread_id": body.thread_id}}
    if body.confirm is not None:
        result = graph.invoke(Command(resume=body.confirm), config=config)
    else:
        result = graph.invoke(
            {"messages": [HumanMessage(body.message)], "user_id": body.user_id, "tool_call_count": 0},
            config=config,
        )
    interrupt = result.get("__interrupt__")
    if interrupt:
        return {"status": "confirm_required", "question": interrupt[0].value["question"]}
    return {"status": "done", "answer": result["messages"][-1].content}
