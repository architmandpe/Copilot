import os
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel
from parser import parse_task, TaskDraft
from rag import ask as rag_ask

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
