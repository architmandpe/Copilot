from fastapi import FastAPI
from pydantic import BaseModel
from parser import parse_task, TaskDraft

app = FastAPI(title="Copilot")

class ParseRequest(BaseModel):
    sentence: str

@app.post("/parse", response_model=TaskDraft)
def parse(body: ParseRequest) -> TaskDraft:
    return parse_task(body.sentence)
