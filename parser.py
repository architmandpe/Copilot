import datetime as dt
from typing import Callable
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

class TaskDraft(BaseModel):
    """The shape we FORCE the model to return."""
    model_config = {"extra": "forbid"}
    title: str = Field(description="short imperative task title")
    due_date: str | None = Field(default=None, description="ISO date YYYY-MM-DD, or null if none implied")
    priority: str = Field(default="normal", description="one of: low, normal, high")

model = ChatGroq(model="openai/gpt-oss-20b", temperature=0, timeout=10, max_retries=2)
structured = model.with_structured_output(TaskDraft, include_raw=True, method="json_schema")

SYSTEM = (
    "You convert a user's sentence into a single task. "
    "Infer a concise title. Only set due_date if a date is clearly and explicitly implied; "
    "vague or uncertain time references (e.g. 'maybe', 'sometime', 'next week maybe') "
    "do NOT count as clearly implied — leave due_date null for those. "
    "Never invent details that aren't in the sentence. "
    "If the sentence describes more than one task, merge them into one "
    "combined title rather than dropping any of them. "
    "Set priority to 'high' for urgent language (e.g. 'ASAP', 'urgent'), "
    "'low' for vague/uncertain/tentative language (e.g. 'maybe', 'whenever'), "
    "and 'normal' otherwise. "
    "Today's date is {today} ({weekday})."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("user", "{sentence}"),
])

chain = prompt | structured


class TaskParseError(Exception):
    """Raised when the model fails to return a usable TaskDraft after retrying."""

def parse_task(sentence: str) -> TaskDraft:
    today_date = dt.date.today()
    today = today_date.isoformat()
    weekday = today_date.strftime("%A")

    last_error = None
    for attempt in range(2):
        try:
            result = chain.invoke({"sentence": sentence, "today": today, "weekday": weekday})
            if result["parsing_error"] is not None:
                last_error = result["parsing_error"]
                continue
            usage = result["raw"].usage_metadata
            print(f"[cost] prompt={usage['input_tokens']} completion={usage['output_tokens']}")
            return result["parsed"]
        except Exception as e:
            last_error = e

    raise TaskParseError(f"Model failed to return a usable task after 2 attempts: {last_error}")


chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a task assistant. Be concise."),
    MessagesPlaceholder("history"),
    ("user", "{input}"),
])

chat_chain = chat_prompt | model

def new_conversation(max_turns: int = 5) -> Callable[[str], str]:
    """Returns a say(text) function with its own private, isolated history, bounded to the last max_turns exchanges."""
    history = []

    def say(text: str) -> str:
        reply = chat_chain.invoke({"history": history, "input": text})
        history.extend([HumanMessage(text), AIMessage(reply.content)])
        history[:] = history[-(max_turns * 2):]
        return reply.content

    return say

