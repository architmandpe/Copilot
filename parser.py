import os
import json
import datetime as dt
from groq import Groq
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

class TaskDraft(BaseModel):
    """The shape we FORCE the model to return."""
    model_config = {"extra": "forbid"}
    title: str = Field(description="short imperative task title")
    due_date: str | None = Field(default=None, description="ISO date YYYY-MM-DD, or null if none implied")
    priority: str = Field(default="normal", description="one of: low, normal, high")

client = Groq(api_key=os.environ["GROQ_API_KEY"])

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
    "and 'normal' otherwise."
)


class TaskParseError(Exception):
    """Raised when the model fails to return a usable TaskDraft after retrying."""

def parse_task(sentence: str) -> TaskDraft:
    schema = TaskDraft.model_json_schema()
    schema["required"] = list(schema["properties"].keys())

    today_date = dt.date.today()
    today = today_date.isoformat()
    weekday = today_date.strftime("%A")
    system_with_date = SYSTEM + f" Today's date is {today} ({weekday})."

    last_error = None
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                temperature=0,
                messages=[
                    {"role": "system", "content": system_with_date},
                    {"role": "user", "content": sentence},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "task_draft",
                        "strict": True,
                        "schema": schema,
                    },
                },
            )
            usage = response.usage
            print(f"[cost] prompt={usage.prompt_tokens} completion={usage.completion_tokens}")
            return TaskDraft.model_validate(json.loads(response.choices[0].message.content))
        except Exception as e:
            last_error = e

    raise TaskParseError(f"Model failed to return a usable task after 2 attempts: {last_error}")

