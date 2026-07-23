import os
import httpx
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from parser import model
load_dotenv()

TASK_TRACKER_URL = os.environ.get("TASK_TRACKER_URL", "http://localhost:8000")
INTERNAL_SECRET = os.environ["INTERNAL_SECRET"]

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

store = PGVector(
    embeddings=embeddings,
    collection_name="tasks",
    connection=os.environ["DATABASE_URL"],
)


def fetch_user_tasks(user_id: int) -> list[dict]:
    resp = httpx.get(
        f"{TASK_TRACKER_URL}/internal/tasks/{user_id}",
        headers={"X-Internal-Secret": INTERNAL_SECRET},
    )
    resp.raise_for_status()
    return resp.json()


def ingest_user_tasks(user_id: int) -> None:
    tasks = fetch_user_tasks(user_id)
    docs = [
        Document(
            page_content=f"Task: {t['title']}. Status: {t['status']}. Due: {t['due_at']}. Priority: {t['priority']}",
            metadata={"user_id": user_id, "task_id": t["id"]},
        )
        for t in tasks
    ]
    ids = [f"user{user_id}-task{t['id']}" for t in tasks]
    if docs:
        store.add_documents(docs, ids=ids)


GROUNDED = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the question using ONLY the context below. "
     "Interpret the question generously: if a task in the context is relevant to what the "
     "user is asking, even if phrased loosely or indirectly, answer using it. "
     "For example, \"what do I need to write?\" should be answered using a task titled "
     "\"write the spec\" if that task is present. "
     "Only say you don't have enough information if NO task in the context is relevant at all. "
     "Cite the task_id(s) you used.\n\nContext:\n{context}"),
    ("user", "{question}"),
])

def ask(user_id: int, question: str) -> dict:
    hits = store.similarity_search(question, k=4, filter={"user_id": user_id})
    context = "\n".join(f"[{d.metadata['task_id']}] {d.page_content}" for d in hits)
    answer = (GROUNDED | model).invoke({"context": context, "question": question})
    return {"answer": answer.content, "sources": [d.metadata["task_id"] for d in hits]}


if __name__ == "__main__":
    ingest_user_tasks(1)
    print("Ingested. Testing ask()...")
    result = ask(1, "what do I need to write?")
    print(result)
