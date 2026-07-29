import os
import httpx
from rag import ingest_user_tasks, remove_task, store, TASK_TRACKER_URL, INTERNAL_SECRET

USER_ID = 11

# (title, question that should retrieve it) - each covers a distinct topic so
# retrieval can't accidentally match the wrong task.
SEED = [
    ("Write the spec", "what do I need to write?"),
    ("Book flights for the trip", "what travel do I need to arrange?"),
    ("Review the PR", "is there any code I need to review?"),
    ("Buy groceries", "what do I need to buy?"),
    ("Prepare the presentation slides", "do I have a presentation to prepare?"),
    ("Fix the login bug", "any bugs to fix?"),
]


def test_retrieval_surfaces_expected_source():
    task_ids = []
    try:
        for title, _ in SEED:
            resp = httpx.post(
                f"{TASK_TRACKER_URL}/internal/tasks/{USER_ID}",
                json={"title": title},
                headers={"X-Internal-Secret": INTERNAL_SECRET},
            )
            resp.raise_for_status()
            task_ids.append(resp.json()["id"])

        ingest_user_tasks(USER_ID)

        for (_, question), expected_task_id in zip(SEED, task_ids):
            hits = store.similarity_search(question, k=4, filter={"user_id": USER_ID})
            retrieved_ids = {d.metadata["task_id"] for d in hits}
            assert expected_task_id in retrieved_ids, f"retrieval missed for: {question}"
    finally:
        for task_id in task_ids:
            httpx.delete(
                f"{TASK_TRACKER_URL}/internal/tasks/{USER_ID}/{task_id}",
                headers={"X-Internal-Secret": INTERNAL_SECRET},
            )
            remove_task(USER_ID, task_id)
