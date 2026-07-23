from rag import ingest_user_tasks, store

USER_ID = 11

EVAL = [
    ("what do I need to write?", 7),
    ("what travel do I need to arrange?", 9),
    ("is there any code I need to review?", 10),
    ("what do I need to buy?", 11),
    ("do I have a presentation to prepare?", 12),
    ("any bugs to fix?", 13),
]

def test_retrieval_surfaces_expected_source():
    ingest_user_tasks(USER_ID)
    for question, expected_task_id in EVAL:
        hits = store.similarity_search(question, k=4, filter={"user_id": USER_ID})
        retrieved_ids = {d.metadata["task_id"] for d in hits}
        assert expected_task_id in retrieved_ids, f"retrieval missed for: {question}"
