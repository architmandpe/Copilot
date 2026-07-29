from langchain_core.documents import Document
from rag import store

USER_A = 1001
USER_B = 1002

def test_retrieval_never_crosses_users():
    ids = [f"isolation-test-user{USER_A}-task1", f"isolation-test-user{USER_B}-task2"]
    try:
        store.add_documents(
            [Document(page_content="Task: launch the rocket. Status: todo.", metadata={"user_id": USER_A, "task_id": 1})],
            ids=[ids[0]],
        )
        store.add_documents(
            [Document(page_content="Task: launch the rocket. Status: todo.", metadata={"user_id": USER_B, "task_id": 2})],
            ids=[ids[1]],
        )

        hits = store.similarity_search("rocket launch", k=4, filter={"user_id": USER_A})

        assert len(hits) > 0
        assert all(d.metadata["user_id"] == USER_A for d in hits)
    finally:
        store.delete(ids=ids)
