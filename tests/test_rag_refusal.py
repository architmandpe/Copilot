from rag import ask

NEVER_INGESTED_USER_ID = 999999

def test_refuses_when_context_is_empty():
    result = ask(NEVER_INGESTED_USER_ID, "what should I do today?")
    assert "enough information" in result["answer"].lower()
    assert result["sources"] == []
