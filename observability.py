import json
import time


def log_turn(user_id: int, thread_id: str, tools_called: list[str], total_tokens: int, latency_s: float, outcome: str) -> None:
    print(json.dumps({
        "timestamp": time.time(),
        "user_id": user_id,
        "thread_id": thread_id,
        "tools_called": tools_called,
        "total_tokens": total_tokens,
        "latency_ms": round(latency_s * 1000, 1),
        "outcome": outcome,
    }))
