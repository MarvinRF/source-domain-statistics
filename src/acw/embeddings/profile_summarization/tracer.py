"""Per-user debug traces (prompt, output, metadata) for inspecting LLM generations."""

import json

from .config import TRACE_DIR


def save_debug_trace(
    heuristic: str,
    user_id: str,
    prompt: str,
    output: str | None,
    metadata: dict,
):
    trace_dir = TRACE_DIR / heuristic
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{user_id}.json"
    path.write_text(
        json.dumps(
            {
                "user_id": user_id,
                "heuristic": heuristic,
                "prompt": prompt,
                "output": output,
                "metadata": metadata,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
