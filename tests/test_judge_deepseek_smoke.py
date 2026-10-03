"""Opt-in real DeepSeek JSON judge smoke. Deselected by default (integration marker).

Requires OPENAI_API_KEY in the process environment (optionally OPENAI_BASE_URL
and JUDGE_OPENAI_MODEL). Exercises the OpenAIJsonJudgeBackend fallback with one
real provider request and prints the structured result as non-sensitive evidence.
"""

import json
import os
import time

import pytest

from agent_system.judge import OpenAIJsonJudgeBackend, JudgeQuery
from agent_system.models import CorrectionComponent, FrameState, TrajectoryTolerance

pytestmark = pytest.mark.integration


def demo_query() -> JudgeQuery:
    """Same bronze-demo numbers as the Jev smoke, for a like-for-like comparison."""
    return JudgeQuery(
        shot_id="bronze_reveal_001", plan_id="plan_001", trajectory_time=2.5,
        expected=FrameState(center_x=0.5, center_y=0.5, subject_height_ratio=0.4),
        measured=FrameState(center_x=0.58, center_y=0.49, subject_height_ratio=0.43),
        tolerance=TrajectoryTolerance(center_x_tolerance=0.05, center_y_tolerance=0.05,
                                      height_ratio_tolerance=0.08, distance_tolerance=None),
        errors=[CorrectionComponent(dimension="CENTER_X", target_value=0.5,
                                    observed_value=0.58, error=0.08)],
        confidence=88, trigger="OUT_OF_TOLERANCE")


def test_deepseek_json_judge_real_smoke():
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        pytest.skip("OPENAI_API_KEY not configured; the real fallback smoke is opt-in")
    from openai import OpenAI
    client = OpenAI(api_key=api_key,
                    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"))
    model = os.environ.get("JUDGE_OPENAI_MODEL", "deepseek-chat")
    backend = OpenAIJsonJudgeBackend(client, model=model)
    result = None
    error = None
    for attempt in range(1, 4):
        started = time.perf_counter()
        try:
            result = backend.judge(demo_query())
            elapsed = time.perf_counter() - started
            print(f"attempt {attempt}: OK in {elapsed:.2f}s")
            break
        except Exception as exc:  # noqa: BLE001 - evidence gathering
            error = exc
            elapsed = time.perf_counter() - started
            print(f"attempt {attempt}: FAILED in {elapsed:.2f}s: {type(exc).__name__}: {exc}")
            time.sleep(2)
    if result is None:
        pytest.fail(f"DeepSeek judge smoke failed after retries; last error: {error}")
    assert result.is_reviewed is True
    assert result.action in ("CONTINUE", "WARN", "PAUSE")
    assert result.backend == "openai_json"
    print(json.dumps(result.model_dump(), ensure_ascii=False))
