"""Opt-in real committee smoke: Typesafe Jev + DeepSeek in parallel, fused.

Deselected by default (integration marker). Requires both TYPESAFE_API_KEY and
OPENAI_API_KEY. Both members run concurrently; the fused verdict follows the
deterministic worst-case rule in fuse_verdicts.
"""

import json
import os
import time

import pytest

from agent_system.judge import (CommitteeJudgeBackend, CommitteeJudgeConfig, JudgeQuery,
                                OpenAIJsonJudgeBackend, TypesafeJevBackend, TypesafeJevConfig)
from agent_system.models import CorrectionComponent, FrameState, TrajectoryTolerance

pytestmark = pytest.mark.integration


def demo_query() -> JudgeQuery:
    """Same bronze-demo deviation as the single-backend smokes, for comparison."""
    return JudgeQuery(
        shot_id="bronze_reveal_001", plan_id="plan_001", trajectory_time=2.5,
        expected=FrameState(center_x=0.5, center_y=0.5, subject_height_ratio=0.4),
        measured=FrameState(center_x=0.58, center_y=0.49, subject_height_ratio=0.43),
        tolerance=TrajectoryTolerance(center_x_tolerance=0.05, center_y_tolerance=0.05,
                                      height_ratio_tolerance=0.08, distance_tolerance=None),
        errors=[CorrectionComponent(dimension="CENTER_X", target_value=0.5,
                                    observed_value=0.58, error=0.08)],
        confidence=88, trigger="OUT_OF_TOLERANCE")


def test_committee_real_smoke():
    typesafe_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not typesafe_key or not openai_key:
        pytest.skip("TYPESAFE_API_KEY and OPENAI_API_KEY are both required for the committee smoke")
    from openai import OpenAI
    jev = TypesafeJevBackend(TypesafeJevConfig(
        api_key=typesafe_key,
        base_url=os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai"),
        model=os.environ.get("TYPESAFE_MODEL", "jev-latest"),
        timeout_seconds=float(os.environ.get("TYPESAFE_SMOKE_TIMEOUT", "10"))))
    deepseek = OpenAIJsonJudgeBackend(
        OpenAI(api_key=openai_key,
               base_url=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")),
        model=os.environ.get("JUDGE_OPENAI_MODEL", "deepseek-chat"))
    committee = CommitteeJudgeBackend([jev, deepseek],
                                      config=CommitteeJudgeConfig(member_timeout_seconds=10.0))
    result = None
    error = None
    for attempt in range(1, 4):
        started = time.perf_counter()
        try:
            result = committee.judge(demo_query())
            elapsed = time.perf_counter() - started
            print(f"attempt {attempt}: OK in {elapsed:.2f}s (parallel wall time)")
            break
        except Exception as exc:  # noqa: BLE001 - evidence gathering
            error = exc
            elapsed = time.perf_counter() - started
            print(f"attempt {attempt}: FAILED in {elapsed:.2f}s: {type(exc).__name__}: {exc}")
            time.sleep(2)
    if result is None:
        pytest.fail(f"committee smoke failed after retries; last error: {error}")
    assert result.backend == "committee"
    assert result.action in ("CONTINUE", "WARN", "PAUSE")
    assert len(result.members) == 2
    print(json.dumps(result.model_dump(), ensure_ascii=False))
