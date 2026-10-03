"""Supervisor fusion rules: local PAUSE is final, judge may only escalate."""

import json
from concurrent.futures import TimeoutError as FuturesTimeoutError

from agent_system.errors import AgentError
from agent_system.judge import FakeJudgeBackend, JudgeConfig, JudgeResult, JudgeSupervisor
from agent_system.models import (AgentDecision, CorrectionComponent, FrameState,
                                 TrajectoryTolerance, TrajectoryCorrectionIntent)


def frame(center_x=0.5, center_y=0.5, ratio=0.4, distance=None):
    return FrameState(center_x=center_x, center_y=center_y,
                      subject_height_ratio=ratio, distance=distance)


def tolerance():
    return TrajectoryTolerance(center_x_tolerance=0.05, center_y_tolerance=0.05,
                               height_ratio_tolerance=0.08, distance_tolerance=None)


def scripted(action="CONTINUE", confidence=0.9):
    return JudgeResult(action=action, confidence=confidence, reason="scripted",
                       backend="fake", is_reviewed=True)


def local_decision(action="CONTINUE"):
    if action == "PAUSE":
        return AgentDecision(decision="PAUSE", reason="target_lost")
    if action == "ADJUST":
        intent = TrajectoryCorrectionIntent(
            shot_id="bronze_reveal_001", plan_id="plan_001", trajectory_time=2.5,
            reason="trajectory_offset",
            components=[CorrectionComponent(dimension="CENTER_X", target_value=0.5,
                                            observed_value=0.6, error=0.1)])
        return AgentDecision(decision="ADJUST", reason="trajectory_offset",
                             correction_intent=intent)
    return AgentDecision(decision="CONTINUE", reason="within_trajectory_tolerance")


_UNSET = object()


def review(supervisor, *, now=10.0, measured=_UNSET, action="CONTINUE", confidence=None,
           shot_id="bronze_reveal_001", plan_id="plan_001"):
    return supervisor.review(
        shot_id=shot_id, plan_id=plan_id, trajectory_time=2.5,
        expected=frame(), measured=frame() if measured is _UNSET else measured,
        tolerance=tolerance(), local_decision=local_decision(action), now=now,
        confidence=confidence)


def make_supervisor(responses, **config_overrides):
    backend = FakeJudgeBackend(responses)
    return JudgeSupervisor(backend, config=JudgeConfig(**config_overrides)), backend


class TestLocalFloorRules:
    def test_local_pause_is_final_and_never_wakes_the_judge(self):
        supervisor, backend = make_supervisor([scripted("CONTINUE")])
        result = review(supervisor, action="PAUSE")
        assert result.fused_action == "PAUSE"
        assert result.reason == "local_pause_final"
        assert result.trigger is None
        assert backend.calls == []

    def test_measurement_missing_skips_review_without_a_call(self):
        supervisor, backend = make_supervisor([scripted()])
        result = review(supervisor, measured=None)
        assert result.reason == "measurement_missing_skipped"
        assert result.fused_action == "CONTINUE"
        assert backend.calls == []

    def test_within_tolerance_wakes_nobody(self):
        supervisor, backend = make_supervisor([scripted()])
        first = review(supervisor, now=0.0)
        second = review(supervisor, now=0.2)
        assert first.trigger is None and second.trigger is None
        assert backend.calls == []
        assert first.reason == "within_tolerance_no_wakeup"


class TestWakeupAndFusion:
    def test_periodic_check_fires_once_per_interval(self):
        supervisor, backend = make_supervisor([scripted("CONTINUE")])
        assert review(supervisor, now=0.0).trigger is None
        periodic = review(supervisor, now=0.5)
        assert periodic.trigger == "PERIODIC_CHECK"
        assert periodic.is_reviewed is True
        assert len(backend.calls) == 1
        assert review(supervisor, now=0.7).trigger is None
        assert len(backend.calls) == 1

    def test_near_tolerance_wakeup_and_judge_warn_escalates(self):
        supervisor, backend = make_supervisor([scripted("WARN", 0.71)])
        result = review(supervisor, now=3.2, measured=frame(center_x=0.536))
        assert result.trigger == "NEAR_TOLERANCE"
        assert result.fused_action == "WARN"
        assert result.reason == "judge_warn_recorded"
        assert result.judge_result.is_reviewed is True
        assert len(backend.calls) == 1

    def test_judge_pause_escalates_a_continue(self):
        supervisor, _ = make_supervisor([scripted("PAUSE", 0.95)])
        result = review(supervisor, now=3.2, measured=frame(center_x=0.6))
        assert result.trigger == "OUT_OF_TOLERANCE"
        assert result.fused_action == "PAUSE"
        assert result.reason == "judge_pause_proposed"

    def test_judge_continue_keeps_the_local_adjust(self):
        supervisor, _ = make_supervisor([scripted("CONTINUE", 0.8)])
        result = review(supervisor, now=3.2, measured=frame(center_x=0.6), action="ADJUST")
        assert result.fused_action == "ADJUST"
        assert result.reason == "judge_continue_kept_local"

    def test_confidence_drop_wakeup_reaches_the_backend(self):
        supervisor, backend = make_supervisor([scripted("WARN")])
        result = review(supervisor, now=1.0, confidence=25)
        assert result.trigger == "CONFIDENCE_DROP"
        assert backend.calls[0]["query"].confidence == 25


class TestDegradation:
    def test_judge_error_degrades_to_the_local_decision(self):
        class ExplodingBackend:
            def judge(self, query):
                raise AgentError("JUDGE_PROVIDER_ERROR", "network down")

        supervisor = JudgeSupervisor(ExplodingBackend(), config=JudgeConfig())
        result = review(supervisor, now=3.2, measured=frame(center_x=0.6))
        assert result.fused_action == "CONTINUE"
        assert result.is_reviewed is False
        assert result.reason == "judge_error_unreviewed"
        assert result.judge_result is None

    def test_judge_timeout_degrades_to_the_local_decision(self):
        class SleepingBackend:
            def judge(self, query):
                raise FuturesTimeoutError("simulated hang")

        supervisor = JudgeSupervisor(SleepingBackend(), config=JudgeConfig(call_timeout_seconds=0.05))
        result = review(supervisor, now=3.2, measured=frame(center_x=0.6))
        assert result.is_reviewed is False
        assert result.reason == "judge_timeout_unreviewed"

    def test_busy_supervisor_skips_without_a_call(self):
        supervisor, backend = make_supervisor([scripted()])
        with supervisor._call_lock:
            result = review(supervisor, now=3.2, measured=frame(center_x=0.6))
        assert result.reason == "judge_busy_skipped"
        assert backend.calls == []


class TestMessageHygieneAndArchive:
    def test_query_carries_only_visual_numbers(self):
        supervisor, backend = make_supervisor([scripted()])
        review(supervisor, now=3.2, measured=frame(center_x=0.6), confidence=88)
        payload = json.dumps(backend.calls[0]["query"].model_dump())
        assert "pwm" not in payload and "pulse" not in payload
        assert "current" not in payload and "voltage" not in payload
        query_keys = set(backend.calls[0]["query"].model_dump())
        assert query_keys == {"shot_id", "plan_id", "trajectory_time", "expected",
                              "measured", "tolerance", "errors", "confidence", "trigger"}

    def test_review_log_is_appended_and_exportable(self, tmp_path):
        supervisor, _ = make_supervisor([scripted("WARN")])
        review(supervisor, now=0.0)
        review(supervisor, now=3.2, measured=frame(center_x=0.536))
        assert len(supervisor.entries) == 2
        export_path = tmp_path / "judge_log.jsonl"
        supervisor.export_jsonl(export_path)
        lines = export_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        restored = json.loads(lines[1])
        assert restored["fused_action"] == "WARN"
        assert restored["shot_id"] == "bronze_reveal_001"
