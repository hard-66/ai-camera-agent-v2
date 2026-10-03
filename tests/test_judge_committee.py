"""Committee backend: parallel members, deterministic worst-case fusion."""

import time

import pytest
from concurrent.futures import TimeoutError as FuturesTimeoutError

from agent_system.errors import AgentError
from agent_system.judge import (CommitteeJudgeBackend, CommitteeJudgeConfig, FakeJudgeBackend, JudgeQuery,
                                JudgeResult, fuse_verdicts)
from agent_system.models import CorrectionComponent, FrameState, TrajectoryTolerance


def verdict(action="CONTINUE", confidence=0.8, quality=None):
    return JudgeResult(action=action, confidence=confidence, reason="scripted",
                       backend="fake", is_reviewed=True, quality_score=quality)


def make_query():
    return JudgeQuery(
        shot_id="bronze_reveal_001", plan_id="plan_001", trajectory_time=2.5,
        expected=FrameState(center_x=0.5, center_y=0.5, subject_height_ratio=0.4),
        measured=FrameState(center_x=0.58, center_y=0.49, subject_height_ratio=0.43),
        tolerance=TrajectoryTolerance(center_x_tolerance=0.05, center_y_tolerance=0.05,
                                      height_ratio_tolerance=0.08, distance_tolerance=None),
        errors=[CorrectionComponent(dimension="CENTER_X", target_value=0.5,
                                    observed_value=0.58, error=0.08)],
        confidence=88, trigger="OUT_OF_TOLERANCE")


class ExplodingBackend:
    def judge(self, query):
        raise AgentError("JUDGE_PROVIDER_ERROR", "simulated outage")


class SleepingBackend:
    def judge(self, query):
        raise FuturesTimeoutError("simulated hang")


class SlowVerdictBackend:
    def __init__(self, seconds, action="CONTINUE"):
        self._seconds = seconds
        self._action = action

    def judge(self, query):
        time.sleep(self._seconds)
        return verdict(self._action)


class TestFuseVerdicts:
    def test_agreeing_members_keep_their_verdict(self):
        result = fuse_verdicts([verdict("CONTINUE"), verdict("CONTINUE", 0.7)],
                               config=CommitteeJudgeConfig())
        assert result.action == "CONTINUE"
        assert result.backend == "committee"
        assert result.reason == "committee_continue"
        assert result.confidence == 0.8

    def test_disagreement_takes_the_stricter_survivor(self):
        assert fuse_verdicts([verdict("CONTINUE"), verdict("WARN")],
                             config=CommitteeJudgeConfig()).action == "WARN"
        assert fuse_verdicts([verdict("WARN"), verdict("PAUSE")],
                             config=CommitteeJudgeConfig()).action == "PAUSE"

    def test_low_confidence_member_is_dropped_but_archived(self):
        result = fuse_verdicts([verdict("PAUSE", 0.2), verdict("WARN", 0.9)],
                               config=CommitteeJudgeConfig())
        assert result.action == "WARN"
        assert len(result.members) == 2

    def test_confidence_comes_from_the_strongest_agreeing_member(self):
        result = fuse_verdicts([verdict("PAUSE", 0.6), verdict("PAUSE", 0.9)],
                               config=CommitteeJudgeConfig())
        assert result.confidence == 0.9

    def test_quality_score_is_the_plain_average(self):
        result = fuse_verdicts([verdict("PAUSE", 0.8, 0.555), verdict("WARN", 0.9, 0.75)],
                               config=CommitteeJudgeConfig())
        assert result.quality_score == pytest.approx(0.6525)

    def test_no_usable_member_raises_for_supervisor_degradation(self):
        with pytest.raises(AgentError) as excinfo:
            fuse_verdicts([None, None], config=CommitteeJudgeConfig())
        assert excinfo.value.code == "JUDGE_MEMBERS_FAILED"


class TestCommitteeBackend:
    def make_committee(self, backends, **config_overrides):
        return CommitteeJudgeBackend(backends, config=CommitteeJudgeConfig(**config_overrides))

    def test_requires_at_least_one_member(self):
        with pytest.raises(AgentError):
            self.make_committee([])

    def test_one_member_failing_degrades_to_the_survivor(self):
        committee = self.make_committee([ExplodingBackend(), FakeJudgeBackend([verdict("WARN")])])
        result = committee.judge(make_query())
        assert result.action == "WARN"
        assert len(result.members) == 1

    def test_one_member_timing_out_degrades_to_the_survivor(self):
        committee = self.make_committee([SleepingBackend(), FakeJudgeBackend([verdict("PAUSE")])],
                                        member_timeout_seconds=0.05)
        result = committee.judge(make_query())
        assert result.action == "PAUSE"

    def test_all_members_failing_propagates_for_unreviewed_degradation(self):
        committee = self.make_committee([ExplodingBackend(), ExplodingBackend()])
        with pytest.raises(AgentError) as excinfo:
            committee.judge(make_query())
        assert excinfo.value.code == "JUDGE_MEMBERS_FAILED"

    def test_members_run_in_parallel_not_in_series(self):
        committee = self.make_committee([SlowVerdictBackend(0.3, "WARN"),
                                         SlowVerdictBackend(0.3, "CONTINUE")],
                                        member_timeout_seconds=5.0)
        started = time.perf_counter()
        result = committee.judge(make_query())
        elapsed = time.perf_counter() - started
        assert result.action == "WARN"
        assert elapsed < 0.55, f"members appear serial: {elapsed:.2f}s"

    def test_fused_result_carries_no_hardware_quantities(self):
        committee = self.make_committee([FakeJudgeBackend([verdict("WARN")]),
                                         FakeJudgeBackend([verdict("PAUSE")])])
        payload = committee.judge(make_query()).model_dump_json()
        assert "pwm" not in payload and "voltage" not in payload and "pulse" not in payload
