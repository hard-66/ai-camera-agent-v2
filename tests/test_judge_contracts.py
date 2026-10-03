"""Judge layer contracts: strict domain, extra-forbid, mock-only backend."""

import pytest
from pydantic import ValidationError

from agent_system.errors import AgentError
from agent_system.judge import (FakeJudgeBackend, JudgeConfig, JudgeQuery, JudgeResult,
                                TypesafeJevConfig)
from agent_system.models import CorrectionComponent, FrameState, TrajectoryTolerance


def frame(center_x=0.5, center_y=0.5, ratio=0.4, distance=None):
    return FrameState(center_x=center_x, center_y=center_y,
                      subject_height_ratio=ratio, distance=distance)


def tolerance(distance=None):
    return TrajectoryTolerance(center_x_tolerance=0.05, center_y_tolerance=0.05,
                               height_ratio_tolerance=0.08, distance_tolerance=distance)


def breach_component():
    return CorrectionComponent(dimension="CENTER_X", target_value=0.5,
                               observed_value=0.6, error=0.1)


def make_query(**overrides):
    values = dict(shot_id="bronze_reveal_001", plan_id="plan_001", trajectory_time=2.5,
                  expected=frame(), measured=frame(center_x=0.6), tolerance=tolerance(),
                  errors=[breach_component()], confidence=None, trigger="OUT_OF_TOLERANCE")
    values.update(overrides)
    return JudgeQuery(**values)


class TestJudgeConfig:
    def test_defaults_match_the_documented_review_policy(self):
        config = JudgeConfig()
        assert config.near_tolerance_ratio == 0.7
        assert config.periodic_interval_seconds == 0.5
        assert config.confidence_floor == 40
        assert config.call_timeout_seconds == 1.2

    def test_rejects_out_of_domain_values(self):
        with pytest.raises(ValidationError):
            JudgeConfig(near_tolerance_ratio=0.0)
        with pytest.raises(ValidationError):
            JudgeConfig(periodic_interval_seconds=-1.0)
        with pytest.raises(ValidationError):
            JudgeConfig(confidence_floor=101)


class TestJudgeQuery:
    def test_accepts_a_numbers_only_review_request(self):
        query = make_query()
        assert query.trigger == "OUT_OF_TOLERANCE"
        assert query.confidence is None

    def test_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            make_query(motor_pwm=120)

    def test_rejects_confidence_outside_0_100(self):
        with pytest.raises(ValidationError):
            make_query(confidence=101)
        with pytest.raises(ValidationError):
            make_query(confidence=-1)

    def test_allows_empty_error_list_for_non_breach_wakeups(self):
        query = make_query(trigger="PERIODIC_CHECK", errors=[])
        assert query.errors == []


class TestJudgeResult:
    def test_accepts_a_reviewed_verdict(self):
        result = JudgeResult(action="WARN", confidence=0.7, reason="near_tolerance_drift",
                             backend="typesafe_jev", is_reviewed=True, quality_score=0.5)
        assert result.action == "WARN"

    def test_rejects_unknown_action_and_out_of_domain_confidence(self):
        with pytest.raises(ValidationError):
            JudgeResult(action="RESUME", confidence=0.5, reason="x", backend="typesafe_jev",
                        is_reviewed=True)
        with pytest.raises(ValidationError):
            JudgeResult(action="WARN", confidence=1.5, reason="x", backend="typesafe_jev",
                        is_reviewed=True)


class TestTypesafeJevConfig:
    def test_requires_an_api_key_and_keeps_defaults(self):
        with pytest.raises(ValidationError):
            TypesafeJevConfig(api_key="")
        config = TypesafeJevConfig(api_key="secret")
        assert config.base_url == "https://api.typesafe.ai"
        assert config.model == "jev-latest"
        assert config.timeout_seconds == 1.2


class TestFakeJudgeBackend:
    def test_replays_scripted_results_and_records_calls(self):
        scripted = JudgeResult(action="CONTINUE", confidence=0.9, reason="ok",
                               backend="fake", is_reviewed=True)
        backend = FakeJudgeBackend([scripted])
        assert backend.judge(make_query()) is scripted
        assert len(backend.calls) == 1
        assert backend.calls[0]["query"].shot_id == "bronze_reveal_001"

    def test_exhaustion_raises_a_structured_error(self):
        backend = FakeJudgeBackend([])
        with pytest.raises(AgentError) as excinfo:
            backend.judge(make_query())
        assert excinfo.value.code == "JUDGE_FIXTURES_EXHAUSTED"
