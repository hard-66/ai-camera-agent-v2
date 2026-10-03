"""Wakeup triggers for the judge layer, in the four documented situations."""

import pytest

from agent_system.errors import AgentError
from agent_system.judge import (JudgeConfig, breach_components, evaluate_trigger,
                                signed_frame_errors)
from agent_system.models import FrameState, TrajectoryTolerance


def frame(center_x=0.5, center_y=0.5, ratio=0.4, distance=None):
    return FrameState(center_x=center_x, center_y=center_y,
                      subject_height_ratio=ratio, distance=distance)


def tolerance(distance=None):
    return TrajectoryTolerance(center_x_tolerance=0.05, center_y_tolerance=0.05,
                               height_ratio_tolerance=0.08, distance_tolerance=distance)


def errors_for(measured, expected=None, tol=None):
    return signed_frame_errors(expected or frame(), measured, tol or tolerance())


class TestSignedFrameErrors:
    def test_within_tolerance_measures_small_signed_errors(self):
        signed = errors_for(frame(center_x=0.52))
        by_dimension = {dimension: (error, allowed) for dimension, error, allowed, _, _ in signed}
        assert by_dimension["CENTER_X"] == (0.02, 0.05)

    def test_distance_only_when_the_plan_carries_a_distance_target(self):
        without_distance = errors_for(frame(), frame(distance=None), tolerance())
        assert all(dimension != "DISTANCE" for dimension, _, _, _, _ in without_distance)
        with_distance = signed_frame_errors(frame(distance=1.5), frame(distance=1.4), tolerance(distance=0.15))
        assert any(dimension == "DISTANCE" for dimension, _, _, _, _ in with_distance)

    def test_distance_target_requires_a_distance_tolerance(self):
        with pytest.raises(AgentError) as excinfo:
            signed_frame_errors(frame(distance=1.5), frame(distance=1.5), tolerance(distance=None))
        assert excinfo.value.code == "INVALID_PLAN"

    def test_missing_measured_distance_is_rejected_before_review(self):
        with pytest.raises(AgentError) as excinfo:
            signed_frame_errors(frame(distance=1.5), frame(distance=None), tolerance(distance=0.15))
        assert excinfo.value.code == "JUDGE_QUERY_INVALID"


class TestBreachComponents:
    def test_beyond_tolerance_becomes_a_correction_component(self):
        signed = errors_for(frame(center_x=0.6))
        components = breach_components(signed)
        assert len(components) == 1
        assert components[0].dimension == "CENTER_X"
        assert components[0].target_value == 0.5
        assert components[0].observed_value == 0.6
        assert components[0].error == 0.1

    def test_within_tolerance_has_no_components(self):
        assert breach_components(errors_for(frame(center_x=0.52))) == []


class TestEvaluateTrigger:
    def test_within_tolerance_with_fresh_anchor_wakes_nobody(self):
        signed = errors_for(frame(center_x=0.52))
        assert evaluate_trigger(signed, None, config=JudgeConfig(),
                                seconds_since_last_check=0.1) is None

    def test_out_of_tolerance_wakes_first(self):
        signed = errors_for(frame(center_x=0.6))
        assert evaluate_trigger(signed, None, config=JudgeConfig(),
                                seconds_since_last_check=0.0) == "OUT_OF_TOLERANCE"

    def test_near_tolerance_uses_the_configured_ratio(self):
        config = JudgeConfig(near_tolerance_ratio=0.7)
        just_below = errors_for(frame(center_x=0.534))
        assert evaluate_trigger(just_below, None, config=config,
                                seconds_since_last_check=0.0) is None
        just_inside = errors_for(frame(center_x=0.536))
        assert evaluate_trigger(just_inside, None, config=config,
                                seconds_since_last_check=0.0) == "NEAR_TOLERANCE"

    def test_confidence_drop_wakes_without_a_breach(self):
        signed = errors_for(frame(center_x=0.52))
        assert evaluate_trigger(signed, 30, config=JudgeConfig(),
                                seconds_since_last_check=0.0) == "CONFIDENCE_DROP"
        assert evaluate_trigger(signed, None, config=JudgeConfig(),
                                seconds_since_last_check=0.0) is None

    def test_periodic_check_fires_after_the_configured_interval(self):
        signed = errors_for(frame(center_x=0.52))
        config = JudgeConfig()
        assert evaluate_trigger(signed, None, config=config,
                                seconds_since_last_check=None) is None
        assert evaluate_trigger(signed, None, config=config,
                                seconds_since_last_check=0.4) is None
        assert evaluate_trigger(signed, None, config=config,
                                seconds_since_last_check=0.6) == "PERIODIC_CHECK"

    def test_breach_outranks_confidence_and_periodic(self):
        signed = errors_for(frame(center_x=0.6))
        assert evaluate_trigger(signed, 5, config=JudgeConfig(),
                                seconds_since_last_check=99.0) == "OUT_OF_TOLERANCE"

    def test_near_tolerance_covers_distance_dimension(self):
        signed = signed_frame_errors(frame(distance=1.5), frame(distance=1.62),
                                     tolerance(distance=0.15))
        assert evaluate_trigger(signed, None, config=JudgeConfig(),
                                seconds_since_last_check=0.0) == "NEAR_TOLERANCE"
