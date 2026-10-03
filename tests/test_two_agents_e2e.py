"""Offline two-agent handshake: Director plan -> observation stream -> judge.

All execution is MOCK ONLY. The Director side replays the scripted demo
fixture through the real Planner V2 pipeline; the judge side runs the real
Gate/Feedback/Supervisor/committee code over a synthetic 5 s shot. The
interface under test is exactly the frozen contract handoff: the judge
reviews the very TargetTrajectory object the Director produced.
"""

from decimal import Decimal

from agent_system.demo_v2 import DEMO_REQUEST, captain_script
from agent_system.feedback import FeedbackConfig, evaluate_feedback_v2
from agent_system.judge import (CommitteeJudgeBackend, CommitteeJudgeConfig, JudgeConfig,
                                JudgeResult, JudgeSupervisor)
from agent_system.llm import FakeDirectorLLM
from agent_system.models import FrameState, Observation, UserRequest
from agent_system.mocks import MockExecutor, MockReachabilityValidator, mock_correction_capabilities
from agent_system.motion_compiler import MockMotionCompiler, demo_registry
from agent_system.planner_v2 import VisualPlanner, VisualPlannerConfig
from agent_system.reachability import ReachabilityResult
from agent_system.state import AgentState

STEP_SECONDS = 0.25
NEAR_OFFSET = Decimal("0.036")   # >= 0.7 x tolerance 0.05, still inside 0.05
OUT_OFFSET = Decimal("0.08")     # beyond tolerance 0.05


class TriggerScriptedJudge:
    """MOCK ONLY committee seat: replies by wakeup trigger, one fixed verdict each."""

    def __init__(self, actions_by_trigger, backend_name):
        self._actions = actions_by_trigger
        self._backend_name = backend_name
        self.calls = []

    def judge(self, query):
        self.calls.append(query.trigger)
        return JudgeResult(action=self._actions[query.trigger], confidence=0.85,
                           reason="scripted_by_trigger", backend=self._backend_name,
                           is_reviewed=True, quality_score=0.6)


def _observation(frame, *, shot_id, plan_id, timestamp, lost=False):
    """MOCK ONLY bbox synthesis, mirroring the demo helper."""
    bbox = None
    if not lost:
        x = Decimal(str(frame.center_x))
        y = Decimal(str(frame.center_y))
        half_height = Decimal(str(frame.subject_height_ratio)) / 2
        bbox = {"x1": float(x - Decimal("0.1")), "x2": float(x + Decimal("0.1")),
                "y1": float(y - half_height), "y2": float(y + half_height)}
    return Observation(shot_id=shot_id, plan_id=plan_id, timestamp=timestamp,
                       bbox=bbox, distance=frame.distance)


def _measured(expected, offset):
    """Shift center_x by the offset; everything else stays on plan."""
    values = expected.model_dump()
    values["center_x"] = float(Decimal(str(expected.center_x)) + offset)
    return FrameState.model_validate(values)


def _offset_at(sample_time):
    if sample_time <= 2.0:
        return Decimal("0")            # on plan
    if sample_time <= 4.0:
        return NEAR_OFFSET             # approaching the tolerance edge
    return OUT_OFFSET                  # beyond tolerance


def run_two_agent_shot():
    registry = demo_registry()
    reachability = MockReachabilityValidator(
        ReachabilityResult(status="REACHABLE", reason="MOCK ONLY fixture verdict"))
    executor = MockExecutor(registry)
    # --- Director agent: user request -> ShotScript 0.2 (scripted LLM, 0 API calls) ---
    script = VisualPlanner(FakeDirectorLLM([captain_script(registry.revision).model_dump()]),
                           config=VisualPlannerConfig()).plan(
        UserRequest(text=DEMO_REQUEST), registry, reachability)
    shot = script.shots[0]
    context = AgentState(plan_id="e2e-two-agents", shot_id=shot.shot_id,
                         registry_revision=registry.revision)
    compiler = MockMotionCompiler(shot.target_trajectory, fixture_id="captain-fixed-object")
    plan = compiler.compile(shot.target_trajectory, context, registry, reachability)
    state = context.mark_ready(plan, registry)
    state = state.on_executor_event(executor.submit_plan(plan))
    state = state.on_executor_event(executor.start(plan.plan_id))

    # --- Judge agent: committee of two scripted seats over the SAME trajectory ---
    jev_seat = TriggerScriptedJudge({"PERIODIC_CHECK": "CONTINUE", "NEAR_TOLERANCE": "WARN",
                                     "OUT_OF_TOLERANCE": "PAUSE"}, "scripted_jev_seat")
    fallback_seat = TriggerScriptedJudge({"PERIODIC_CHECK": "CONTINUE", "NEAR_TOLERANCE": "CONTINUE",
                                          "OUT_OF_TOLERANCE": "WARN"}, "scripted_fallback_seat")
    committee = CommitteeJudgeBackend([jev_seat, fallback_seat],
                                      config=CommitteeJudgeConfig(member_timeout_seconds=1.0))
    supervisor = JudgeSupervisor(committee, config=JudgeConfig())

    # --- One shot: deterministic observation stream through Gate/Feedback/Judge ---
    feedback_config = FeedbackConfig(max_corrections=3, max_age_seconds=1.0)
    frames = []
    timeline = [round(i * STEP_SECONDS, 2) for i in range(int(5.0 / STEP_SECONDS) + 1)]
    for sample_time in timeline:
        expected = shot.target_trajectory.evaluate(sample_time)
        lost = sample_time == timeline[-1]
        measured = None if lost else _measured(expected, _offset_at(sample_time))
        observation = _observation(expected if lost else measured, shot_id=shot.shot_id,
                                   plan_id=plan.plan_id, timestamp=100.0 + sample_time,
                                   lost=lost)
        gate, decision, state = evaluate_feedback_v2(
            shot.target_trajectory, sample_time, observation, state, feedback_config,
            mock_correction_capabilities()["ALL"], trajectory_plan_id=plan.plan_id,
            now=observation.timestamp)
        assert gate.accepted, f"frame at t={sample_time} rejected: {gate.code}"
        review = supervisor.review(
            shot_id=shot.shot_id, plan_id=plan.plan_id, trajectory_time=sample_time,
            expected=expected, measured=observation.to_frame_state(),
            tolerance=shot.target_trajectory.tolerance, local_decision=decision,
            now=observation.timestamp)
        frames.append({"t": sample_time, "decision": decision.decision, "review": review})
    return shot, plan, frames, supervisor, jev_seat, fallback_seat


class TestTwoAgentHandshake:
    def test_judge_reviews_the_directors_own_trajectory_object(self):
        shot, plan, frames, supervisor, jev_seat, fallback_seat = run_two_agent_shot()
        # The interface is the frozen contract handoff: one trajectory object flows
        # from the Director's script into every judge wakeup, never a re-planned copy.
        assert shot.shot_id == "object-shot"
        assert all(entry.shot_id == shot.shot_id and entry.plan_id == plan.plan_id
                   for entry in supervisor.entries)
        assert len(supervisor.entries) == len(frames) == 21

    def test_within_tolerance_wakes_only_on_the_periodic_clock(self):
        _, _, frames, _, jev_seat, _ = run_two_agent_shot()
        on_plan = [f for f in frames if f["t"] <= 2.0]
        # 2.0 s on plan at 0.25 s steps: periodic fires at 0.5 s intervals only.
        woken = [f for f in on_plan if f["review"].trigger is not None]
        assert [f["t"] for f in woken] == [0.5, 1.0, 1.5, 2.0]
        assert all(f["review"].trigger == "PERIODIC_CHECK" for f in woken)
        assert all(f["review"].fused_action == "CONTINUE" for f in on_plan)
        assert jev_seat.calls[:4] == ["PERIODIC_CHECK"] * 4

    def test_near_tolerance_escalates_to_warn(self):
        _, _, frames, _, _, _ = run_two_agent_shot()
        near = [f for f in frames if 2.0 < f["t"] <= 4.0]
        assert all(f["review"].trigger == "NEAR_TOLERANCE" for f in near)
        assert all(f["review"].fused_action == "WARN" for f in near)
        assert near[0]["review"].judge_result.reason == "committee_continue_warn"

    def test_out_of_tolerance_fuses_to_pause_and_keeps_members(self):
        _, _, frames, _, _, _ = run_two_agent_shot()
        breach = [f for f in frames if f["t"] > 4.0 and f["t"] < 5.0]
        assert all(f["review"].trigger == "OUT_OF_TOLERANCE" for f in breach)
        assert all(f["review"].fused_action == "PAUSE" for f in breach)
        fused = breach[0]["review"].judge_result
        assert fused.reason == "committee_pause_warn"
        assert [member.backend for member in fused.members] == ["scripted_jev_seat",
                                                                "scripted_fallback_seat"]

    def test_local_pause_is_final_and_skips_the_committee(self):
        _, _, frames, supervisor, jev_seat, _ = run_two_agent_shot()
        last = frames[-1]
        assert last["decision"] == "PAUSE"                 # Feedback: target lost
        assert last["review"].fused_action == "PAUSE"
        assert last["review"].reason == "local_pause_final"
        assert last["review"].judge_result is None
        # 15 judge wakeups happened (4 periodic + 8 near + 3 breach); the lost frame added none.
        assert len(jev_seat.calls) == 15

    def test_archive_round_trips_through_jsonl(self, tmp_path):
        _, _, frames, supervisor, _, _ = run_two_agent_shot()
        export_path = tmp_path / "two_agents.jsonl"
        supervisor.export_jsonl(export_path)
        lines = export_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 21
        restored_last = lines[-1]
        assert "local_pause_final" in restored_last
        fused_actions = [f["review"].fused_action for f in frames]
        assert fused_actions.count("WARN") == 8 and fused_actions.count("PAUSE") == 4
