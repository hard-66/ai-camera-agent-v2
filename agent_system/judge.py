"""Jev judge layer: low-frequency second-agent review above Feedback V2.

Deterministic Feedback remains the only safety floor. This module adds the four
documented wakeup situations (out of tolerance, near tolerance, confidence drop,
about 0.5 s periodic spot check), never wakes the judge while the composition is
inside tolerance, and fuses the judge verdict under two rules: local PAUSE is
final, and the judge may only escalate. Queries carry visual numbers and
tolerances only — never images and never hardware quantities. Provider outage,
timeout or a busy in-flight call degrades to the local decision marked as
unreviewed, so the pipeline keeps running without the judge in the loop.
"""

import json
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from copy import deepcopy
from decimal import Decimal
from typing import Annotated, Literal, Protocol

from pydantic import Field, ValidationError

from .errors import AgentError
from .models import (AgentDecision, ContractModel, CorrectionComponent, FrameState,
                     Identifier, PositiveTime, TrajectoryTolerance)

JudgeTrigger = Literal["NEAR_TOLERANCE", "OUT_OF_TOLERANCE", "CONFIDENCE_DROP", "PERIODIC_CHECK"]
JudgeAction = Literal["CONTINUE", "WARN", "PAUSE"]
FusedAction = Literal["CONTINUE", "ADJUST", "WARN", "PAUSE"]


class JudgeConfig(ContractModel):
    near_tolerance_ratio: Annotated[float, Field(gt=0.0, le=1.0, allow_inf_nan=False)] = 0.7
    periodic_interval_seconds: PositiveTime = 0.5
    confidence_floor: Annotated[int, Field(ge=0, le=100)] = 40
    call_timeout_seconds: PositiveTime = 1.2


class JudgeQuery(ContractModel):
    """One review request. Visual numbers and tolerances only, never images."""

    shot_id: Identifier
    plan_id: Identifier
    trajectory_time: Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
    expected: FrameState
    measured: FrameState
    tolerance: TrajectoryTolerance
    errors: list[CorrectionComponent] = Field(default_factory=list)
    confidence: Annotated[int, Field(ge=0, le=100)] | None = None
    trigger: JudgeTrigger


class JudgeResult(ContractModel):
    """One structured verdict. A judge never carries hardware quantities.

    ``members`` is reserved for committee backends: the archived per-member
    verdicts behind a fused decision. Single backends leave it empty.
    """

    action: JudgeAction
    confidence: Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
    reason: Identifier
    backend: Identifier
    is_reviewed: bool
    quality_score: Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)] | None = None
    members: list["JudgeResult"] = Field(default_factory=list)


class JudgeReview(ContractModel):
    """Supervisor output: local decision plus the optional judge verdict, fused."""

    shot_id: Identifier
    plan_id: Identifier
    trajectory_time: Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
    now: Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
    local_action: Literal["CONTINUE", "ADJUST", "PAUSE"]
    fused_action: FusedAction
    trigger: JudgeTrigger | None
    judge_result: JudgeResult | None
    is_reviewed: bool
    reason: Identifier


class JudgeBackend(Protocol):
    def judge(self, query: JudgeQuery) -> JudgeResult: ...


class FakeJudgeBackend:
    """MOCK ONLY scripted replies; exhaustion raises to exercise degradation."""

    def __init__(self, responses):
        self._responses = iter(responses)
        self.calls = []

    def judge(self, query):
        self.calls.append({"query": deepcopy(query)})
        try:
            return next(self._responses)
        except StopIteration as exc:
            raise AgentError("JUDGE_FIXTURES_EXHAUSTED", "Fake judge fixtures exhausted") from exc


# Stable contract order mirroring Feedback V2 — never a priority ranking.
_CONTRACT_ORDER = ("CENTER_X", "CENTER_Y", "SUBJECT_HEIGHT_RATIO", "DISTANCE")
_DIMENSION_FIELD = {"CENTER_X": "center_x", "CENTER_Y": "center_y",
                    "SUBJECT_HEIGHT_RATIO": "subject_height_ratio", "DISTANCE": "distance"}
_TOLERANCE_FIELD = {"CENTER_X": "center_x_tolerance", "CENTER_Y": "center_y_tolerance",
                    "SUBJECT_HEIGHT_RATIO": "height_ratio_tolerance",
                    "DISTANCE": "distance_tolerance"}


def signed_frame_errors(expected: FrameState, measured: FrameState,
                        tolerance: TrajectoryTolerance) -> list[tuple]:
    """(dimension, signed_error, allowed, target, observed) in stable contract order.

    DISTANCE participates only when the plan carries a distance target. A distance
    target without a distance tolerance is an invalid plan; a missing measured
    distance under a distance target must have paused in Feedback before review.
    """
    if expected.distance is not None and tolerance.distance_tolerance is None:
        raise AgentError("INVALID_PLAN", "Distance target requires distance_tolerance for review")
    if expected.distance is not None and measured.distance is None:
        raise AgentError("JUDGE_QUERY_INVALID",
                         "Measured distance missing for a distance target; Feedback must pause before review")
    rows = []
    for dimension in _CONTRACT_ORDER:
        target_value = getattr(expected, _DIMENSION_FIELD[dimension])
        if target_value is None:
            continue
        observed_value = getattr(measured, _DIMENSION_FIELD[dimension])
        allowed = getattr(tolerance, _TOLERANCE_FIELD[dimension])
        error = float(Decimal(str(observed_value)) - Decimal(str(target_value)))
        rows.append((dimension, error, float(allowed), float(target_value), float(observed_value)))
    return rows


def breach_components(signed_errors: list[tuple]) -> list[CorrectionComponent]:
    """Dimensions beyond tolerance as reusable Feedback correction components."""
    components = []
    for dimension, error, allowed, target_value, observed_value in signed_errors:
        if abs(Decimal(str(error))) > Decimal(str(allowed)):
            components.append(CorrectionComponent(dimension=dimension, target_value=target_value,
                                                  observed_value=observed_value, error=error))
    return components


def evaluate_trigger(signed_errors: list[tuple], confidence: int | None, *, config: JudgeConfig,
                     seconds_since_last_check: float | None) -> JudgeTrigger | None:
    """Pure wakeup check. None means: do not wake the judge this observation.

    Stable situation order: OUT_OF_TOLERANCE, then CONFIDENCE_DROP, then
    NEAR_TOLERANCE, then PERIODIC_CHECK. seconds_since_last_check=None means the
    anchor has just been set and the periodic check has no case yet.
    """
    if any(abs(Decimal(str(error))) > Decimal(str(allowed))
           for _, error, allowed, _, _ in signed_errors):
        return "OUT_OF_TOLERANCE"
    if confidence is not None and confidence < config.confidence_floor:
        return "CONFIDENCE_DROP"
    if any(abs(Decimal(str(error))) >= Decimal(str(allowed)) * Decimal(str(config.near_tolerance_ratio))
           for _, error, allowed, _, _ in signed_errors):
        return "NEAR_TOLERANCE"
    if seconds_since_last_check is not None and seconds_since_last_check >= config.periodic_interval_seconds:
        return "PERIODIC_CHECK"
    return None


class JudgeSupervisor:
    """Single-flight, timeout-guarded review above Feedback V2.

    The first observation anchors the periodic clock; every attempted call re-anchors
    it. PAUSE stays local and final; the judge can escalate CONTINUE/ADJUST to WARN
    or PAUSE, never downgrade. Fusion results and skipped wakeups are all logged for
    the shot archive.
    """

    def __init__(self, backend: JudgeBackend, *, config: JudgeConfig):
        self._backend = backend
        self._config = config
        self._anchor_time: float | None = None
        self._call_lock = threading.Lock()
        self.entries: list[JudgeReview] = []

    def review(self, *, shot_id: str, plan_id: str, trajectory_time: float,
               expected: FrameState, measured: FrameState | None,
               tolerance: TrajectoryTolerance, local_decision: AgentDecision, now: float,
               confidence: int | None = None) -> JudgeReview:
        def record(**fields) -> JudgeReview:
            entry = JudgeReview(shot_id=shot_id, plan_id=plan_id, trajectory_time=trajectory_time,
                                now=now, local_action=local_decision.decision, **fields)
            self.entries.append(entry)
            return entry

        if local_decision.decision == "PAUSE":
            return record(fused_action="PAUSE", trigger=None, judge_result=None,
                          is_reviewed=False, reason="local_pause_final")
        if measured is None:
            return record(fused_action=local_decision.decision, trigger=None, judge_result=None,
                          is_reviewed=False, reason="measurement_missing_skipped")
        if self._anchor_time is None:
            self._anchor_time = now
        seconds_since_last_check = now - self._anchor_time
        signed = signed_frame_errors(expected, measured, tolerance)
        trigger = evaluate_trigger(signed, confidence, config=self._config,
                                   seconds_since_last_check=seconds_since_last_check)
        if trigger is None:
            return record(fused_action=local_decision.decision, trigger=None, judge_result=None,
                          is_reviewed=False, reason="within_tolerance_no_wakeup")
        if not self._call_lock.acquire(blocking=False):
            return record(fused_action=local_decision.decision, trigger=trigger,
                          judge_result=None, is_reviewed=False, reason="judge_busy_skipped")
        try:
            self._anchor_time = now
            query = JudgeQuery(shot_id=shot_id, plan_id=plan_id, trajectory_time=trajectory_time,
                               expected=expected, measured=measured, tolerance=tolerance,
                               errors=breach_components(signed), confidence=confidence,
                               trigger=trigger)
            judge_result, failure_reason = self._invoke(query)
        finally:
            self._call_lock.release()
        if judge_result is None:
            return record(fused_action=local_decision.decision, trigger=trigger,
                          judge_result=None, is_reviewed=False, reason=failure_reason)
        if judge_result.action == "PAUSE":
            fused, fuse_reason = "PAUSE", "judge_pause_proposed"
        elif judge_result.action == "WARN":
            fused, fuse_reason = "WARN", "judge_warn_recorded"
        else:
            fused, fuse_reason = local_decision.decision, "judge_continue_kept_local"
        return record(fused_action=fused, trigger=trigger, judge_result=judge_result,
                      is_reviewed=True, reason=fuse_reason)

    def _invoke(self, query: JudgeQuery) -> tuple[JudgeResult | None, str | None]:
        """One isolated call; a hung worker cannot block later reviews."""
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jev-judge")
        try:
            future = executor.submit(self._backend.judge, query)
            result = future.result(timeout=self._config.call_timeout_seconds)
        except FuturesTimeoutError:
            return None, "judge_timeout_unreviewed"
        except Exception:
            return None, "judge_error_unreviewed"
        finally:
            executor.shutdown(wait=False)
        try:
            candidate = result.model_dump() if isinstance(result, JudgeResult) else result
            return JudgeResult.model_validate(candidate), None
        except (TypeError, ValidationError):
            return None, "judge_error_unreviewed"

    def export_jsonl(self, path) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            for entry in self.entries:
                handle.write(json.dumps(entry.model_dump(), ensure_ascii=False, allow_nan=False))
                handle.write("\n")


_ACTION_INSTRUCTIONS = (
    "Choose the safest disposition for this camera composition deviation: continue, warn, "
    "or pause. Hardware safety already runs locally; your pause is a proposal, not a command.")
_ACTION_CRITERIA = {
    "continue": "Composition still matches the plan within tolerance; keep executing.",
    "warn": "Drifting toward the tolerance edge but still recoverable; record a warning without stopping.",
    "pause": "No longer acceptable against the plan; propose pausing execution.",
}
_CONFORMS_INSTRUCTIONS = ("Probability that the measured composition still conforms to the "
                          "planned composition at this trajectory time.")
_QUALITY_INSTRUCTIONS = "Rate how well the current composition matches the planned look."
_QUALITY_CRITERIA = ["composition far from the planned look",
                     "composition partially matches the planned look",
                     "composition closely matches the planned look"]


def _frame_values(frame_state: FrameState) -> dict:
    return {"center_x": frame_state.center_x, "center_y": frame_state.center_y,
            "subject_height_ratio": frame_state.subject_height_ratio,
            "distance_m": frame_state.distance}


def _tolerance_values(tolerance: TrajectoryTolerance) -> dict:
    return {"center_x": tolerance.center_x_tolerance, "center_y": tolerance.center_y_tolerance,
            "height_ratio": tolerance.height_ratio_tolerance,
            "distance_m": tolerance.distance_tolerance}


def build_jev_payload(query: JudgeQuery, *, model: str) -> dict:
    """Official /v1/systemone request shape.

    Wire format follows the public quickstart (awesome-jev-zh). TypeSafe's judge
    interface is explicitly early-stage, so the design doc defers to the vendor's
    current docs; the parser below also tolerates the gateway boolean/probability
    variant. Confirm the exact shape once with a real-key smoke before demo day.
    """
    state = {"shot_id": query.shot_id, "plan_id": query.plan_id,
             "trajectory_time_s": query.trajectory_time, "wakeup_reason": query.trigger,
             "expected": _frame_values(query.expected), "measured": _frame_values(query.measured),
             "tolerance": _tolerance_values(query.tolerance),
             "errors": [component.model_dump() for component in query.errors],
             "confidence": query.confidence}
    questions = {
        "action": {"type": "choice", "instructions": _ACTION_INSTRUCTIONS,
                   "criteria": dict(_ACTION_CRITERIA)},
        "conforms": {"type": "noul", "instructions": _CONFORMS_INSTRUCTIONS},
        "quality": {"type": "score", "instructions": _QUALITY_INSTRUCTIONS,
                    "criteria": list(_QUALITY_CRITERIA)},
    }
    return {"model": model,
            "state": json.dumps(state, ensure_ascii=False, allow_nan=False, sort_keys=True),
            "questions": questions}


def _conforms_probability(answer) -> float | None:
    if not isinstance(answer, dict):
        return None
    for key in ("noul", "probability"):
        if key in answer:
            value = float(answer[key])
            return value if 0.0 <= value <= 1.0 else None
    return None


def _quality_score(answer) -> float | None:
    if not isinstance(answer, dict) or "score" not in answer:
        return None
    score = float(answer["score"])
    return max(0.0, min(1.0, score / 2.0))


def parse_jev_response(payload) -> dict:
    """Map the three answers to action/confidence/conforms/quality; strict on action."""
    answers = payload.get("answers") if isinstance(payload, dict) else None
    if not isinstance(answers, dict) or "action" not in answers:
        raise AgentError("JUDGE_SCHEMA_ERROR", "Jev response is missing the action answer")
    action_answer = answers["action"]
    if not isinstance(action_answer, dict):
        raise AgentError("JUDGE_SCHEMA_ERROR", "Jev action answer is not an object")
    choice = str(action_answer.get("choice", "")).strip().lower()
    if choice not in ("continue", "warn", "pause"):
        raise AgentError("JUDGE_SCHEMA_ERROR", f"Unknown Jev action choice: {choice!r}")
    confidence = float(action_answer.get("confidence", 0.0))
    if not 0.0 <= confidence <= 1.0:
        raise AgentError("JUDGE_SCHEMA_ERROR", "Jev confidence outside 0..1")
    return {"action": choice, "confidence": confidence,
            "conforms_probability": _conforms_probability(answers.get("conforms")),
            "quality_score": _quality_score(answers.get("quality"))}


class TypesafeJevConfig(ContractModel):
    api_key: Annotated[str, Field(min_length=1)]
    base_url: str = "https://api.typesafe.ai"
    model: str = "jev-latest"
    timeout_seconds: PositiveTime = 1.2


class TypesafeJevBackend:
    """Official TypeSafe Jev decision endpoint; stdlib HTTP only, no new deps."""

    backend_name = "typesafe_jev"

    def __init__(self, config: TypesafeJevConfig):
        self._config = config

    def judge(self, query: JudgeQuery) -> JudgeResult:
        payload = build_jev_payload(query, model=self._config.model)
        request = urllib.request.Request(
            self._config.base_url.rstrip("/") + "/v1/systemone",
            data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self._config.api_key}"},
            method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self._config.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise AgentError("JUDGE_PROVIDER_ERROR", "Typesafe Jev request failed",
                             {"exception_type": type(exc).__name__}) from exc
        parsed = parse_jev_response(body)
        return JudgeResult(action=parsed["action"].upper(), confidence=parsed["confidence"],
                           reason=f"jev_choice_{parsed['action']}", backend=self.backend_name,
                           is_reviewed=True, quality_score=parsed["quality_score"])


JUDGE_JSON_INSTRUCTIONS = (
    "You are the composition review judge for a camera robot shot. "
    "Judge only the supplied numbers against the tolerances; you receive no images. "
    "Escalate freely: answer warn when a deviation is drifting toward the tolerance edge, "
    "and pause when the composition is no longer acceptable. Never emit hardware quantities "
    "or motion commands. Respond with exactly one JSON object: "
    '{"action": "continue"|"warn"|"pause", "confidence": 0..1, '
    '"reason": short_snake_case_reason, "quality_score": 0..1}.')


class OpenAIJsonJudgeBackend:
    """OpenAI-compatible JSON fallback (e.g. DeepSeek); injected SDK client."""

    backend_name = "openai_json"

    def __init__(self, client, *, model: str):
        if not isinstance(model, str) or not model.strip():
            raise AgentError("SCHEMA_ERROR", "An explicit model is required")
        self._client = client
        self._model = model

    def judge(self, query: JudgeQuery) -> JudgeResult:
        payload = build_jev_payload(query, model=self._model)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "system", "content": JUDGE_JSON_INSTRUCTIONS},
                          {"role": "user", "content": payload["state"]}],
                response_format={"type": "json_object"})
        except Exception as exc:
            raise AgentError("JUDGE_PROVIDER_ERROR", "OpenAI-compatible judge request failed",
                             {"exception_type": type(exc).__name__}) from exc
        try:
            reply = json.loads(response.choices[0].message.content)
            action = str(reply["action"]).strip().lower()
            confidence = float(reply["confidence"])
            if action not in ("continue", "warn", "pause"):
                raise ValueError("action outside the allowed set")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence outside 0..1")
            quality = reply.get("quality_score")
            quality_score = float(quality) if isinstance(quality, (int, float)) else None
            if quality_score is not None and not 0.0 <= quality_score <= 1.0:
                raise ValueError("quality_score outside 0..1")
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise AgentError("SCHEMA_ERROR", "Malformed judge reply") from exc
        reason = str(reply.get("reason") or "judge_json_reply")
        return JudgeResult(action=action.upper(), confidence=confidence, reason=reason,
                           backend=self.backend_name, is_reviewed=True,
                           quality_score=quality_score)


class CommitteeJudgeConfig(ContractModel):
    member_confidence_floor: Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)] = 0.5
    member_timeout_seconds: PositiveTime = 1.2


_SEVERITY_ORDER = {"CONTINUE": 0, "WARN": 1, "PAUSE": 2}


def fuse_verdicts(member_results: list[JudgeResult | None], *,
                  config: CommitteeJudgeConfig) -> JudgeResult:
    """Deterministic committee fusion — plain code, never another model.

    Rules, in the project's safety order: low-confidence members are dropped;
    the surviving verdict with the highest severity wins (worst-case bias);
    confidence is the strongest agreeing member; quality is the plain average.
    Raising here lets the supervisor degrade to its unreviewed local decision.
    """
    valid = [member for member in member_results
             if isinstance(member, JudgeResult) and member.confidence >= config.member_confidence_floor]
    if not valid:
        raise AgentError("JUDGE_MEMBERS_FAILED", "No committee member produced a usable verdict")
    top_severity = max(_SEVERITY_ORDER[member.action] for member in valid)
    fused_action = next(action for action, severity in _SEVERITY_ORDER.items() if severity == top_severity)
    agreeing = [member for member in valid if member.action == fused_action]
    confidence = max(member.confidence for member in agreeing)
    scores = [member.quality_score for member in valid if member.quality_score is not None]
    quality_score = sum(scores) / len(scores) if scores else None
    reason = "committee_" + "_".join(sorted({member.action.lower() for member in valid}))
    return JudgeResult(action=fused_action, confidence=confidence, reason=reason,
                       backend="committee", is_reviewed=True, quality_score=quality_score,
                       members=[member for member in member_results if member is not None])


class CommitteeJudgeBackend:
    """Run every member judge in parallel and fuse with fuse_verdicts.

    The fusion adds zero serial latency: wall time is the slowest member.
    A failing or timing-out member is dropped and the survivors decide; when
    no member answers the AgentError propagates so the supervisor marks the
    frame unreviewed. Wiring note: the supervisor's call_timeout_seconds must
    stay above this member_timeout_seconds (add about 0.3 s of headroom).
    """

    backend_name = "committee"

    def __init__(self, member_backends: list[JudgeBackend], *,
                 config: CommitteeJudgeConfig | None = None):
        if not member_backends:
            raise AgentError("SCHEMA_ERROR", "Committee requires at least one member backend")
        self._members = list(member_backends)
        self._config = config or CommitteeJudgeConfig()

    def judge(self, query: JudgeQuery) -> JudgeResult:
        executor = ThreadPoolExecutor(max_workers=len(self._members),
                                      thread_name_prefix="jev-committee")
        futures = {executor.submit(member.judge, query): index
                   for index, member in enumerate(self._members)}
        member_results: list[JudgeResult | None] = [None] * len(self._members)
        try:
            for future in futures:
                try:
                    member_results[futures[future]] = future.result(
                        timeout=self._config.member_timeout_seconds)
                except FuturesTimeoutError:
                    member_results[futures[future]] = None
                except Exception:
                    member_results[futures[future]] = None
        finally:
            executor.shutdown(wait=False)
        return fuse_verdicts(member_results, config=self._config)
