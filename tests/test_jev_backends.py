"""Provider backends: offline payload/parse tests plus an opt-in real smoke."""

import json
import urllib.error

import pytest

from agent_system.errors import AgentError
from agent_system.judge import (JudgeQuery, OpenAIJsonJudgeBackend, TypesafeJevBackend,
                                TypesafeJevConfig, build_jev_payload, parse_jev_response)
from agent_system.models import (CorrectionComponent, FrameState, TrajectoryTolerance)


def frame(center_x=0.5, center_y=0.5, ratio=0.4, distance=None):
    return FrameState(center_x=center_x, center_y=center_y,
                      subject_height_ratio=ratio, distance=distance)


def make_query():
    return JudgeQuery(shot_id="bronze_reveal_001", plan_id="plan_001", trajectory_time=2.5,
                      expected=frame(), measured=frame(center_x=0.6),
                      tolerance=TrajectoryTolerance(center_x_tolerance=0.05,
                                                    center_y_tolerance=0.05,
                                                    height_ratio_tolerance=0.08,
                                                    distance_tolerance=None),
                      errors=[CorrectionComponent(dimension="CENTER_X", target_value=0.5,
                                                  observed_value=0.6, error=0.1)],
                      confidence=88, trigger="OUT_OF_TOLERANCE")


OFFICIAL_REPLY = {
    "model": "jev-latest",
    "answers": {
        "action": {"type": "choice", "choice": "warn", "confidence": 0.71},
        "conforms": {"type": "noul", "noul": 0.44},
        "quality": {"type": "score", "score": 1.2, "confidence": 0.8},
    },
    "usage": {"input_tokens": 210, "output_tokens": 16},
}


class TestBuildJevPayload:
    def test_state_is_numbers_only_and_questions_cover_three_primitives(self):
        payload = build_jev_payload(make_query(), model="jev-latest")
        assert set(payload) == {"model", "state", "questions"}
        state = json.loads(payload["state"])
        assert set(state) == {"shot_id", "plan_id", "trajectory_time_s", "wakeup_reason",
                              "expected", "measured", "tolerance", "errors", "confidence"}
        assert state["trajectory_time_s"] == 2.5 and state["confidence"] == 88
        assert {name: question["type"] for name, question in payload["questions"].items()} == {
            "action": "choice", "conforms": "noul", "quality": "score"}

    def test_payload_never_contains_the_api_key(self):
        payload = build_jev_payload(make_query(), model="jev-latest")
        assert "super-secret-key" not in json.dumps(payload)


class TestParseJevResponse:
    def test_official_shape_maps_choice_confidence_and_score(self):
        parsed = parse_jev_response(OFFICIAL_REPLY)
        assert parsed["action"] == "warn"
        assert parsed["confidence"] == 0.71
        assert parsed["conforms_probability"] == 0.44
        assert parsed["quality_score"] == 0.6

    def test_gateway_boolean_variant_is_tolerated(self):
        reply = json.loads(json.dumps(OFFICIAL_REPLY))
        reply["answers"]["conforms"] = {"type": "boolean", "probability": 0.31}
        assert parse_jev_response(reply)["conforms_probability"] == 0.31

    def test_missing_answers_is_a_schema_error(self):
        with pytest.raises(AgentError) as excinfo:
            parse_jev_response({"model": "jev-latest"})
        assert excinfo.value.code == "JUDGE_SCHEMA_ERROR"

    def test_unknown_choice_is_a_schema_error(self):
        reply = json.loads(json.dumps(OFFICIAL_REPLY))
        reply["answers"]["action"]["choice"] = "reschedule"
        with pytest.raises(AgentError) as excinfo:
            parse_jev_response(reply)
        assert excinfo.value.code == "JUDGE_SCHEMA_ERROR"


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self._body).encode("utf-8")


class TestTypesafeJevBackend:
    def test_roundtrip_against_a_fake_http_layer(self, monkeypatch):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["headers"] = {key.lower(): value for key, value in request.header_items()}
            captured["timeout"] = timeout
            captured["body"] = request.data
            return _FakeResponse(OFFICIAL_REPLY)

        monkeypatch.setattr("agent_system.judge.urllib.request.urlopen", fake_urlopen)
        backend = TypesafeJevBackend(TypesafeJevConfigSafe())
        result = backend.judge(make_query())
        assert result.action == "WARN"
        assert result.is_reviewed is True
        assert result.backend == "typesafe_jev"
        assert result.quality_score == 0.6
        assert captured["url"].endswith("/v1/systemone")
        assert captured["headers"]["authorization"] == "Bearer super-secret-key"
        assert b"super-secret-key" not in captured["body"]

    def test_network_failure_wraps_into_a_provider_error(self, monkeypatch):
        def fake_urlopen(request, timeout=None):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr("agent_system.judge.urllib.request.urlopen", fake_urlopen)
        backend = TypesafeJevBackend(TypesafeJevConfigSafe())
        with pytest.raises(AgentError) as excinfo:
            backend.judge(make_query())
        assert excinfo.value.code == "JUDGE_PROVIDER_ERROR"


def TypesafeJevConfigSafe():
    from agent_system.judge import TypesafeJevConfig
    return TypesafeJevConfig(api_key="super-secret-key",
                             base_url="https://api.typesafe.example")


class TestOpenAIJsonJudgeBackend:
    def _client_with(self, content):
        captured = {}

        class Message:
            pass

        class Choice:
            pass

        class Reply:
            pass

        message = Message()
        message.content = content
        choice = Choice()
        choice.message = message
        reply = Reply()
        reply.choices = [choice]

        class Completions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return reply

        class Chat:
            completions = Completions()

        class Client:
            chat = Chat()

        return Client(), captured

    def test_parses_a_structured_json_reply(self):
        client, captured = self._client_with(
            '{"action": "pause", "confidence": 0.8, "reason": "subject_lost_risk", '
            '"quality_score": 0.2}')
        backend = OpenAIJsonJudgeBackend(client, model="deepseek-flash")
        result = backend.judge(make_query())
        assert result.action == "PAUSE"
        assert result.is_reviewed is True
        assert result.backend == "openai_json"
        assert result.quality_score == 0.2
        assert captured["response_format"] == {"type": "json_object"}
        assert captured["messages"][0]["role"] == "system"

    def test_malformed_replies_raise_schema_errors(self):
        for content in ("not json", '{"action": "fly", "confidence": 0.5}'):
            client, _ = self._client_with(content)
            backend = OpenAIJsonJudgeBackend(client, model="deepseek-flash")
            with pytest.raises(AgentError) as excinfo:
                backend.judge(make_query())
            assert excinfo.value.code in ("SCHEMA_ERROR", "JUDGE_PROVIDER_ERROR")
