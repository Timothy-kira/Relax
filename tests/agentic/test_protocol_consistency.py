# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Canonical request golden tests for the three agentic ingress protocols.

``relax/agentic/session/service.py`` exposes Chat Completions, Responses and
Anthropic Messages endpoints that all converge on
``_normalized_generation_request`` and therefore on ``check_messages``,
``normalize_tools`` and ``normalize_template_kwargs``. The canonical triple they
produce -- ``messages``, ``tools`` and ``chat_template_kwargs`` -- is hashed by
``_messages_tools_template_state_hash`` into the SessionForest prefix-match key,
so a representation that drifts between protocols silently destroys prefix
reuse for a session that mixes them.

These tests pin that contract from committed goldens:

* ``goldens/requests.<protocol>.json`` hold the same semantic scenario written
  idiomatically for each protocol; ``goldens/canonical_state.json`` holds the
  single expected canonical triple all three must produce.
* ``goldens/errors.<protocol>.json`` hold malformed requests together with the
  exact ``param`` and the exact full-path error message each must raise.

Everything here is a pure function call: no network, no Ray actor, no tokenizer
and no model service.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from relax.agentic.session.service import (
    AgenticChatRequestError,
    _normalized_anthropic_request,
    _normalized_chat_request,
    _normalized_responses_request,
)
from relax.agentic.session.state import _messages_tools_template_state_hash


_GOLDEN_DIR = Path(__file__).parent / "goldens"
# The three fields that define one SessionForest state identity. Adding a field
# here requires the Forest hash and every protocol projection to agree on it.
_CANONICAL_FIELDS = ("messages", "tools", "chat_template_kwargs")
_NORMALIZERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "anthropic_messages": _normalized_anthropic_request,
    "chat_completions": _normalized_chat_request,
    "responses": _normalized_responses_request,
}


def _load_golden(name: str) -> dict[str, Any]:
    return json.loads((_GOLDEN_DIR / name).read_text(encoding="utf-8"))


_REQUESTS = {protocol: _load_golden(f"requests.{protocol}.json") for protocol in _NORMALIZERS}
_ERRORS = {protocol: _load_golden(f"errors.{protocol}.json") for protocol in _NORMALIZERS}
_CANONICAL_STATE = _load_golden("canonical_state.json")

_SCENARIOS = sorted(_CANONICAL_STATE)
_SCENARIO_CASES = [(protocol, scenario) for protocol in sorted(_NORMALIZERS) for scenario in _SCENARIOS]
_ERROR_CASES = [(protocol, case) for protocol in sorted(_ERRORS) for case in sorted(_ERRORS[protocol])]


def _canonical_state(protocol: str, scenario: str) -> dict[str, Any]:
    """Return the canonical triple one protocol produces for one scenario."""

    normalized = _NORMALIZERS[protocol](_REQUESTS[protocol][scenario])
    return {field: normalized[field] for field in _CANONICAL_FIELDS}


def _stable_dump(state: dict[str, Any]) -> str:
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


@pytest.mark.parametrize(("protocol", "scenario"), _SCENARIO_CASES, ids=lambda value: value)
def test_protocol_consistency_canonical_state_matches_golden(protocol: str, scenario: str) -> None:
    assert _canonical_state(protocol, scenario) == _CANONICAL_STATE[scenario]


@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_protocol_consistency_equivalent_requests_share_state_hash(scenario: str) -> None:
    hashes = {}
    for protocol in _NORMALIZERS:
        state = _canonical_state(protocol, scenario)
        hashes[protocol] = _messages_tools_template_state_hash(
            state["messages"],
            state["tools"],
            state["chat_template_kwargs"],
        )
    assert len(set(hashes.values())) == 1, hashes


@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_protocol_consistency_canonical_state_serializes_stably(scenario: str) -> None:
    dumps = {protocol: _stable_dump(_canonical_state(protocol, scenario)) for protocol in _NORMALIZERS}
    assert len(set(dumps.values())) == 1, dumps
    assert set(dumps.values()) == {_stable_dump(_CANONICAL_STATE[scenario])}


@pytest.mark.parametrize(("protocol", "case"), _ERROR_CASES, ids=lambda value: value)
def test_protocol_consistency_rejects_invalid_requests(protocol: str, case: str) -> None:
    expected = _ERRORS[protocol][case]
    with pytest.raises(AgenticChatRequestError) as excinfo:
        _NORMALIZERS[protocol](expected["request"])
    assert excinfo.value.message == expected["message"]
    assert excinfo.value.param == expected["param"]
    assert excinfo.value.status_code == 400


def test_protocol_consistency_golden_scenarios_cover_all_protocols() -> None:
    expected = set(_CANONICAL_STATE)
    for protocol, requests in _REQUESTS.items():
        assert set(requests) == expected, protocol


def test_protocol_consistency_images_normalize_to_canonical_block() -> None:
    image_scenarios = [scenario for scenario in _SCENARIOS if scenario.startswith("image_")]
    assert image_scenarios, "the golden corpus must keep at least one image scenario"
    for scenario in image_scenarios:
        for protocol in _NORMALIZERS:
            parts = []
            for message in _canonical_state(protocol, scenario)["messages"]:
                if not isinstance(message["content"], list):
                    continue
                parts.extend(part for part in message["content"] if part["type"] == "image_url")
            assert parts, f"{protocol}/{scenario} projected away its image"
            for part in parts:
                assert set(part) == {"type", "image_url"}
                assert set(part["image_url"]) == {"url"}
                assert isinstance(part["image_url"]["url"], str) and part["image_url"]["url"]
