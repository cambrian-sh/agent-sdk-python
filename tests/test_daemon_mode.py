"""Tests for daemon mode support in the Cambrian Agent SDK. ADR-0033."""

import json
import sys
import unittest.mock as mock

import pytest

from cambrian_agent_sdk.server import is_daemon_mode, _parse_daemon_cli


# ── is_daemon_mode ────────────────────────────────────────────────────────────

def test_is_daemon_mode_false_by_default():
    """Without --daemon-mode flag the function returns False."""
    with mock.patch.object(sys, "argv", ["agent.py"]):
        assert is_daemon_mode() is False


def test_is_daemon_mode_true_when_flag_present():
    """With --daemon-mode flag the function returns True."""
    with mock.patch.object(sys, "argv", ["agent.py", "--daemon-mode"]):
        assert is_daemon_mode() is True


# ── _parse_daemon_cli ─────────────────────────────────────────────────────────

def test_parse_daemon_cli_defaults():
    """Without flags, returns empty stream_id, default substrate_addr, empty params."""
    with mock.patch.object(sys, "argv", ["agent.py", "--daemon-mode"]):
        stream_id, substrate_addr, params = _parse_daemon_cli()
    assert stream_id == ""
    assert "localhost:50051" in substrate_addr
    assert params == {}


def test_parse_daemon_cli_stream_id():
    """--stream-id value is returned as stream_id."""
    with mock.patch.object(sys, "argv", ["agent.py", "--daemon-mode", "--stream-id", "gold_tracker"]):
        stream_id, _, _ = _parse_daemon_cli()
    assert stream_id == "gold_tracker"


def test_parse_daemon_cli_daemon_params():
    """--daemon-params JSON is deserialised into a dict."""
    params_json = json.dumps({"interval_seconds": 30, "currency": "USD"})
    with mock.patch.object(sys, "argv", ["agent.py", "--daemon-mode", "--daemon-params", params_json]):
        _, _, params = _parse_daemon_cli()
    assert params == {"interval_seconds": 30, "currency": "USD"}


def test_parse_daemon_cli_invalid_json_params_fallback():
    """Invalid --daemon-params JSON yields empty dict (no crash)."""
    with mock.patch.object(sys, "argv", ["agent.py", "--daemon-mode", "--daemon-params", "not-json"]):
        _, _, params = _parse_daemon_cli()
    assert params == {}


def test_parse_daemon_cli_substrate_addr():
    """The daemon reads --substrate-addr — the flag the Go AgentManager actually
    injects (instance_manager.go); --substrate-socket is not part of the contract."""
    with mock.patch.object(sys, "argv",
                           ["agent.py", "--daemon-mode", "--stream-id", "conv:acme",
                            "--substrate-addr", "unix:/tmp/cambrian.sock"]):
        stream_id, substrate_addr, _ = _parse_daemon_cli()
    assert stream_id == "conv:acme"
    assert substrate_addr == "unix:/tmp/cambrian.sock"
