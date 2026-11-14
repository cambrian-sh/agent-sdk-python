"""HTTPChatIngress is deleted (owner decision 2026-11-14, ENG-1449).

It was an unauthenticated loopback HTTP chat endpoint in the public API. People and
harnesses reach Cambrian through the operator plane's OpenConversation / SendTurn.
"""
import importlib

import pytest

import cambrian_agent_sdk


def test_http_chat_ingress_is_not_exported():
    assert not hasattr(cambrian_agent_sdk, "HTTPChatIngress")
    assert "HTTPChatIngress" not in cambrian_agent_sdk.__all__


def test_http_chat_ingress_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("cambrian_agent_sdk.http_chat_ingress")
