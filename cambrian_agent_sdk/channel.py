"""The ONE way the SDK dials the kernel (ADR-0130 D4).

Every gRPC channel the SDK opens to the Substrate — the agent server's backchannel
(``Agent.serve``), the daemon ``SignalStream`` (``daemon.py``, ``server.py``) and the
lazily-dialed ``SubstrateClient`` — is built here, so the credential and transport
decisions are made once and the kernel's requirements reach every caller.

Two facts the kernel establishes on the agent plane, and this module supplies:

* **Credential** — ``CAMBRIAN_AGENT_TOKEN``, the agent's own named secret
  (``agent:<id>:token`` in the kernel's ADR-0101 store), presented on every RPC as
  ``authorization: Bearer <token>``. The kernel injects it into the agent's process
  environment at spawn; an agent started by hand sets it. ``CAMBRIAN_AGENT_ID`` rides
  with it as the ``x-agent-id`` assertion the kernel checks the token against — the
  header is never the identity, the token is.
* **Transport** — ``CAMBRIAN_TLS_CA``, a PEM root the kernel's certificate chains to.
  Set, the channel is TLS; unset, it is plaintext and the SDK says so ONCE at WARN.
  ``CAMBRIAN_TLS_SERVER_NAME`` overrides the name the certificate is verified
  against (a kernel reached as ``localhost`` with a certificate for its hostname).

A loopback kernel with ``server.insecure_localhost=true`` requires neither: the
plaintext channel with no token keeps working exactly as before ADR-0130, which is
how the benchmark harness and the integration tests run.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Callable, List, Optional, Sequence, Tuple

import grpc

logger = logging.getLogger("cambrian.channel")

#: The agent-plane credential (ADR-0130 D1). Presented as a bearer on every RPC.
TOKEN_ENV = "CAMBRIAN_AGENT_TOKEN"
#: The agent the credential was issued to; sent as ``x-agent-id`` when the caller did not.
AGENT_ID_ENV = "CAMBRIAN_AGENT_ID"
#: A PEM root the kernel's TLS certificate chains to. Set ⇒ TLS channel.
TLS_CA_ENV = "CAMBRIAN_TLS_CA"
#: Overrides the hostname the certificate is verified against.
TLS_SERVER_NAME_ENV = "CAMBRIAN_TLS_SERVER_NAME"

_warned_insecure = False
_warn_lock = threading.Lock()


def _warn_insecure_once(addr: str) -> None:
    """One WARN per process when the kernel is dialed without TLS (ADR-0130 D4)."""
    global _warned_insecure
    with _warn_lock:
        if _warned_insecure:
            return
        _warned_insecure = True
    logger.warning(
        "dialing the kernel at %s in PLAINTEXT; set %s to the kernel's PEM root for TLS "
        "(fine on loopback with server.insecure_localhost=true)",
        addr, TLS_CA_ENV,
    )


def _reset_insecure_warning_for_tests() -> None:
    global _warned_insecure
    with _warn_lock:
        _warned_insecure = False


class _Credentials:
    """What the environment says about this process's identity and transport."""

    def __init__(self, agent_id: str = "") -> None:
        self.token = (os.environ.get(TOKEN_ENV) or "").strip()
        self.agent_id = (agent_id or os.environ.get(AGENT_ID_ENV) or "").strip()
        self.tls_ca = (os.environ.get(TLS_CA_ENV) or "").strip()
        self.tls_server_name = (os.environ.get(TLS_SERVER_NAME_ENV) or "").strip()

    def metadata(self, existing: Optional[Sequence[Tuple[str, str]]]) -> List[Tuple[str, str]]:
        """The call metadata with the credential added — never duplicating a key the
        caller already set. An explicit ``x-agent-id`` on the call wins over the env."""
        md: List[Tuple[str, str]] = list(existing or [])
        keys = {k.lower() for k, _ in md}
        if self.token and "authorization" not in keys:
            md.append(("authorization", "Bearer " + self.token))
        if self.agent_id and "x-agent-id" not in keys:
            md.append(("x-agent-id", self.agent_id))
        return md


class _CredentialInterceptor(
    grpc.UnaryUnaryClientInterceptor,
    grpc.UnaryStreamClientInterceptor,
    grpc.StreamUnaryClientInterceptor,
    grpc.StreamStreamClientInterceptor,
):
    """Adds the bearer and the ``x-agent-id`` assertion to every call, all four
    arities. A client interceptor rather than ``grpc.metadata_call_credentials``
    because call credentials only ride a SECURE channel, and the loopback
    ``insecure_localhost`` deployment is plaintext by design."""

    def __init__(self, creds: _Credentials) -> None:
        self._creds = creds

    def _with(self, details):
        md = self._creds.metadata(details.metadata)
        return _ClientCallDetails(details, md)

    def intercept_unary_unary(self, continuation, details, request):
        return continuation(self._with(details), request)

    def intercept_unary_stream(self, continuation, details, request):
        return continuation(self._with(details), request)

    def intercept_stream_unary(self, continuation, details, request_iterator):
        return continuation(self._with(details), request_iterator)

    def intercept_stream_stream(self, continuation, details, request_iterator):
        return continuation(self._with(details), request_iterator)


class _ClientCallDetails(grpc.ClientCallDetails):
    """A ClientCallDetails copy with replaced metadata (the base type is immutable)."""

    def __init__(self, base, metadata) -> None:
        self.method = base.method
        self.timeout = base.timeout
        self.metadata = metadata
        self.credentials = base.credentials
        self.wait_for_ready = getattr(base, "wait_for_ready", None)
        self.compression = getattr(base, "compression", None)


def transport_mode(creds: Optional[_Credentials] = None) -> str:
    """``"tls"`` when a root is configured, else ``"insecure"``. Exposed for logs and tests."""
    creds = creds or _Credentials()
    return "tls" if creds.tls_ca else "insecure"


def build_channel(
    addr: str,
    options: Optional[Sequence[Tuple[str, object]]] = None,
    agent_id: str = "",
    _secure_channel: Optional[Callable] = None,
    _insecure_channel: Optional[Callable] = None,
) -> grpc.Channel:
    """Dial the kernel at ``addr`` with the transport and credential the environment
    configures (module docstring). ``agent_id`` is the caller's own id, used as the
    ``x-agent-id`` assertion when ``CAMBRIAN_AGENT_ID`` is unset.

    The two ``_*_channel`` parameters are test seams for the grpc constructors.
    """
    creds = _Credentials(agent_id=agent_id)
    secure = _secure_channel or grpc.secure_channel
    insecure = _insecure_channel or grpc.insecure_channel
    if creds.tls_ca:
        with open(creds.tls_ca, "rb") as fh:
            root = fh.read()
        ssl = grpc.ssl_channel_credentials(root_certificates=root)
        opts = list(options or [])
        if creds.tls_server_name:
            opts.append(("grpc.ssl_target_name_override", creds.tls_server_name))
        channel = secure(addr, ssl, options=opts)
    else:
        _warn_insecure_once(addr)
        channel = insecure(addr, options=list(options or [])) if options else insecure(addr)
    if creds.token or creds.agent_id:
        channel = grpc.intercept_channel(channel, _CredentialInterceptor(creds))
    return channel
