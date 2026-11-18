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
  Set, the channel is TLS; unset, it is plaintext and the SDK says so ONCE at WARN —
  except for a ``unix:`` socket target, which is local by construction (a kernel-only
  sandboxed agent's socket, ADR-0157 S2) and needs no TLS advice.
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


#: gRPC target schemes that name a local socket, never a network address.
_LOCAL_SOCKET_SCHEMES = ("unix:", "unix-abstract:")


def _is_local_socket(addr: str) -> bool:
    return addr.strip().lower().startswith(_LOCAL_SOCKET_SCHEMES)


def _warn_insecure_once(addr: str) -> None:
    """One WARN per process when the kernel is dialed without TLS over the network
    (ADR-0130 D4).

    A ``unix:`` target is skipped: the socket is local by construction — the kernel
    hands a kernel-only (network-denied) sandboxed agent ``unix:<dir>/kernel.sock``
    (ADR-0157 S2) and treats a unix peer as local (``authz.routablePeer``) — so
    plaintext is the intended transport and the TLS advice would mislead operators
    (the kernel forwards it as an ERROR line per agent). It logs at DEBUG and does
    not use up the once-per-process WARN, so a later TCP plaintext dial still warns.
    """
    global _warned_insecure
    if _is_local_socket(addr):
        logger.debug("dialing the kernel at %s over a local unix socket (plaintext by design)", addr)
        return
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


_warned_identity: set = set()


def kernel_agent_id(declared: str) -> str:
    """The id this process must assert to the kernel: ``CAMBRIAN_AGENT_ID`` when set,
    else ``declared`` (ENG-2129).

    The kernel mints the per-instance token for the agent's REGISTRY id (derived from
    its filename) and passes that id in ``CAMBRIAN_AGENT_ID``. A class that declares a
    different ``agent_id`` would assert the wrong ``x-agent-id`` and every
    agent→kernel call (memory, LLM gateway, tools) is refused UNAUTHENTICATED: the
    token names the registry id. So the registry id wins, the way ``daemon.py`` adopts
    ``--agent-id``. Unset (an agent started by hand, outside the kernel), the declared
    id is kept. A mismatch is logged ONCE per (declared, registry) pair.
    """
    declared = (declared or "").strip()
    registry = (os.environ.get(AGENT_ID_ENV) or "").strip()
    if not registry:
        return declared
    if declared and declared != registry:
        with _warn_lock:
            first = (declared, registry) not in _warned_identity
            _warned_identity.add((declared, registry))
        if first:
            logger.warning(
                "agent declares agent_id '%s' but the kernel launched it as '%s' (%s); "
                "using '%s' for the kernel credential and x-agent-id",
                declared, registry, AGENT_ID_ENV, registry,
            )
    return registry


def _reset_identity_warning_for_tests() -> None:
    with _warn_lock:
        _warned_identity.clear()


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
    configures (module docstring). ``agent_id`` is the caller's own id, the
    ``x-agent-id`` assertion; it wins over ``CAMBRIAN_AGENT_ID``, which is used only
    when it is empty. A kernel-launched agent resolves its id with
    :func:`kernel_agent_id` first (ENG-2129), so the two agree.

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
