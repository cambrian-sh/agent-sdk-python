"""The ``google.rpc.RetryInfo`` detail of a gRPC status, read and written (ENG-2125).

The kernel's LLM gateway answers a provider 429 that outlived the transport backoff
(and a 5xx) with ``RESOURCE_EXHAUSTED`` / ``UNAVAILABLE`` plus the provider's
Retry-After as a ``google.rpc.RetryInfo`` detail (cambrian-core
``internal/substrate/network/gateway_request.go: withRetryInfo``). The detail travels
in the ``grpc-status-details-bin`` trailer as a serialized ``google.rpc.Status``.

The SDK reads it into ``GenerateError.retry_delay`` and, when the agent's ``run()``
ends on that error, writes it back on the worker's own status so the kernel sees the
same advice from either source.

Why not ``googleapis-common-protos`` / ``grpcio-status``: they would be two new hard
dependencies for three message types. The messages are declared here instead, in a
PRIVATE descriptor pool, so they never collide with those packages when an author's
environment has them installed. The wire format is the stable public one.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

#: The trailer gRPC carries a rich ``google.rpc.Status`` in.
STATUS_DETAILS_KEY = "grpc-status-details-bin"
RETRY_INFO_TYPE_URL = "type.googleapis.com/google.rpc.RetryInfo"

_classes = None


def _messages():
    """(Status, RetryInfo, Any, Duration) message classes from a private pool, built once."""
    global _classes
    if _classes is not None:
        return _classes
    from google.protobuf import any_pb2, descriptor_pb2, descriptor_pool, duration_pb2, message_factory

    pool = descriptor_pool.DescriptorPool()
    for dep in (any_pb2, duration_pb2):
        fdp = descriptor_pb2.FileDescriptorProto()
        dep.DESCRIPTOR.CopyToProto(fdp)
        pool.Add(fdp)

    F = descriptor_pb2.FieldDescriptorProto
    status = descriptor_pb2.FileDescriptorProto(
        name="google/rpc/status.proto", package="google.rpc", syntax="proto3",
        dependency=["google/protobuf/any.proto"])
    m = status.message_type.add(name="Status")
    m.field.add(name="code", number=1, type=F.TYPE_INT32, label=F.LABEL_OPTIONAL)
    m.field.add(name="message", number=2, type=F.TYPE_STRING, label=F.LABEL_OPTIONAL)
    m.field.add(name="details", number=3, type=F.TYPE_MESSAGE, label=F.LABEL_REPEATED,
                type_name=".google.protobuf.Any")
    pool.Add(status)

    details = descriptor_pb2.FileDescriptorProto(
        name="google/rpc/error_details.proto", package="google.rpc", syntax="proto3",
        dependency=["google/protobuf/duration.proto"])
    r = details.message_type.add(name="RetryInfo")
    r.field.add(name="retry_delay", number=1, type=F.TYPE_MESSAGE, label=F.LABEL_OPTIONAL,
                type_name=".google.protobuf.Duration")
    pool.Add(details)

    get = message_factory.GetMessageClass
    _classes = (
        get(pool.FindMessageTypeByName("google.rpc.Status")),
        get(pool.FindMessageTypeByName("google.rpc.RetryInfo")),
        get(pool.FindMessageTypeByName("google.protobuf.Any")),
        get(pool.FindMessageTypeByName("google.protobuf.Duration")),
    )
    return _classes


def retry_delay_from_rpc(exc) -> Optional[float]:
    """The RetryInfo delay, in seconds, of a ``grpc.RpcError`` from a call; None when
    the status carries none (or the trailer cannot be read — the delay is advice)."""
    try:
        trailers = exc.trailing_metadata() or ()
    except Exception:  # noqa: BLE001 — not a grpc.Call (a bare RpcError, a test double)
        return None
    raw = None
    for key, value in trailers:
        if key == STATUS_DETAILS_KEY:
            raw = value
            break
    if not raw:
        return None
    try:
        Status, RetryInfo, _, _ = _messages()
        st = Status.FromString(bytes(raw))
        for any_msg in st.details:
            if any_msg.type_url.endswith("/google.rpc.RetryInfo"):
                info = RetryInfo.FromString(any_msg.value)
                delay = info.retry_delay.seconds + info.retry_delay.nanos / 1e9
                return delay if delay > 0 else None
    except Exception as e:  # noqa: BLE001
        logger.debug("unreadable %s trailer: %s", STATUS_DETAILS_KEY, e)
    return None


def status_details_bin(code: int, message: str, retry_delay: float) -> bytes:
    """A serialized ``google.rpc.Status`` with one RetryInfo detail. ``code`` and
    ``message`` must equal the status the call ends with: grpc-go replaces a status
    whose details trailer disagrees on the code with an INTERNAL "mismatch" error."""
    Status, RetryInfo, Any, Duration = _messages()
    seconds = int(retry_delay)
    nanos = int(round((retry_delay - seconds) * 1e9))
    info = RetryInfo(retry_delay=Duration(seconds=seconds, nanos=nanos))
    st = Status(code=code, message=message)
    st.details.append(Any(type_url=RETRY_INFO_TYPE_URL, value=info.SerializeToString()))
    return st.SerializeToString()
