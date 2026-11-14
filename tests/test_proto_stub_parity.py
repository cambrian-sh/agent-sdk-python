"""The vendored ``_proto/cambrian_pb2.py`` describes the same schema as core's proto.

The SDK ships generated stubs of ``cambrian-core/api/proto/cambrian.proto``;
core CI's proto-sync checks the ``.proto`` copies but not generated stubs
(T-20261031-no-automated-check-that-vendored-proto-copies). This compares the
stub's embedded descriptor with the source compiled by ``grpc_tools.protoc``:
messages, fields, enum values and RPCs — insensitive to line endings and to the
protoc version. It reads the sibling ``../cambrian-core`` checkout and skips
when it is absent, so a standalone SDK checkout still runs green.
"""
from __future__ import annotations

from pathlib import Path

import pytest

descriptor_pb2 = pytest.importorskip("google.protobuf.descriptor_pb2")
protoc = pytest.importorskip("grpc_tools.protoc")
import grpc_tools  # noqa: E402

from cambrian_agent_sdk._proto import cambrian_pb2  # noqa: E402

_CORE_PROTO = Path(__file__).resolve().parents[2] / "cambrian-core" / "api" / "proto"


def _schema(fdp):
    items = {("package", fdp.package)}

    def walk(messages, prefix):
        for m in messages:
            name = prefix + m.name
            items.add(("message", name))
            for f in m.field:
                items.add(("field", name, f.name, f.number, f.type, f.label, f.type_name))
            for e in m.enum_type:
                for v in e.value:
                    items.add(("enum", f"{name}.{e.name}", v.name, v.number))
            walk(m.nested_type, name + ".")

    walk(fdp.message_type, "")
    for e in fdp.enum_type:
        for v in e.value:
            items.add(("enum", e.name, v.name, v.number))
    for s in fdp.service:
        for r in s.method:
            items.add(("rpc", s.name, r.name, r.input_type, r.output_type,
                       r.client_streaming, r.server_streaming))
    return items


def test_vendored_cambrian_stub_matches_core_proto(tmp_path: Path) -> None:
    if not (_CORE_PROTO / "cambrian.proto").is_file():
        pytest.skip("cambrian-core is not checked out next to the SDK")
    out = tmp_path / "set.pb"
    well_known = Path(grpc_tools.__file__).parent / "_proto"
    rc = protoc.main(["protoc", f"-I{_CORE_PROTO}", f"-I{well_known}",
                      f"--descriptor_set_out={out}", "cambrian.proto"])
    assert rc == 0
    fds = descriptor_pb2.FileDescriptorSet()
    fds.ParseFromString(out.read_bytes())
    source = next(f for f in fds.file if f.name == "cambrian.proto")
    vendored = descriptor_pb2.FileDescriptorProto()
    cambrian_pb2.DESCRIPTOR.CopyToProto(vendored)

    want, have = _schema(source), _schema(vendored)
    missing = sorted(want - have, key=str)[:20]
    extra = sorted(have - want, key=str)[:20]
    assert not missing and not extra, (
        "sdk/cambrian_agent_sdk/_proto/cambrian_pb2.py is stale against core's "
        f"cambrian.proto — regenerate it.\nmissing from stub: {missing}\nonly in stub: {extra}"
    )
