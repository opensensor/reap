import hashlib
from types import SimpleNamespace

import pytest
import torch

from reap.muse_native import NativeW4, check_manifest


def test_offset8_manifest_and_validation(tmp_path):
    binary = tmp_path / "native.bin"
    binary.write_bytes(b"test")
    entry = {"path": str(binary), "sha256": hashlib.sha256(b"test").hexdigest()}
    manifest = dict(
        name="symmetric",
        namespace="muse_w4_test",
        library=entry,
        kernel=entry,
        tile=1024,
        abi="muse-w4-offset8-v1",
    )
    assert check_manifest(manifest)
    backend = SimpleNamespace(validated_offsets={})
    offsets = torch.full((8, 32, 2), 8, dtype=torch.int8)
    NativeW4.require_offset8(backend, offsets)
    assert backend.validated_offsets[id(offsets)][0] is offsets
    NativeW4.require_offset8(backend, offsets)
    offsets[-1, -1, -1] = 7
    with pytest.raises(ValueError, match="every weight offset"):
        NativeW4.require_offset8(backend, offsets)
    with torch.inference_mode():
        immutable = torch.full((8, 32, 2), 8, dtype=torch.int8)
        NativeW4.require_offset8(backend, immutable)
        NativeW4.require_offset8(backend, immutable)
    assert backend.validated_offsets[id(immutable)] == (immutable, None)
