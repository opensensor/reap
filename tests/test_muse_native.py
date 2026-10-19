import hashlib
from collections import OrderedDict
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


def test_specialized_linear_rejects_offsets_before_native_launch():
    backend = NativeW4.__new__(NativeW4)
    device = torch.device("cpu")
    backend.manifest = {"abi": "muse-w4-offset8-v1"}
    backend.kernels = {device: None}
    backend.scratch = {device: torch.empty(0)}
    backend.geometry = OrderedDict()
    backend.validated_offsets = {}
    launches = []

    def launch(kernel, arguments, cores):
        launches.append(cores)
        arguments[5].zero_()

    backend.launch = launch
    codes = torch.zeros(1, 32, 128, dtype=torch.int8)
    scales = torch.ones(1, 32, 2, dtype=torch.float16)
    offsets = torch.full_like(scales, 8, dtype=torch.int8)
    offsets[0, -1, -1] = 7
    bank = SimpleNamespace(inputs=256, outputs=32, parts=[(codes, scales, offsets)])
    x = torch.ones(1, 256, dtype=torch.float16)
    routes = torch.zeros(1, dtype=torch.int32)
    with pytest.raises(ValueError, match="every weight offset"):
        backend.linear(bank, x, ids=routes)
    assert not launches
    offsets.fill_(8)
    assert backend.linear(bank, x, ids=routes).shape == (1, 32)
    assert launches == [8]
    offsets[0, 0, 0] = 0
    with pytest.raises(ValueError, match="every weight offset"):
        backend.linear(bank, x, ids=routes)
    assert launches == [8]
    backend.manifest = {"abi": "muse-w4-v1"}
    backend.linear(bank, x, ids=routes)
    assert launches == [8, 8]
