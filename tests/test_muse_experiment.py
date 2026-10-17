import importlib.util
import json

import pytest
import torch

if importlib.util.find_spec("transformers.models.muse_spark") is None:
    pytest.skip("requires the Muse-capable environment", allow_module_level=True)

from tests.test_muse_evaluate import native_checkpoint
from tests.test_muse_streaming import cpu_threads  # noqa: F401
from reap.muse_experiment import run_experiment


def test_complete_pipeline_resumes_and_compares_both_rankings(tmp_path):
    _, root, batches = native_checkpoint(tmp_path, torch.float32)
    corpus = tmp_path / "corpus.json"
    records = [
        {
            "id": str(i),
            "group": str(i),
            "token_sha256": str(i),
            "category": "test",
            "input_ids": batch["input_ids"][0].tolist(),
        }
        for i, batch in enumerate(batches)
    ]
    corpus.write_text(
        json.dumps({"calibration": records[:1], "validation": records[1:]})
    )
    output = tmp_path / "experiment"
    first = run_experiment(
        root,
        corpus,
        output,
        keeps=[3],
        stop_after_calibration_layer=1,
        progress=lambda x: None,
    )
    assert not first["complete"]
    final = run_experiment(
        root, corpus, output, keeps=[3], resume=True, progress=lambda x: None
    )
    assert final["complete"]
    assert set(final["results"]) == {"baseline", "reap-3", "weighted_ean_sum-3"}
    for name in ("reap-3", "weighted_ean_sum-3"):
        assert (
            final["results"][name]["baseline_mean_nll"]
            == final["results"]["baseline"]["mean_nll"]
        )
        assert final["results"][name]["per_batch"][0]["record_id"] == "1"
        assert (output / "plans" / f"{name}.json").exists()
    with pytest.raises(ValueError, match="identity"):
        run_experiment(root, corpus, output, keeps=[4], resume=True)


def test_pipeline_rejects_split_leakage_before_work(tmp_path):
    _, root, _ = native_checkpoint(tmp_path, torch.float32)
    record = {
        "id": "same",
        "group": "same",
        "token_sha256": "same",
        "input_ids": [1, 3, 4],
    }
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps({"calibration": [record], "validation": [record]}))
    output = tmp_path / "experiment"
    with pytest.raises(ValueError, match="overlap"):
        run_experiment(root, corpus, output, keeps=[3])
    assert not output.exists()
