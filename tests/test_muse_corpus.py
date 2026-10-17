import hashlib
import json

import pytest

from reap.muse_corpus import magicoder_candidates, select_records, split_for_group


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return type("Encoding", (), {"ids": [ord(c) for c in text]})()

    def decode(self, ids, skip_special_tokens=False):
        return "".join(map(chr, ids))


def candidates():
    return [
        {
            "id": str(i),
            "group": f"group-{i}",
            "category": "test",
            "source": {"row": i},
            "text": (str(i) + " ") * 100,
            "chunkable": True,
        }
        for i in range(100)
    ]


def test_reproducible_split_and_exact_saved_tokens():
    records = candidates()
    kwargs = dict(
        tokenizer=TinyTokenizer(),
        bos=[1],
        max_tokens=128,
        quotas={"calibration": 4, "validation": 3},
    )
    first = select_records(records, seen=set(), **kwargs)
    second = select_records(list(reversed(records)), seen=set(), **kwargs)
    assert first == second
    assert not {r["group"] for r in first["calibration"]} & {
        r["group"] for r in first["validation"]
    }
    for split, selected in first.items():
        for record in selected:
            assert split_for_group(record["group"]) == split
            assert len(record["input_ids"]) == 128
            assert record["input_ids"] == [1] + [ord(c) for c in record["text"]]
            assert (
                record["token_sha256"]
                == hashlib.sha256(json.dumps(record["input_ids"]).encode()).hexdigest()
            )


def test_whole_records_never_silently_truncated():
    records = [{**r, "chunkable": False} for r in candidates()]
    with pytest.raises(ValueError, match="not enough"):
        select_records(
            records,
            TinyTokenizer(),
            [1],
            128,
            {"calibration": 1, "validation": 1},
            set(),
        )


def test_magicoder_ids_include_language_and_source_index(tmp_path):
    source = tmp_path / "data.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "lang": language,
                    "raw_index": 10,
                    "index": 1,
                    "seed": language,
                    "problem": "task",
                    "solution": "solution",
                }
            )
            for language in ("python", "cpp")
        )
    )
    records = [
        next(magicoder_candidates(source, language)) for language in ("python", "cpp")
    ]
    assert records[0]["id"] != records[1]["id"]
    assert records[0]["source"]["raw_index"] == 10
