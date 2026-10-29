"""Build a deterministic, short-context coding pilot from local pinned inputs.

Download sources separately with hf. This builder makes no network calls and
does not execute dataset content. The output contains token IDs and provenance;
records use plain task/solution text, not the model's chat template.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

from tokenizers import Tokenizer


MAGICODER_REVISION = "5f839b1f368a76b161028bb9edff055db34022b2"
GLAIVE_REVISION = "e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac"


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def normalized(value):
    return " ".join(value.split())


def split_for_group(group):
    return (
        "validation"
        if int(digest("muse-coding-v2:" + group)[:8], 16) % 5 == 0
        else "calibration"
    )


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def select_records(candidates, tokenizer, bos, max_tokens, quotas, seen):
    """Group-disjoint hash split; whole records unless explicitly chunkable."""
    output = {"calibration": [], "validation": []}
    groups = set()
    for record in sorted(candidates, key=lambda r: digest("order:" + r["id"])):
        group = record["group"]
        split = split_for_group(group)
        if len(output[split]) == quotas[split] or group in groups:
            continue
        text = record["text"]
        text_digest = digest(normalized(text))
        prompt_digest = digest(normalized(record.get("prompt", text)))
        if text_digest in seen or prompt_digest in seen:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if len(ids) < 96:
            continue
        offset = 0
        if len(ids) + len(bos) > max_tokens:
            if not record.get("chunkable"):
                continue
            offset = min(len(ids) // 3, len(ids) - max_tokens + len(bos))
            ids = ids[offset : offset + max_tokens - len(bos)]
            text = tokenizer.decode(ids, skip_special_tokens=False)
        token_digest = digest(json.dumps(bos + ids))
        if token_digest in seen:
            continue
        groups.add(group)
        seen.update((text_digest, prompt_digest, token_digest))
        output[split].append(
            {
                "id": record["id"],
                "group": group,
                "category": record["category"],
                "source": record["source"],
                "text": text,
                "input_ids": bos + ids,
                "source_text_sha256": text_digest,
                "token_sha256": token_digest,
                "token_offset": offset,
            }
        )
        if all(len(output[k]) == quotas[k] for k in output):
            break
    if any(len(output[k]) != quotas[k] for k in output):
        raise ValueError(
            f"not enough suitable disjoint records: { {k: len(v) for k, v in output.items()} }, wanted {quotas}"
        )
    return output


def magicoder_candidates(path, language):
    for line in path.open():
        row = json.loads(line)
        if row["lang"] != language:
            continue
        yield {
            "id": f"magicoder:{language}:{row['raw_index']}:{row['index']}",
            "group": "magicoder-seed:" + digest(normalized(row["seed"])),
            "category": f"coding_{language}",
            "source": {
                "dataset": "ise-uiuc/Magicoder-OSS-Instruct-75K",
                "revision": MAGICODER_REVISION,
                "index": row["index"],
                "raw_index": row["raw_index"],
                "language": language,
                "license": "mit",
            },
            "prompt": row["problem"],
            "text": "Task:\n" + row["problem"] + "\n\nSolution:\n" + row["solution"],
        }


def glaive_candidates(path):
    for index, row in enumerate(json.loads(path.read_text())):
        if "<functioncall>" not in row["chat"]:
            continue
        # Focus the general tool corpus on data/files/search/programming tools.
        if not re.search(
            r"file|directory|code|repository|search|database|sql|json|test|api",
            row["system"],
            re.I,
        ):
            continue
        first_user = row["chat"].split("ASSISTANT:", 1)[0]
        yield {
            "id": f"glaive:{index}",
            "group": "glaive-user:" + digest(normalized(first_user)),
            "category": "tool_use",
            "prompt": first_user,
            "source": {
                "dataset": "glaiveai/glaive-function-calling-v2",
                "revision": GLAIVE_REVISION,
                "row": index,
                "license": "apache-2.0",
            },
            "text": row["system"] + "\n" + row["chat"],
        }


def repository_candidates(repo, revision, category):
    paths = git(repo, "ls-tree", "-r", "--name-only", revision).decode().splitlines()
    for path in paths:
        if not path.startswith("tests/") or not path.endswith(".py"):
            continue
        group = "repository-file:" + str(repo.resolve()) + ":" + path
        if category == "repository_tests":
            text = git(repo, "show", f"{revision}:{path}").decode(errors="replace")
            source_revision = revision
        else:
            source_revision = (
                git(repo, "log", "-1", "--format=%H", revision, "--", path)
                .decode()
                .strip()
            )
            # Include one real fix/test diff and its commit message, bounded to this file.
            text = git(
                repo,
                "show",
                "--format=medium",
                "--no-ext-diff",
                "--no-renames",
                "--unified=3",
                source_revision,
                "--",
                path,
            ).decode(errors="replace")
            if not text or "diff --git" not in text:
                continue
        yield {
            "id": f"{category}:{source_revision}:{path}",
            "group": group,
            "category": category,
            "chunkable": category == "repository_tests",
            "source": {
                "repository": str(repo.resolve()),
                "revision": source_revision,
                "path": path,
            },
            "text": text,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--magicoder", type=Path, required=True)
    parser.add_argument("--glaive", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a new corpus version")
    if args.max_tokens < 128:
        parser.error("max-tokens must be at least 128")
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    config = json.loads((args.model / "config.json").read_text())
    bos_id = config.get("text_config", config).get("bos_token_id")
    bos = [] if bos_id is None else [bos_id]
    corpus = {
        "schema": 1,
        "format": "plain-text with BOS; no chat template",
        "max_tokens": args.max_tokens,
        "calibration": [],
        "validation": [],
    }
    seen = set()
    sources = [
        (magicoder_candidates(args.magicoder, language), 16, 4)
        for language in ("python", "cpp", "typescript", "rust")
    ]
    sources.append((glaive_candidates(args.glaive), 16, 4))
    sources.extend(
        (repository_candidates(args.repo, args.revision, category), 8, 2)
        for category in ("repository_tests", "repository_patches")
    )
    for candidates, cal, val in sources:
        selected = select_records(
            candidates,
            tokenizer,
            bos,
            args.max_tokens,
            {"calibration": cal, "validation": val},
            seen,
        )
        for split in selected:
            corpus[split].extend(selected[split])
    cal_groups = {r["group"] for r in corpus["calibration"]}
    val_groups = {r["group"] for r in corpus["validation"]}
    if cal_groups & val_groups:
        raise ValueError("calibration/validation source groups overlap")
    ids = [r["id"] for split in ("calibration", "validation") for r in corpus[split]]
    if len(ids) != len(set(ids)):
        raise ValueError("corpus record IDs must be globally unique")
    corpus["inputs"] = {}
    for name, path in (
        ("magicoder", args.magicoder),
        ("glaive", args.glaive),
        ("tokenizer", args.model / "tokenizer.json"),
    ):
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        corpus["inputs"][name] = {"path": str(path.resolve()), "sha256": sha}
    corpus["summary"] = {
        split: {
            "records": len(corpus[split]),
            "tokens": sum(len(r["input_ids"]) for r in corpus[split]),
            "categories": dict(Counter(r["category"] for r in corpus[split])),
        }
        for split in ("calibration", "validation")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(corpus, stream, indent=2)
        stream.write("\n")
    print(json.dumps(corpus["summary"], indent=2))


if __name__ == "__main__":
    main()
