"""Compare the same short coding completions across Muse runtime revisions."""

import argparse, json, time, urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--base-url", default="http://127.0.0.1:8002")
p.add_argument("--output", type=Path, required=True)
p.add_argument("--compare", type=Path)
a = p.parse_args()
cases = [
    ("add", "def add(a, b):\n    ", 16),
    ("reverse", "def reverse_string(s):\n    ", 8),
    (
        "factorial",
        'def factorial(n):\n    """Return n! for a nonnegative integer."""\n    ',
        8,
    ),
]
results = []
for name, prompt, limit in cases:
    body = {"prompt": prompt, "max_tokens": limit, "temperature": 0}
    started = time.monotonic()
    request = urllib.request.Request(
        a.base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        r = json.load(response)
    item = {
        "name": name,
        "prompt": prompt,
        "response": r,
        "seconds": time.monotonic() - started,
    }
    if a.compare:
        reference = json.loads(a.compare.read_text())[len(results)]
        item["text_matches_baseline"] = (
            r["choices"][0]["text"] == reference["response"]["choices"][0]["text"]
        )
        item["speedup"] = reference["seconds"] / item["seconds"]
    results.append(item)
    a.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(item), flush=True)
