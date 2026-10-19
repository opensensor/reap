#!/usr/bin/env python3
"""Control a trusted local Muse resident server without exposing its secret in argv."""

import argparse
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--token-file", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    load = sub.add_parser("load-native")
    load.add_argument("manifest", type=Path)
    switch = sub.add_parser("switch")
    switch.add_argument("--native")
    switch.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    body = None
    if args.command == "load-native":
        body = json.loads(args.manifest.read_text())
    elif args.command == "switch":
        body = {
            "native": args.native,
            "source": args.candidate.read_text() if args.candidate else None,
        }
    request = urllib.request.Request(
        args.url.rstrip("/") + "/resident/" + args.command,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + args.token_file.read_text().strip(),
        },
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        print(json.dumps(json.load(response), indent=2))


if __name__ == "__main__":
    main()
