"""Wait for a verified export and free NPUs, then launch and smoke-test Muse.

Never stops another service. Run under the same CANN/custom-operator environment
as muse_serve. Status and logs persist across client disconnects.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

from reap.muse_streaming import _save_json


def http_json(url, body=None, timeout=10):
    request = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def run(args):
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with (args.state_dir / "deploy.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {
            "supervisor_pid": os.getpid(),
            "model": str(args.model),
            "started_unix": time.time(),
            "serving_ready": False,
        }
        child = None

        def report(phase, **values):
            state.update(phase=phase, updated_unix=time.time(), **values)
            _save_json(args.state_dir / "status.json", state)
            print(json.dumps(state), flush=True)

        try:
            report("waiting_for_verified_model")
            while not (args.model / "transfer-verified.json").exists():
                time.sleep(30)
            receipt = json.loads((args.model / "transfer-verified.json").read_text())
            if not receipt.get("complete"):
                raise ValueError("transfer verification is incomplete")
            import torch
            import torch_npu
            from reap.muse_runtime import placement_plan

            torch_npu.npu.set_compile_mode(jit_compile=False)
            plan = placement_plan(
                args.model,
                args.devices,
                args.max_context,
                optimized=args.optimized,
                flash_attention=args.flash_attention,
                prefill_chunk=args.prefill_chunk,
            )
            while True:
                memory = []
                for stage in plan:
                    with torch.npu.device(stage["device"]):
                        free, total = torch.npu.mem_get_info()
                    memory.append({**stage, "free_bytes": free, "total_bytes": total})
                if all(item["free_bytes"] >= item["required_bytes"] for item in memory):
                    break
                report("waiting_for_free_npus", memory=memory)
                time.sleep(30)
            command = [
                sys.executable,
                "-u",
                "-m",
                "reap.muse_serve",
                "--model",
                str(args.model),
                "--devices",
                *args.devices,
                "--library",
                str(args.library),
                "--max-context",
                str(args.max_context),
                "--prefill-chunk",
                str(args.prefill_chunk),
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
            ]
            if args.optimized:
                command.append("--optimized")
            if args.flash_attention:
                command.append("--flash-attention")
            with (args.state_dir / "server.log").open("a") as log:
                child = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                report(
                    "loading_model",
                    server_pid=child.pid,
                    command=command,
                    memory=memory,
                )
                base = f"http://127.0.0.1:{args.port}"
                deadline = time.monotonic() + 1800
                while True:
                    if child.poll() is not None:
                        raise RuntimeError(
                            f"server exited {child.returncode}; inspect server.log"
                        )
                    try:
                        health = http_json(base + "/health")
                        if (
                            health.get("model") != args.model.name
                            or health.get("pid") != child.pid
                        ):
                            raise RuntimeError("port belongs to a different server")
                        break
                    except urllib.error.URLError:
                        if time.monotonic() > deadline:
                            raise TimeoutError("model loading exceeded 30 minutes")
                        time.sleep(10)
                report("checking_first_response", health=health)
                prompt = "def add(a, b):\n    "
                started = time.monotonic()
                response = http_json(
                    base + "/v1/completions",
                    {
                        "model": args.model.name,
                        "prompt": prompt,
                        "max_tokens": 16,
                        "temperature": 0,
                    },
                    timeout=900,
                )
                if (
                    not response.get("choices")
                    or not response["choices"][0].get("text", "").strip()
                ):
                    raise RuntimeError("first generation returned no text")
                _save_json(
                    args.state_dir / "first-response.json",
                    {
                        "prompt": prompt,
                        "response": response,
                        "seconds": time.monotonic() - started,
                        "scope": "generation/HTTP smoke; not coding quality certification",
                    },
                )
                report(
                    "serving",
                    serving_ready=True,
                    first_response=str(args.state_dir / "first-response.json"),
                )
                code = child.wait()
                report("stopped", serving_ready=False, server_exit_code=code)
        except BaseException as exc:
            report("failed", serving_ready=False, error=str(exc))
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=[f"npu:{i}" for i in range(6)])
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--prefill-chunk", type=int, default=64)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--optimized", action="store_true")
    parser.add_argument("--flash-attention", action="store_true")
    args = parser.parse_args()
    if args.flash_attention and (
        not args.optimized or not 1 <= args.max_context <= 65536
    ):
        parser.error(
            "--flash-attention requires --optimized and context between 1 and 65536"
        )
    if any(not device.startswith("npu:") for device in args.devices):
        parser.error("deployment supervisor requires explicit NPU devices")
    run(args)


if __name__ == "__main__":
    main()
