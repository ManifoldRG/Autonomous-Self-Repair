"""Retry driver for outreach scenarios 8 (line) and 9 (tree-150).

Re-simulates until the run reconnects (PyBullet trials are not
bit-deterministic, and scenario 9 additionally scans structure seeds),
then leaves the winning frame cache in videos/ for a --rerender pass.
"""
import gzip
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
VIDEOS = os.path.join(REPO, "videos")


def reconnected(name: str) -> bool:
    cache = os.path.join(VIDEOS, f"{name}_frames.json.gz")
    if not os.path.exists(cache):
        return False
    with gzip.open(cache, "rt", encoding="utf-8") as fh:
        return bool(json.load(fh).get("reconnected"))


def attempt(scenario: int, name: str, seed=None) -> bool:
    cache = os.path.join(VIDEOS, f"{name}_frames.json.gz")
    if os.path.exists(cache):
        os.remove(cache)
    cmd = [sys.executable, os.path.join(HERE, "video_abstract.py"),
           "--scenario", str(scenario), "--sim-only"]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    subprocess.run(cmd, cwd=REPO, check=True)
    ok = reconnected(name)
    print(f"== scenario {scenario} seed={seed}: reconnected={ok}", flush=True)
    return ok


def main():
    # scenario 8: fixed structure, retry on sim non-determinism
    if not reconnected("scenario8_line20"):
        for k in range(6):
            print(f"== scenario 8 attempt {k + 1}/6", flush=True)
            if attempt(8, "scenario8_line20"):
                break
        else:
            print("== scenario 8: NO reconnecting run in 6 attempts", flush=True)

    # scenario 9: scan structure seeds
    if not reconnected("scenario9_tree150"):
        for seed in (42, 43, 44, 45, 46, 47, 48, 49):
            print(f"== scenario 9 seed {seed}", flush=True)
            if attempt(9, "scenario9_tree150", seed=seed):
                break
        else:
            print("== scenario 9: NO reconnecting seed found", flush=True)

    print("== driver done", flush=True)


if __name__ == "__main__":
    main()
