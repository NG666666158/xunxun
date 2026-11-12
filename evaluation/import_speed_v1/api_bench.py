"""How fast SiliconFlow's bge-m3 embeds for indexing: latency by batch size, throughput by
concurrency, and where the account's rate limit (429) starts.

    python -I evaluation/import_speed_v1/api_bench.py <crud_rag/80000_docs> <out.json> [--levels 4x32,8x32,8x64]

Reads the API key from SILICONFLOW_API_KEY or ``--env <.env file>`` (never printed). Texts are
public news cut to ``--chars`` characters, like the app's children. Connections go direct (no
proxy), as the app does since answer_speed_v2.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import statistics
import threading
import time

import httpx

URL = "https://api.siliconflow.cn/v1/embeddings"
MODEL = "BAAI/bge-m3"


def api_key(env_file: Path | None) -> str:
    key = os.environ.get("SILICONFLOW_API_KEY", "")
    if not key and env_file and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "SILICONFLOW_API_KEY":
                key = value.strip().strip('"').strip("'")
    if not key:
        raise SystemExit("no SILICONFLOW_API_KEY")
    return key


def texts(source: Path, chars: int, count: int) -> list[str]:
    out: list[str] = []
    for part in sorted(source.iterdir()):
        for line in part.read_text(encoding="utf-8", errors="ignore").splitlines():
            body = line.split("正文：", 1)[-1].strip()
            for k in range(0, len(body) - chars // 2, chars):
                out.append(body[k : k + chars])
                if len(out) >= count:
                    return out
    return out


class Bench:
    def __init__(self, key: str):
        self.client = httpx.Client(
            trust_env=False, timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32, keepalive_expiry=50),
            headers={"Authorization": f"Bearer {key}"},
        )

    def call(self, batch: list[str]) -> dict:
        started = time.perf_counter()
        try:
            response = self.client.post(URL, json={"model": MODEL, "input": batch, "encoding_format": "float"})
            seconds = time.perf_counter() - started
            if response.status_code != 200:
                return {"ok": False, "status": response.status_code, "seconds": seconds,
                        "detail": response.text[:160]}
            payload = response.json()
            return {"ok": True, "seconds": seconds, "n": len(payload["data"]),
                    "tokens": int((payload.get("usage") or {}).get("total_tokens") or 0)}
        except httpx.HTTPError as exc:
            return {"ok": False, "status": type(exc).__name__, "seconds": time.perf_counter() - started}


def latency(bench: Bench, pool: list[str], sizes: list[int], repeats: int) -> dict:
    results = {}
    at = 0
    bench.call(pool[:1])  # open the connection
    for size in sizes:
        runs = []
        for _ in range(repeats):
            batch = pool[at : at + size]
            at += size
            runs.append(bench.call(batch))
        ok = [r for r in runs if r["ok"]]
        results[size] = {
            "ok": len(ok), "failed": [r.get("status") for r in runs if not r["ok"]],
            "median_s": round(statistics.median(r["seconds"] for r in ok), 3) if ok else None,
            "tokens_per_text": round(statistics.mean(r["tokens"] / r["n"] for r in ok), 1) if ok else None,
            "detail": next((r.get("detail") for r in runs if not r["ok"]), None),
        }
        print("batch", size, results[size], flush=True)
    return results


def throughput(bench: Bench, pool: list[str], workers: int, size: int, seconds: float, start: int) -> dict:
    lock = threading.Lock()
    state = {"at": start, "calls": [], "stop": time.monotonic() + seconds}

    def worker() -> None:
        while time.monotonic() < state["stop"]:
            with lock:
                at = state["at"]
                state["at"] += size
            result = bench.call(pool[at % len(pool) : at % len(pool) + size] or pool[:size])
            result["t"] = time.monotonic()
            with lock:
                state["calls"].append(result)
            if not result["ok"] and result.get("status") == 429:
                time.sleep(2.0)

    began = time.monotonic()
    with ThreadPoolExecutor(workers) as pool_:
        for _ in range(workers):
            pool_.submit(worker)
    elapsed = time.monotonic() - began
    calls = state["calls"]
    ok = [c for c in calls if c["ok"]]
    out = {
        "workers": workers, "batch": size, "seconds": round(elapsed, 1), "calls": len(calls), "ok": len(ok),
        "429": sum(1 for c in calls if c.get("status") == 429),
        "other_errors": sorted({str(c.get("status")) for c in calls if not c["ok"] and c.get("status") != 429}),
        "texts_per_s": round(sum(c["n"] for c in ok) / elapsed, 1),
        "tokens_per_min": round(sum(c["tokens"] for c in ok) / elapsed * 60),
        "requests_per_min": round(len(ok) / elapsed * 60, 1),
        "median_s": round(statistics.median(c["seconds"] for c in ok), 3) if ok else None,
        "p90_s": round(sorted(c["seconds"] for c in ok)[int(0.9 * (len(ok) - 1))], 3) if ok else None,
        "detail_429": next((c.get("detail") for c in calls if c.get("status") == 429), None),
        # Tokens embedded and 429s per 10 s, to see the window the limit works over.
        "timeline": [
            [sum(c.get("tokens", 0) for c in calls if c["ok"] and k <= c["t"] - began < k + 10),
             sum(1 for c in calls if c.get("status") == 429 and k <= c["t"] - began < k + 10)]
            for k in range(0, int(elapsed) + 1, 10)
        ],
    }
    print("throughput", out, flush=True)
    return out, state["at"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--env", type=Path)
    parser.add_argument("--chars", type=int, default=650)
    parser.add_argument("--sizes", default="1,8,32,64,128")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--levels", default="4x32,8x32,8x64")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--rest", type=float, default=20.0)
    args = parser.parse_args()
    bench = Bench(api_key(args.env))
    pool = texts(args.source, args.chars, 60000)
    report: dict = {"chars": args.chars, "texts": len(pool)}
    sizes = [int(s) for s in args.sizes.split(",") if s]
    if sizes:
        report["latency"] = latency(bench, pool, sizes, args.repeats)
    report["throughput"] = []
    at = 5000
    for level in [x for x in args.levels.split(",") if x]:
        workers, size = (int(v) for v in level.split("x"))
        time.sleep(args.rest)
        result, at = throughput(bench, pool, workers, size, args.seconds, at)
        report["throughput"].append(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
