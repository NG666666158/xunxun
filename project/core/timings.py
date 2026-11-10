"""Where one answer's time went, stage by stage.

A turn's ``TurnTimings`` is current (a context variable) while its graph runs. Graph nodes,
the embedding and rerank clients, the store searches and every chat-model call add to it, so
a slow answer can be explained afterwards: the last turns are kept in
``logs/answer_timings.jsonl`` and go into 关于 › 导出诊断信息. A record holds numbers, the
question type and the model name only, never the question or the answer.

Stage names (milliseconds, summed when a stage runs more than once, e.g. a second search):

* graph nodes: ``analyze_query`` (the planner), ``retrieve``, ``build_context``,
  ``generate_answer``, ``fact_check``, ``rewrite_search``, ``search_files``,
  ``read_whole_file``, ``query_table`` …
* inside them: ``embed`` (query embedding), ``search`` (hybrid/BM25/vector search in the
  store), ``rerank``; ``prepare`` is the time before the graph starts (start-up, history).

Each model call is listed with its node, duration, time to its first token and token counts.
``first_token_ms`` is the time from the request to the first answer text the user sees.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable, Iterable, Iterator, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_CURRENT: ContextVar["TurnTimings | None"] = ContextVar("turn_timings", default=None)
_NODE: ContextVar[str] = ContextVar("timing_node", default="")

# The answer node: its first token is the first answer text the user sees.
ANSWER_NODE = "generate_answer"
LOG_NAME = "answer_timings.jsonl"
KEEP_RECORDS = 500


def _ms(seconds: float) -> float:
    return round(seconds * 1000.0, 1)


class TurnTimings:
    def __init__(self, kind: str = "answer", started: float | None = None):
        self.kind = kind
        self.started = time.perf_counter() if started is None else started
        self._lock = threading.Lock()
        self.stages: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.calls: list[dict[str, Any]] = []
        self.first_token_ms: float | None = None
        self.meta: dict[str, Any] = {}
        self.total_ms: float | None = None

    def elapsed_ms(self) -> float:
        return _ms(time.perf_counter() - self.started)

    def add(self, stage: str, ms: float) -> None:
        with self._lock:
            self.stages[stage] = round(self.stages.get(stage, 0.0) + ms, 1)
            self.counts[stage] = self.counts.get(stage, 0) + 1

    def mark_first_token(self) -> None:
        with self._lock:
            if self.first_token_ms is None:
                self.first_token_ms = self.elapsed_ms()

    def model_call(self, ms: float, first_token_ms: float | None, usage: dict[str, Any] | None) -> None:
        call = {"node": _NODE.get() or "other", "ms": round(ms, 1)}
        if first_token_ms is not None:
            call["first_token_ms"] = round(first_token_ms, 1)
        if usage:
            call["input_tokens"] = int(usage.get("input_tokens") or 0)
            call["output_tokens"] = int(usage.get("output_tokens") or 0)
        with self._lock:
            self.calls.append(call)

    def finish(self, **meta: Any) -> dict[str, Any]:
        self.meta.update({key: value for key, value in meta.items() if value not in (None, "")})
        if self.total_ms is None:
            self.total_ms = self.elapsed_ms()
        return self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "kind": self.kind,
                **self.meta,
                "total_ms": self.total_ms if self.total_ms is not None else self.elapsed_ms(),
                "first_token_ms": self.first_token_ms,
                "stages": dict(self.stages),
                "counts": {k: v for k, v in self.counts.items() if v > 1},
                "calls": list(self.calls),
            }


def current() -> TurnTimings | None:
    return _CURRENT.get()


def note(**values: Any) -> None:
    """Add facts about this turn to its record (counts, choices; never text)."""
    timings = _CURRENT.get()
    if timings is not None:
        with timings._lock:
            timings.meta.update(values)


@contextmanager
def activate(timings: TurnTimings | None) -> Iterator[TurnTimings | None]:
    token = _CURRENT.set(timings)
    try:
        yield timings
    finally:
        _CURRENT.reset(token)


def iterate(iterator: Iterable[T], timings: TurnTimings | None) -> Iterator[T]:
    """Yield from ``iterator`` with ``timings`` current while each item is produced.

    A generator may be resumed in a different thread each time (the SSE response steps it
    from a thread pool), so the variable is set around every step, not once.
    """
    iterator = iter(iterator)
    while True:
        token = _CURRENT.set(timings)
        try:
            item = next(iterator)
        except StopIteration:
            return
        finally:
            _CURRENT.reset(token)
        yield item


@contextmanager
def stage(name: str) -> Iterator[None]:
    timings = _CURRENT.get()
    if timings is None:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        timings.add(name, _ms(time.perf_counter() - started))


def timed_node(name: str, fn: Callable[..., T]) -> Callable[..., T]:
    """A graph node that adds its own duration, and names the node for the model calls in it."""

    def node(state):
        token = _NODE.set(name)
        try:
            with stage(name):
                return fn(state)
        finally:
            _NODE.reset(token)

    node.__name__ = getattr(fn, "__name__", name)
    node.__doc__ = getattr(fn, "__doc__", None)
    return node


def bind_context(fn: Callable[..., T]) -> Callable[..., T]:
    """``fn`` run in a worker thread still adds to this turn (thread pools do not copy context)."""
    timings, node = _CURRENT.get(), _NODE.get()
    if timings is None:
        return fn

    def bound(*args, **kwargs):
        first, second = _CURRENT.set(timings), _NODE.set(node)
        try:
            return fn(*args, **kwargs)
        finally:
            _NODE.reset(second)
            _CURRENT.reset(first)

    return bound


class ModelCallTimer:
    """Wraps one streamed model call: total time, time to the first token, token usage."""

    def __init__(self) -> None:
        self.timings = _CURRENT.get()
        self.started = time.perf_counter()
        self.first: float | None = None
        self.usage: dict[str, Any] | None = None

    def token(self) -> None:
        if self.first is None:
            self.first = time.perf_counter()
            if self.timings is not None and _NODE.get() == ANSWER_NODE:
                self.timings.mark_first_token()

    def done(self) -> None:
        if self.timings is None:
            return
        first = _ms(self.first - self.started) if self.first is not None else None
        self.timings.model_call(_ms(time.perf_counter() - self.started), first, self.usage)


# ---------------- the record kept for diagnostics ----------------

_log_lock = threading.Lock()


def summary_line(record: dict[str, Any]) -> str:
    """One log line: the total, the first token and the stages over 50 ms."""
    stages = " ".join(
        f"{name}={value / 1000:.1f}s" for name, value in sorted(record.get("stages", {}).items(), key=lambda kv: -kv[1])
        if value >= 50
    )
    first = record.get("first_token_ms")
    head = f"{record.get('kind', 'answer')} {record.get('intent', '')}/{record.get('profile', '')}".rstrip("/ ")
    calls = len(record.get("calls") or [])
    return (
        f"{head} total={record.get('total_ms', 0) / 1000:.1f}s"
        + (f" first_token={first / 1000:.1f}s" if first is not None else "")
        + f" model_calls={calls} {stages}"
    ).strip()


def save(record: dict[str, Any], log_dir: str | Path) -> None:
    """Append ``record`` to the timing log (the newest ``KEEP_RECORDS`` are kept) and log one line."""
    logger.info("Answer timings: %s", summary_line(record))
    path = Path(log_dir) / LOG_NAME
    line = json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}, ensure_ascii=False)
    try:
        with _log_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            if path.stat().st_size > 400 * KEEP_RECORDS:
                lines = path.read_text(encoding="utf-8").splitlines()
                if len(lines) > KEEP_RECORDS:
                    path.write_text("\n".join(lines[-KEEP_RECORDS:]) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.warning("Answer timings not saved: %s", exc)


def load(log_dir: str | Path, limit: int = KEEP_RECORDS) -> list[dict[str, Any]]:
    path = Path(log_dir) / LOG_NAME
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records = []
    for line in lines[-limit:]:
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return records


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return round(ordered[index], 1)


def overview(records: list[dict[str, Any]]) -> dict[str, Any]:
    """p50/p90 of the total, the first token and each stage, over the answers in ``records``."""
    answers = [r for r in records if r.get("kind", "answer") == "answer"]
    if not answers:
        return {"answers": 0}
    stages: dict[str, list[float]] = {}
    for record in answers:
        for name, value in (record.get("stages") or {}).items():
            stages.setdefault(name, []).append(float(value))
    totals = [float(r.get("total_ms") or 0) for r in answers]
    firsts = [float(r["first_token_ms"]) for r in answers if r.get("first_token_ms") is not None]
    return {
        "answers": len(answers),
        "total_ms": {"p50": _percentile(totals, 0.5), "p90": _percentile(totals, 0.9)},
        "first_token_ms": {"p50": _percentile(firsts, 0.5), "p90": _percentile(firsts, 0.9)},
        "stages_ms": {
            name: {"n": len(values), "p50": _percentile(values, 0.5), "p90": _percentile(values, 0.9)}
            for name, values in sorted(stages.items())
        },
    }
