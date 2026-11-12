"""Keeps the library's embedding requests at the account's rate limit instead of past it.

SiliconFlow limits bge-m3 by tokens per minute (TPM). Measured on the account tested
(evaluation/import_speed_v1/REPORT.md), it behaves like a bucket of about 1.6 million tokens
refilled at about 500 thousand a minute; one request of 32 children takes 0.44 s, so two
requests side by side already ask for 4 million a minute. Without pacing, four files indexed side by side get 429 on
most requests; each waits out its retries (1, 2, 4 ... 20 s) while the workers idle together,
and a file whose sixth try fails loses every batch it had embedded and is read again later.

The pacer is a token bucket in front of those requests:

* a request made while the library reads a file (inside ``paced()``) waits for its tokens
  (estimated from the text: about 0.6 per Chinese character, 0.3 per other character); the
  bucket holds ``burst`` tokens, like the service's, so a rested account starts at full speed;
* a 429 from the embeddings endpoint -- anyone's, a question's too -- lowers the rate by a
  fifth (once per burst of 429s) and holds paced requests for a few seconds;
* every ``GROW_SECONDS`` without a 429 in which the library used most of the rate, it grows
  by a tenth up to ``ceiling``: an account with a higher limit is found within minutes.

Growing past the service's rate first spends what is left in the service's own bucket, so a
rate is only taken as the account's (``known``: what the estimate uses, and what is kept in
``<data_dir>/cache/embed_rate.json`` for the next start) after a 429 has cut it, or after
``PROVEN_SECONDS`` of busy sending at it without one.

It is installed as event hooks on the shared HTTP client, so the model clients need no change,
and requests outside ``paced()`` (a question being answered) are never held.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import logging
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Iterable, Iterator

logger = logging.getLogger(__name__)

_local = threading.local()
_CJK = re.compile(r"[⺀-鿿豈-﫿＀-￯]")


@contextmanager
def paced() -> Iterator[None]:
    """Embedding requests made by this thread meanwhile wait for the pacer (library jobs)."""
    depth = getattr(_local, "depth", 0)
    _local.depth = depth + 1
    try:
        yield
    finally:
        _local.depth = depth


def is_paced() -> bool:
    return getattr(_local, "depth", 0) > 0


def estimate_tokens(texts: Iterable[str]) -> float:
    """bge-m3 tokens of these texts, roughly (api_bench.py: 370 for 650 characters of news)."""
    total = 0.0
    for text in texts:
        text = str(text)
        cjk = len(_CJK.findall(text))
        total += 0.6 * cjk + 0.3 * (len(text) - cjk) + 2
    return total


class EmbedPacer:
    # A burst of 429s (several requests in flight) cuts the rate once.
    CUT_WINDOW = 3.0
    CUT = 0.8
    HOLD_SECONDS = 4.0
    GROW_SECONDS = 20.0
    GROW = 1.1
    # The rate grows after GROW_SECONDS in which the library used at least this share of it
    # while the bucket was nearly empty (it was the rate, not the bucket, that held it back).
    GROW_WHEN_USED = 0.8
    GROW_BELOW_LEVEL = 0.25
    # Busy seconds without a 429 after which a grown rate is the account's own (the service's
    # bucket is spent long before: 1.6 million tokens at 350 thousand a minute over its rate).
    PROVEN_SECONDS = 600.0
    # A rate kept from an earlier run is used for this long.
    REMEMBER_DAYS = 30

    def __init__(
        self,
        per_minute: float = 450_000,
        ceiling: float = 4_000_000,
        floor: float = 50_000,
        burst: float = 1_500_000,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.floor = float(floor)
        self.ceiling = max(float(ceiling), self.floor)
        self.rate = min(max(float(per_minute), self.floor), self.ceiling)
        self.clock = clock
        self.sleep = sleep
        self._lock = threading.Lock()
        now = clock()
        # Like the service: a bucket of ``burst`` tokens (about 1.6 million there), full when
        # rested and refilled at the rate. A rested account's first import starts with it.
        self.burst = max(0.0, float(burst))
        self.level = self.capacity
        self._at = now
        self._hold_until = 0.0
        self._last_cut = float("-inf")
        # Tokens sent since ``_window`` (for growing the rate).
        self._window = now
        self._used = 0.0
        self._low = False
        self.stats = {"requests": 0, "tokens": 0.0, "waited": 0.0, "limited": 0, "cuts": 0}
        # The account's sustained rate as far as it is known (see the module docstring), and
        # the busy seconds without a 429 since the last one.
        self.known = self.rate
        self._proven = 0.0
        # Where ``known`` is kept for the next start (``remember``).
        self.memory: Path | None = None

    @property
    def capacity(self) -> float:
        """Tokens that may go out at once: the burst, and never less than 15 seconds' worth."""
        return max(self.burst, self.rate / 4)

    def configure(self, per_minute: float | None = None, ceiling: float | None = None,
                  burst: float | None = None) -> None:
        with self._lock:
            if ceiling is not None:
                self.ceiling = max(float(ceiling), self.floor)
            if per_minute is not None:
                self.rate = self.known = min(max(float(per_minute), self.floor), self.ceiling)
            self.rate = min(self.rate, self.ceiling)
            if burst is not None:
                rested = self.level >= self.capacity
                self.burst = max(0.0, float(burst))
                if rested:
                    self.level = self.capacity
            self.level = min(self.level, self.capacity)

    def _update(self, now: float) -> None:
        if now - self._window >= self.GROW_SECONDS:
            if self._low and self._used >= self.GROW_WHEN_USED * self.rate * (now - self._window) / 60.0:
                self._proven += now - self._window
                if self._proven >= self.PROVEN_SECONDS and self.rate > self.known:
                    self.known = self.rate
                    self._save()
                self.rate = min(self.ceiling, self.rate * self.GROW)
            self._window, self._used, self._low = now, 0.0, False
        self.level = min(self.capacity, self.level + (now - self._at) * self.rate / 60.0)
        self._at = now

    def acquire(self, tokens: float) -> float:
        """Wait until ``tokens`` may be sent; returns the seconds waited."""
        waited = 0.0
        while True:
            with self._lock:
                now = self.clock()
                self._update(now)
                need = min(float(tokens), self.capacity)
                if now >= self._hold_until and self.level >= need:
                    self.level -= need
                    self._used += float(tokens)
                    self._low = self._low or self.level < self.GROW_BELOW_LEVEL * self.capacity
                    self.stats["requests"] += 1
                    self.stats["tokens"] += float(tokens)
                    self.stats["waited"] += waited
                    return waited
                wait = max(self._hold_until - now, (need - self.level) * 60.0 / self.rate, 0.01)
            wait = min(wait, 1.0)
            self.sleep(wait)
            waited += wait

    def limited(self) -> None:
        """The service said 429: slow down, and let the window it counts over pass."""
        with self._lock:
            now = self.clock()
            self.stats["limited"] += 1
            self._window, self._used, self._proven = now, 0.0, 0.0
            if now - self._last_cut > self.CUT_WINDOW:
                self.rate = max(self.floor, self.rate * self.CUT)
                self._last_cut = now
                self.stats["cuts"] += 1
                logger.info("Embedding rate limited; indexing now paced at %.0f tokens/min", self.rate)
                self.known = self.rate
                self._save()
            # The service's bucket is empty: so is ours.
            self.level = min(self.level, 0.0)
            self._hold_until = max(self._hold_until, now + self.HOLD_SECONDS)

    def remember(self, path: Path) -> None:
        """Keep ``known`` in ``path`` from now on, and start from the one kept there (if recent)."""
        with self._lock:
            self.memory = Path(path)
            try:
                kept = json.loads(self.memory.read_text(encoding="utf-8"))
                age_days = (time.time() - float(kept["saved"])) / 86400
                if 0 <= age_days <= self.REMEMBER_DAYS and self.stats["requests"] == 0:
                    rate = float(kept["tokens_per_minute"])
                    self.rate = self.known = min(max(rate, self.floor), self.ceiling)
                    self.level = min(self.level, self.capacity)
            except (OSError, ValueError, KeyError, TypeError):
                pass

    def _save(self) -> None:
        """Write ``known`` (called with the lock held, when it changed)."""
        if self.memory is None:
            return
        try:
            self.memory.parent.mkdir(parents=True, exist_ok=True)
            self.memory.write_text(json.dumps({"tokens_per_minute": round(self.known), "saved": time.time()}),
                                   encoding="utf-8")
        except OSError as exc:
            logger.debug("Embedding rate not kept: %s", exc)

    def available_burst(self) -> float:
        """Tokens the next requests may send at once."""
        with self._lock:
            self._update(self.clock())
            return self.level

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"tokens_per_minute": round(self.rate), "known": round(self.known),
                    **{k: round(v, 2) for k, v in self.stats.items()}}


PACER = EmbedPacer()
_installed: dict[int, Any] = {}
_install_lock = threading.Lock()


def _texts(request: Any) -> list[str]:
    try:
        body = json.loads(request.content or b"{}")
    except Exception:
        return []
    texts = body.get("input") if isinstance(body, dict) else None
    if isinstance(texts, str):
        return [texts]
    return [str(text) for text in texts] if isinstance(texts, list) else []


def _is_embeddings(request: Any) -> bool:
    try:
        return str(request.url.path).rstrip("/").endswith("/embeddings")
    except Exception:
        return False


def install(client: Any, pacer: EmbedPacer = PACER) -> bool:
    """Put ``pacer`` in front of ``client``'s embedding requests (once per client)."""
    with _install_lock:
        if _installed.get(id(client)) is client:
            return False
        hooks = client.event_hooks

        def on_request(request: Any) -> None:
            if is_paced() and _is_embeddings(request):
                pacer.acquire(estimate_tokens(_texts(request)))

        def on_response(response: Any) -> None:
            if response.status_code == 429 and _is_embeddings(response.request):
                pacer.limited()

        client.event_hooks = {
            "request": [*hooks.get("request", []), on_request],
            "response": [*hooks.get("response", []), on_response],
        }
        _installed[id(client)] = client
        return True


def install_for(settings: Any, pacer: EmbedPacer = PACER) -> bool:
    """Install on the shared model-service client of ``settings``; configure from settings
    (``index_embed_tpm``: where the rate starts, ``index_embed_tpm_max``: how high it may go)."""
    try:
        from services.model_clients import shared_http_client
    except Exception:  # an older model_clients without the shared pool: nothing to pace
        return False
    start = getattr(settings, "index_embed_tpm", None)
    ceiling = getattr(settings, "index_embed_tpm_max", None)
    if pacer.stats["requests"] == 0:
        pacer.configure(per_minute=start, ceiling=ceiling, burst=getattr(settings, "index_embed_burst", None))
    else:
        pacer.configure(ceiling=ceiling)
    data_dir = getattr(settings, "data_dir", None)
    if data_dir and pacer.memory is None:
        pacer.remember(Path(data_dir) / "cache" / "embed_rate.json")
    try:
        return install(shared_http_client(settings), pacer)
    except Exception as exc:
        logger.warning("Embedding pacing not installed: %s", exc)
        return False
