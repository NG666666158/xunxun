"""SiliconFlow adapters with timeouts, retries and typed return values."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import logging
import queue
import threading
import time
from typing import Iterable, Iterator, Sequence
import urllib.request

import httpx
import openai
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI
from openai import OpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from config import MissingApiKeyError, Settings, get_settings
from core import timings

logger = logging.getLogger(__name__)


# Failures of the model service itself, as opposed to a bad answer or bad JSON.
MODEL_SERVICE_ERRORS = (openai.APIError, httpx.TransportError)


def transient_error(exc: BaseException) -> bool:
    """Worth trying again: the connection dropped or timed out (a pooled connection the
    server had closed shows up as "Connection error."), or the service was busy or failed
    (429, 5xx). A bad key, a bad request or a refused input is not."""
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, (httpx.TransportError, TimeoutError, ConnectionError))


# ---------------- what went wrong, for the user and for retrying ----------------

# busy: the model's servers are overloaded (429, code 50609); rate_limit: the account's per-minute
# budget is used up (429); network: dropped or timed out; server: 5xx; auth: bad key or no access;
# balance: 402; bad_request: the request itself (other 4xx); unknown: anything else.
ERROR_KINDS = ("busy", "rate_limit", "network", "server", "auth", "balance", "bad_request", "unknown")
# The app waits and tries again by itself (within the turn's budget).
AUTO_RETRY_KINDS = frozenset({"busy", "rate_limit", "network", "server"})
# Asking again may help, so the answer offers "重新提问"; auth and balance need the settings.
MANUAL_RETRY_KINDS = AUTO_RETRY_KINDS | {"unknown"}


def _status_kind(status: int, busy: bool) -> str:
    if status == 429:
        return "busy" if busy else "rate_limit"
    if status in (401, 403):
        return "auth"
    if status == 402:
        return "balance"
    if status >= 500:
        return "server"
    return "bad_request"


def error_kind(exc: BaseException) -> str:
    """One of ERROR_KINDS for an exception raised while answering."""
    if isinstance(exc, MissingApiKeyError):
        return "auth"
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return "network"
    if isinstance(exc, openai.APIStatusError):
        return _status_kind(exc.status_code, _server_busy(exc))
    if isinstance(exc, httpx.HTTPStatusError):
        return _status_kind(exc.response.status_code, "too busy" in exc.response.text.lower())
    if isinstance(exc, (httpx.TransportError, TimeoutError, ConnectionError)):
        return "network"
    return "unknown"


def retry_after(exc: BaseException, cap: float = 60.0) -> float | None:
    """The wait a 429 / 503 asked for (``Retry-After`` in seconds), at most ``cap``."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        value = float(headers.get("retry-after") or headers.get("Retry-After") or "")
    except (TypeError, ValueError):
        return None
    return min(max(value, 0.0), cap)


class ServiceBusy:
    """The chat model said it is overloaded or rate limited: for a while, background model
    work (document cards, home-page suggestions, follow-ups written ahead) leaves the model
    to the question being answered instead of competing with its retries."""

    def __init__(self, clock=time.monotonic, sleep=time.sleep):
        self._until: dict[str, float] = {}
        self._lock = threading.Lock()
        self.clock, self.sleep = clock, sleep

    def note(self, model: str, seconds: float = 60.0) -> None:
        with self._lock:
            self._until[model] = max(self._until.get(model, 0.0), self.clock() + seconds)

    def left(self, model: str) -> float:
        with self._lock:
            return max(0.0, self._until.get(model, 0.0) - self.clock())

    def wait(self, model: str, limit: float = 120.0, step: float = 1.0) -> float:
        """Sleep while ``model`` is noted busy, at most ``limit`` seconds; returns the wait."""
        waited = 0.0
        while waited < limit and (left := self.left(model)) > 0:
            pause = min(step, left, limit - waited)
            self.sleep(pause)
            waited += pause
        return waited

    def clear(self) -> None:
        with self._lock:
            self._until.clear()


chat_busy = ServiceBusy()


# A question's model calls share one budget for waiting (turn_retry_budget_seconds): past it the
# turn ends with a retryable error and the user can ask again, instead of waiting for minutes.
_turn_deadline: ContextVar[float | None] = ContextVar("turn_deadline", default=None)


@contextmanager
def turn_budget(seconds: float | None) -> Iterator[None]:
    token = _turn_deadline.set(time.monotonic() + seconds if seconds and seconds > 0 else None)
    try:
        yield
    finally:
        _turn_deadline.reset(token)


def budget_left() -> float | None:
    """Seconds of waiting this turn may still spend; None outside a budgeted turn."""
    deadline = _turn_deadline.get()
    return None if deadline is None else deadline - time.monotonic()


def within_budget(iterator: Iterable, seconds: float | None) -> Iterator:
    """Yield from ``iterator`` with one turn budget current while each item is produced (a
    streamed answer may be stepped from a different thread each time, as in timings.iterate)."""
    deadline = time.monotonic() + seconds if seconds and seconds > 0 else None
    iterator = iter(iterator)
    while True:
        token = _turn_deadline.set(deadline)
        try:
            item = next(iterator)
        except StopIteration:
            return
        finally:
            _turn_deadline.reset(token)
        yield item


# Background work (indexing) waits longer between tries than a question being answered.
RETRY_BACKGROUND = dict(
    retry=retry_if_exception(transient_error),
    wait=wait_exponential_jitter(multiplier=1, max=20, jitter=1),
    stop=stop_after_attempt(6),
    reraise=True,
)
RETRY_INTERACTIVE = dict(
    retry=retry_if_exception(transient_error),
    wait=wait_exponential_jitter(multiplier=0.5, max=4, jitter=0.5),
    stop=stop_after_attempt(3),
    reraise=True,
)


class CallLimiter:
    """At most ``limit`` model calls from background indexing at once (embeddings, vision),
    so files indexed side by side do not trip SiliconFlow's rate limits. Answering a
    question never waits for a slot."""

    def __init__(self, limit: int = 4):
        self.limit = max(1, int(limit))
        self._slots = threading.Semaphore(self.limit)
        self._lock = threading.Lock()

    def resize(self, limit: int) -> None:
        with self._lock:
            limit = max(1, int(limit))
            for _ in range(limit - self.limit):
                self._slots.release()
            for _ in range(self.limit - limit):
                self._slots.acquire()
            self.limit = limit

    @contextmanager
    def slot(self) -> Iterator[None]:
        with self._slots:
            yield


background_calls = CallLimiter()


# Every call to the model service goes through one connection pool. A question makes four or
# five calls in a row (query embedding, rerank, the answer, the check, follow-ups), and each new
# HTTPS connection costs a handshake first, more so through a proxy. httpx drops idle
# connections after 5 s by default, less than the time it takes to read an answer and type the
# next question; KEEPALIVE_SECONDS keeps them a while, under the ~60 s after which servers
# commonly close idle connections themselves.
KEEPALIVE_SECONDS = 50.0
# How the model service is reached when a system or environment proxy is set: "auto" goes
# direct when a direct connection opens and through the proxy otherwise; "proxy" / "direct"
# fix the route. Measured on a machine whose system proxy is Clash (evaluation/answer_speed_v2
# conn_bench.py): a query embedding took 4.1 s through the proxy on a new connection, 0.23 s
# direct; a chat call's first token 2.4 s vs 0.40 s.
MODEL_CONNECTIONS = ("auto", "direct", "proxy")
# A direct connection that does not open within this time falls back to the proxy ...
DIRECT_CONNECT_SECONDS = 3.0
# ... which is then used for this long before direct is tried again.
ROUTE_RECHECK_SECONDS = 600.0
_pools: dict[tuple[str, str], httpx.Client] = {}
_pools_lock = threading.Lock()


def system_proxy(url: str) -> str | None:
    """The proxy the environment (HTTPS_PROXY ...) or the Windows system settings name for ``url``."""
    parsed = httpx.URL(url)
    try:
        if urllib.request.proxy_bypass(parsed.host):
            return None
    except Exception:
        pass
    proxies = urllib.request.getproxies()
    proxy = proxies.get(parsed.scheme) or proxies.get("all") or ""
    if proxy and "://" not in proxy:
        proxy = "http://" + proxy
    return proxy or None


class RoutedTransport(httpx.BaseTransport):
    """Sends requests directly when that works, through the proxy otherwise (see MODEL_CONNECTIONS)."""

    def __init__(self, proxy: str | None, mode: str, limits: httpx.Limits):
        self.mode = mode if mode in MODEL_CONNECTIONS else "auto"
        self.proxy = proxy
        self.direct = httpx.HTTPTransport(limits=limits)
        self.proxied = httpx.HTTPTransport(proxy=proxy, limits=limits) if proxy else None
        self._lock = threading.Lock()
        self._route: str | None = None
        self._decided_at = 0.0

    @property
    def route(self) -> str:
        """"direct", "proxy", or "" while auto has not decided yet."""
        if self.proxied is None or self.mode == "direct":
            return "direct"
        if self.mode == "proxy":
            return "proxy"
        with self._lock:
            if self._route == "proxy" and time.monotonic() - self._decided_at > ROUTE_RECHECK_SECONDS:
                self._route = None
            return self._route or ""

    def _decide(self, route: str, why: str) -> None:
        with self._lock:
            changed = route != self._route
            self._route, self._decided_at = route, time.monotonic()
        if changed:
            how = "directly" if route == "direct" else f"through the proxy {self.proxy}"
            logger.info("Model service reached %s (%s)", how, why or f"system proxy {self.proxy} bypassed")

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        route = self.route
        if route == "proxy":
            return self.proxied.handle_request(request)
        if self.proxied is None or self.mode == "direct":
            return self.direct.handle_request(request)
        original = dict(request.extensions.get("timeout") or {})
        if route == "":
            # Not decided yet: a direct connection must open quickly to be chosen.
            limit = min(original.get("connect") or DIRECT_CONNECT_SECONDS, DIRECT_CONNECT_SECONDS)
            request.extensions["timeout"] = {**original, "connect": limit}
        try:
            response = self.direct.handle_request(request)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            # Nothing was sent: the same request goes through the proxy.
            self._decide("proxy", f"no direct connection: {type(exc).__name__}")
            request.extensions["timeout"] = original
            return self.proxied.handle_request(request)
        request.extensions["timeout"] = original
        if route == "":
            self._decide("direct", "")
        return response

    def close(self) -> None:
        self.direct.close()
        if self.proxied is not None:
            self.proxied.close()


def shared_http_client(settings: Settings) -> httpx.Client:
    base = str(settings.siliconflow_base_url).rstrip("/")
    mode = str(getattr(settings, "model_connection", "auto") or "auto")
    with _pools_lock:
        client = _pools.get((base, mode))
        if client is None or client.is_closed:
            limits = httpx.Limits(max_connections=64, max_keepalive_connections=16, keepalive_expiry=KEEPALIVE_SECONDS)
            client = httpx.Client(
                transport=RoutedTransport(system_proxy(base), mode, limits),
                timeout=httpx.Timeout(float(settings.request_timeout_seconds), connect=10.0),
                follow_redirects=True,
                # The transport applies the proxy settings itself.
                trust_env=False,
            )
            _pools[(base, mode)] = client
    return client


def connection_route(settings: Settings) -> str:
    """How the model service is reached now: "direct", "proxy" or "" (not decided yet)."""
    transport = getattr(shared_http_client(settings), "_transport", None)
    return getattr(transport, "route", "")


def open_connections(settings: Settings, count: int = 2) -> int:
    """Open ``count`` connections to the model service ahead of a question (a free model
    listing each); returns how many answered. Never raises."""
    client = shared_http_client(settings)
    url = str(settings.siliconflow_base_url).rstrip("/") + "/models"

    def one(_: int) -> bool:
        try:
            response = client.get(url, params={"type": "text", "sub_type": "reranker"},
                                  auth=CurrentKeyAuth(settings), timeout=10.0)
            return response.status_code < 500
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=max(1, count)) as pool:
        return sum(pool.map(one, range(max(1, count))))


def _server_busy(exc: BaseException) -> bool:
    """SiliconFlow answers 429 with code 50609 "System is too busy" when the model is overloaded."""
    body = getattr(exc, "body", None)
    code = body.get("code") if isinstance(body, dict) else None
    return code == 50609 or "too busy" in str(exc).lower()


def describe_model_error(exc: BaseException) -> str:
    """Turn a model-service failure into a message the user can act on."""
    if isinstance(exc, MissingApiKeyError):
        return str(exc)
    if isinstance(exc, openai.RateLimitError) and _server_busy(exc):
        return "模型服务繁忙：硅基流动这个模型的服务器暂时过载（不是你的账号额度用完）。请稍后重试，或在设置中换一个模型。"
    if isinstance(exc, openai.RateLimitError):
        return "模型服务限流：每分钟的 token 用量已达账号上限。请等待约一分钟后重试，或在设置中换用限额更高的模型。"
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return "模型服务拒绝了请求：API Key 无效或没有该模型的权限，请在设置页检查 Key 和模型。"
    if isinstance(exc, openai.APIStatusError) and exc.status_code == 402:
        return "硅基流动账户余额不足，请充值后重试。"
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError, httpx.TransportError)):
        return "与模型服务的连接中断了，请检查网络后重试。"
    if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
        return "模型服务暂时故障，请稍后重试。"
    return "问答服务暂时不可用，请稍后重试。"


class CurrentKeyAuth(httpx.Auth):
    """Bearer auth with whatever key the settings hold at request time."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def auth_flow(self, request):
        request.headers["Authorization"] = f"Bearer {self.settings.require_api_key()}"
        yield request


@dataclass(frozen=True)
class RerankItem:
    index: int
    score: float


class SiliconFlowChat(ChatOpenAI):
    """ChatOpenAI whose streamed token usage is counted once, and which does not wait out a
    stuck request.

    SiliconFlow repeats the running usage totals on every streamed chunk, and LangChain
    adds the chunks' usage together, which inflates the counts by the number of chunks.
    Only the last reported totals are kept.

    Now and then a request gets no first token at all and only fails at the read timeout
    (30 s) before the client retries (seen in evaluation/answer_speed_v2). With
    ``hedge_after`` set, an identical request is sent alongside once that many seconds pass
    without a first token; whichever answers first is used and the other is dropped.
    """

    hedge_after: float = 0.0

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        usage = None
        timer = timings.ModelCallTimer()
        if self.hedge_after and self.hedge_after > 0:
            source = self._first_to_answer(messages, stop, kwargs)
        else:
            source = super()._stream(messages, stop=stop, run_manager=None, **kwargs)
        try:
            for chunk in source:
                message = chunk.message
                if getattr(message, "usage_metadata", None):
                    usage = message.usage_metadata
                    message.usage_metadata = None
                if message.content:
                    timer.token()
                # Only the chunks that are used reach the callbacks (the streamed draft).
                if run_manager is not None:
                    run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                yield chunk
        finally:
            source.close()
            timer.usage = dict(usage) if usage else None
            timer.done()
        if usage:
            yield ChatGenerationChunk(message=AIMessageChunk(content="", usage_metadata=usage))

    def _first_to_answer(self, messages, stop, kwargs):
        """The chunks of the request that answers first (see the class docstring)."""
        parent_stream = super()._stream
        events: queue.Queue = queue.Queue()
        cancels: list[threading.Event] = []

        def attempt() -> None:
            number, cancel = len(cancels), threading.Event()
            cancels.append(cancel)

            def run() -> None:
                stream = None
                try:
                    stream = parent_stream(messages, stop=stop, run_manager=None, **kwargs)
                    for chunk in stream:
                        if cancel.is_set():
                            break
                        events.put((number, "chunk", chunk))
                    events.put((number, "end", None))
                except BaseException as exc:  # handed to the consumer below
                    events.put((number, "error", exc))
                finally:
                    if stream is not None:
                        try:
                            stream.close()
                        except Exception:
                            pass

            threading.Thread(target=run, name=f"model-call-{number}", daemon=True).start()

        attempt()
        started = time.monotonic()
        winner: int | None = None
        held: dict[int, list] = {0: [], 1: []}
        failures: dict[int, BaseException] = {}
        try:
            while True:
                wait = None
                if winner is None and len(cancels) == 1:
                    wait = max(0.0, self.hedge_after - (time.monotonic() - started))
                try:
                    number, kind, payload = events.get(timeout=wait)
                except queue.Empty:
                    logger.info("No first token after %.0f s: sending the same request alongside", self.hedge_after)
                    timings.note(hedged=True)
                    attempt()
                    continue
                if winner is not None and number != winner:
                    continue
                if kind == "chunk":
                    if winner is None:
                        if not payload.message.content:
                            held[number].append(payload)
                            continue
                        winner = number
                        for other, cancel in enumerate(cancels):
                            if other != number:
                                cancel.set()
                        yield from held[number]
                    yield payload
                elif kind == "end":
                    if winner is None:
                        yield from held[number]
                    return
                else:
                    failures[number] = payload
                    if winner == number or len(failures) == len(cancels):
                        raise failures.get(0, payload)
        finally:
            for cancel in cancels:
                cancel.set()


def build_chat_model(
    settings: Settings | None = None,
    max_tokens: int | None = None,
    *,
    model: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> ChatOpenAI:
    """Create the OpenAI-compatible chat client used by LangGraph."""
    settings = settings or get_settings()
    return SiliconFlowChat(
        model=model or settings.llm_model,
        # A callable: the key is read per request, so Settings changes apply at once.
        api_key=settings.require_api_key,
        base_url=settings.siliconflow_base_url,
        temperature=settings.llm_temperature,
        timeout=timeout or settings.request_timeout_seconds,
        max_retries=settings.request_max_retries if max_retries is None else max_retries,
        seed=42,
        max_tokens=max_tokens or settings.llm_max_tokens,
        extra_body={"enable_thinking": False},
        http_client=shared_http_client(settings),
        hedge_after=float(getattr(settings, "first_token_hedge_seconds", 0.0) or 0.0),
        streaming=True,
        # Token counts arrive in the last streamed chunk (response.usage_metadata).
        stream_usage=True,
    )


class SiliconFlowEmbeddings:
    """Small LangChain-compatible embedding adapter for BAAI/bge-m3."""

    def __init__(self, settings: Settings | None = None, batch_size: int = 32):
        self.settings = settings or get_settings()
        self.batch_size = batch_size
        self.client = OpenAI(
            api_key=self.settings.require_api_key,
            base_url=self.settings.siliconflow_base_url,
            timeout=self.settings.request_timeout_seconds,
            max_retries=0,
            http_client=shared_http_client(self.settings),
        )

    def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.settings.embedding_model,
            input=list(texts),
            encoding_format="float",
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors = [item.embedding for item in ordered]
        for vector in vectors:
            if len(vector) != self.settings.embedding_dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: configured "
                    f"{self.settings.embedding_dimension}, received {len(vector)}"
                )
        return vectors

    @retry(**RETRY_BACKGROUND)
    def _embed_for_index(self, texts: Sequence[str]) -> list[list[float]]:
        with background_calls.slot():
            return self._embed_batch(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Indexing: one request per batch, each retried on dropped connections and 429/5xx,
        and only a few at once across all files being indexed."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_for_index(texts[start : start + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        with timings.stage("embed"):
            return self._embed_query(text)

    @retry(**RETRY_INTERACTIVE)
    def _embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """A question and its parts in one request (retrieval/multipart.py)."""
        with timings.stage("embed"):
            return self._embed_queries(list(texts))

    @retry(**RETRY_INTERACTIVE)
    def _embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed_batch(texts)


class SiliconFlowReranker:
    """Cross-encoder reranking through SiliconFlow's official rerank endpoint."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.client = shared_http_client(self.settings)
        self.auth = CurrentKeyAuth(self.settings)
        self.url = str(self.settings.siliconflow_base_url).rstrip("/") + "/rerank"

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[RerankItem]:
        if not documents:
            return []
        with timings.stage("rerank"):
            return self._rerank(query, documents, top_n)

    @retry(**RETRY_INTERACTIVE)
    def _rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[RerankItem]:
        response = self.client.post(
            self.url,
            auth=self.auth,
            json={
                "model": self.settings.rerank_model,
                "query": query,
                "documents": list(documents),
                "top_n": min(top_n, len(documents)),
                "return_documents": False,
            },
        )
        response.raise_for_status()
        payload = response.json()
        return [
            RerankItem(index=int(item["index"]), score=float(item["relevance_score"]))
            for item in payload.get("results", [])
        ]

    def close(self) -> None:
        """The connection pool is shared with the other model clients and stays open."""


def batched(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    """Public utility used by ingestion tests and scripts."""
    for start in range(0, len(values), size):
        yield values[start : start + size]
