"use strict";

/* What a failed answer offers (kept free of the page so tests can run it under Node).
   The backend says what failed (services/model_clients.py ERROR_KINDS) and whether asking
   again may help; while the model service is overloaded, rate limited or unreachable, the
   page also asks again once by itself after a short countdown, which the user can cancel. */
(function (root) {
  // Kinds the page asks again by itself, once (the backend already waited within its budget).
  const AUTO = new Set(["busy", "rate_limit", "network", "server"]);
  // Kinds where asking again may help; anything unknown is worth a manual try.
  const MANUAL = new Set([...AUTO, "unknown"]);
  // A bad key or no balance: the settings, not another try.
  const SETTINGS = new Set(["auth", "balance"]);
  const AUTO_SECONDS = 10;

  /** {action: "retry" | "settings" | null, auto: seconds before asking again by itself (0: no)}.
      ``retryable`` is what the backend said (null: decide from the kind); ``autoUsed``: the
      page already asked again once for this question. */
  function plan(kind, { retryable = null, autoUsed = false } = {}) {
    const k = kind || "unknown";
    if (SETTINGS.has(k)) return { action: "settings", auto: 0 };
    const canRetry = retryable === null || retryable === undefined ? MANUAL.has(k) : !!retryable;
    if (!canRetry) return { action: null, auto: 0 };
    return { action: "retry", auto: AUTO.has(k) && !autoUsed ? AUTO_SECONDS : 0 };
  }

  /** A bot message that failed (an error, or a refusal because the model or the search failed). */
  function failed(m) {
    if (!m || m.role !== "bot") return false;
    if (m.error) return m.error !== "已停止生成。";
    return !!(m.done && m.refused && m.errorKind);
  }

  /** The waiting line under a failed answer: "10 秒后自动再问一次". */
  function countdown(seconds) {
    return `${Math.max(0, Math.ceil(seconds))} 秒后自动再问一次`;
  }

  root.Retry = { plan, failed, countdown, AUTO_SECONDS };
})(typeof window !== "undefined" ? window : globalThis);
