"""A local stand-in for SiliconFlow, so the app's main flows run with no key and no network.

It proves the plumbing only, nothing about answer quality, cost or speed. Vectors come from
hashed character pairs and scores from shared pairs, so a question still finds the passage
that shares its words. The key decides how the account behaves:

* ``sk-bad...``   rejected (401)
* ``sk-empty...`` valid, but out of balance (402 on everything except listing models)
"""

from __future__ import annotations

import json
import math
import re
import zlib

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

DIM = 1024
CHAT_MODELS = ["Qwen/Qwen3-8B", "Qwen/Qwen3-30B-A3B-Instruct-2507", "deepseek-ai/DeepSeek-V4-Flash"]


def _pairs(text: str) -> list[str]:
    text = re.sub(r"\s+", "", text)
    return [text[i:i + 2] for i in range(max(0, len(text) - 1))] or [text or " "]


def embed(text: str) -> list[float]:
    vector = [0.0] * DIM
    for pair in _pairs(text):
        vector[zlib.crc32(pair.encode("utf-8")) % DIM] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def score(query: str, document: str) -> float:
    wanted = set(_pairs(query))
    return len(wanted & set(_pairs(document))) / (len(wanted) or 1)


def reply(messages: list[dict]) -> str:
    system = str(messages[0].get("content", ""))
    body = str(messages[-1].get("content", ""))
    if "查询规划器" in system:
        question = re.sub(r"\s+", " ", body)[-100:] or "问题"
        return json.dumps({"intent": "knowledge", "rewritten_query": question}, ensure_ascii=False)
    if "事实一致性检查器" in system:
        return '{"grounded":true,"unsupported_claims":[]}'
    if "为后面的全文总结做笔记" in system:
        return "\n".join(f"P{number}: 第{number}段的要点" for number in re.findall(r"\[P(\d+)｜", body))
    if "翻译成一个查询" in system:
        return '{"op":"count"}'
    if "基于用户知识库文档回答问题" in system:
        evidence = re.findall(r"\[证据 \d+｜来源：([^｜]+)｜位置：([^\]]+)\]\n([^\n]*)", body)
        if not evidence:
            return "当前知识库没有足够依据"
        source, where, text = evidence[0]
        return f"{text[:60]} [来源：{source}｜位置：{where}]"
    return "好"


def create_app() -> FastAPI:
    app = FastAPI()

    @app.middleware("http")
    async def account(request: Request, call_next):
        key = request.headers.get("authorization", "").removeprefix("Bearer ")
        if key.startswith("sk-bad"):
            return JSONResponse({"code": 20007, "message": "Invalid token"}, status_code=401)
        if key.startswith("sk-empty") and not request.url.path.endswith("/models"):
            return JSONResponse({"code": 30001, "message": "Sorry, your account balance is insufficient"}, status_code=402)
        return await call_next(request)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model, "object": "model"} for model in CHAT_MODELS]}

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        data = await request.json()
        inputs = data["input"] if isinstance(data["input"], list) else [data["input"]]
        rows = [{"object": "embedding", "index": i, "embedding": embed(str(text))} for i, text in enumerate(inputs)]
        return {"object": "list", "data": rows, "model": data["model"], "usage": {"prompt_tokens": 1, "total_tokens": 1}}

    @app.post("/v1/rerank")
    async def rerank(request: Request):
        data = await request.json()
        scored = sorted(((score(data["query"], doc), i) for i, doc in enumerate(data["documents"])), reverse=True)
        return {"results": [{"index": i, "relevance_score": s} for s, i in scored[: data.get("top_n") or None]]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        data = await request.json()
        text = reply(data["messages"])
        usage = {"prompt_tokens": 1, "completion_tokens": len(text), "total_tokens": 1 + len(text)}
        if not data.get("stream"):
            return {
                "id": "fake", "object": "chat.completion", "created": 0, "model": data["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": usage,
            }

        def events():
            base = {"id": "fake", "object": "chat.completion.chunk", "created": 0, "model": data["model"]}
            for start in range(0, len(text), 8):
                chunk = {**base, "choices": [{"index": 0, "delta": {"content": text[start:start + 8]}, "finish_reason": None}]}
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], 'usage': usage})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app
