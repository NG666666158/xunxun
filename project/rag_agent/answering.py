"""Answer profiles by question type, and the citation bookkeeping of the fact check."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from retrieval.hybrid_retriever import comparison_topic


@dataclass(frozen=True)
class AnswerProfile:
    """How much evidence a question gets and how long its answer may be."""

    name: str
    # Estimated tokens of parent passages given to the model (see AgentNodes._estimate_tokens).
    context_budget: int
    # Hard cap on generated tokens; the length guidance below keeps answers well under it.
    max_tokens: int
    guidance: str
    # Most reranked passages the retriever keeps for this type (its score cutoffs may keep
    # fewer); build_context then fits their parents into context_budget.
    max_passages: int = 6
    # Most passages from one document while other documents qualify (None: the setting
    # rerank_per_document).
    per_document: int | None = None


PROFILES: dict[str, AnswerProfile] = {
    "fact": AnswerProfile(
        "fact", 3500, 900,
        "本题是事实定位类问题：第一句直接给出答案，必要时再用不超过 3 条要点补充条件、数字或出处细节；全文控制在 250 字以内，不要展开背景。",
        max_passages=6,
    ),
    # Several asks in one question ("…对哪些省份启动了何种响应，并且安徽省预计会…？"): each part
    # may come from another document, so more passages and fewer from any one document
    # (evaluation/suite_v1/REPORT.md, fix 2).
    "multi": AnswerProfile(
        "multi", 7000, 1800,
        "本题包含几个小问：按提问的顺序逐一回答，每一问单独一段，先给出直接答案，再补充必要的条件、数字或出处；"
        "不要补充问题没有问到的内容；全文控制在 500 字左右。",
        max_passages=10,
        per_document=2,
    ),
    "explain": AnswerProfile(
        "explain", 7000, 1800,
        "本题是解释类问题：先用一两句话直接回答，再分 3～5 点说明原理、作用、步骤或示例；全文控制在 600 字左右，不要大段复述证据原文。",
        max_passages=10,
    ),
    "compare": AnswerProfile(
        "compare", 9000, 2400,
        "本题是对比类问题：先用一两句话概括核心差异，再按维度逐项对比（可用列表或表格），最后可补一句适用场景；全文控制在 900 字左右。",
        max_passages=12,
    ),
    "summary": AnswerProfile(
        "summary", 0, 2400,
        "本题要总结整份文件，证据是按文件顺序排列的全文笔记：先用两三句话说明这份文件是什么、核心内容是什么；"
        "再按文件的结构分点写主要内容，从头到尾都要覆盖，不要只写开头；数字、金额、人名保留原样；全文控制在 800 字左右。",
    ),
}

_COMPARE = re.compile(
    r"区别|对比|比较|异同|相同点|不同点|有何不同|有什么不同|不同之处|不同侧重|侧重点|差异|优缺点|优劣|利弊|相比|"
    r"\bvs\.?\b|\bversus\b|各有什么|哪个更|还是.{0,12}更"
)
_EXPLAIN = re.compile(
    r"^\s*(?:什么是|何为|何谓|请?介绍|请?解释|讲讲|说说|谈谈|概述|总结|展开)|为什么|为何|原理|机制|怎么理解|如何理解|"
    r"是如何|如何工作|如何实现|如何影响|如何结合|作用|意义|有什么用|好处|优点|缺点|局限|风险|危险|思路|本质|"
    r"主要解决|解决什么|哪些方面|考察|知识点|有哪些.{0,8}(?:方法|方式|技术|类型|策略|步骤|阶段|原则)|流程|步骤|过程|"
    r"举例|例子|展开讲|详细"
)


# ---------------- several asks in one question ----------------

# Interrogatives. "几" and "哪" only with what follows them in a question ("几天", "哪一年"),
# so "几乎" / "哪怕" do not count.
_ASK_ZH = re.compile(
    r"哪(?:些|个|里|儿|一|两|三|几|位|家|种|所|条|项|年|月|天|次|国|部|款|类|方面)|什么|啥|如何|怎样|怎么|多少|"
    r"多(?:久|长|大|高|远)|"
    r"几(?:个|项|条|种|类|次|天|年|月|号|岁|名|家|人|点|成|倍|位|所|部|款|期|批|十|百|千|万)|"
    r"何(?:种|时|地|处|人|在|方)|为何|为什么|是否|谁"
)
# Where one ask ends and the next may begin: punctuation and joining words.
_CLAUSE_ZH = re.compile(r"[，,；;？?。！!]|并且|而且|同时|此外|另外|以及|还有")
# "发射时间、着陆地点以及发射年份是什么": one ask over a list of several things.
_LISTED_ZH = re.compile(r"以及|、[^，,；;？?。]*(?:和|及|与)")
_CJK = re.compile(r"[㐀-鿿]")
_WH_EN = r"(?:who|whom|whose|what|which|when|where|why|how)"
_YES_NO_EN = r"(?:does|did|do|is|was|were|are|has|have|had|can|could|will|would|should)"
# An ask opens a clause: the question's start, or after ", ; and or" ("…, and when did it …").
_ASK_EN = re.compile(rf"(?:^|[,;:]|\band\b|\bor\b)\s*(?:{_WH_EN}|{_YES_NO_EN})\b", re.I)
# A second part after the first: "…, and is also …", "… while the other …", "both … and …".
_JOIN_EN = re.compile(
    r",\s*and\b|\band also\b|\bas well as\b|\bwhile\b|\bwhereas\b|\bcompared (?:to|with)\b|\bin contrast\b|"
    r"\bboth\b.+\band\b|\bbetween\b.+\band\b|\bbut also\b",
    re.I,
)


def multi_part_question(question: str) -> bool:
    """Does the question ask several things at once (each may come from another document)?

    Chinese: two clauses that each hold an interrogative ("…在哪个地点举办，并且有多少家企业参展？",
    "…由哪个团队维护，失败后最多重试几次？"), or one ask over a list ("A、B 以及 C 是什么").
    English: two asks, each opening a clause ("Who … , and when …"), or one ask with a second
    part joined to it ("Which company …, and is also …", "Does … while …", "both … and …").
    Two interrogatives inside one clause ("4 什么加 9 什么", "怎么看网卡多少兆") are one ask.
    """
    text = question.strip()
    if _CJK.search(text):
        clauses = [part for part in _CLAUSE_ZH.split(text) if _ASK_ZH.search(part)]
        if len(clauses) >= 2:
            return True
        return len(clauses) == 1 and bool(_LISTED_ZH.search(text))
    asks = len(_ASK_EN.findall(text))
    return asks >= 2 or (asks >= 1 and bool(_JOIN_EN.search(text)))


def classify_question(question: str) -> str:
    """"compare" | "explain" | "multi" | "fact" from the wording alone (no model call).

    "multi" is a fact question that asks several things; an explanation or a comparison
    already gets as many passages and keeps its own answer shape.
    """
    if comparison_topic(question) is not None or _COMPARE.search(question):
        return "compare"
    if _EXPLAIN.search(question):
        return "explain"
    if multi_part_question(question):
        return "multi"
    return "fact"


def profile_for(name: str | None) -> AnswerProfile:
    return PROFILES.get(name or "", PROFILES["explain"])


def retrieval_limits(name: str | None) -> dict[str, int]:
    """``HybridRetriever.retrieve`` keyword arguments for a question type (the retrieve node's).

    ``per_document`` only when the type has its own (else the setting applies).
    """
    profile = profile_for(name)
    limits = {"max_keep": profile.max_passages}
    if profile.per_document is not None:
        limits["per_document"] = profile.per_document
    return limits


# "[来源：文件名｜位置：page:3]"; the UI (project/ui/static/citations.js) accepts the same forms.
CITATION = re.compile(
    r"[\[【]\s*来源\s*[:：]\s*([^\]】]+?)\s*(?:[｜|，,;；]\s*位置\s*[:：]\s*([^\]】]+?)\s*)?[\]】]"
)


def _loose(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()


def cited_claims(answer: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """``[(claim, [(source, locator), ...]), ...]`` for every cited statement of an answer.

    A citation closes the text before it, back to the previous citation or the start of
    its line; a run of adjacent citations belongs to the same claim. Uncited lines
    (an opening summary, a heading, "知识库中没有找到…") are not returned.
    """
    claims: list[tuple[str, list[tuple[str, str]]]] = []
    for line in answer.splitlines():
        position = 0
        pending: list[tuple[str, str]] = []
        claim = ""
        for match in CITATION.finditer(line):
            text = line[position : match.start()]
            if text.strip(" \t，,。；;、"):
                if pending:
                    claims.append((claim, pending))
                claim, pending = text, []
            pending.append((match.group(1).strip(), (match.group(2) or "").strip()))
            position = match.end()
        if pending:
            claims.append((claim, pending))
    cleaned = []
    for claim, refs in claims:
        claim = re.sub(r"^[\s>*#\-+]*(?:\d+[.、)]\s*)?", "", claim).strip(" \t，,；;")
        claim = re.sub(r"\*\*|__", "", claim)
        if claim:
            cleaned.append((claim, refs))
    return cleaned


def evidence_for(
    refs: list[tuple[str, str]], blocks: list[dict[str, Any]]
) -> tuple[list[int], bool]:
    """Indexes of the evidence blocks a claim cites, and whether every citation resolved.

    A citation matches a block by source and locator (spacing and case ignored); with a
    locator that matches nothing, the source's blocks are used instead.
    """
    found: list[int] = []
    resolved = True
    for source, locator in refs:
        if not locator and not any(_loose(b["source"]) == _loose(source) for b in blocks):
            # "[来源：差旅制度.docx，section:2]": the locator without its "位置：" label.
            parts = re.split(r"\s*[｜|，,;；]\s*", source, maxsplit=1)
            if len(parts) == 2:
                source, locator = parts
        same_source = [i for i, b in enumerate(blocks) if _loose(b["source"]) == _loose(source)]
        if not same_source:
            # File names the model shortened.
            same_source = [i for i, b in enumerate(blocks) if _loose(source) and _loose(source) in _loose(b["source"])]
        exact = [i for i in same_source if _loose(blocks[i]["locator"]) == _loose(locator)] if locator else []
        chosen = exact or same_source
        if not chosen:
            resolved = False
        found.extend(i for i in chosen if i not in found)
    return found, resolved


# ---------------- statements copied from their passages ----------------

# A bigram holding one of these must appear in the passage exactly as written: a changed
# number, negation, degree or comparison changes what a statement says even when every other
# word is copied ("不需要" / "需要", "超过三倍" / "三倍", "最高" / "较高").
_EXACT_CHARS = frozenset(
    "不没无未非否勿禁仅只均皆全最极更较超高低多少大小增减升降倍约近余半零"
    "〇一二两三四五六七八九十百千万亿首唯"
)
# Joining words left out of the overlap: "粒径为 2nm" copies "粒径 2nm".
_JOINERS = re.compile(r"[的地得了着是为在和与及或等也都就把被将对于其之而并且即]")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_WORD_CHARS = re.compile(r"[^\W_]+")
# Below this many characters a statement says too little to judge by its words.
_MIN_LITERAL_CHARS = 6


def _plain(text: str) -> str:
    return "".join(_WORD_CHARS.findall(CITATION.sub("", text).lower()))


def _bigrams(text: str) -> set[str]:
    return {text[i : i + 2] for i in range(len(text) - 1)}


def literally_supported(claim: str, passages: list[str], min_overlap: float = 0.85) -> bool:
    """True when ``claim`` is, near enough, a copy of the passages it cites.

    Such a statement needs no model to confirm it: every number in it appears in the
    passages, every bigram around a negation, number or degree word is found there as it is,
    and at least ``min_overlap`` of its other bigrams too. Anything reworded fails and goes
    to the model check; a failure here never marks a statement as unsupported.
    """
    plain_claim = _plain(claim)
    if len(plain_claim) < _MIN_LITERAL_CHARS or not passages:
        return False
    passage_numbers = {number for passage in passages for number in _NUMBER.findall(passage)}
    if not set(_NUMBER.findall(CITATION.sub("", claim))) <= passage_numbers:
        return False
    plain_passages = [_plain(passage) for passage in passages]
    grams = set().union(*(_bigrams(passage) for passage in plain_passages))
    if any(gram not in grams for gram in _bigrams(plain_claim) if any(char in _EXACT_CHARS for char in gram)):
        return False
    claim_grams = _bigrams(_JOINERS.sub("", plain_claim))
    if not claim_grams:
        return False
    loose = set().union(*(_bigrams(_JOINERS.sub("", passage)) for passage in plain_passages))
    return sum(gram in loose for gram in claim_grams) / len(claim_grams) >= min_overlap
