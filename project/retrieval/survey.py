"""Survey questions: "一般的创新项目申报能给多少补贴", "各个项目的补贴额度分别是多少".

Such a question is answered by several files, one figure (or condition, or date) each. Scored
against the whole question, the passages that echo its words win (a FAQ that asks "能拿多少
补贴？", a collection that says 补贴 in every line), and the usual cut keeps six passages from
three files (evaluation/recall_wording_v1/REPORT.md, step 5). For these questions the
retriever takes a wider pool with few candidates per file, and ``select_survey`` keeps the
best passage of each of the first files instead of the global top k:

* a passage only counts when it has the shape of an answer: an amount for "多少钱", a date or
  period for "什么时候";
* files whose type states rules (a guide, measures, a notice, a plan: by file name or
  document card) come before files that only mention them (an application form, a report);
  the two best files stay first whatever their type.

Detection is by wording only, and narrow (the regression's questions showed what else these
words mean): a short question with a generality word (一般 / 通常 / 普遍 …), or with an "each /
all / which" form (各…, 都…, 有哪些) that asks for an amount, a time or a condition and names no
particular place, person, organisation, code or file. "面试题里有哪些关于 RLHF 的题目" (a list
from one file), "纪念币有哪些" (a plain list) and "A 和 B 分别是多少" (two named things) are not
survey questions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Mapping

from .question_parts import ASK_ZH, _has_name
from .selection import SelectionPolicy, _score, no_evidence
from .types import SearchHit

# "一般纳税人", "一般贸易", "一般现在时" are terms, not a generality.
_GENERAL = re.compile(r"一般(?!纳税人|贸易|公共|现在时|过去时|将来时|等价物|性|化)|通常|普遍|大多|大都|大部分|往往|基本上?都")
_EACH = re.compile(r"各(?:个|地|类|项|家|种|自|有|是|省|市|区|部门|单位)|都(?:有|是|要|能|给|需要|得)|有哪些")
# "产品线 P-039 的收入和毛利率…": one identified thing, not what several files say.
_CODE = re.compile(r"[A-Za-z]+[-_]?\d+|\d{3,}")
MAX_CHARS = 32
_TITLE = re.compile(r"《[^》]{1,40}》")
_AMOUNT_ASK = re.compile(r"多少|几成|几万|比例|上限|封顶|金额|额度|标准")
_TIME_ASK = re.compile(r"什么时候|何时|几号|哪天|几月|截止|截至|期限|多久|时间")
_WHO_ASK = re.compile(r"条件|资格|门槛|要求|对象|什么人|谁(?:能|可以|可)|哪些(?:人|单位|企业|主体)")

# How an answer of each kind looks in a passage.
_AMOUNT = re.compile(r"\d+(?:\.\d+)?\s*(?:万元|亿元|元|万美元|美元|%|％|个基点|万|倍)|[一二两三四五六七八九十百千]+(?:万元|元)|百分之")
_TIME = re.compile(
    r"[\d一二两三四五六七八九十半]+\s*个?(?:交易日|工作日|自然日|小时|分钟|星期|天|日|周|月|年)|\d{1,2}\s*[月号]|"
    r"每(?:年|季度?|月|半年|周)|常年|长期有效|即日起|截止|截至"
)

# Document types (retrieval/doc_cards.py) that state the rules a survey question asks about.
RULE_TYPES = frozenset({"制度规范", "说明书", "通知公告", "方案"})
# File names that end the way rule-stating papers are named, beyond the card vocabulary
# ("…工作指引", "…申报须知", "…指导意见", "1、政策集合"), with an optional "（2025年修订）".
_RULE_NAME = re.compile(
    r"(?:指南|办法|细则|规定|规程|指引|须知|意见|措施|通知|方案|条例|制度|标准|政策(?:集合|汇编|汇总|清单|合集)?)"
    r"(?:[（(][^（()）]{1,12}[)）])?$"
)


def rule_name(name: str) -> bool:
    """Does the file's own name say it states rules?"""
    stem = name.rsplit(".", 1)[0] if "." in name else name
    return bool(_RULE_NAME.search(stem.strip()))


@dataclass(frozen=True)
class Survey:
    ask: str  # "amount" | "time" | "who" | "" (what the answer of each file looks like)
    cue: str  # the wording that made it a survey question


def survey_question(question: str) -> Survey | None:
    """The question asks for what several files say (see the module text), else None."""
    text = (question or "").strip()
    if not text or len(text) > MAX_CHARS or _TITLE.search(text) or not ASK_ZH.search(text):
        return None
    general = _GENERAL.search(text)
    each = None if general else _EACH.search(text)
    if not general and not each:
        return None
    ask = "amount" if _AMOUNT_ASK.search(text) else "time" if _TIME_ASK.search(text) else "who" if _WHO_ASK.search(text) else ""
    if each and (not ask or _CODE.search(text) or _has_name(text)):
        # A plain list ("纪念币有哪些"), one identified thing, or named things ("广东省和河南省…"):
        # the usual, multi-part and comparison paths.
        return None
    return Survey(ask, (general or each).group(0))


# What a survey question says besides its topic: the generality / "each" wording, the ask
# ("能给多少", "什么时候", "有哪些门槛") and words every programme paper uses (补贴, 申报, 政策).
_NOT_TOPIC = frozenset(
    "一般 一般来说 通常 能领 能补 能拿 各有 搞 做 普遍 大多 大都 大部分 往往 基本上 基本 都 各 各个 各地 各类 各项 各种 各家 各自 分别 有 有没有 哪些 有哪些 哪 什么 "
    "什么样 多少 多少钱 几 几月 几月份 几号 哪天 何时 时候 什么时候 能 能够 可以 可 会 要 需要 得 给 拿 拿到 领 补 报 是 的 了 吗 呢 "
    "这 这些 这类 那些 类 方面 上 对 和 与 及 或 等 一下 大概 最高 最多 金额 额度 标准 比例 上限 门槛 条件 要求 资格 对象 "
    "满足 达到 符合 具备 截止 期限 时间 申请 申报 申领 补贴 资助 扶持 奖补 奖励 补助 支持 政策 项目 钱 人 单位 谁".split()
)


# Words that list several topics: "参展、办展、跨境电商", "技术改造和绿色工厂".
_LIST_WORDS = frozenset("、 ， , 和 与 及 以及 或 或者 跟 同".split())
MAX_TOPICS = 3


def survey_topics(question: str) -> list[str]:
    """The question's topics without its survey wording: "考技能证书一般能领多少补贴？" ->
    ["考 技能 证书"], "参展、办展、跨境电商一般各补贴多少？" -> ["参展", "办展", "跨境 电商"]
    (at most MAX_TOPICS; [] when nothing is left). Searched alone a topic finds the files
    about it; the whole question also matches every passage that says 补贴 or 申报. Each
    listed topic is its own: a passage about 跨境电商 scores low against "参展 办展 跨境 电商"."""
    from .embedded_store import _segmenter

    groups: list[list[str]] = [[]]
    for word in _segmenter().lcut(question or ""):
        if word in _LIST_WORDS:
            groups.append([])
        elif re.search(r"\w", word) and word not in _NOT_TOPIC:
            groups[-1].append(word)
    topics = [" ".join(words) for words in groups if words]
    if len(topics) > MAX_TOPICS:
        topics = topics[: MAX_TOPICS - 1] + [" ".join(topics[MAX_TOPICS - 1:])]
    return topics


def survey_topic(question: str) -> str:
    """All of the question's topics in one text ('' when nothing is left)."""
    return " ".join(survey_topics(question))


def has_shape(text: str, ask: str) -> bool:
    """Could ``text`` hold an answer of this kind (an amount, a date)? Other kinds: always."""
    if ask == "amount":
        return bool(_AMOUNT.search(text))
    if ask == "time":
        return bool(_TIME.search(text))
    return True


def select_survey(
    ranked: list[SearchHit],
    policy: SelectionPolicy,
    survey: Survey,
    files: int = 8,
    floor: float = 0.01,
    types: Mapping[str, str] | None = None,
) -> list[SearchHit]:
    """The best answer-shaped passage of each of the first ``files`` documents, then the next
    best passages of those documents while there is room (``policy.max_keep`` passages,
    ``policy.per_document`` per document). ``ranked`` is best first.

    Documents come in this order: the two best whatever their type, then those of a
    rule-stating type (``types``: doc id -> type), then the others; each by its best passage,
    which must score at least ``floor``. Nothing at all when even the best passage is below the
    policy's no-evidence floor.
    """
    if not ranked or no_evidence(ranked, policy):
        return []
    shaped = [hit for hit in ranked if has_shape(hit.text, survey.ask)]
    if not shaped:
        shaped = list(ranked)  # no passage looks like an answer: the reranker's order is all there is
    best: dict[str, SearchHit] = {}
    for hit in shaped:
        if _score(hit) >= floor:
            best.setdefault(hit.doc_id, hit)
    order = sorted(best.values(), key=_score, reverse=True)
    lead, rest = order[:2], order[2:]
    typed = [hit for hit in rest if (types or {}).get(hit.doc_id, "") in RULE_TYPES]
    others = [hit for hit in rest if hit not in typed]
    chosen = (lead + typed + others)[: max(1, files)]
    if not chosen:
        chosen = ranked[: max(1, policy.min_keep)]
    kept = chosen[: policy.max_keep]
    ids = {hit.id for hit in kept}
    docs = {hit.doc_id for hit in kept}
    per_doc = {hit.doc_id: 1 for hit in kept}
    cutoff = max(policy.min_score, policy.relative * _score(ranked[0]))
    for hit in shaped:
        if len(kept) >= policy.max_keep:
            break
        if hit.id in ids or hit.doc_id not in docs or _score(hit) < cutoff:
            continue
        if policy.per_document > 0 and per_doc[hit.doc_id] >= policy.per_document:
            continue
        kept.append(hit)
        ids.add(hit.id)
        per_doc[hit.doc_id] += 1
    return sorted(kept, key=_score, reverse=True)


# ---------------------------------------------------------------- files first (survey_file_first)
def topic_files(hits: list[SearchHit], limit: int, per_file: int = 2) -> list[str]:
    """Doc ids of the files a search for the topic found, best first: a file's weight is the
    sum of its ``per_file`` best fused scores (one strong passage, or two good ones)."""
    scores: dict[str, list[float]] = {}
    for hit in hits:
        scores.setdefault(hit.doc_id, []).append(hit.score)
    weight = {doc_id: sum(sorted(values, reverse=True)[:per_file]) for doc_id, values in scores.items()}
    return sorted(weight, key=lambda doc_id: -weight[doc_id])[: max(1, limit)]


def select_by_files(
    pool: list[SearchHit],
    topic: Mapping[str, float],
    asked: Mapping[str, float],
    survey: Survey,
    policy: SelectionPolicy,
    files: int = 8,
    topic_floor: float = 0.05,
    file_floor: float = 0.01,
    topic_relative: float = 0.05,
    types: Mapping[str, str] | None = None,
) -> list[SearchHit]:
    """The files about the topic first, then each file's passage that answers the question.

    ``topic`` and ``asked`` are each passage's reranker score (by chunk id) against the topic
    alone and against the question. Nothing is kept when no file's best topic score reaches
    ``topic_floor`` (the library has no paper on the topic); otherwise a file is about the topic
    when its best topic score reaches ``file_floor`` and ``topic_relative`` x the best file's.
    Files of a rule-stating type (``types``) come first, then the others, each by topic score;
    the first ``files`` are kept. Of each, the answer-shaped passage that scores best against
    the question, then the next ones while there is room (``policy.max_keep``,
    ``policy.per_document``). A kept passage's score is the better of its two.
    """
    if not pool:
        return []
    by_file: dict[str, list[SearchHit]] = {}
    for hit in pool:
        by_file.setdefault(hit.doc_id, []).append(hit)
    relevance = {doc_id: max(topic.get(hit.id, 0.0) for hit in hits) for doc_id, hits in by_file.items()}
    if max(relevance.values()) < topic_floor:
        return []
    cut = max(file_floor, topic_relative * max(relevance.values()))
    about = sorted((doc_id for doc_id, score in relevance.items() if score >= cut), key=lambda d: -relevance[d])
    typed = [doc_id for doc_id in about if (types or {}).get(doc_id, "") in RULE_TYPES]
    chosen = (typed + [doc_id for doc_id in about if doc_id not in typed])[: max(1, files)]

    def score(hit: SearchHit) -> float:
        return asked.get(hit.id, 0.0)

    ranked: dict[str, list[SearchHit]] = {}
    for doc_id in chosen:
        hits = sorted(by_file[doc_id], key=score, reverse=True)
        shaped = [hit for hit in hits if has_shape(hit.text, survey.ask)]
        ranked[doc_id] = shaped + [hit for hit in hits if hit not in shaped] if shaped else hits
    kept = [ranked[doc_id][0] for doc_id in chosen][: policy.max_keep]
    room = policy.max_keep - len(kept)
    per_doc = policy.per_document if policy.per_document > 0 else policy.max_keep
    texts = {hit.text for hit in kept}
    extras = sorted(
        (hit for doc_id in chosen for hit in ranked[doc_id][1:per_doc] if has_shape(hit.text, survey.ask) and hit.text not in texts),
        key=score, reverse=True,
    )
    kept += extras[: max(0, room)]
    order = {doc_id: i for i, doc_id in enumerate(chosen)}
    kept.sort(key=lambda hit: (order[hit.doc_id], -score(hit)))
    return [replace(hit, rerank_score=max(score(hit), topic.get(hit.id, 0.0))) for hit in kept]
