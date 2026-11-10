"""A question that asks several things, split into its parts by rules (no model call).

"…黑龙江省发挥了哪些作用，并在最近的大会上取得了哪些具体成果？" asks two things, and the
passages answering each may sit in different documents. The cross-encoder scores every
passage against the whole question, so a passage that answers one part scores low and the
cutoffs drop it (evaluation/suite_v1/REPORT.md). The parts let retrieval search, score and
keep evidence for each ask on its own (evaluation/multidoc_v1/REPORT.md).

Chinese:
* clauses that each hold an interrogative: "A 有哪些作用，并且 B 取得了哪些成果";
  a later part gets the leading context ("在…中") and the first ask's subject;
* "X 和 Y 分别 / 各自 …": one part per listed item ("湖北省政协的协商会和湖北代表团在湖南的
  座谈会中，分别有哪些领导人讲话");
* "A、B 以及 C 是什么" / "新出生人口和死亡人口的数量分别是多少": one part per item.

English: clauses opening with a question word ("Who …, and when …"), "… while / whereas /
compared to …", "both X and Y", "between A and B", and a list of comma-separated clues
("Who is the person facing …, once likened to …, but now accused of …").

``key_terms`` gives the distinctive words of a part (for coverage checks and lexical matches).
"""

from __future__ import annotations

import re

MAX_PARTS = 4
_MIN_PART_CHARS = 4

# Interrogatives. "几" and "哪" only with what follows them in a question ("几天", "哪一年"),
# so "几乎" / "哪怕" do not count.
ASK_ZH = re.compile(
    r"哪(?:些|个|里|儿|一|两|三|几|位|家|种|所|条|项|年|月|天|次|国|部|款|类|方面)|什么|啥|如何|怎样|怎么|多少|"
    r"多(?:久|长|大|高|远)|"
    r"几(?:个|项|条|种|类|次|天|年|月|号|岁|名|家|人|点|成|倍|位|所|部|款|期|批|十|百|千|万)|"
    r"何(?:种|时|地|处|人|在|方)|有何|为何|为什么|是否|谁"
)
_CJK = re.compile(r"[㐀-鿿]")
_CLAUSE_ZH = re.compile(r"[，,；;？?。！!]|并且|而且|同时|此外|另外|以及|还有")
# Words that open a clause and carry nothing to search for.
_LEAD_ZH = re.compile(r"^(?:并|且|又|还|而|那么|那|则|也|再|请问|请)+")
# The end of a subject: what follows the first ask's noun phrase.
_SUBJECT_TAIL = re.compile(r"(?:是|有|的|了|在|对|于|将|会|被|把|与|和|及|都|也|又|还|共|一共|分别|各自|具体|主要)+$")
_EACH = re.compile(r"分别|各自|各有|各是")
# "X 和 Y": the joining words of a list, strongest first ("与" often sits inside a name:
# "鹏华基金与北京银行合作的…").
_JOINERS = ("和", "以及", "及", "与", "、")
_CONTEXT_OPEN = re.compile(r"^(?:在|于|据|从|结合|根据|针对|关于|对于|对|随着)")
_CONTEXT_CLOSE = re.compile(r"(?:中|里|上|下|期间|方面|背景下|之际|过程中|时)$")

_WH_EN = r"(?:who|whom|whose|what|which|when|where|why|how)"
_AUX_EN = r"(?:does|did|do|is|was|were|are|has|have|had|can|could|will|would|should)"
_SPLIT_EN = re.compile(
    rf"\s*(?:;|,\s*and\s+(?={_WH_EN}\b|{_AUX_EN}\b)|,?\s+while\s+|,?\s+whereas\s+|,?\s+compared (?:to|with)\s+"
    rf"|,?\s+in contrast to\s+|,?\s+but\s+(?=\w+\s+\w+\s+\w+))",
    re.I,
)
_BOTH_EN = re.compile(r"^(?P<head>.*?)\bboth\s+(?P<x>.+?)\s+and\s+(?P<y>.+?)(?P<tail>[,?].*)?$", re.I)
# "Between A, and B, was there …" (the question opens with it).
_BETWEEN_EN = re.compile(
    rf"^between\s+(?P<x>.+?),\s+and\s+(?P<y>.+?),\s*(?P<tail>(?:{_AUX_EN}|{_WH_EN})\b.+)$", re.I
)
# Clues that only frame the question ("indicating a difference in …", "was there a change …").
_FRAMING_EN = re.compile(r"^(?:indicating|resulting in|showing|was there|is there|were there|are there)\b", re.I)
# Where a new clue starts after a comma: a relative word, a verb or a participle
# ("…, once likened to …", "…, facing allegations …", "…, and is accused of …"). Other
# commas stay inside a clue: dates ("September 26, 2023") and lists ("developers, OEMs, and carriers").
_CLUE_START_EN = re.compile(
    rf"^(?:and\s+|but\s+|yet\s+)?(?:who|which|that|whose|while|once|now|also|then|later|recently|according|as|"
    rf"known|not|after|before|regarding|specifically|with|in|{_AUX_EN}|\w+ed|\w+ing)\b",
    re.I,
)
_MONTHS_EN = frozenset(
    "january february march april may june july august september october november december "
    "jan feb mar apr jun jul aug sep sept oct nov dec monday tuesday wednesday thursday friday saturday sunday "
    "day same later earlier".split()
)
_WORD_EN = re.compile(r"[A-Za-z0-9][\w'’-]*")

_STOP_EN = frozenset(
    "a an the of in on at to for from by with and or but as is was were are be been being do does did has have had "
    "which who whom whose what when where why how that this these those it its their his her they them he she "
    "both between while whereas compared than also not only any all some such there here about after before "
    "article articles report reports reported according source sources news mentioned covered featured discussed "
    "published noted highlighted stated stating suggest suggests suggesting claim claims reporting coverage respectively".split()
)


def _dedupe(parts: list[str]) -> list[str]:
    out: list[str] = []
    for part in parts:
        part = part.strip(" ，,、；;：:。？?！!")
        if len(part) >= _MIN_PART_CHARS and part not in out:
            out.append(part)
    return out[:MAX_PARTS] if len(out) >= 2 else []


# ---------------------------------------------------------------- Chinese
def _subject(ask: str) -> str:
    """The first ask's subject: its text before the interrogative ("黑龙江省发挥了哪些作用" -> "黑龙江省发挥")."""
    match = ASK_ZH.search(ask)
    head = ask[: match.start()] if match else ask
    head = _SUBJECT_TAIL.sub("", head.strip(" ，,"))
    return head if len(head) >= 2 else ""


def _split_list(text: str) -> tuple[str, list[str], str] | None:
    """``prefix, items, suffix`` of the first list "X 和 Y" / "X、Y 以及 Z" in ``text``."""
    for joiner in _JOINERS:
        if joiner not in text:
            continue
        pieces = text.split(joiner)
        if joiner != "、" and "、" in pieces[0]:
            # "A、B 以及 C": the earlier items are joined by "、".
            pieces = pieces[0].split("、") + pieces[1:]
        if len(pieces) < 2:
            continue
        first, last = pieces[0], pieces[-1]
        # The first item starts after the last "的 / 在 / 中 / 对" before it ("2022年日本的新出生人口").
        cut = max(first.rfind(mark) for mark in ("的", "在", "对", "中", "，", "让", "了"))
        prefix, first_item = (first[: cut + 1], first[cut + 1 :]) if cut >= 0 and len(first) - cut - 1 >= 2 else ("", first)
        # The last item ends at the first "的 / 中 / 分别 / 在" after it ("死亡人口的数量分别是多少").
        ends = [m.start() for m in re.finditer(r"的|中|里|分别|各自|在|是|有|对|如何|怎么|哪", last)]
        end = next((e for e in ends if e >= 2), len(last))
        last_item, suffix = last[:end], last[end:]
        items = [first_item, *pieces[1:-1], last_item]
        items = [item.strip(" ，,") for item in items]
        if all(len(item) >= 2 for item in items) and len(items) <= MAX_PARTS:
            return prefix, items, suffix
    return None


def _parts_zh(question: str) -> list[str]:
    text = question.strip().rstrip("？?。！! ")
    clauses = [_LEAD_ZH.sub("", c).strip() for c in _CLAUSE_ZH.split(text)]
    clauses = [c for c in clauses if c]
    asks = [c for c in clauses if ASK_ZH.search(c)]
    first_ask = next((i for i, c in enumerate(clauses) if ASK_ZH.search(c)), len(clauses))
    context = "".join(clauses[:first_ask])
    if len(asks) >= 2:
        subject = _subject(asks[0])
        parts = [context + asks[0]]
        for ask in asks[1:]:
            # "…，这些政策何时生效" needs the first ask's subject; "…，苏宁易购预计带动多少消费" has its own.
            own = ask[: ASK_ZH.search(ask).start()]
            head = subject if subject and subject not in ask and not _has_name(own) else ""
            parts.append(f"{context}{head} {ask}".strip())
        return _dedupe(parts)
    if len(asks) != 1:
        return []
    ask = asks[0]
    # "在 X 和 Y 中，分别有哪些…": the list is in the context clause.
    if _EACH.search(ask) and context and (listed := _split_list(context)):
        prefix, items, suffix = listed
        return _dedupe([f"{prefix}{item}{suffix}，{ask}" for item in items])
    # "X 和 Y 分别是什么" / "新出生人口和死亡人口的数量分别是多少" / "A、B 以及 C 是什么".
    each = _EACH.search(ask)
    scope = ask[: each.start()] if each else ask[: ASK_ZH.search(ask).start()]
    if each or re.search(r"、|以及", scope):
        if listed := _split_list(scope):
            prefix, items, suffix = listed
            rest = ask[len(scope):]
            rest = _EACH.sub("", rest)
            return _dedupe([f"{context}{prefix}{item}{suffix}{rest}" for item in items])
    # "描述习水县和临洮县中，社区成员如何…", "恒大汽车的股权变动和长安汽车在泰国的投资计划反映了…":
    # a list of named things (places, organisations, names), each likely in its own report.
    tail = ask[ASK_ZH.search(ask).start():]
    for clause, before, after in ((context, "", f"，{ask}"), (ask, context, "")):
        pieces = _named_pieces(clause)
        if pieces:
            # Each piece keeps the ask: the last one already ends with it when the list is in the ask.
            ending = after or f" {tail}"
            parts = [f"{before}{piece}{ending}" for piece in pieces[:-1]]
            parts.append(f"{before}{pieces[-1]}{after}")
            return _dedupe(parts)
    return []


def _named_pieces(clause: str) -> list[str]:
    """``clause`` cut at "和 / 与 / 及" when every piece names something (2 to MAX_PARTS)."""
    for joiner in ("和", "与", "及"):
        if joiner not in clause:
            continue
        pieces = [p.strip(" ，,") for p in clause.split(joiner)]
        if 2 <= len(pieces) <= MAX_PARTS and all(len(p) >= 3 and _has_name(p) for p in pieces):
            return pieces
        return []
    return []


_NAME_TAGS = frozenset({"ns", "nr", "nt", "nz", "nrt", "nrfg"})


def _has_name(text: str) -> bool:
    """Does ``text`` hold a proper name (jieba's place / person / organisation / other name tags)?"""
    from .embedded_store import _segmenter

    _segmenter()  # jieba initialised once, with the app's dictionary cache
    import jieba.posseg as posseg

    return any(pair.flag in _NAME_TAGS and len(pair.word) >= 2 for pair in posseg.cut(text))


# ---------------------------------------------------------------- English
def _clean_en(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" ,;:?.!")


def _content_words(text: str) -> list[str]:
    """Words that say what a clue is about: not stop words, months, numbers or source names."""
    words = []
    for word in _WORD_EN.findall(text):
        low = word.lower().strip("'’")
        if low in _STOP_EN or low in _MONTHS_EN or low.isdigit() or re.fullmatch(r"\d+(?:st|nd|rd|th)", low):
            continue
        words.append(low)
    return words


def _clues(text: str) -> list[str]:
    """``text`` cut at the commas that open a new clue; clues that say almost nothing
    beyond a source and a date ("After The Verge reported on September 26, 2023") are dropped."""
    clues: list[str] = []
    for piece in re.split(r",\s+", text):
        if clues and not _CLUE_START_EN.match(piece):
            clues[-1] = f"{clues[-1]}, {piece}"
        else:
            clues.append(piece)
    return [_clean_en(c) for c in clues if len(_content_words(c)) >= 3 and not _FRAMING_EN.match(c)]


def _parts_en(question: str) -> list[str]:
    text = _clean_en(question)
    if match := _BETWEEN_EN.match(text):
        tail = match["tail"]
        return _dedupe([_clean_en(f"{match['x']} {tail}"), _clean_en(f"{match['y']} {tail}")])
    pieces = [p for p in _SPLIT_EN.split(text) if p and p.strip()]
    if len(pieces) < 2 and (match := _BOTH_EN.match(text)):
        head, tail = match["head"], match["tail"] or ""
        pieces = [f"{head} {match['x']} {tail}", f"{head} {match['y']} {tail}"]
    # Within each ask, clues separated by commas: "Who is the individual facing …, once
    # likened to …, but now accused of …" (each clue may come from another article).
    parts = [clue for piece in pieces for clue in _clues(_clean_en(piece))]
    return _dedupe(parts)


def question_parts(question: str) -> list[str]:
    """The asks of a question that asks several things (2 to MAX_PARTS), else []."""
    text = (question or "").strip()
    if not text or len(text) > 600:
        return []
    return _parts_zh(text) if _CJK.search(text) else _parts_en(text)


# ---------------------------------------------------------------- key terms
_ZH_STOP = frozenset(
    "哪些 哪个 什么 如何 怎样 怎么 多少 是否 为什么 为何 有何 以及 并且 而且 同时 此外 另外 还有 分别 各自 "
    "具体 主要 相关 情况 方面 问题 进行 通过 采取 发挥 取得 作用 影响 措施 内容 这些 那些 这个 那个 这次 "
    "其中 他们 我们 你们 它们 以及 可以 能够 哪里 哪儿 一些 有关 关于 对于 根据 结合 最近 目前 当前 本月 "
    "今年 去年 之前 之后 期间 过程 背景 计划 预计 实现 提供 表示 认为 发表 提出 带来 推动 促进 了解".split()
)


def key_terms(text: str, limit: int = 8) -> list[str]:
    """Distinctive words of ``text`` (names, numbers, longer words), longest first.

    Chinese uses jieba (already loaded for BM25); English keeps capitalised words, numbers
    and words of five letters or more outside a stop list.
    """
    if _CJK.search(text):
        from .embedded_store import tokenize

        quoted = re.findall(r"[“\"《「『']([^”\"》」』']{2,20})[”\"》」』']", text)
        words = [w for w in tokenize(ASK_ZH.sub(" ", text)) if len(w) >= 2 and w not in _ZH_STOP]
        words += [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9.+-]{1,}|\d+(?:\.\d+)?%?", text)]
        terms = list(dict.fromkeys([*quoted, *words]))
    else:
        terms = []
        for word in _WORD_EN.findall(text):
            low = word.lower().strip("'’")
            if low in _STOP_EN:
                continue
            if word[0].isupper() or word[0].isdigit() or len(low) >= 5:
                terms.append(low)
        terms = list(dict.fromkeys(terms))
    # Drop terms contained in a longer kept term ("俄青" in "中俄青年企业家").
    kept = [t for t in terms if not any(t != o and t in o for o in terms)]
    return sorted(kept, key=len, reverse=True)[:limit]


def mentions(text: str, terms: list[str]) -> float:
    """Share of ``terms`` found in ``text`` (case-insensitive)."""
    if not terms:
        return 0.0
    body = text.lower()
    return sum(term.lower() in body for term in terms) / len(terms)
