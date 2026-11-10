"""Checks around the answer model that need no model call (evaluation/generation_v1/REPORT.md).

* ``near_term_notes``: a key term of the question that the evidence never uses, while it
  does use a near term ("质保期" asked, "保质期" written). Retrieval and passage selection
  never filter on terms (the hybrid search finds the near term on purpose); the answer only
  says, in one line, which word the documents use.
* ``rule_check``: numbers, model names (A100, GPT-4) and quoted names in cited statements
  that appear in none of the evidence passages. It stands in for the model check when that
  check cannot run.
* ``missing_anchors``: a year, place, name or 《title》 of the question that the answer talks
  about although none of its cited passages mention it.
* ``conflicting_values``: the answer's value for the question's fact, and a different value
  another passage about the same subject gives for it ("10.48‰" / "5.48‰").
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

from .answering import CITATION, cited_claims

# Interrogatives and question furniture: never a term the documents are expected to contain.
_ASKING = re.compile(
    r"什么时候|时候|什么|多少|哪(?:一|些|个|里|儿|位|家|种|所|条|项|款|类|年|月|天|次|国|部)?|"
    r"几(?:个|项|条|种|类|次|天|年|月|号|岁|名|家|人|点|成|倍|位|所|部|款|期|批|场|届)?|谁|吗|呢|是否|如何|怎么|怎样|"
    r"为何|为什么|何时|何地|何人|何种|请问|是不是|有没有|多久|多长|多大|多高|多远"
)
# Units and measure words: after a number they belong to it ("2022年特斯拉" is not a term "年特斯拉").
_UNITS = frozenset("年月日号个位名届次部场家台辆座架只条项些种期批元岁天周季代")
_NUMERALS = re.compile(r"[0-9零〇一二两三四五六七八九十百千万亿第]")
# A count opening a word ("两个项目", "三家公司", "单个项目"): an amount of something, not its name
# (零 is left out: "零部件" is a word, not a count).
# Without this "两个项目" asked and "单个项目" written read as near terms (user data, 1.3.0 check).
_COUNT = re.compile(r"^[一二两三四五六七八九十百千万亿几多单双每各半数][个位名家条项种只台部批次届场所款类]")
_CJK_RUN = re.compile(r"[㐀-鿿]+")
_STOP = re.compile(r"^(?:[的了和与及或在是有为对把被将从向于就都也还而且并]|一个|这个|那个|我们|你们|他们|公司|资料|文件|文档)$")
# Where a note's quoted span ends.
_SPAN_END = re.compile(r"[，,。；;！!？?\n、（(）)]")
MAX_TERM = 6
MAX_SPAN = 18


@dataclass(frozen=True)
class TermNote:
    term: str  # the question's word, absent from every passage
    near: str  # the near word the passages use
    span: str  # the near word with what follows it, as written ("保质期：未开封 12 个月")
    passage: int  # index of the passage it was found in

    def line(self) -> str:
        return f"资料里写的是“{self.span}”，没有提到“{self.term}”。"


def _near(term: str, other: str) -> bool:
    """Same length, different words: the same characters reordered ("质保期" / "保质期"), or for
    three characters or more, one character different with the last one shared ("保修期" /
    "质保期", "年增长率" / "月增长率": Chinese compounds keep their head noun last)."""
    if len(term) != len(other) or term == other:
        return False
    if Counter(term) == Counter(other):
        return True
    if len(term) < 3 or term[-1] != other[-1]:
        return False
    return sum((Counter(term) & Counter(other)).values()) == len(term) - 1


# Function characters a term never starts or ends with ("产品的", "期是"), and never holds.
_EDGE = frozenset("的了着过地得之其及与和或而且也都就还又在于从向对把被将给让由是有为这那此该各每某你我他她它们吧啊呀吗呢")
_INNER = frozenset("的了吗呢吧啊呀")


def _term_like(text: str) -> bool:
    if not _CJK_RUN.fullmatch(text) or text[0] in _EDGE or text[-1] in _EDGE or _COUNT.match(text):
        return False
    return not (set(text) & _INNER) and not _ASKING.search(text) and not _STOP.match(text)


def _aligned_terms(text: str, sizes) -> dict[str, int]:
    """Term-like words of the given sizes, with the offset where each first ends: one to three
    consecutive words of jieba's dictionary-only segmentation (no HMM guessing), and the
    sub-words its search mode lists. A known word stays whole ("香港", "神舟") so no term starts
    inside it ("港第七届", "舟十四"); its sub-words still count ("广州市" inside
    "广州市中级人民法院"); an unknown word falls apart into characters, which leaves it free to
    be a term ("质保期" from 质 保 期)."""
    from retrieval.embedded_store import _segmenter

    jieba = _segmenter()
    wanted = set(sizes)
    found: dict[str, int] = {}

    def add(start: int, end: int) -> None:
        term = text[start:end]
        if len(term) not in wanted or term in found or not _term_like(term):
            return
        if term[0] in _UNITS and start > 0 and _NUMERALS.match(text[start - 1]):
            return  # the unit of a number before it
        found[term] = end

    words = list(jieba.tokenize(text, HMM=False))
    for i in range(len(words)):
        for j in range(i, min(i + 3, len(words))):
            add(words[i][1], words[j][2])
    for _, start, end in jieba.tokenize(text, mode="search", HMM=False):
        add(start, end)
    return found


def _question_terms(question: str) -> list[str]:
    """Candidate key terms of the question, longest first: 2..6 CJK characters on word
    boundaries ("港第七届" in "香港第七届" is not one), no interrogative or function character
    at either end."""
    # Interrogatives split the question ("什么时候上映" leaves "上映", not "候上映").
    masked = _ASKING.sub("，", question)
    terms = _aligned_terms(masked, range(MAX_TERM, 1, -1))
    return sorted(terms, key=len, reverse=True)


def _differing_part_present(term: str, near: str, evidence: str) -> bool:
    """The evidence does name what makes ``term`` different: "女单亚军" asked, "女单冠军" and
    "亚军" written. Only words of the term that hold its differing character count."""
    different = Counter(term) - Counter(near)
    if not different:
        return False  # the same characters reordered ("质保期" / "保质期")
    for size in range(2, len(term)):
        for start in range(len(term) - size + 1):
            part = term[start:start + size]
            if (Counter(part) & different) and part in evidence and _term_like(part):
                return True
    return False


def _made_of_present_words(term: str, evidence: str) -> bool:
    """Every word of ``term`` is in the evidence, only not side by side ("冬奥会男子" asked,
    "男子" and "冬奥会" written): a different wording, not a different term."""
    from retrieval.embedded_store import _segmenter

    words = _segmenter().lcut(term, HMM=False)
    return len(words) > 1 and all(word in evidence for word in words)


def near_term_notes(question: str, passages: list[str], limit: int = 2) -> list[TermNote]:
    """Key terms of ``question`` that no passage contains while one uses a near term instead.

    An exact match anywhere in the evidence means no note. The near term, too, must sit on word
    boundaries of its passage, so a window across two words ("美国中部" -> "国中") is never
    taken for one ("中国").
    """
    texts = [p or "" for p in passages]
    if not question or not any(texts):
        return []
    joined = "\n".join(texts)
    absent = [t for t in _question_terms(question) if t not in joined]
    if not absent:
        return []
    sizes = sorted({len(t) for t in absent}, reverse=True)
    # Candidate words of each passage: (passage, end offset) of the first occurrence.
    windows: dict[str, tuple[int, int]] = {}
    for index, text in enumerate(texts):
        for term, end in _aligned_terms(text, sizes).items():
            windows.setdefault(term, (index, end))
    notes: list[TermNote] = []
    covered: list[str] = []
    for term in absent:  # longest first
        if any(term in longer for longer in covered):
            continue  # "质保" inside an already noted "质保期"
        match = next((w for w in windows if _near(term, w)), None)
        if match is None or _differing_part_present(term, match, joined) or _made_of_present_words(term, joined):
            continue
        index, end = windows[match]
        tail = _SPAN_END.split(texts[index][end:end + MAX_SPAN], maxsplit=1)[0]
        notes.append(TermNote(term, match, (match + tail).strip(), index))
        covered.append(term)
        if len(notes) >= limit:
            break
    return notes


def note_present(answer: str, note: TermNote) -> bool:
    """Does the answer already say that the documents use the other word?"""
    body = CITATION.sub("", answer or "")
    said_absent = re.search(rf"(?:没有|未|并未|没)(?:提到|提及|出现|写到|说明)[“\"'‘]?{re.escape(note.term)}", body)
    return bool(said_absent) and note.near in body


def hint(notes: list[TermNote]) -> str:
    """What the answer prompt says about the notes (the model is told the exact line to open with)."""
    lines = []
    for note in notes:
        lines.append(
            f"问题里的“{note.term}”在证据中没有出现，证据里只有相近的“{note.near}”。两者不是同一个词，不要把“{note.near}”说成“{note.term}”。"
            f"回答第一句写：{note.line()}然后给出证据里关于“{note.near}”的内容。"
        )
    return "用词提醒：" + "".join(lines)


# ---------------------------------------------------------------- rule check
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_MODEL_NAME = re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z\-]*\d[A-Za-z0-9\-.]*|(?<![A-Za-z0-9])\d+[A-Za-z]{1,3}\d*(?![A-Za-z])")
_QUOTED = re.compile(r"《([^》]{2,30})》")
_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
# Chinese numerals only before a unit ("十二个月", "二〇二三年"): "十分重要" is not a number.
_CN_NUMBER = re.compile(r"[零〇一二两三四五六七八九十百千]+(?=[年月日号个人名家次天岁倍项条位元万亿%％点])")


def _cn_value(text: str) -> str:
    if not any(ch in "十百千" for ch in text):
        return "".join(str(_CN_DIGITS[ch]) for ch in text)
    total = current = 0
    for ch in text:
        if ch in _CN_DIGITS:
            current = _CN_DIGITS[ch]
        else:
            total += (current or 1) * {"十": 10, "百": 100, "千": 1000}[ch]
            current = 0
    return str(total + current)


def _normalized(text: str) -> str:
    text = CITATION.sub("", text or "")
    text = text.translate({ord(f): ord(h) for f, h in zip("０１２３４５６７８９．", "0123456789.")})
    text = re.sub(r"(?<=\d)[,，](?=\d{3})", "", text)
    text = re.sub(r"\s+", "", text)
    return _CN_NUMBER.sub(lambda m: _cn_value(m.group(0)), text)


def _numbers(text: str) -> set[str]:
    """Numbers worth checking: two digits or more, or a decimal (a lone "3" says too little).
    Not years: answers work them out from the passages ("同比" -> "与2021年相比"), which made
    most of the rule check's false alarms on RGB (evaluation/generation_v1/REPORT.md)."""
    out = set()
    for number in _NUMBER.findall(text):
        stripped = number.rstrip("0").rstrip(".") if "." in number else number
        if _YEAR.fullmatch(number):
            continue
        if len(number) >= 2 or "." in number:
            out.add(stripped)
    return out


_YEAR = re.compile(r"(?:19|20)\d\d")


def _terms(claim: str) -> set[str]:
    raw = CITATION.sub("", claim)
    return {t.lower().strip(".-") for t in _MODEL_NAME.findall(raw)} | {q for q in _QUOTED.findall(raw)}


def rule_check(answer: str, passages: list[str], question: str = "") -> list[str]:
    """Statements of the answer with a number, model name or 《title》 found in no passage.

    Lenient on purpose: the whole evidence counts (a statement citing the wrong passage is
    not an invented fact), Chinese numerals and thousands separators are normalized, and only
    numbers of two digits or more are checked. Uncited answers are checked as a whole. What
    the question itself says counts as given (an answer restating the question invents nothing).
    """
    evidence = _normalized("\n".join([*passages, question])).lower()
    evidence_numbers = {n.rstrip("0").rstrip(".") if "." in n else n for n in _NUMBER.findall(evidence)}
    claims = [claim for claim, _ in cited_claims(answer)] or [line for line in (answer or "").splitlines() if line.strip()]
    problems = []
    for claim in claims:
        body = _normalized(claim)
        terms = sorted(t for t in _terms(claim) if _normalized(t).lower() not in evidence)
        missing = [n for n in sorted(_numbers(body)) if n not in evidence_numbers and not any(n in t for t in terms)]
        missing += terms
        if missing:
            problems.append(f"证据里没有“{'、'.join(missing[:3])}”：{CITATION.sub('', claim).strip()[:80]}")
    return problems


# ---------------------------------------------------------------- anchors
# What the question is about that the cited passages must also be about: years, 《titles》, and
# places, people and organizations (jieba's dictionary tags; its POS tagger is not shipped).
# A year, or a season ("2020-2021", "2020-21赛季"); "2023-12" is a month and not an anchor.
_ANCHOR_YEAR = re.compile(r"(?:19|20)\d{2}(?=年)|(?:19|20)\d{2}[-—–](?:19|20)\d{2}(?!\d)|(?:19|20)\d{2}[-—–]\d{2}(?=赛季)")
_ANCHOR_TAGS = frozenset({"ns", "nt", "nr", "nrt", "nrfg"})
_ANCHOR_STOP = frozenset({"中国", "全国", "我国", "国家", "国内", "城市", "地区", "世界", "全球", "国际"})
# A place or team named with or without its last character ("德国队" / "德国", "北京市" / "北京").
_ANCHOR_SUFFIX = ("队", "市", "省", "区", "县", "州", "镇", "村", "地区")
# Common words the jieba dictionary tags as places or names: "设计深度" names no place, and a
# question about it must not get "资料里没有找到关于“深度”的内容" (release_1_4).
_MISTAGGED = frozenset({"深度", "长度", "广度", "高原", "平原", "山区", "新区"})
_tags: dict[str, str] | None = None


def _dictionary_tags() -> dict[str, str]:
    global _tags
    if _tags is None:
        from retrieval.embedded_store import _segmenter

        jieba = _segmenter()
        tags: dict[str, str] = {}
        try:
            with jieba.get_dict_file() as handle:
                for raw in handle:
                    parts = raw.decode("utf-8", "ignore").split()
                    if len(parts) == 3 and parts[2] in _ANCHOR_TAGS and parts[0] not in _MISTAGGED:
                        tags[parts[0]] = parts[2]
        except OSError:
            pass
        _tags = tags
    return _tags


def question_anchors(question: str) -> list[str]:
    from retrieval.embedded_store import _segmenter

    found = set(_ANCHOR_YEAR.findall(question)) | {t for t in _QUOTED.findall(question)}
    tags = _dictionary_tags()
    for word in _segmenter().cut(question, HMM=False):
        if len(word) >= 2 and word in tags and word not in _ANCHOR_STOP:
            found.add(word)
    return sorted(found)


def _anchor_present(anchor: str, text: str) -> bool:
    from retrieval.embedded_store import _segmenter

    anchor, text = anchor.lower(), text.lower()
    if anchor in text:
        return True
    if _ANCHOR_YEAR.fullmatch(anchor):
        return False  # "2021" is not "2022"
    for suffix in _ANCHOR_SUFFIX:
        if anchor.endswith(suffix) and len(anchor) - len(suffix) >= 2 and anchor[: -len(suffix)] in text:
            return True
    parts = [w for w in _segmenter().cut_for_search(anchor, HMM=False) if len(w) >= 2 and w != anchor]
    return bool(parts) and all(part in text for part in parts)


def missing_anchors(question: str, answer: str, blocks: list[dict]) -> list[str]:
    """Anchors of the question the answer's statements name although none of the passages it
    cites mention them: "印控克什米尔的踩踏事故死亡 3 人" citing only a passage about a temple in
    拉贾斯坦邦, or project A's figures cited from project B's report. Uncited answers (a refusal)
    have nothing to check."""
    from .answering import evidence_for

    claims = cited_claims(answer)
    used: set[int] = set()
    for _, refs in claims:
        used.update(evidence_for(refs, blocks)[0])
    if not used:
        return []
    said = "".join(claim for claim, _ in claims).lower()
    text = "\n".join(blocks[i].get("text", "") for i in sorted(used))
    return [a for a in question_anchors(question) if a.lower() in said and not _anchor_present(a, text)]


def anchor_line(missing: list[str]) -> str:
    names = "、".join(f"“{a}”" for a in missing[:2])
    return f"注意：引用的资料里没有出现{names}，下面的内容可能讲的是别的对象，请点开出处核对。"



# ---------------------------------------------------------------- conflicting values
# The answer gives one value ("人口出生率为5.48‰", "在成都举办") while another passage about the
# same subject gives a different one ("10.48‰", "武汉") and never mentions the answer's. The
# weak answer model cannot tell such a contradiction from passages that only say more or less
# (evaluation/generation_v1/REPORT.md, levers 5 and 5'), so the code finds it, and the answer
# opens with one line naming both values and where each is written.
@dataclass(frozen=True)
class Conflict:
    values: tuple[tuple[str, int], ...]  # (value as written, passage index), the answer's first

    def line(self, blocks: list[dict]) -> str:
        parts = [f"{_source_name(blocks[index])}写的是“{value}”" for value, index in self.values]
        return f"注意：资料里的说法不一致，{'，'.join(parts)}，请点开出处核对。"


def _source_name(block: dict) -> str:
    source = str(block.get("source") or "").strip()
    stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", source) or "资料"
    locator = str(block.get("locator") or "").strip()
    return f"《{stem}》" + (locator if locator and locator != "未标注位置" else "")


# What kind of value the question asks for (none of them: any kind).
_ASKS_NUMBER = re.compile(r"多少|几(?![月日号])|多大|多高|多长|多远|多重|增长率|增速|比例|比重|震级|人数|数量|总量|产量|销量|票房|营收|收入|价格|售价|金额|率")
_ASKS_TIME = re.compile(r"什么时候|何时|时间|日期|哪一?年|哪天|几月|几号|哪个月|年份|上映|发射|开幕|闭幕|发售|发布|成立|去世|出生")
_ASKS_NAME = re.compile(r"谁|哪|什么|冠军|导演|作者|编剧|主席|总统|首相|市长|创始人|处理器|芯片|名字|原名|首都|举办|主办|地点|城市|国家")
# Which names. A team, a winner: any proper name (the dictionary calls "利物浦" a place,
# "切尔西" a person, "苏宁" a place). Otherwise a company, a place ("在哪举办", "首都",
# "举办国") or a person ("导演是谁"); 《titles》 and “quoted names” always count.
_ASKS_TEAM = re.compile(r"冠军|球队|哪支|队|得主")
_ASKS_ORG = re.compile(r"公司|哪家|机构|组织|品牌|企业")
_ASKS_PLACE = re.compile(r"哪里|在哪|哪儿|哪个(?:城市|国家|省|地方|地区)|哪国|举办国|举办地|主办国|主办城市|首都|地点|位于|城市|国家|主席国")
_ASKS_PERSON = re.compile(r"谁|导演|作者|编剧|得主|创始人|创立|创办|总统|首相|总理|市长|议长|行政长官|原名|扮演|饰演|演的")
# A question asking for a list ("RCEP包括哪些国家", a cast): passages naming different members do
# not contradict each other.
_ASKS_LIST = re.compile(r"哪些|有哪|包括|包含|都有|列举|主演|演员|成员|嘉宾|有谁|[二三四五六七八十]强|前[一二三四五六七八九十\d]+")
# Words of the question that are never a slot: question furniture and generic words.
_SLOT_STOP = frozenset({"分别", "多少", "什么", "哪个", "哪里", "是否", "以及", "还有", "现在", "目前", "资料", "文档", "中国", "我国",
                        "全国", "国内", "世界", "全球", "情况", "问题", "哪些", "有关", "相关", "主要"})
_JOINERS = frozenset({"和", "与", "及", "以及", "跟", "同", "、", "或"})
_PLACE_TAGS = frozenset({"ns"})
_PERSON_TAGS = frozenset({"nr", "nrt", "nrfg"})
_ORG_TAGS = frozenset({"nt", "nz"})
_NAME_TAGS = _PLACE_TAGS | _PERSON_TAGS | _ORG_TAGS
_PROPER = frozenset({"place", "person", "dictname", "org"})
_UNIT = (r"万亿元|亿美元|万美元|亿元|万元|美元|欧元|日元|港元|亿人|万人|万吨|万辆|万台|万户|万亩|万亿|亿|万|千|"
         r"%|‰|元|人|名|个|家|辆|台|吨|架|艘|枚|席|座|所|项|级|岁|倍|平方公里|公里|千米|米|亩|斤|公斤|千克|克")
# A number with its unit; not a rank or an ordinal ("前10名", "第75届").
_CONFLICT_NUMBER = re.compile(rf"(?<![\d.前第])(\d+(?:\.\d+)?)({_UNIT})")
_CONFLICT_DATE = re.compile(r"(?<!\d)(?:((?:19|20)\d{2})年)?(\d{1,2})月(?:(\d{1,2})[日号])?")
_CONFLICT_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})年(?!\d{1,2}月)")
# Latin names worth comparing: with a digit ("A15", "GPT-3"), four letters or more ("Meta"), or a
# letter label ("E组", "B轮"); not short acronyms ("PC", "CTO").
_LATIN_NAME = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z][A-Za-z0-9\-+]*\d[A-Za-z0-9\-+]*|[A-Za-z][A-Za-z\-+]{3,}|[A-Z](?=[组轮类级区]))(?![A-Za-z0-9])")
# A model name of two characters and a number ("骁龙845"); not a year, a rank or an amount.
# A count is not a model number: "覆盖3000多家门店", "1200余项".
_MODEL_CJK = re.compile(r"([㐀-鿿]{2})(\d{3,4})(?![\d.%‰年月日号届次个名人万亿元级岁多余家项台件辆位种条套座只艘架款户间所米吨千百])")
_DOT_NAME = re.compile(r"([A-Za-z㐀-鿿]{1,6})((?:[·•・．][A-Za-z㐀-鿿]{1,8})+)")
_TITLE = re.compile(r"《([^》]{2,20})》")
_QUOTE = re.compile(r"“([^”]{2,20})”")
# Characters a transliterated name does not hold ("辛格导演的", "是布莱恩"): where it is cut.
_NAME_CUT = frozenset("的是在于由被将和与及为导执主编演饰任拍监制获夺等所曾已也以拿当成出带率称说表指向对把从让给用后前时中上下才就都还又很最更")
# Surnames that are rarely a word of their own (so "时", "安", "华", "关" are left out).
_SURNAMES = frozenset(
    "王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾肖田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆"
    "汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤柯滕殷甄童雍文舒乔房葛隋闵骆祁"
    "邢裴岳鲁庞栾冉聂晏柴巫翟仇鞠阎詹伍涂饶窦喻蓝凌霍糜粟桂匡桑敖廉"
)
# A known word after a surname that can still be a given name ("蒋万安", "叶伟民", "文牧野"): rare
# in the dictionary (common words like "中国", "进行" are not).
_GIVEN_NAME_FREQ = 600
# Words the dictionary tags as names that news uses as common words ("夺得澳网公开赛男单桂冠").
_NOT_NAMES = frozenset({"公开赛", "桂冠", "卫冕", "冠军", "亚军", "季军", "大满贯", "锦标赛", "世锦赛", "决赛", "半决赛", "晋级",
                        "青少年", "保姆", "城市", "地区", "国家", "世界", "全球", "国际", "全国", "我国", "国内", "中华人民共和国"})
# Numbered subjects ("神舟十六号", "第六届", "19号线", "第四任"): a sentence about another number is
# about another subject.
_NUMBERED = re.compile(r"(?:(第)|(?<![月\d]))([0-9零〇一二两三四五六七八九十百]+)(?(1)[㐀-鿿]|(号|届|代|期|任))")
# A running total ("截至8月14日，累计开行4025列，完成货运160万吨"), in the clause of the value or the
# two before it: figures of different dates differ without contradicting each other.
_RUNNING_TOTAL = re.compile(r"截至|截止|累计|迄今|至今")
_SENTENCE = re.compile(r"[。！!？?；;\n]|\.{3}|…+")
_CLAUSE = re.compile(r"[，,（）()]")
_ASK_SPLIT = re.compile(r"[，,；;？?。！!]")


@dataclass
class _Value:
    kind: str  # "number" | "time" | "name"
    key: tuple  # what is compared: (unit, amount), (year, month, day), (sort of name,)
    text: str  # as written
    start: int = 0
    end: int = 0
    last: str = ""  # a joined name's last part ("湖北省武汉市" -> "武汉市")
    lead: str = ""  # its clause and the two before it


def _value_kinds(question: str) -> set[str]:
    question = re.sub(r"(?:票房|销量|营收|收入|产量)(?=冠军|第一|最高|最多)", "", question)  # "票房冠军" asks a name
    kinds = {kind for kind, pattern in (("number", _ASKS_NUMBER), ("time", _ASKS_TIME), ("name", _ASKS_NAME))
             if pattern.search(question)}
    return kinds or {"number", "time", "name"}


def _name_sorts(question: str) -> set[str]:
    """The sorts of names the question asks for. The dictionary tags many places and companies
    as people ("圣保罗", "华为"), so its person-tagged words ("dictname") count for those too."""
    always = {"title", "quote", "latin", "model"}
    if _ASKS_TEAM.search(question):
        return always | _PROPER
    if _ASKS_ORG.search(question):
        return always | {"org", "place", "dictname"}
    if _ASKS_PLACE.search(question):
        return {"place", "dictname", "title", "quote", "latin"}
    if _ASKS_PERSON.search(question):
        return {"person", "dictname", "title", "quote", "latin"}
    return always | _PROPER


def _family(value: _Value, sorts: set[str]) -> str:
    """Names of these sorts stand for the same slot: the proper names the question accepts."""
    sort = value.key[0]
    return "proper" if sort in _PROPER and sort in sorts else sort


# A place before one of these is a nationality, not a value ("澳大利亚选手", "美国总统").
_PEOPLE_NOUNS = frozenset({"选手", "球员", "名将", "队员", "运动员", "籍", "人", "总统", "政府", "官方", "方面", "媒体", "网友",
                           "球迷", "主帅", "教练", "代表团", "队友", "外长", "总理", "首相", "官员"})


def _question_words(question: str) -> list[str]:
    from retrieval.embedded_store import _segmenter

    words: list[str] = []
    for word in _segmenter().lcut(_normalized(question), HMM=False):
        if word == "年" and words and re.fullmatch(r"(?:19|20)\d{2}", words[-1]):
            words[-1] += "年"  # "2022" + "年"
        elif word.strip():
            words.append(word)
    return words


def _slot_words(question: str, anchors: list[str]) -> list[str]:
    """The question's words a value may belong to, and their parts ("发售日期" -> "发售", "日期"):
    two characters or more, no interrogative, no anchor (a year, place or title), no named entity."""
    from retrieval.embedded_store import _segmenter

    tags = _dictionary_tags()
    out = []
    for word in _segmenter().cut_for_search(_normalized(question), HMM=False):
        if len(word) < 2 or word in _SLOT_STOP or _ASKING.match(word) or word in anchors:
            continue
        if tags.get(word) in _NAME_TAGS or re.search(r"\d", word) or not _CJK_RUN.fullmatch(word):
            continue
        out.append(word)
    return list(dict.fromkeys(out))


def _parts(question: str, anchors: list[str]) -> list[frozenset[str]] | None:
    """What the question asks about side by side, each as the words that name it: "香港和澳门",
    "2022年和2023年", "开始和结束", or two asks ("…是什么类型的电影？它的导演是谁？"). A value is
    only compared with values about the same part. ``None``: the parts cannot be told apart."""
    from .answering import _ASK_ZH

    words = _question_words(question)
    parts: list[frozenset[str]] = []
    for i, word in enumerate(words):
        if word in _JOINERS and 0 < i < len(words) - 1:
            left, right = words[i - 1], words[i + 1]
            if len(left) == 1 and i >= 2:
                left = words[i - 2] + left  # "冬" + "残奥会"
            if len(right) == 1 and i + 2 < len(words):
                right += words[i + 2]
            for side in (left, right):
                side = side[:-1] if re.fullmatch(r"(?:19|20)\d{2}年", side) else side
                if len(side) >= 2 and side not in _SLOT_STOP and not _ASKING.match(side):
                    parts.append(frozenset({side}))
    years = [y for y in _ANCHOR_YEAR.findall(question) if re.fullmatch(r"\d{4}", y)]
    if len(set(years)) >= 2:
        parts += [frozenset({y}) for y in dict.fromkeys(years)]
    if parts:
        parts = list(dict.fromkeys(parts))
        return parts if len(parts) >= 2 else None
    asks = [a for a in _ASK_SPLIT.split(question) if _ASK_ZH.search(a)]
    if len(asks) >= 2:
        slots = [set(_slot_words(a, anchors)) for a in asks]
        own = [frozenset(s - set().union(*(o for j, o in enumerate(slots) if j != i))) for i, s in enumerate(slots)]
        return own if all(own) else None
    return None if "分别" in question else []


def _names_part(word: str, text: str) -> bool:
    if re.fullmatch(r"\d{4}", word):  # a year, not the year of a page date ("Feb11,2022")
        return bool(re.search(rf"(?<![\d,，]){word}(?!\d)", text))
    if re.search(r"[A-Za-z]", word):  # "GPT4" names "GPT-4"
        return re.sub(r"[\s\-_]", "", word).lower() in re.sub(r"[\s\-_]", "", text).lower()
    return word in text


def _which(parts: list[frozenset[str]], *texts: str) -> set[int]:
    """The parts the first of ``texts`` that names any names (a clause, then its sentence, ...)."""
    for text in texts:
        named = {k for k, words in enumerate(parts) if any(_names_part(w, text) for w in words)}
        if named:
            return named
    return set()


def _dot_name(head: str, tail: str) -> tuple[str, str]:
    """A transliterated name around its dots, without the words before and after it
    ("设计总监杰夫·威尔科克斯" -> "杰夫·威尔科克斯", "布莱恩·辛格导演的" -> "布莱恩·辛格")."""
    from retrieval.embedded_store import _segmenter

    tags = _dictionary_tags()
    kept = ""
    for word in reversed(_segmenter().lcut(head, HMM=False)):
        if not _CJK_RUN.fullmatch(word):
            kept = word + kept
        elif len(word) == 1 and word not in _NAME_CUT and word not in _EDGE and len(kept) < 4:
            kept = word + kept
        elif len(word) > 1 and tags.get(word) in _NAME_TAGS and not kept.strip("·"):
            kept = word + kept
            break
        else:
            break
    last = re.split(r"[·•・．]", tail)[-1]
    cut = next((i for i, ch in enumerate(last) if ch in _NAME_CUT), len(last))
    return kept, tail[: len(tail) - len(last) + min(cut, 5)]


def _names(clause: str) -> list[_Value]:
    """Names in a clause, with their sort: dictionary places, people and organizations (adjacent
    ones of one sort joined: "江苏苏宁"; a nationality is left out: "澳大利亚选手"), "·" names,
    《titles》, “quoted names”,
    Latin names, model names ("骁龙845"), and unknown Chinese names (a surname followed by one or
    two characters the dictionary-only segmentation leaves unknown: "郭帆", "林郑月娥", "蒋万安")."""
    from retrieval.embedded_store import _segmenter

    jieba = _segmenter()
    tags = _dictionary_tags()
    found: list[_Value] = []

    def add(text: str, start: int, end: int, sort: str) -> None:
        if len(text) >= 2 and not any(v.start < end and start < v.end for v in found):
            found.append(_Value("name", (sort,), text, start, end))

    for pattern, sort in ((_TITLE, "title"), (_QUOTE, "quote")):
        for match in pattern.finditer(clause):
            add(match.group(1), match.start(), match.end(), sort)
    for match in _DOT_NAME.finditer(clause):
        head, tail = _dot_name(match.group(1), match.group(2))
        if head.strip("·") and tail.strip("·•・．"):
            start = match.end(1) - len(head)
            add(head + tail, start, start + len(head + tail), "person")
    for match in _LATIN_NAME.finditer(clause):
        add(match.group(0) + (clause[match.end()] if len(match.group(0)) == 1 else ""), *match.span(), "latin")
    for match in _MODEL_CJK.finditer(clause):
        prefix, digits = match.group(1), match.group(2)
        if not (set(prefix) & (_EDGE | _NAME_CUT | set("第达约共近超逾"))) and not _YEAR.fullmatch(digits):
            add(prefix + digits, *match.span(), "model")
    tokens = list(jieba.tokenize(clause, HMM=False))
    proper: list[_Value] = []
    for n, (word, start, end) in enumerate(tokens):
        tag = tags.get(word) if len(word) >= 2 and word not in _NOT_NAMES else None
        following = tokens[n + 1][0] if n + 1 < len(tokens) else ""
        if tag in _NAME_TAGS and len(following) == 1 and _CJK_RUN.fullmatch(following) and not (
                following in _EDGE or following in _NAME_CUT or following in _UNITS or following in "队市省区县州镇村胜败赢输外内间里旁边"):
            continue  # a longer unknown name cut short ("阿德" of "阿德恩")
        if tag in _NAME_TAGS:
            sort = "place" if tag in _PLACE_TAGS else "dictname" if tag in _PERSON_TAGS else "org"
            first = n
            while first > 0 and n - first < 2 and len(tokens[first - 1][0]) == 1 and _CJK_RUN.fullmatch(tokens[first - 1][0]) \
                    and tokens[first - 1][0] not in _EDGE and tokens[first - 1][0] not in _NAME_CUT \
                    and tokens[first - 1][0] not in _UNITS and not (proper and proper[-1].end > tokens[first - 1][1]):
                first -= 1
            head = clause[tokens[first][1]:start]
            proper.append(_Value("name", (sort,), head + word, tokens[first][1], end, word))
            continue
        before = tokens[n - 1][0] if n else ""
        if len(word) == 1 and word in _SURNAMES and (not _CJK_RUN.fullmatch(before) or len(before) > 1 or before in _EDGE):
            name, last = word, end
            for following, _, following_end in tokens[n + 1:]:
                if not _CJK_RUN.fullmatch(following) or following in _EDGE or set(following) & _NAME_CUT:
                    break
                if len(following) > 1 and (tags.get(following) in _NAME_TAGS
                                           or getattr(jieba, "dt", jieba).FREQ.get(following, 0) > _GIVEN_NAME_FREQ):
                    break
                if len(name) + len(following) > (4 if len(following) == 1 and len(name) >= 3 else 3):
                    break
                name, last = name + following, following_end
            if len(name) >= 2:
                proper.append(_Value("name", ("person",), name, start, last, name))
    joined: list[_Value] = []
    for value in proper:  # "江苏" + "苏宁" -> "江苏苏宁", "湖北省" + "武汉市" -> "湖北省武汉市"
        if joined and joined[-1].end == value.start and joined[-1].key == value.key:
            joined[-1].text, joined[-1].end, joined[-1].last = joined[-1].text + value.text, value.end, value.last
        else:
            joined.append(value)
    for value in joined:
        after = next((w for w, s, _ in tokens if s == value.end), "")
        if value.key[0] == "place" and after in _PEOPLE_NOUNS:
            continue  # "澳大利亚选手": a nationality
        if not any(v.start < value.end and value.start < v.end for v in found):
            found.append(value)
    return found


def _values(text: str, kinds: set[str], sorts: set[str], listed_too: bool = False) -> list[_Value]:
    out: list[_Value] = []

    def free(start: int, end: int) -> bool:
        return not any(v.start < end and start < v.end for v in out)

    if "time" in kinds:
        for match in _CONFLICT_DATE.finditer(text):
            year, month, day = match.group(1), int(match.group(2)), match.group(3)
            if text[match.end():match.end() + 1] == "电":
                continue  # a news dateline ("新华社酒泉5月29日电"): when it was reported
            if text[max(0, match.start() - 1):match.start()] in ("-", "—", "–", "~", "至", "到"):
                continue  # the end of a range ("2008年05月12日-2008年12月31日")
            if 1 <= month <= 12 and (day is None or 1 <= int(day) <= 31):
                out.append(_Value("time", (year, month, int(day) if day else None), match.group(0), *match.span()))
        for match in _CONFLICT_YEAR.finditer(text):
            if free(*match.span()):
                out.append(_Value("time", (match.group(1), None, None), match.group(0), *match.span()))
    if "number" in kinds:
        for match in _CONFLICT_NUMBER.finditer(text):
            if free(*match.span()):
                out.append(_Value("number", (match.group(2), float(match.group(1))), match.group(0), *match.span()))
    if "name" in kinds:
        for value in _names(text):
            before = text[max(0, value.start - 8):value.start]
            listed = before.endswith("、") or text[value.end:value.end + 1] in ("、", "等") or (
                before[-1:] in ("和", "与", "及") and "、" in before)
            if value.key[0] in sorts and free(value.start, value.end) and (listed_too or not listed):
                out.append(value)  # a name in an enumeration ("杜琪峰、徐克等") is one of several
    return out


def _plain_name(text: str) -> str:
    text = text.lower()
    for suffix in _ANCHOR_SUFFIX:
        if text.endswith(suffix) and len(text) - len(suffix) >= 2:
            return text[: -len(suffix)]
    return text


def _same_value(a: _Value, b: _Value, sorts: frozenset[str] | set[str] = frozenset()) -> bool | None:
    """True / False, or None when the two cannot be compared (another kind, unit or sort of
    name, a year against a date)."""
    if a.kind != b.kind:
        return None
    if a.kind == "number":
        return a.key == b.key if a.key[0] == b.key[0] else None
    if a.kind == "time":
        if (a.key[1] is None) != (b.key[1] is None):
            return None
        pairs = [(x, y) for x, y in zip(a.key, b.key) if x is not None and y is not None]
        return all(x == y for x, y in pairs) if pairs else None
    ta, tb = _plain_name(a.text), _plain_name(b.text)
    if ta in tb or tb in ta:
        return True
    return False if _family(a, sorts) == _family(b, sorts) else None


def _amount_in(amount: float, text: str) -> bool:
    """The number as written in any of its forms ("1.2" / "1.20", "43214.85"), not part of a
    longer number ("12" is not in "12.5" or "112")."""
    written = f"{amount:.6f}".rstrip("0").rstrip(".")
    whole, _, fraction = written.partition(".")
    tail = rf"\.{fraction}0*" if fraction else r"(?:\.0+)?"
    return bool(re.search(rf"(?<![\d.]){whole}{tail}(?!\.?\d)", text))


def _present(value: _Value, text: str) -> bool:
    """Does the (normalized) passage carry this value anywhere?"""
    if value.kind == "number":
        return _amount_in(value.key[1], text)
    if value.kind == "time":
        year, month, day = value.key
        if month is None:
            return f"{year}年" in text
        core = rf"(?<!\d){month}月" + (rf"{day}(?!\d)" if day else "")
        return bool(re.search(core, text)) and (year is None or year in text)
    lowered = text.lower()
    return _plain_name(value.text) in lowered or (len(value.last) >= 2 and _plain_name(value.last) in lowered)


def _numbered(text: str) -> set[tuple[str, str]]:
    """("第", n) for an ordinal of any kind ("第四任", "第1位": one ranking), else (unit, n) ("十六号")."""
    return {(ordinal or unit, number if number.isdigit() else _cn_value(number))
            for ordinal, number, unit in _NUMBERED.findall(text)}


def _other_subject(sentence: str, start: int, end: int, years: set[str], numbered: set[tuple[str, str]]) -> bool:
    """The sentence is about another year or another numbered subject than the question's
    ("2019年出生人口1465万人" for a question about 2018, "神舟17号" for one about 神舟十六号)."""
    rest = sentence[:start] + "，" + sentence[end:]
    if years:
        named = set(re.findall(r"(?<!\d)((?:19|20)\d{2})(?=年)", rest))
        if named and not (named & years):
            return True
    for unit in {u for u, _ in numbered}:
        named = {n for u, n in _numbered(rest) if u == unit}
        if named and not (named & {n for u, n in numbered if u == unit}):
            return True
    return False


def _slot_values(text: str, kinds: set[str], sorts: set[str], slots: list[str], years: set[str],
                 numbered: set[tuple[str, str]], parts: list[frozenset[str]], part: int | None) -> list[_Value]:
    """Values in clauses that hold a slot word of the question, not about another year or
    numbered subject than the question's, and (when the question asks about several things side
    by side) about ``part`` alone, as the clause, else its sentence, names it."""
    out: list[_Value] = []
    for sentence in _SENTENCE.split(text):
        position = 0
        clauses = _CLAUSE.split(sentence)
        for n, clause in enumerate(clauses):
            base = sentence.find(clause, position)
            position = base + len(clause)
            if not clause or not any(w in clause for w in slots):
                continue
            if part is not None and _which(parts, clause, sentence) != {part}:
                continue
            for value in _values(clause, kinds, sorts):
                if value.kind == "time" and years and value.key[0] and value.key[0] not in years:
                    continue  # a date of another year than the question's
                if not _other_subject(sentence, base + value.start, base + value.end, years, numbered):
                    value.lead = "，".join(clauses[max(0, n - 2):n + 1])
                    out.append(value)
    return out


def _leads(value: _Value, text: str) -> list[str]:
    """Each clause of ``text`` that carries ``value``, with the two clauses before it."""
    out = []
    for sentence in _SENTENCE.split(text):
        clauses = _CLAUSE.split(sentence)
        out += ["，".join(clauses[max(0, n - 2):n + 1]) for n, clause in enumerate(clauses) if _present(value, clause)]
    return out


def _answer_sentences(answer: str) -> list[str]:
    """The answer's sentences without citation marks and the notes the code put before it."""
    out = []
    for sentence in _SENTENCE.split(_normalized(answer)):
        sentence = sentence.strip(" -*#>：:")
        if len(sentence) >= 4 and not sentence.startswith(("注意", "资料里写的是")):
            out.append(sentence)
    return out


def conflicting_values(question: str, answer: str, blocks: list[dict]) -> list[Conflict]:
    """The answer's value against a different value for the same fact in another passage.

    The answer's value: in its first sentence (for a question about several things side by side,
    "香港" / "澳门", its first sentence about each), of the kind the question asks for (a number
    with its unit, a date or year, a place, a person, another name), not a word of the question,
    and written in a passage the answer cites. The other passage never mentions it, names the
    question's anchors (year, place, title) that the cited one names, and, in a clause with a word
    of the question that is not about another year, gives exactly one value of the same kind that
    the cited passages never mention. A question asking for a list, and an answer citing nothing
    (a refusal), have nothing to check.
    """
    from .answering import classify_question, evidence_for

    if not question or _ASKS_LIST.search(question) or classify_question(question) == "explain":
        return []  # a list, or how / why: not one value per fact
    cited: set[int] = set()
    for _, refs in cited_claims(answer or ""):
        cited.update(evidence_for(refs, blocks)[0])
    anchors = question_anchors(question)
    slots = _slot_words(question, anchors)
    parts = _parts(question, anchors)
    if not cited or not slots or parts is None:
        return []
    kinds, sorts = _value_kinds(question), _name_sorts(question)
    years = {a for a in anchors if re.fullmatch(r"(?:19|20)\d{2}", a)}
    numbered = _numbered(_normalized(question))
    asked = _normalized(question).lower()
    texts = [_normalized(str(b.get("text", ""))) for b in blocks]

    def own_word(value: _Value) -> bool:
        return value.kind == "name" and (_plain_name(value.text) in asked or any(value.text in w for w in slots))

    statements: list[tuple[int | None, str]] = []
    sentences = _answer_sentences(answer)
    if not parts:
        statements = [(None, sentences[0])] if sentences else []
    else:
        for sentence in sentences:
            named = _which(parts, sentence)
            if len(named) == 1 and next(iter(named)) not in {p for p, _ in statements}:
                statements.append((next(iter(named)), sentence))
    conflicts: list[Conflict] = []
    for part, sentence in statements:
        for value in (v for v in _slot_values(sentence, kinds, sorts, slots, years, numbered, parts, part) if not own_word(v)):
            source = [i for i in sorted(cited) if _present(value, texts[i])]
            if not source:
                continue  # not taken from a cited passage
            required = [a for a in anchors if any(_anchor_present(a, texts[i]) for i in source)]
            # A running total where the cited passage gives the value.
            running = value.kind == "number" and any(
                _RUNNING_TOTAL.search(lead) for i in source for lead in _leads(value, texts[i]))
            found = None
            for other, text in enumerate(texts):
                if other in source or _present(value, text) or not all(_anchor_present(a, text) for a in required):
                    continue
                rivals: list[_Value] = []
                for rival in _slot_values(text, kinds, sorts, slots, years, numbered, parts, part):
                    if own_word(rival) or _same_value(rival, value, sorts) is not False:
                        continue
                    if rival.kind == "number" and (_RUNNING_TOTAL.search(rival.lead) or running):
                        continue  # running totals of different dates
                    if any(_present(rival, texts[i]) for i in source):
                        continue  # the cited passage mentions it too: another fact, not a rival
                    if not any(_same_value(rival, seen, sorts) for seen in rivals):
                        rivals.append(rival)
                if len(rivals) == 1:
                    found = Conflict(((value.text, source[0]), (rivals[0].text, other)))
                    break
            if found:
                conflicts.append(found)
                break
    return conflicts


# ---------------------------------------------------------------- unanswered constraints
# The library holds passages on the question's topic but not its answer (RGB reject, CRUD news
# missing from the library). The weak answer model then answers about a similar subject (another
# year, place or edition) or from its own knowledge while citing an on-topic passage, and the fact
# check lets it through (evaluation/generation_v1/REPORT.md: RGB reject 0.15-0.24). Telling it
# not to in the prompt was measured and did not help. Here the code compares what the question
# pins down (its years and months, numbered subjects such as "第七届" or "神舟十四号", places and
# names) and the answer's own value with the evidence; when the evidence lacks one, the answer
# opens with one plain line saying so. Retrieval and the answer are left as they are
# (evaluation/rejection_v1/REPORT.md).
# A year: not inside a longer number or a decimal ("3.2021"), not an amount ("2022万元"); a dated
# page ("Feb12,2022...", "2022.02.14") counts.
_Q_YEAR = re.compile(r"(?<!\d)(?<!\d\.)((?:19|20)\d{2})(?![\d%‰万亿元人米个家名件台辆吨])")
_Q_MONTH = re.compile(r"(?<!\d)((?:19|20)\d{2})年(\d{1,2})月")
# A season ("2019-2020赛季", "2020-21"): its first year stands for it.
_SEASON = re.compile(r"((?:19|20)\d{2})[-—–/](?:(?:19|20)?\d{2})(?!\d)")
_SELF_REFUSAL = re.compile(r"资料里没有找到|没有找到能回答|资料里的内容不足以|当前知识库没有足够依据")
_COMPARED_YEAR = re.compile(r"(?:较|比|相比|相较|对比|与|超过|高于|低于)(?:19|20)\d{2}年?|(?:19|20)\d{2}年?(?:同期|全年同期)")
_CJK = re.compile(r"[㐀-鿿]")
# How / why: the answer is an explanation, not one value ("如何通过…提升…效果" is classified "multi").
_HOW_WHY = re.compile(r"如何|怎么|怎样|为什么|为何|为啥")


@dataclass(frozen=True)
class Gap:
    kind: str  # "year" | "month" | "numbered" | "place" | "name" | "value"
    asked: str  # as the question writes it ("2022年", "克什米尔地区"); for "value", the answer's value
    nearest: str = ""  # what the evidence has instead ("2023年", "拉贾斯坦邦")

    def label(self) -> str:
        return self.asked if self.kind in ("year", "month") else f"“{self.asked}”"

    def nearest_label(self) -> str:
        return self.nearest if self.kind in ("year", "month") else f"“{self.nearest}”"


@dataclass(frozen=True)
class Unanswered:
    gaps: tuple[Gap, ...]
    every: bool  # every constraint the question names is missing from the evidence

    def line(self) -> str:
        if self.gaps[0].kind == "attribute":
            gap = self.gaps[0]
            return f"资料里没有找到这个问题问的{gap.asked}，下面回答里的“{gap.nearest}”在资料里说的不是{gap.asked}，请点开出处核对。"
        constraints = [gap for gap in self.gaps if gap.kind != "value"]
        if constraints:
            first = constraints[0]
            asked = "、".join(gap.label() for gap in constraints[:2])
            about = "" if first.kind in ("year", "month") else "关于"
            near = next((gap for gap in constraints if gap.nearest), None)
            if near is not None:
                rest = f"下面的回答讲的可能是{near.nearest_label()}的情况"
            elif first.kind in ("year", "month"):
                rest = "下面的回答讲的可能是别的年份"
            else:
                rest = "下面的回答讲的可能是别的对象"
            return f"资料里没有找到{about}{asked}的内容，{rest}，请点开出处核对。"
        return f"资料里没有找到这个问题的答案，下面回答里的“{self.gaps[0].asked}”在资料中没有出现，请核对。"

    def refusal(self) -> str:
        asked = "、".join(gap.label() for gap in self.gaps if gap.kind != "value")
        return f"资料里没有找到{asked}的内容，所以没有作答。可以换个说法再问，或者把相关的文件加进资料库。"


def _years(text: str) -> list[str]:
    return list(dict.fromkeys(_Q_YEAR.findall(_SEASON.sub(r"\1", text))))


def _constraints(question: str) -> list[Gap]:
    """What the question pins down: its years ("2022年", "2022清明档", a season by its first year),
    year-months, numbered subjects ("第七届", "14号"), and the places, people and 《titles》 of
    question_anchors (not organizations: "波兰政府", "香港立法会" are mostly a place and a common
    word; not the words the dictionary wrongly tags as names: "世锦赛")."""
    asked = _normalized(question)
    out: list[Gap] = []
    for year in _years(asked):
        out.append(Gap("year", f"{year}年"))
    for year, month in _Q_MONTH.findall(asked):
        out.append(Gap("month", f"{year}年{int(month)}月"))
    for match in _NUMBERED.finditer(asked):
        out.append(Gap("numbered", match.group(0)))
    tags = _dictionary_tags()
    titles = set(_QUOTED.findall(question))
    for anchor in question_anchors(question):
        if _ANCHOR_YEAR.fullmatch(anchor) or anchor in _NOT_NAMES or re.search(r"\d", anchor):
            continue
        if anchor in titles:
            out.append(Gap("name", anchor))
        elif tags.get(anchor) in _PLACE_TAGS:
            out.append(Gap("place", anchor))
        elif tags.get(anchor) in _PERSON_TAGS:
            out.append(Gap("name", anchor))
    for match in _COMPANY.finditer(question):
        if 2 <= len(_company_core(match.group(0))) <= 6:
            out.append(Gap("org", match.group(0)))
    return list(dict.fromkeys(out))


# A company the question names ("杭州麦刻物联科技有限公司"), known by its own word ("麦刻"): the
# name without the place before it and the trade and legal form after it.
_COMPANY = re.compile(r"[㐀-鿿A-Za-z]{2,24}?(?:股份有限公司|有限责任公司|有限公司|集团公司|集团)")
_COMPANY_TAIL = re.compile(
    r"(?:股份有限公司|有限责任公司|有限公司|集团公司|集团|控股|科技|物联|网络|信息|技术|文化|传媒|实业|投资|发展|贸易|电子商务|电子|"
    r"智能|数字|管理|咨询|服务|工程|建设|新能源|生物|医药|食品|物流|供应链|股份|国际|中国)+$")


def _company_core(name: str) -> str:
    from retrieval.embedded_store import _segmenter

    tags = _dictionary_tags()
    words = _segmenter().lcut(name, HMM=False)
    while words and (tags.get(words[0]) in _PLACE_TAGS or words[0] in ("中国", "全国")):
        words = words[1:]
    return _COMPANY_TAIL.sub("", "".join(words))


# What the question asks about its subject, and the words a passage uses for it: the sentence
# that holds the answer's value has to say one of them ("徐磊" in an interview with the cast is
# not "《四海》的导演"; "香港GDP只是大陆的2%" is not a growth rate).
_ATTRIBUTES = (
    (re.compile(r"导演|执导"), re.compile(r"导演|执导|导筒|掌镜|编导")),
    (re.compile(r"主演|演员|谁演|(?<!上)演的是"), re.compile(r"主演|饰演|饰|出演|领衔|演员|扮演|主角|参演|加盟|联袂|携手|搭档")),
    (re.compile(r"编剧"), re.compile(r"编剧|执笔|剧本")),
    (re.compile(r"冠军|金牌|得主|夺冠"), re.compile(r"冠军|夺冠|夺得|摘得|摘金|金牌|桂冠|折桂|捧杯|加冕|卫冕|称王|登顶|击败|战胜|胜出|问鼎|获胜|赢得|\d+[-:：比]\d+")),
    (re.compile(r"死亡|遇难|死了|丧生"), re.compile(r"死亡|遇难|丧生|罹难|身亡|遇害|丧命|殒命|死")),
    (re.compile(r"增长|增速|增幅|涨了|上涨|涨幅"), re.compile(r"增长|增速|增幅|增加|上升|提高|上涨|涨|增|下降|下滑|回落|减少|下跌|降")),
    (re.compile(r"票房"), re.compile(r"票房")),
    (re.compile(r"创立|创办|创建|创始"), re.compile(r"创立|创办|创建|成立|创始|发起|组建|建立|创")),
)


_ATTRIBUTE_WINDOW = 1


def _asked_attribute(question: str) -> re.Pattern | None:
    asked = [words for cue, words in _ATTRIBUTES if cue.search(question)]
    return asked[0] if len(asked) == 1 else None


def _constraint_in(gap: Gap, text: str) -> bool:
    """Is the constraint named in this (normalized) text?"""
    if gap.kind == "year":
        return gap.asked[:4] in _years(text)
    if gap.kind == "month":
        return gap.asked[:4] in text and bool(re.search(rf"(?<!\d){gap.asked[5:-1]}月", text))
    if gap.kind == "numbered":
        return bool(_numbered(gap.asked) & _numbered(text))
    if gap.kind == "org":
        return _company_core(gap.asked).lower() in text.lower()
    return _anchor_present(_normalized(gap.asked), text)  # "三年行动计划" is written "3年行动计划"


def _nearest_year(year: str, texts: list[str]) -> str:
    seen = [y for text in texts for y in _years(text) if y != year]
    if not seen:
        return ""
    return f"{min(seen, key=lambda y: (abs(int(y) - int(year)), -seen.count(y)))}年"


def _found(value: _Value, text: str) -> bool:
    """Is the value in this (normalized) text? More lenient than ``_present``: a date by its
    month and day (another year is the subject check's job), a "·" name by its last part
    ("德约科维奇" for "诺瓦克·德约科维奇"), a name read with one character too many ("王暖暖坠"
    for "王暖暖")."""
    if value.kind == "time" and value.key[1]:
        _, month, day = value.key
        text = re.sub(r"(\d{1,2}[:：]\d{2})(?=\d)", r"\1 ", text)  # "14:373月10日": a clock time first
        return bool(re.search(rf"(?<!\d){month}月" + (rf"{day}(?!\d)" if day else ""), text))
    if _present(value, text):
        return True
    if value.kind == "name":
        lowered = text.lower()
        last = re.split(r"[·•・．]", value.text)[-1]
        if len(last) >= 2 and last != value.text and last.lower() in lowered:
            return True
        return len(value.text) >= 3 and bool(_CJK_RUN.fullmatch(value.text)) and value.text[:-1] in text
    return False


def _occurrence_subject(value: _Value, text: str, clauses: list[str], n: int, constraints: list[Gap]) -> Gap | None:
    """The question's constraint this occurrence of the value is about something else than, with
    what it is about instead (``Gap(kind, asked, nearest)``), or None (about the question's subject,
    or not saying). Looks at the value's clause and the two before it, then the sentence up to it;
    a place only counts as another place when the passage never names the question's (a district
    of the city asked is still the city)."""
    lead = "，".join(clauses[max(0, n - 2):n + 1])
    lead_rest = lead.replace(value.text, "，")
    head = "，".join(clauses[:n + 1]).replace(value.text, "，")
    # A question about several years or numbers ("2022和2023的举办地"): any of them is the subject.
    years = {gap.asked[:4] for gap in constraints if gap.kind == "year"}
    numbered = set().union(*(_numbered(gap.asked) for gap in constraints if gap.kind == "numbered"))
    for gap in constraints:
        if gap.kind == "year":
            # A year the figure is compared with ("较2022年", "2022年同期") is not its year.
            named = _years(_COMPARED_YEAR.sub("，", lead_rest)) or _years(_COMPARED_YEAR.sub("，", head))
            if named and not (set(named) & years):
                return Gap("year", gap.asked, f"{min(named, key=lambda y: abs(int(y) - int(gap.asked[:4])))}年")
        elif gap.kind == "numbered":
            asked = numbered
            units = {unit for unit, _ in asked}
            named = {(u, k) for u, k in _numbered(lead_rest) if u in units}
            if named and not (named & asked):
                other = next((m.group(0) for m in _NUMBERED.finditer(lead_rest) if _numbered(m.group(0)) & named), "")
                return Gap("numbered", gap.asked, other)
        elif gap.kind == "place" and not _anchor_present(gap.asked, text):
            places = [v.text for v in _names(lead_rest) if v.key[0] == "place" and v.text not in _ANCHOR_STOP
                      and not _anchor_present(v.text, gap.asked) and not _anchor_present(gap.asked, v.text)]
            if places:
                return Gap("place", gap.asked, places[-1])
    return None


def _answer_values(question: str, sentence: str) -> list[_Value]:
    """The values the answer's first sentence gives of the kind the question asks for, without
    the question's own words ("2022年", "北京", "吴易昺" read as "吴易昺均") and with names listed
    together ("杨紫和邓伦")."""
    asked = _normalized(question).lower()
    anchors = question_anchors(question)
    slots = _slot_words(question, anchors)
    years = set(_years(asked))
    listing = bool(_ASKS_LIST.search(question))
    from retrieval.embedded_store import _segmenter

    jieba = _segmenter()
    frequency = getattr(jieba, "dt", jieba).FREQ
    out = []
    for value in _values(sentence, _value_kinds(question), _name_sorts(question), listed_too=True):
        text = _normalized(value.text).lower()
        if text in asked or (value.kind == "time" and value.key[1] is None and value.key[0] in years):
            continue
        if value.kind == "name" and (_plain_name(value.text) in asked or any(value.text in w for w in slots)
                                     or (len(text) >= 3 and text[:3] in asked) or len(text) > 24):
            continue
        # The dictionary calls common words people or organizations ("荣誉", "新进展"): a frequent
        # word, or an "organization" among the things a list question asks for, is not a value.
        if value.key[0] in ("dictname", "org") and (frequency.get(value.text, 0) > _GIVEN_NAME_FREQ
                                                    or (listing and value.key[0] == "org")):
            continue
        out.append(value)
    return out


def _chinese(question: str) -> bool:
    letters = re.findall(r"[A-Za-z]+|[㐀-鿿]", question)
    cjk = sum(1 for part in letters if _CJK.fullmatch(part))
    return cjk >= 4 and cjk >= len(letters) / 2


def unanswered(question: str, answer: str, blocks: list[dict], scope: str = "value") -> Unanswered | None:
    """What the question asks about that the evidence does not have, as one line for the answer.

    ``scope``:
    * ``all``: a year, month, numbered subject, place, person or 《title》 of the question that no
      passage names ("2021年" asked, the passages are about 2020 and 2023);
    * ``cited``: the same, in the passages the answer cites;
    * ``value``: ``all``, and the answer's own value (first sentence, of the kind the question
      asks for): in no passage at all (the model's own knowledge with a citation added), or only
      in sentences about another year, numbered subject or place than the question's ("2023年清明档
      … 1.92亿元" for a question about 2022). Not for how / why questions.

    An answer that cites nothing (a refusal) or opens by saying it found nothing has nothing to
    check, nor has a question that is not mainly Chinese (the rules read Chinese).
    """
    if not question or not answer or scope not in ("all", "cited", "value") or not _chinese(question):
        return None
    from .answering import classify_question, evidence_for

    cited: set[int] = set()
    for _, refs in cited_claims(answer):
        cited.update(evidence_for(refs, blocks)[0])
    sentences = _answer_sentences(answer)
    if not cited or not sentences or _SELF_REFUSAL.search(sentences[0]):
        return None
    texts = [_normalized(str(block.get("text", ""))) for block in blocks]
    pool = [texts[i] for i in sorted(cited)] if scope == "cited" else texts
    constraints = _constraints(question)
    gaps: list[Gap] = []
    for gap in constraints:
        if not any(_constraint_in(gap, text) for text in pool):
            nearest = _nearest_year(gap.asked[:4], [texts[i] for i in sorted(cited)]) if gap.kind == "year" else ""
            gaps.append(Gap(gap.kind, gap.asked, nearest))
    every = bool(constraints) and len(gaps) == len(constraints)
    if scope == "value" and not gaps and classify_question(question) != "explain" and not _HOW_WHY.search(question):
        values = _answer_values(question, sentences[0])
        found = [v for v in values if any(_found(v, text) for text in texts)]
        if values and not found:
            gaps.append(Gap("value", values[0].text))
        elif found:
            # Every sentence that holds the answer's value is about another year, numbered subject
            # or place, or (when the question asks for a role or a measure) never says it.
            attribute = _asked_attribute(question)
            cue = next((c.search(question).group(0) for c, w in _ATTRIBUTES if w is attribute), "")
            others: list[Gap | None] = []
            for value in found:
                for text in texts:
                    split = _SENTENCE.split(text)
                    for k, sentence in enumerate(split):
                        clauses = _CLAUSE.split(sentence)
                        around = "。".join(split[max(0, k - _ATTRIBUTE_WINDOW):k + _ATTRIBUTE_WINDOW + 1])
                        for n, clause in enumerate(clauses):
                            if not _found(value, clause):
                                continue
                            other = _occurrence_subject(value, text, clauses, n, constraints)
                            if other is None and attribute is not None and not attribute.search(around.replace(value.text, "，")):
                                other = Gap("attribute", cue, value.text)
                            others.append(other)
            if others and all(others):
                gaps.append(next((g for g in others if g.kind != "attribute"), others[0]))
    return Unanswered(tuple(gaps), every) if gaps else None
