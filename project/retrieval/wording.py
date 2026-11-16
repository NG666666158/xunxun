"""Everyday wording to the words documents use, without a model call.

"一般的创新项目申报能给多少补贴" asks about 补贴; the guide that answers it says "单个项目资助
资金不超过…万元" under "支持强度与方式". Neither BM25 nor the vector search ranks that passage
high for the question as asked, so it never reaches the reranker
(evaluation/recall_wording_v1/REPORT.md).

``expand`` finds the subject groups a question touches (money, applying, deadlines, who may
apply, what to hand in) and returns the question followed by the group's other words. The
retriever searches that text beside the question itself and adds its new candidates to the
pool; the reranker still scores every candidate against the question as asked.

The table is small on purpose: funding and application papers. A question outside it (no word
for money or applying in it) is not expanded at all.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

# Words for the same thing in policy papers, application guides and measures. The first
# group is the one a question must touch (or the next) for any expansion to happen.
FUNDING = ("补贴", "资助", "奖励", "扶持", "补助", "经费", "资金", "支持强度", "额度", "津贴", "奖补", "贴息")
# Added when the question asks how much: how the papers state an amount.
AMOUNT = ("标准", "万元", "最高", "不超过", "比例")
APPLYING = ("申报", "申请", "立项", "评审", "受理")
DEADLINE = ("截止", "期限", "受理时间", "申报时间", "时限", "逾期")
ELIGIBLE = ("条件", "对象", "资格", "范围", "主体", "要求")
MATERIALS = ("材料", "资料", "清单", "附件", "提交")

_FUNDING_CUE = re.compile(
    "|".join(FUNDING) + r"|报销|多少钱|(?:拿|领|给|补|贷|抵|发)(?:到|了)?多少|补(?:几成|多少)|有没有钱|给钱"
)
_HOW_MUCH = re.compile(r"多少|几成|几万|比例|上限|封顶|最高|最多")
_APPLYING_CUE = re.compile("|".join(APPLYING) + r"|报名|(?:能|可以|怎么|如何|去)报")
_DEADLINE_CUE = re.compile(r"截止|截至|期限|最晚|最迟|逾期|过期|几号|哪天|哪一天|什么时候|何时|几月|多久")
_ELIGIBLE_CUE = re.compile(r"条件|资格|门槛|要求|对象|谁(?:能|可以|可)|什么人|哪些(?:人|单位|企业|主体)|能不能|可以.{0,6}吗")
_MATERIALS_CUE = re.compile(r"材料|资料|附件|清单|要交|要准备|要提供|提交什么|交什么|交哪些")
_CJK = re.compile(r"[㐀-鿿]")


@dataclass(frozen=True)
class Expansion:
    groups: tuple[str, ...]
    terms: tuple[str, ...]  # the words added (none of them is in the question)
    text: str  # the question followed by them: what is searched beside the question

    def to_dict(self) -> dict:
        return {"groups": list(self.groups), "terms": list(self.terms)}


def expand(question: str, limit: int = 18) -> Expansion | None:
    """The question with its subject groups' other words, or None when it touches none.

    Only questions about money or applying are expanded; within those, a deadline,
    eligibility or materials ask adds that group's words too.
    """
    text = (question or "").strip()
    if not text or len(text) > 200 or not _CJK.search(text):
        return None
    funding = bool(_FUNDING_CUE.search(text))
    applying = bool(_APPLYING_CUE.search(text))
    if not (funding or applying):
        return None
    chosen: list[tuple[str, tuple[str, ...]]] = []
    if funding:
        # A "how much" question: the amount words go in before the rarer words for money.
        how_much = bool(_HOW_MUCH.search(text))
        chosen.append(("funding", FUNDING[:8] + AMOUNT[:4] + FUNDING[8:] + AMOUNT[4:] if how_much else FUNDING))
    if applying:
        chosen.append(("applying", APPLYING))
    for name, cue, words in (
        ("deadline", _DEADLINE_CUE, DEADLINE),
        ("eligible", _ELIGIBLE_CUE, ELIGIBLE),
        ("materials", _MATERIALS_CUE, MATERIALS),
    ):
        if cue.search(text):
            chosen.append((name, words))
    terms: list[str] = []
    # One word of each group in turn, so a long first group does not use up the limit.
    for index in range(max(len(words) for _, words in chosen)):
        for _, words in chosen:
            if index < len(words) and words[index] not in text and words[index] not in terms:
                terms.append(words[index])
    terms = terms[:limit]
    if not terms:
        return None
    return Expansion(tuple(name for name, _ in chosen), tuple(terms), f"{text} {' '.join(terms)}")
