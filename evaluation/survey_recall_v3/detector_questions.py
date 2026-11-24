"""Questions for the survey detector, written before it was widened (round three, item 2).

* survey: what several files of the library answer, asked without 一般 / 通常 / 各 / 分别 / 都
  ("哪些企业能申请创新补贴", "创新项目补贴有多少", "有什么补贴可以申请"). ``ask`` is the kind of
  answer each file gives (amount / time / who / '' for a list of programmes); ``group`` the
  synthetic library's group whose guides answer it (recall_wording_v1/corpus_spec.py).
* fact: look-alikes that must stay fact questions: one place, office, programme title, file,
  code or person; "这个 / 这份 / 文件里"; the user's own record; a definition, a how-to, a
  yes / no; questions that are not about programmes at all.

  python evaluation/survey_recall_v3/detector_questions.py   # writes detector.jsonl and wide.jsonl

wide.jsonl: the survey ones with a group, with round two's kind of gold (every guide of the
group; amounts for amount asks), to measure what the survey route does for them.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

V1 = Path(__file__).resolve().parent.parent / "recall_wording_v1"
sys.path.insert(0, str(V1))

SURVEY = [
    ("D01", "哪些企业能申请创新补贴？", "who", "innovation"),
    ("D02", "创新项目补贴有多少？", "amount", "innovation"),
    ("D03", "有什么补贴可以申请？", "", ""),
    ("D04", "什么企业能申请技术改造补贴？", "who", "upgrade"),
    ("D05", "人才补贴能拿多少？", "amount", "talent"),
    ("D06", "技能培训补贴什么时候报？", "time", "training"),
    ("D07", "专利补贴能拿多少钱？", "amount", "ip"),
    ("D08", "商标注册有补贴吗，补多少？", "amount", "ip"),
    ("D09", "创业补贴哪些人可以领？", "who", "startup"),
    ("D10", "创业能拿到哪些扶持？", "", "startup"),
    ("D11", "技改项目最多补多少？", "amount", "upgrade"),
    ("D12", "跨境电商有什么扶持政策？", "", "market"),
    ("D13", "高新技术企业认定有什么奖励？", "", "rnd"),
    ("D14", "众创空间有什么补贴？", "", "platform"),
    ("D15", "实验室建设补贴多少钱？", "amount", "platform"),
    ("D16", "哪些单位可以申报科技项目？", "who", "innovation"),
    ("D17", "人才政策的申请条件是什么？", "who", "talent"),
    ("D18", "创业补贴什么时候申请？", "time", "startup"),
    ("D19", "技能补贴申领截止到什么时候？", "time", "training"),
    ("D20", "申请创新补贴要什么条件？", "who", "innovation"),
    ("D21", "企业上新设备能补贴多少？", "amount", "upgrade"),
    ("D22", "外贸企业有哪些补贴？", "", "market"),
    ("D23", "青年人才能领什么补贴？", "", "talent"),
    ("D24", "知识产权方面能申请什么资助？", "", "ip"),
    ("D25", "什么人可以申请人才补贴？", "who", "talent"),
    ("D26", "办展会有补贴吗？", "", "market"),
    ("D27", "研发投入大的企业能拿什么奖励？", "", "rnd"),
    ("D28", "参展补贴最高多少？", "amount", "market"),
]

FACT = [
    ("F01", "澜州青年创新项目最多能拿多少补贴？", "a place and its programme"),
    ("F02", "岚江新区榜单攻关项目能补贴多少？", "a place and its programme"),
    ("F03", "青屿科技创新专项哪些单位可以报？", "a place and its programme"),
    ("F04", "《临溪市众创空间备案及绩效考核办法》的补贴是多少？", "a title"),
    ("F05", "这个创新项目补贴有多少？", "this one"),
    ("F06", "这份指南里的申报条件是什么？", "this file"),
    ("F07", "文件里说的补贴多少钱？", "the file"),
    ("F08", "第三条规定的补贴标准是多少？", "an article of one paper"),
    ("F09", "P-039项目的补贴是多少？", "a code"),
    ("F10", "大学生创新训练计划国家级项目给多少钱？", "one programme by its full name"),
    ("F11", "高新技术企业培育入库工作方案要求什么条件？", "one file by its name"),
    ("F12", "张伟的创业补贴批下来了吗？", "a person"),
    ("F13", "我们公司去年拿了多少补贴？", "the user's own record"),
    ("F14", "创新券怎么使用？", "a how-to"),
    ("F15", "什么是创新券？", "a definition"),
    ("F16", "申报材料需要盖章吗？", "yes / no"),
    ("F17", "补贴申请被驳回了怎么办？", "a how-to"),
    ("F18", "苹果手机多少钱？", "not a programme"),
    ("F19", "Python 列表有哪些方法？", "not a programme"),
    ("F20", "去年的补贴报告里写的金额是多少？", "one report"),
    ("F21", "博士后科研资助工作指引里在站补贴多少？", "one file by its name"),
    ("F22", "临溪的青年人才生活津贴要交多久社保？", "a place and its programme"),
    ("F23", "上次会议定的补贴标准是多少？", "one meeting"),
    ("F24", "合同里约定的违约金是多少？", "one contract"),
    ("F25", "这笔补贴什么时候到账？", "this one"),
    ("F26", "望海市产学研合作项目政府出多少钱？", "a place and its programme"),
    ("F27", "马桶多少钱？", "not a programme"),
    ("F28", "研发费用加计扣除比例是多少？", "one national rule, no programme word"),
]


def main() -> None:
    out = Path(__file__).resolve().parent / "detector.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for qid, text, ask, group in SURVEY:
            handle.write(json.dumps({"id": qid, "question": text, "label": "survey", "ask": ask, "group": group}, ensure_ascii=False) + "\n")
        for qid, text, why in FACT:
            handle.write(json.dumps({"id": qid, "question": text, "label": "fact", "why": why}, ensure_ascii=False) + "\n")
    import corpus_spec as spec  # recall_wording_v1
    from make_corpus import EXTENSION

    files = {g["id"]: g["title"] + EXTENSION[g["fmt"]] for g in spec.GUIDES}
    kinds = {"amount": "fund", "time": "time", "who": "who", "": "fund"}
    wide = []
    for qid, text, ask, group in SURVEY:
        if not group:
            continue  # "有什么补贴可以申请": every programme of the library
        kind = kinds[ask]
        gold = [{"file": files[g["id"]], "marker": g[kind][2], "amounts": g["amounts"] if ask == "amount" else []}
                for g in spec.GUIDES if g["cat"] == group]
        wide.append({"id": qid, "type": "survey", "kind": kind, "group": group, "question": text, "gold": gold})
    with (out.parent / "wide.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in wide:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(len(SURVEY), "survey,", len(FACT), "fact ->", out, ";", len(wide), "with gold -> wide.jsonl")


if __name__ == "__main__":
    main()
