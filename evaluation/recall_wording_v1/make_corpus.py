"""Write the synthetic library of recall_wording_v1 and its questions (fixed seed).

  python evaluation/recall_wording_v1/make_corpus.py

work/corpus/      the files (.docx, .txt, .md, one .pdf), regenerable, not committed
questions.jsonl   one question per line: id, kind, type, question, gold = [{file, marker, amounts}], group
                  type "fact": names the place and the programme; "generic": asked the way the
                  failing question was, without the place ("青年创新项目申报能给多少补贴？");
                  "survey": the answer draws on every guide of a group (3 to 5 files)

Files: 31 guides (corpus_spec.GUIDES), 3 policy collections, 8 application forms,
6 award notices, 25 unrelated files, and two companions per guide (a FAQ and a year-end
summary in the user's words, without the guide's facts).
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import corpus_spec as spec  # noqa: E402

WORK = HERE / "work"
CORPUS = WORK / "corpus"
SEED = 20261116
EXTENSION = {"bold": ".docx", "style": ".docx", "txt": ".txt", "md": ".md"}


def guide_sections(guide: dict, rng: random.Random) -> list[tuple[str, list[str]]]:
    """(heading, paragraphs) in the order of a real guide: the money comes after the
    purpose, who may apply, the conditions and what is supported."""
    slots = {"city": guide["city"], "org": guide["org"], "topic": guide["title"].lstrip("0123456789年")}
    directions = rng.sample(spec.DIRECTIONS[guide["cat"]], k=3)
    conditions = [text.format(**slots) for text in rng.sample(spec.CONDITIONS, k=3)]
    materials = rng.sample(spec.MATERIALS[:-1], k=5) + [spec.MATERIALS[-1]]
    review = [text.format(**slots) for text in rng.sample(spec.REVIEW, k=2)]
    supervision = [text.format(**slots) for text in rng.sample(spec.SUPERVISION, k=3)]
    who_head, who_text, _ = guide["who"]
    fund_head, fund_text, _ = guide["fund"]
    time_head, time_text, _ = guide["time"]
    sections = [
        (rng.choice(["总体要求", "设定依据", "目的和依据"]), [rng.choice(spec.BASIS).format(**slots)]),
        (who_head, [who_text] + conditions[:1]),
        (rng.choice(["基本条件", "其他条件", "申请要求"]), [f"（{'一二三'[i]}）{text}" for i, text in enumerate(conditions)]),
        (rng.choice(["支持方向", "申请内容", "重点领域"]), [f"（{'一二三'[i]}）{text}。" for i, text in enumerate(directions)]),
        (fund_head, [fund_text]),
        (rng.choice(["申请材料", "提交资料", "材料清单"]), [f"{i + 1}. {text}；" for i, text in enumerate(materials)]),
        (time_head, [time_text]),
        (rng.choice(["评审流程", "办理程序", "审核方式"]), review),
        (rng.choice(["监督管理", "绩效管理", "责任追究"]), supervision),
        ("联系方式", [f"{guide['org']}业务受理窗口，咨询电话：0{rng.randint(510, 599)}-5550{rng.randint(100, 999)}，"
                  f"受理地点：{guide['city']}市民服务中心{rng.randint(2, 9)}楼{rng.randint(1, 30)}号窗口。本文件自印发之日起施行，由{guide['org']}负责解释。"]),
    ]
    return sections


def write_guide(guide: dict, rng: random.Random) -> str:
    name = guide["title"] + EXTENSION[guide["fmt"]]
    path = CORPUS / name
    sections = guide_sections(guide, rng)
    numbered = [(f"{spec.SECTION_NUMBERS[i]}、{heading}", paragraphs) for i, (heading, paragraphs) in enumerate(sections)]
    lead = f"{guide['org']}关于印发《{guide['title']}》的通知。各有关单位：现将《{guide['title']}》印发给你们，请遵照执行。"
    if guide["fmt"] in ("bold", "style"):
        from docx import Document

        document = Document()
        title = document.add_paragraph()
        title.add_run(guide["title"]).bold = True
        document.add_paragraph(lead)
        for heading, paragraphs in numbered:
            if guide["fmt"] == "style":
                document.add_heading(heading, level=1)
            else:
                document.add_paragraph().add_run(heading).bold = True
            for text in paragraphs:
                document.add_paragraph(text)
        document.save(path)
    elif guide["fmt"] == "md":
        lines = [f"# {guide['title']}", "", lead, ""]
        for heading, paragraphs in numbered:
            lines += [f"## {heading}", "", *[f"{text}\n" for text in paragraphs]]
        path.write_text("\n".join(lines), encoding="utf-8")
    else:
        lines = [guide["title"], "", lead, ""]
        for heading, paragraphs in numbered:
            lines += [heading, *paragraphs, ""]
        path.write_text("\n".join(lines), encoding="utf-8")
    return name


def compilation_item(number: int, rng: random.Random) -> str:
    name = rng.choice(spec.ITEM_NAMES)
    city = rng.choice(spec.ITEM_CITIES)
    amount = rng.choice([f"每人每月{rng.randint(2, 15) * 100}元", f"每户最高{rng.randint(1, 9)}000元", f"按实际支出的{rng.randint(1, 5) * 10}%",
                         f"每家企业每年最高{rng.randint(2, 30)}万元", f"每个项目{rng.randint(1, 20)}万元", f"每千瓦时{rng.randint(1, 9) / 10:.1f}元"])
    month = rng.randint(1, 12)
    return (
        f"{number}. 【{city}{name}补贴】补贴对象：{rng.choice(spec.ITEM_TARGETS)}。补贴标准：{amount}给予{name}补贴，"
        f"补贴期限最长{rng.randint(1, 3)}年。申报时间：每年{month}月1日至{month}月{rng.choice([15, 20, 25, 28])}日。"
        f"申报方式：{rng.choice(spec.ITEM_HOW)}，补贴资金直接拨付至申报单位或个人账户。政策依据：《{city}{name}补贴申报实施办法》。"
    )


def write_compilations(rng: random.Random) -> list[str]:
    from docx import Document
    import pymupdf

    names = []
    for index, (title, count, kind) in enumerate([("1、惠企补贴政策集合", 900, "pdf"), ("2、各类补贴项目申报汇总表", 800, "docx"),
                                                  ("3、补贴政策问答手册", 700, "docx")]):
        items = [compilation_item(i + 1, rng) for i in range(count)]
        if index == 2:
            items = [f"问：{item.split('】')[0].split('【')[1]}怎么申报，能补多少？\n答：{item.split('】', 1)[1]}" for item in items]
        name = f"{title}.{kind}"
        if kind == "docx":
            document = Document()
            document.add_paragraph().add_run(title).bold = True
            document.add_paragraph("本汇总收录了各地现行有效的补贴项目，供企业和个人查询补贴标准和申报方式，具体以主管部门发布的申报通知为准。")
            for item in items:
                for line in item.split("\n"):
                    document.add_paragraph(line)
            document.save(CORPUS / name)
        else:
            pdf = pymupdf.open()
            lines: list[str] = [title, ""]
            for item in items:
                lines += [item[i:i + 38] for i in range(0, len(item), 38)] + [""]
            for start in range(0, len(lines), 44):
                page = pdf.new_page()
                for row, line in enumerate(lines[start:start + 44]):
                    if line:
                        page.insert_text((56, 60 + row * 16.5), line, fontname="china-s", fontsize=11)
            pdf.save(CORPUS / name)
            pdf.close()
        names.append(name)
    return names


def write_forms(rng: random.Random) -> list[str]:
    from docx import Document

    names = []
    for index, (project, field) in enumerate(spec.FORM_PROJECTS, start=1):
        name = f"{index:02d} {project}项目申报书.docx"
        document = Document()
        document.add_paragraph().add_run(f"创新项目申报书：{project}").bold = True
        budget = [(item, rng.randint(2, 30) / 10) for item in ("设备费", "材料费", "测试化验加工费", "差旅费", "劳务费", "专家咨询费")]
        total = sum(value for _, value in budget)
        body = [
            f"一、项目基本情况。项目名称：{project}；所属领域：{field}；申报单位：{rng.choice(spec.ITEM_CITIES)}某某科技有限公司；项目周期：12个月；申请资助金额：{total:.1f}万元。",
            f"二、立项依据。{field}领域现有方案成本高、部署周期长，本项目拟通过自主研发解决上述问题，具有明确的应用场景和市场需求，符合创新项目申报方向。",
            "三、研究内容和创新点。项目拟完成原型开发、小批量试制和用户试用三个阶段的工作；创新点在于将低成本传感与轻量化算法结合，降低了部署门槛。",
            "四、经费预算。" + "；".join(f"{item}{value:.1f}万元" for item, value in budget) + f"；合计{total:.1f}万元，其中申请财政补贴{total * 0.6:.1f}万元，自筹{total * 0.4:.1f}万元。",
            "五、进度安排。第1至3个月完成方案设计和核心模块开发；第4至8个月完成样机试制与测试；第9至12个月完成试点应用、申报专利并提交结题报告。",
            "六、预期成果。申请发明专利1件、实用新型专利2件，形成样机3台，在2家以上单位试用，带动就业5人。",
            "七、申报单位承诺。本单位承诺申报材料真实有效，项目获得补贴后按规定使用资金并接受监督检查。",
        ]
        for text in body:
            document.add_paragraph(text)
        document.save(CORPUS / name)
        names.append(name)
    return names


def write_notices(rng: random.Random) -> list[str]:
    names = []
    for index in range(1, 7):
        city = spec.ITEM_CITIES[index % len(spec.ITEM_CITIES)]
        item = rng.choice(spec.ITEM_NAMES)
        name = f"关于{city}{2020 + index}年度{item}补贴拟发放名单的公示.txt"
        rows = [f"{i}. {city}{rng.choice('华盛瑞恒宏嘉博远')}{rng.choice('达通科信创新联')}"
                f"{rng.choice(['科技', '实业', '贸易', '制造', '物流'])}有限公司，拟发放补贴{rng.randint(1, 60)}.{rng.randint(0, 9)}万元。" for i in range(1, 41)]
        text = (f"关于{city}{2020 + index}年度{item}补贴拟发放名单的公示\n\n根据《{city}{item}补贴申报实施办法》，经企业申报、部门审核，"
                f"现将拟发放{item}补贴的企业名单公示如下，公示期5个工作日。\n\n" + "\n".join(rows))
        (CORPUS / name).write_text(text, encoding="utf-8")
        names.append(name)
    return names


def write_others(rng: random.Random) -> list[str]:
    names = []
    for kind, sentences in spec.OTHER_TOPICS:
        for index in range(1, 6):
            name = f"{kind}-{index:02d}.txt"
            lines = [f"{kind} 第{index}期", ""]
            for paragraph in range(8):
                lines.append("".join(rng.choice(sentences) for _ in range(rng.randint(3, 5))))
                lines.append("")
            (CORPUS / name).write_text("\n".join(lines), encoding="utf-8")
            names.append(name)
    return names


def write_siblings(rng: random.Random) -> list[str]:
    """Two companions per guide: a FAQ and a year-end summary (see corpus_spec.FAQ)."""
    names = []
    for guide in spec.GUIDES:
        slots = {"title": guide["title"], "org": guide["org"], "city": guide["city"]}
        faq = f"{guide['title']}常见问题解答.txt"
        lines = [f"《{guide['title']}》常见问题解答", "", f"以下问题由{guide['org']}受理窗口整理，供申报人参考。", ""]
        for number, (asked, answer) in enumerate(rng.sample(spec.FAQ, k=10), start=1):
            lines += [f"问题{number}：{asked}", f"答：{answer.format(**slots)}", ""]
        (CORPUS / faq).write_text("\n".join(lines), encoding="utf-8")
        passed = rng.randint(20, 200)
        total = rng.randint(100, 5000)
        numbers = {
            "year": rng.randint(2022, 2025), "applied": passed * rng.randint(2, 5), "passed": passed, "rate": rng.randint(18, 55),
            "total": total, "average": round(total / passed, 1), "growth": rng.randint(3, 40), "late": rng.randint(30, 60),
            "talks": rng.randint(4, 30), "calls": rng.randint(300, 5000), "days": rng.randint(3, 15),
            "audited": rng.randint(10, 40), "issues": rng.randint(1, 5),
        }
        summary = f"{guide['title']}年度工作总结.txt"
        text = [f"{guide['title']}年度工作总结", ""] + [line.format(**slots, **numbers) for line in spec.SUMMARY]
        (CORPUS / summary).write_text("\n\n".join(text), encoding="utf-8")
        names += [faq, summary]
    return names


def questions(files: dict[str, str]) -> list[dict]:
    """Fact questions (one gold passage) and survey questions (every guide of the group)."""
    rows = []
    for guide in spec.GUIDES:
        for number, (kind, question) in enumerate(guide["questions"], start=1):
            heading, _, marker = guide[kind]
            rows.append({
                "id": f"{guide['id']}-{number}", "type": "fact", "kind": kind, "group": guide["cat"], "question": question,
                "format": guide["fmt"], "doc_heading": heading,
                "gold": [{"file": files[guide["id"]], "marker": marker, "amounts": guide["amounts"] if kind == "fund" else []}],
            })
        question, marker = guide["generic"]
        rows.append({
            "id": f"{guide['id']}-g", "type": "generic", "kind": "fund", "group": guide["cat"], "question": question,
            "format": guide["fmt"], "doc_heading": guide["fund"][0],
            "gold": [{"file": files[guide["id"]], "marker": marker, "amounts": []}],
        })
    for number, (group, kind, question) in enumerate(spec.SURVEYS, start=1):
        gold = [{"file": files[g["id"]], "marker": g[kind][2], "amounts": g["amounts"] if kind == "fund" else []}
                for g in spec.GUIDES if g["cat"] == group]
        rows.append({"id": f"S{number:02d}", "type": "survey", "kind": kind, "group": group, "question": question, "gold": gold})
    return rows


def main() -> None:
    rng = random.Random(SEED)
    if CORPUS.exists():
        shutil.rmtree(CORPUS)
    CORPUS.mkdir(parents=True)
    files = {guide["id"]: write_guide(guide, rng) for guide in spec.GUIDES}
    counts = {"guides": len(files), "collections": len(write_compilations(rng)), "forms": len(write_forms(rng)),
              "notices": len(write_notices(rng)), "others": len(write_others(rng)), "siblings": len(write_siblings(rng))}
    rows = questions(files)
    with (HERE / "questions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts["questions"] = {kind: sum(r["type"] == kind for r in rows) for kind in ("fact", "generic", "survey")}
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
