from ingestion.chunker import HeadingAwareChunker
from ingestion.models import DocumentUnit, ParsedDocument


class ChunkSettings:
    parent_chunk_size = 40
    child_chunk_size = 20
    child_chunk_overlap = 5


def test_heading_and_locator_are_preserved():
    document = ParsedDocument(
        doc_id="doc-1",
        source="policy.docx",
        file_type=".docx",
        units=[DocumentUnit(text="第一条规定。" * 10, heading="报销 > 住宿", locator="page:2")],
    )
    chunks = HeadingAwareChunker(ChunkSettings()).split(document)
    assert len(chunks) > 1
    assert all(chunk.heading == "报销 > 住宿" for chunk in chunks)
    assert all(chunk.locator == "page:2" for chunk in chunks)
    assert len({chunk.id for chunk in chunks}) == len(chunks)


def test_children_map_to_complete_parents():
    document = ParsedDocument(
        doc_id="a" * 24,
        source="policy.txt",
        file_type=".txt",
        units=[DocumentUnit(text="甲乙丙丁戊己庚辛壬癸" * 8, heading="制度")],
    )
    parents, children = HeadingAwareChunker(ChunkSettings()).split_parent_child(document)
    by_id = {parent.parent_id: parent for parent in parents}

    assert parents
    assert children
    assert all(child.parent_id in by_id for child in children)
    assert all(child.text in by_id[child.parent_id].text for child in children)


def test_sliding_window_has_overlap():
    windows = HeadingAwareChunker._windows("0123456789ABCDEFGHIJ", size=10, overlap=2)
    assert windows == ["0123456789", "89ABCDEFGH", "GHIJ"]


def _children(text, size=700):
    settings = type("S", (), {"parent_chunk_size": 1800, "child_chunk_size": size, "child_chunk_overlap": 120})()
    document = ParsedDocument(doc_id="e" * 32, source="面试题.md", file_type=".md",
                              units=[DocumentUnit(text=text, heading="面试题 > 第2 章")])
    return [child.text for child in HeadingAwareChunker(settings).split_parent_child(document)[1]]


def test_numbered_question_lists_get_one_child_per_question():
    topics = ["KV 缓存", "分词器", "位置编码", "RAG 召回", "LoRA 微调", "MoE 路由"]
    for style in ("{n}. ", "{n}、", "（{n}）", "第{n}题 "):
        lines = [style.format(n=n) + f"{topic} 的原理是什么？" for n, topic in enumerate(topics, 1)]
        # Single newlines (Markdown/Word lists) and blank lines both split items.
        for joiner in ("\n", "\n\n"):
            children = _children("第2 章 模型结构\n\n" + joiner.join(lines))
            assert children == ["第2 章 模型结构\n\n" + lines[0], *lines[1:]], (style, joiner)


def test_policy_clauses_and_decimals_are_not_question_lists():
    clause = "员工出差应提前三个工作日在系统中提交申请，写明目的地、行程和预算，经部门负责人审批后方可预订交通与住宿。" * 3
    long_items = "\n\n".join(f"{n}. {clause}" for n in range(1, 6))
    assert HeadingAwareChunker._item_spans(long_items, 700) is None
    decimals = "\n".join(f"{n}.5 版本的变化很小。" for n in range(1, 7))
    assert HeadingAwareChunker._item_spans(decimals, 700) is None
    shuffled = "\n".join(f"{n}. 简短条目。" for n in (7, 3, 9, 1, 12, 5))
    assert HeadingAwareChunker._item_spans(shuffled, 700) is None
    # Three numbered lines in prose stay prose.
    assert HeadingAwareChunker._item_spans("正文。\n\n1. 甲\n2. 乙\n3. 丙\n\n" + "正文。" * 5, 700) is None


def test_chunker_version_reaches_every_format():
    from ingestion.chunker import CHUNKER_VERSION
    from ingestion.parsers import parser_version

    for name in ("a.md", "a.docx", "a.txt", "a.xlsx"):
        assert parser_version(name) == 1000 + CHUNKER_VERSION * 10
