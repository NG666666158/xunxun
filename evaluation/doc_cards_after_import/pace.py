"""How long the cards of a big folder take, old builder against new, on a virtual clock.

    python -I evaluation/doc_cards_after_import/pace.py --old-project <main's project folder> \
        --out evaluation/doc_cards_after_import/results/pace.json

Both builders run their own code (retrieval/doc_cards.DocCardBuilder of each checkout); only the
clock, the model and the store are stand-ins. A call takes ``--call-seconds`` and costs
``--tokens`` (measured: one file per call, 1.8 s and 760 to 810 tokens on the public set). Questions
come every ``--question-every`` seconds and take 10 s to answer (0: nobody asks anything).
Files whose own name gives the type: ``--named`` of them (4.6 % of the synthetic news corpus,
22 % of a real 123-file library).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I leaves the script's folder out
from common import HERE, PROJECT, use_project  # noqa: E402

ANSWER_SECONDS = 10.0


class World:
    """The virtual clock, the questions and the model."""

    def __init__(self, call_seconds: float, tokens: int, question_every: float):
        self.now = 0.0
        self.call_seconds = call_seconds
        self.tokens = tokens
        self.question_every = question_every
        self.calls = 0

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, seconds)

    def answering_until(self) -> float | None:
        if not self.question_every:
            return None
        start = (self.now // self.question_every) * self.question_every
        return start + ANSWER_SECONDS if start > 0 and self.now < start + ANSWER_SECONDS else None

    def pause(self) -> None:
        end = self.answering_until()
        if end is not None:
            self.now = end

    def idle_seconds(self) -> float:
        if not self.question_every or self.now < self.question_every:
            return float("inf")
        if self.answering_until() is not None:
            return 0.0
        last = (self.now // self.question_every) * self.question_every + ANSWER_SECONDS
        return self.now - last

    def invoke(self, messages):
        self.calls += 1
        self.now += self.call_seconds
        reply = '{"类型": "新闻", "对象": "", "时间": "", "摘要": "x"}'
        return SimpleNamespace(content=reply, usage_metadata={"total_tokens": self.tokens})


class Store:
    def __init__(self):
        self.cards = {}

    def upsert_doc_card(self, doc_id, card):
        self.cards[doc_id] = card

    def delete_doc_card(self, doc_id):
        self.cards.pop(doc_id, None)

    def doc_card_hashes(self):
        return {doc_id: card["input_hash"] for doc_id, card in self.cards.items()}

    def renormalize_doc_cards(self, normalize):
        return 0


def load(project: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, project / "retrieval" / "doc_cards.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(module, new: bool, files: dict[str, str], world: World) -> dict:
    settings = SimpleNamespace(parsed_dir=".", llm_model="Qwen/Qwen3-8B", doc_card_chars=900,
                               doc_card_interval_seconds=3.0, doc_card_tokens_per_minute=9000.0,
                               doc_card_idle_tokens_per_minute=15000.0, doc_card_cool_down_seconds=60.0,
                               doc_card_batch=1)
    store = Store()
    kwargs = dict(chat_model=lambda: world, files=lambda: files, sleep=world.sleep)
    if new:
        library = SimpleNamespace(importing=lambda: False, idle_seconds=world.idle_seconds,
                                  pause_for_questions=world.pause, registry=None)
        builder = module.DocCardBuilder(store, settings, library=library, clock=world.clock, **kwargs)
    else:
        builder = module.DocCardBuilder(store, settings, pause=world.pause, **kwargs)
        module.time.monotonic = world.clock  # the old builder reads the clock itself
    builder.card_input = lambda doc_id, name: ("开头", f"h-{doc_id}-{name}")
    queued = builder.sync()
    builder.run_pending()
    hours = world.now / 3600
    return {"queued": queued, "calls": world.calls, "hours": round(hours, 2),
            "files_per_minute": round(world.calls / world.now * 60, 1) if world.now else None,
            "tokens_per_minute": round(world.calls * world.tokens / world.now * 60) if world.now else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-project", type=Path, required=True)
    parser.add_argument("--files", type=int, default=9347)
    parser.add_argument("--named", default="0.046,0.22")
    parser.add_argument("--call-seconds", type=float, default=1.8)
    parser.add_argument("--tokens", type=int, default=810)
    parser.add_argument("--question-every", default="0,300")
    parser.add_argument("--out", type=Path, default=HERE / "results" / "pace.json")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    use_project()  # services.model_clients for both builders
    old = load(args.old_project.resolve(), "old_doc_cards")
    new = load(PROJECT, "new_doc_cards")
    rows = []
    for every in (float(v) for v in args.question_every.split(",")):
        for share in (float(v) for v in args.named.split(",")):
            named = round(args.files * share)
            files = {f"{n:032d}": (f"{n:05d}_资料 申报书.docx" if n < named else f"{n:05d}_资料.docx")
                     for n in range(args.files)}
            for label, module, is_new in (("old", old, False), ("new", new, True)):
                world = World(args.call_seconds, args.tokens, every)
                result = run(module, is_new, files, world)
                rows.append({"builder": label, "question_every_seconds": every, "named_share": share, **result})
                print(rows[-1], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"files": args.files, "call_seconds": args.call_seconds, "tokens": args.tokens,
                                    "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
