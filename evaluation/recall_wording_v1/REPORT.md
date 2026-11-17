# Questions in everyday words, documents in their own (recall_wording_v1)

Branch `feat/recall-wording-v1`, from `main` at 1.0.4. Models as shipped: Qwen3-8B, bge-m3,
bge-reranker-v2-m3. Scoring is by rules; no judge model. `python evaluation/recall_wording_v1/summary.py`
prints every table below from `results/*.json`.

## What was wrong

A user's library holds an application guide that says "优质档…单个项目资助资金不超过5万元；潜力档…
2万元；圆梦档…0.5万元". Asked "一般的创新项目申报能给多少补贴", the app answered from a policy
collection and never showed the guide. On a copy of that library (4,172 passages, 123 files):

* the passage ranks 63 in the vector search and 35 in BM25; 40 candidates go to the reranker, 37 of
  them from three files that say 补贴 in every line, so the passage is never scored;
* scored anyway, it gets 0.198 against the question (the best passage 0.529): the question says
  补贴 / 申报, the guide says 资助资金 / 申请 / 立项;
* the question asks what several files say, and was answered as one fact from one file.

## Result

| | main | this branch (new defaults) |
|---|---:|---:|
| Synthetic library, fact questions: gold passage kept | 0.952 | **0.984** |
| Synthetic library, survey questions: gold passages kept | 0.075 | **0.588** |
| … questions with 3 or more gold files kept | 0 / 22 | **11 / 22** |
| Qwen3-8B answers to survey questions: gold files cited | 6 / 80 | **38 / 80** |
| … amounts stated, with their file cited | 7 / 137 | **75 / 137** |
| … refused | 5 / 22 | 4 / 22 |
| Qwen3-8B answers to 16 fact questions: amounts stated | 38 / 47 | 41 / 47 |
| Regression, 5,756 questions: share that keeps what it needs | — | the same on all seven sets |
| Retrieval time, fact question (median, live) | 0.71 s | 0.91 s |
| Retrieval time, survey question (median, live) | 0.71 s | 1.44 s |

On the copy of the user's library, the failing question now takes the survey path: 10 passages from
8 files reach the answer model, the answer cites 6 files, the guide among them, states its 5 万 /
2 万 / 0.5 万 beside the other files' amounts, and passes the fact check.

Switches (all in `project/config.py`):

| Switch | Default | Why |
|---|---|---|
| `wording_expansion` | **on** | gets the passages into the pool; regression: 3 rankings changed, nothing lost |
| `wording_rescore` | **on** | fact kept 0.952 → 0.984, survey kept 0.388 → 0.588; +0.2 s; regression unchanged in what is kept |
| `survey_route` (`survey_pool` 60, `survey_files` 8) | **on** | the lever that fixes the failing case |
| `candidate_per_document` | off (0) | changes 531 regression rankings for no net gain (books 0.942 → 0.935) |
| larger pool (`fusion_top_k` 80 / 150) | unchanged (40) | 2 to 4 times the rerank tokens, nothing more kept |
| `section_headings` | off | a real but small gain, and switching it re-indexes every .docx and .txt (see step 4) |
| Qwen3-8B rewrite (HyDE / paraphrases) | not added | nothing more kept, +1.1 to 2.0 s and a call on a rate-limited model |

## Step 1: the test set

`make_corpus.py` writes an invented library (fixed seed; places, offices, programmes and amounts are
made up) of 135 files, 1,569 passages:

* 31 application guides and measures in nine groups (innovation projects, talent, training, patents
  and trademarks, start-ups, equipment upgrades, fairs and export, R&D, platforms). Each states who
  may apply, the money and the dates in an office's own words (资助资金 / 支持强度 / 扶持标准 / 经费安排
  / 奖励 / 配套 / 津贴 / 激励 …). None of the money sections says 补贴. Half are .docx files whose
  headings are bold paragraphs, the rest .docx with heading styles, .txt and .md;
* three long policy collections (2,400 short items about other programmes, 补贴 in every line: one
  PDF, two .docx), a FAQ and a year-end summary per guide (the user's words, never the guide's
  facts; the summaries give totals and averages), eight application forms, six award notices, 25
  unrelated files.

`questions.jsonl`, 115 questions, each with its gold file and a string only the gold passage has:

* 62 named fact questions ("澜州的青年创新项目最多能拿多少补贴？", "…几号截止申报？", "…谁能报？");
* 31 un-named ones, asked the way the failing question was ("青年创新项目申报能给多少补贴？");
* 22 survey questions whose answer needs every guide of a group, 3 to 5 files ("一般的创新项目申报能
  给多少补贴？", "各个创新项目的补贴额度分别是多少？").

Baseline: fact questions are found (the gold passage is in the pool for all of them; it is kept for
0.952, and loses to the FAQ or the summary of its own guide in the rest). Survey questions fail as
the real one did: 0.30 of the gold passages reach the reranker, 0.075 are kept, no question keeps
three gold files.

The regression is the existing one (`regression.py`): the seven prepared sets of `evaluation/suite`,
5,756 questions, replayed with this branch's code against cached model outputs, pairs the cache lacks
scored live. With every switch off it reproduces the earlier reference run exactly (0 rankings
changed, no model call).

## Retrieval, step by step

| Configuration | Fact: kept | Fact: rank ≤ 3 | Un-named: kept | Survey: in pool | Survey: kept | Survey: gold files kept (of 3.64) | Survey: ≥ 3 files | Passages scored | Rerank tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main (all switches off) | 0.952 | 0.935 | 0.968 | 0.300 | 0.075 | 0.32 | 0.00 | 41 | 14,309 |
| step 2: wording_expansion | 0.952 | 0.935 | 0.968 | 0.537 | 0.125 | 0.50 | 0.04 | 56 | 19,244 |
| step 2: + wording_rescore | 0.984 | 0.935 | 0.968 | 0.537 | 0.150 | 0.64 | 0.09 | 56 | 19,244 |
| step 3: candidate_per_document=8 alone | 0.952 | 0.935 | 0.968 | 0.375 | 0.100 | 0.41 | 0.00 | 41 | 14,688 |
| step 3: pool 80 alone | 0.952 | 0.935 | 0.968 | 0.400 | 0.113 | 0.45 | 0.04 | 82 | 30,846 |
| step 3: pool 150 alone | 0.952 | 0.919 | 0.968 | 0.662 | 0.125 | 0.50 | 0.04 | 154 | 62,069 |
| step 3: wording + cap 8 + pool 150 | 0.952 | 0.919 | 0.968 | 0.863 | 0.113 | 0.45 | 0.04 | 161 | 48,509 |
| step 5: survey_route alone | 0.952 | 0.935 | 0.968 | 0.588 | 0.250 | 0.95 | 0.14 | 44 | 14,374 |
| step 5: wording + rescore + survey_route = new defaults | 0.984 | 0.935 | 0.968 | 0.838 | 0.588 | 2.23 | 0.50 | 60 | 19,903 |
| step 5: new defaults without rescore | 0.952 | 0.935 | 0.968 | 0.838 | 0.388 | 1.45 | 0.23 | 60 | 19,903 |
| step 5: new defaults, survey_pool=100 | 0.984 | 0.935 | 0.968 | 0.963 | 0.650 | 2.41 | 0.59 | 71 | 22,883 |
| step 5: new defaults, survey_files=10 | 0.984 | 0.935 | 0.968 | 0.838 | 0.625 | 2.36 | 0.55 | 60 | 19,903 |
| step 4: section_headings alone (re-indexed) | 1.000 | 0.984 | 0.968 | 0.350 | 0.150 | 0.59 | 0.04 | 41 | 11,972 |
| step 4: section_headings + new defaults | 1.000 | 0.968 | 0.968 | 0.750 | 0.575 | 2.14 | 0.46 | 60 | 14,339 |
| step 4: the same, survey_candidates_per_document=5 | 1.000 | 0.968 | 0.968 | 0.762 | 0.588 | 2.18 | 0.46 | 60 | 14,607 |

"Passages scored" and "rerank tokens" are means over all 115 questions and count the first scoring
only; `wording_rescore` scores the pool of an expanded question a second time (see latency).

Regression: share of answerable questions that keep what they need, rankings changed in brackets.

| Configuration | Questions | Rankings changed | More gold | Fewer gold | books | complex | benchmark | CRUD | MultiHop | Du | T2 |
|---|---:|---:|---:|---:|---|---|---|---|---|---|---|
| main (reference run) | 5,756 | 0 | 0 | 0 | 0.9420 | 0.9524 | 1.0000 | 0.9065 | 0.5747 | 0.9741 | 0.9741 |
| step 2: wording_expansion | 5,756 | 3 | 0 | 0 | 0.9420 (0) | 0.9524 (0) | 1.0000 (1) | 0.9065 (2) | 0.5747 (0) | 0.9741 (0) | 0.9741 (0) |
| step 2: + wording_rescore | 5,756 | 66 | 3 | 2 | 0.9420 (0) | 0.9524 (0) | 1.0000 (21) | 0.9065 (14) | 0.5747 (0) | 0.9741 (15) | 0.9741 (16) |
| step 3: candidate_per_document=8 | 5,756 | 531 | 26 | 18 | 0.9348 (48) | 0.9524 (19) | 1.0000 (46) | 0.9065 (0) | 0.5827 (360) | 0.9741 (10) | 0.9748 (48) |
| steps 2 + 5: new defaults | 5,756 | 71 | 4 | 2 | 0.9420 (0) | 0.9524 (0) | 1.0000 (21) | 0.9065 (14) | 0.5747 (0) | 0.9741 (19) | 0.9741 (17) |

With the new defaults no question gains or loses what it needs, and the unanswerable questions that
get nothing back stay the same on every set (3, 14, 271, 22, 14). The 71 changed rankings are
reorderings among kept passages of the roughly 100 questions the table touches, plus five Du / T2
questions that take the survey path. Step 4 does not touch these sets: their corpora are indexed as
prepared text, not through the .docx / .txt parsers.

### Step 2: the documents' words, without a model call

`retrieval/wording.py` is a small table: money (补贴 资助 奖励 扶持 补助 经费 资金 支持强度 额度 津贴 奖补
贴息, and for a "how much" question 标准 万元 最高 不超过 比例), applying (申报 申请 立项 评审 受理),
deadlines, who may apply, what to hand in. Only a question that itself talks about money or applying
is expanded; the other three groups are added only beside one of those, so "唐朝是什么时候建立的" and
"这个反应需要什么条件" are left alone. About 100 of the regression's 4,000 Chinese questions are touched.

* `wording_expansion`: the question followed by the group's other words is embedded in the same
  request as the question and searched beside it; its first 20 new candidates join the pool through
  the path the question parts already use (`_expansions`), and are scored against the question as
  asked. Nothing is taken out of the pool, which is why the regression does not move.
* `wording_rescore`: the pool is also scored against the expanded text (a second rerank request, in
  parallel) and a passage keeps the better score. This is what lets a passage in the documents'
  words outrank one that only echoes the question: fact questions kept 0.952 → 0.984, and with the
  survey path 0.388 → 0.588. A question with no evidence as asked (best score under 0.1) is not
  rescored outside the survey path, so refusals stay refusals. Added while measuring step 5; it sits
  in the step 5 commit.

Seven guides use money words that are not in the table (激励, 配套, plain 支持, 抵扣): the table was
written from the task's list and general usage, not from the test files.

A Qwen3-8B rewrite on top of the table (`rewrite_eval.py`; one call per question, each rewrite's
first 20 new candidates added the same way):

| | Fact: kept | Un-named: kept | Survey: in pool | Seconds per call (median) | Tokens per call |
|---|---:|---:|---:|---:|---:|
| table only | 0.952 | 0.968 | 0.537 | — | — |
| + HyDE sentence | 0.952 | 0.968 | 0.787 | 1.11 | 104 |
| + three paraphrases | 0.952 | 0.968 | 0.675 | 1.96 | 109 |

It only widens the pool of survey questions, which the survey path does on its own (0.838), and
keeps nothing more. Not added.

### Step 3: pool size and crowding

`candidate_per_document` (a pool that one file crowds is searched four times deeper and cut with a
cap per file; a pool nobody crowds is untouched) and larger pools both raise pool recall of survey
questions and keep nothing more: the gold passages that get in score 0.01 to 0.15 against the
everyday question and the cut drops them. On the regression the cap changes 531 rankings: MultiHop
gains (0.5747 → 0.5827), books lose a question (0.942 → 0.935), 26 questions keep more gold and 18
less. That is not "unchanged or better", so it stays off; the code and its tests are in. The survey
path has its own cap (3 per file), where it is needed.

### Step 4: headings written as ordinary paragraphs

`_parse_docx` only cut at Word heading styles; `_parse_text` not at all. A guide whose headings are
bold or numbered paragraphs was one unit cut every 700 characters: in the test library the amounts
of a guide sat at character 567 of a 704-character passage that opened with the title, the notice
and two other sections (the user's file: character 415 of 514). `ingestion/sections.py` cuts such
files at "三、…", "第二章 …" and, when there are several, at bold lines; a long section is cut again at
"（一）…" sub-headings, a short one stays whole. The heading stays in the text and becomes the
passage's heading ("【指南 · 五、支持强度与方式】"), so the amounts now start a 179-character passage.
A sentence or list item wrongly taken for a heading can only add a boundary.

Measured by re-indexing the test library (1,569 → 1,875 passages):

* alone: fact kept 0.952 → 1.000, gold among the first three 0.935 → 0.984, survey kept 0.075 → 0.150;
* on top of the new defaults: fact kept 0.984 → 1.000 (one question of 62), survey kept the same
  (0.588 with 5 candidates per file; sections make more, smaller passages per file);
* answers: survey gold files cited 46 / 80 against 40 / 80 (both at `survey_pool` 100), refusals 2
  against 4, amounts 83 against 88, fact questions the same 41 / 47.

Is it worth the cost? The switch changes the index version of .docx and .txt files, so on the upgrade
that turns it on every such file is read and embedded again (PDF, slides, sheets and Markdown are
not). After steps 2 and 5 the remaining gain is one fact question in 62 and a few more files cited,
so it is **off**, and worth turning on together with the next change that re-indexes these files
anyway.

### Step 5: survey questions

`retrieval/survey.py`. Detection is by wording and deliberately narrow: a question of at most 32
characters with a generality word (一般, 通常, 普遍, 大多, 往往 …; not "一般纳税人"), or with 各… / 都… /
有哪些 when it also asks for an amount, a time or a condition and names no place, person,
organisation or code. A first, wider version (any 有哪些 / 分别) sent 84 regression questions down
this path and lost gold on lists from one file ("面试题里有哪些关于…的题目") and on long news
questions; with the narrow version five do, none worse.

For such a question:

* the pool is 60 candidates with at most 3 per file, and the expanded search adds up to 30 more, so
  twenty or more files get passages scored instead of three;
* `select_survey` keeps the best passage of each of the first 8 files instead of the global top k. A
  passage counts only if it has the shape of an answer (an amount for 多少, a date or period for
  什么时候): a FAQ that repeats "能拿多少补贴？" without a figure is out. The two best files stay
  first; after them, files that state rules come before files that mention them: by the type in the
  file's name or its document card (制度规范 / 说明书 / 通知公告 / 方案), or a name ending the way such
  papers are named (指南, 办法, 细则, 指引, 须知, 意见, 政策集合 …);
* the answer profile `survey` (10 passages, 2 per file) gives the model the passages themselves, not
  their 1,800-character parents (five parents filled the budget), asks for one line per file with
  its citation, and skips the "资料里的说法不一致" line, which read different programmes' amounts as
  a contradiction.

`survey_pool` 100 keeps a little more (0.650 against 0.588) but its two scoring requests of 150
passages took 8.9 s on the live service against 1.44 s for 90, so the default is 60. `survey_files`
10 keeps more passages (0.625) and the answers got worse (amounts 61 / 137 against 88 / 137 at the
same pool): Qwen3-8B handles eight files better than ten.

## Answers on Qwen3-8B

`answers.py`: the app's graph (analyse, retrieve, context, answer, fact check, strict mode), 30 s
between questions. 22 survey questions and 16 named fact questions about money as a guard.

| Configuration | Survey: refused | Survey: gold files cited | per question | ≥ 3 files cited | Fund surveys: amounts stated | Fact guard: files cited | Fact guard: amounts stated | Median seconds (survey / fact) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| main | 5 / 22 | 6 / 80 | 0.27 | 0 / 22 | 7 / 137 | 14 / 16 | 38 / 47 | 9.3 / 8.0 |
| new defaults | 4 / 22 | 38 / 80 | 1.73 | 8 / 22 | 75 / 137 | 15 / 16 | 41 / 47 | 23.5 / 8.7 |
| new defaults, survey_pool=100 | 4 / 22 | 40 / 80 | 1.82 | 8 / 22 | 88 / 137 | 15 / 16 | 41 / 47 | 21.0 / 9.4 |
| new defaults, survey_pool=100, survey_files=10 | 4 / 22 | 37 / 80 | 1.68 | 6 / 22 | 61 / 137 | - | - | 21.2 / - |
| new defaults + section_headings (survey_pool=100, 5 per file) | 2 / 22 | 46 / 80 | 2.09 | 7 / 22 | 83 / 137 | 15 / 16 | 41 / 47 | 21.4 / 6.8 |

An amount counts when the answer cites its file and contains the figure. A survey answer takes about
14 s longer than before: it is a longer answer from ten passages, and is checked in full.

## Latency

`latency.py`: 24 questions, every model call live, nothing cached, main and the new defaults
alternating. Retriever wall time (query embedding, searches, reranking):

| | main: median / slowest | new defaults: median / slowest | Passages scored |
|---|---:|---:|---:|
| named fact (12) | 0.71 s / 1.95 s | 0.91 s / 1.29 s | 40 → 93 |
| un-named (6) | 0.69 s / 0.83 s | 1.26 s / 5.94 s | 40 → 120 |
| survey (6) | 0.71 s / 1.07 s | 1.44 s / 5.84 s | 47 → 180 |

A question outside the table (most questions) costs what it did. An expanded question is scored
twice, in two parallel requests: about three times the rerank tokens for that question.

## What is not solved

* The same question without a generality word ("创新项目申报能给多少补贴") stays a fact question.
  On the copy of the user's library its retrieval now keeps the guide's passage (rank 8, score
  0.43), but only two parent passages fit the fact budget and the answer cites one file, not the
  guide. Telling "no particular programme is meant" from the wording alone is the open part.
* Survey questions about conditions ("…通常有哪些门槛") are still the weak kind: of the four survey
  questions still refused, three are of this kind (the fourth asks about money); nothing in a
  passage marks it as "a condition" the way a figure marks an amount.
* 0.588 of the gold passages kept is not all of them: a guide whose money word is outside the table
  and whose name gives no type still loses to files that echo the question.
* The test library is invented and by the author of the table. The figures that do not depend on it
  are the regression and the copy of the real library.
* `tests/test_ocr_runtime.py::test_pinned_manifest_matches_this_python` fails in this environment
  on `main` as well (the manifest pins another Python); every other test passes (1,343).

## Files

* `corpus_spec.py`, `make_corpus.py`, `questions.jsonl`: the library and the questions.
* `harness.py`, `run.py`: index and retrieval runs (`--set field=value`, `--index sections`).
* `probe.py`: one question on any index folder, numbers only (used on the copy of the real library).
* `regression.py`, `rewrite_eval.py`, `answers.py`, `latency.py`, `summary.py`.
* `results/`: counts only. `work/` (corpus, indexes, caches, answers, the regression copy) is not committed.
