# Survey questions, round two (survey_recall_v2)

Branch `feat/survey-recall-v2`, from `main` at 1.0.5 (`8db0b5d`). Models as shipped: Qwen3-8B,
bge-m3, bge-reranker-v2-m3; scoring by rules, no judge model. `python evaluation/survey_recall_v2/summary.py`
prints every table below from `results/*.json`.

Round one (`evaluation/recall_wording_v1`) gave survey questions ("一般…能给多少补贴", "各…分别是多少")
their own route. It left half of them short: 11 of 22 kept three or more gold files, the answers cited
38 of 80 gold files and stated 75 of 137 amounts. This round finds out why, file by file, and fixes the
largest causes.

## Result

| | main (1.0.5) | this branch (new defaults) |
|---|---:|---:|
| Survey questions (22): gold files kept by retrieval | 47 / 80 | **68 / 80** |
| … questions keeping 3 or more gold files | 11 / 22 | **16 / 22** |
| Qwen3-8B answers: gold files cited | 38 / 80 | **67 / 80** |
| … amounts stated, with their file cited | 75 / 137 | **119 / 137** |
| … questions citing 3 or more gold files | 8 / 22 | **15 / 22** |
| … refused | 4 / 22 | **0 / 22** |
| Held-out survey questions (16, other wording): gold files kept | 20 / 56 | **41 / 56** |
| … answers: gold files cited (amounts stated) | 21 / 56 (16 / 39) | **45 / 56 (25 / 39)** |
| Questions about programmes the library lacks (8): nothing kept | 4 / 8 | **7 / 8** |
| … answer refused | 6 / 8 | **7 / 8** |
| Fact questions (62): gold passage kept | 0.984 | 0.984 |
| Fact guard answers (16): amounts stated | 41 / 47 | 41 / 47 |
| Regression, 5,756 questions: share that keeps what it needs | — | the same on all seven sets |
| Retrieval time, survey question (median / slowest, live) | 1.25 s / 2.65 s | 1.57 s / 6.17 s |
| Answer time, survey question (median) | 23.8 s | 18.9 s |

Every switch below is on by default on this branch except `survey_about_asked`.

## Why the other half fell short

`losses.py` follows each gold file of each survey question through the pipeline and records the first
step that loses it:

| Step | What it means |
|---|---|
| search | its gold passage is not among the candidates of a search four times as deep as the pool |
| cut | found by that deeper search, but not sent to the reranker (pool size; with files first: its file is not a candidate) |
| file_passage | other passages of its file were scored, not the gold one (the per-file cap) |
| refused | scored, but the whole question was dropped as "no evidence" (best score under 0.1) |
| floor | scored under `survey_floor` (0.01) |
| files | its file was not among the files kept (`survey_files`, the passage limit) |
| passage | its file was kept, with another passage |
| kept | the gold passage is in what the retriever returns |

and, for the kept files, what the answer did with them: not cited, cited without its amounts, cited
with some, cited with all.

On main, the 22 survey questions lose 33 of their 80 gold files before the answer:

* **11 refused** — all of them questions about conditions ("…通常有哪些门槛", "…要满足哪些要求").
  Against such a question every passage scores under 0.1 (the best 0.006 to 0.017): the reranker sees a
  700-character passage of boilerplate around the one condition sentence. The whole question then counts as
  "no evidence".
* **9 file_passage** — the file was in the pool with three other passages; the per-file cap of three
  (needed to make room for other files) kept the purpose, the materials and the contacts instead of the
  money or the dates.
* **5 floor**, **5 files**, **2 cut**, **1 search**.

The answer then loses more: of the 47 files kept, 14 were not cited. Five of those (S08, S09) are one
cause in the graph, not the model: the survey route returns its passages file by file, and the graph's relevance gate
(`relevance_threshold` 0.2 on the first passage) sent three questions back to a search rewrite. The
rewrite drops "一般", the second search is not a survey, and the answer is about a policy collection or a
refusal (S08 "考技能证书一般能领多少补贴" kept all three gold files and was refused). The other nine were
in the context and not used: Qwen3-8B filled its list from the collections' one-line items, which echo the
question, and skipped the guides' 700-character passages.

Per question (main), where each gold file of the eleven short questions is lost:

| Question | Gold files | Kept | Each gold file not kept: where it is lost | Kept but not cited by the answer |
|---|---:|---:|---|---|
| S06 各地对人才的补贴都有多少？ | 4 | 2 | 澜州市高层次人才安居办法实施细则: file_passage; 望海市技能大师工作室建设项目实施方案: floor | - |
| S07 人才方面的政策通常有哪些门槛？ | 4 | 0 | 澜州市高层次人才安居办法实施细则: refused; 青屿市博士后科研资助工作指引: file_passage; 临溪市青年人才生活津贴发放规程: refused; 望海市技能大师工作室建设项目实施方案: search | - |
| S09 技能培训方面各有多少补贴？ | 3 | 2 | 北渚市企业新型学徒制培养工作指引: file_passage | 栖川市职业技能提升行动资金使用规定; 澜州市紧缺工种培训激励办法 |
| S11 专利商标这些补贴通常什么时候申请？ | 3 | 1 | 青屿市知识产权高质量发展资金操作规程: file_passage; 临溪市高价值专利培育项目指南: floor | - |
| S14 创业政策通常要满足哪些要求？ | 4 | 0 | 北渚市创业担保贷款办事指南: refused; 望海市初创企业扶持办法: refused; 栖川市创业孵化基地认定和运营评价细则: refused; 澜州市返乡入乡创业支持计划实施方案: refused | - |
| S15 企业搞技术改造、上新设备一般能补多少？ | 3 | 2 | 北渚市绿色制造体系建设实施方案: cut | - |
| S16 技术改造和绿色工厂这类项目都有什么门槛？ | 3 | 0 | 青屿市工业企业技术改造项目资金管理细则: refused; 临溪市首台（套）重大技术装备推广应用指导意见: refused; 北渚市绿色制造体系建设实施方案: refused | - |
| S18 展会和外贸这类补贴一般什么时候报？ | 3 | 0 | 望海市外贸企业开拓国际市场项目申报须知: file_passage; 岚江新区会展业发展专项资金实施细则: file_passage; 澜州市跨境电商发展若干措施操作指引: file_passage | - |
| S19 企业做研发、评高新一般有多少补贴？ | 3 | 2 | 青屿市科技型中小企业创新券使用指南: files | 北渚市高新技术企业培育入库工作方案 |
| S20 研发和高新方面的政策通常对企业有哪些要求？ | 3 | 0 | 栖川市企业研究开发费用奖励管理办法: refused; 北渚市高新技术企业培育入库工作方案: refused; 青屿市科技型中小企业创新券使用指南: file_passage | - |
| S22 建实验室、众创空间通常要达到什么条件？ | 3 | 1 | 岚江新区重点实验室和工程技术研究中心组建办法: floor; 望海市产学研合作项目联合资助计划指南: floor | - |

The file_passage losses are guides whose money or date passage the per-file cap left out (S18 lost
all three this way); the condition questions S07, S14, S16, S20 lost every gold file to the "no evidence" decision.

## The levers

Every lever has a switch in `project/config.py`; one commit each.

### 1. Files first (`survey_file_first`)

The question's topic is taken out of its survey wording, without a model: "考技能证书一般能领多少补贴？" ->
"考 技能 证书", "参展、办展、跨境电商一般各补贴多少？" -> "参展" / "办展" / "跨境 电商" (`survey.survey_topics`:
a stop list of generality words, asks and the words every programme paper uses — 补贴, 申报, 政策, 项目).
Then:

1. each topic is searched alone (hybrid, 240 deep) and its files ranked by their two best passages; the
   first 16 files, taken from the topics in turns, are the candidates;
2. each candidate file gives its 3 passages closest to the question (with the wording expansion of round
   one), searched inside the file;
3. the reranker scores these passages against each topic and against the question;
4. no file whose best topic score reaches 0.05: no evidence. Otherwise the files reaching 0.01 and 5 % of
   the best are "about the topic"; rule-stating types first (round one's order), at most 8; each keeps its
   best answer-shaped passage.

Scored against the topic alone, a guide that never says 补贴 is plainly about 创业 (gold files' best topic
score median 0.17, other files 0.02), and the eight questions about programmes the library lacks score under
0.05 (their best 0.046). Scoring each listed topic alone matters: "参展 办展 跨境 电商" against a passage
about 跨境电商 scores 0.007, "跨境 电商" alone 0.23.

| | tuning: kept | ≥ 3 files | held out: kept | ≥ 3 files | negative: nothing kept |
|---|---:|---:|---:|---:|---:|
| main | 47 / 80 | 11 / 22 | 20 / 56 | 0 / 16 | 4 / 8 |
| files first | 65 / 80 | 14 / 22 | 35 / 56 | 5 / 16 | 7 / 8 |

The refusals are gone (the no-evidence decision is the topic's), and so is most of the pool cut. Tried on
top, not kept: 24 candidate files (65 / 35, the same), the ask's words for who / time questions in the
in-file search (63 / 35; removed). `survey_question_files=8` (the first 8 files of a search for the question
itself join the candidates) changes nothing on the synthetic library and is kept for a real one, below.

### 2. Sentences, not passages (`survey_excerpts`, `survey_excerpt_scoring`, `survey_answering_files`)

`survey.excerpt` cuts a passage to the sentences that answer the ask: an amount next to a word for money
("资助 / 补 / 奖 / 扶持 / 配套 / 激励 …", and the next amounts of the same section), a date tied to applying
("4月17日…系统关闭", "5个工作日内报送", not "五年内不再受理" or "3号窗口"), or the section on who may apply
(headings with 对象 / 主体 / 条件 / 范围 / 资格 …), each under its section heading, after the file's name.
The 139 gold passages of the 38 tuning and held-out questions keep their gold sentence in 137 excerpts,
121 characters on average instead of about 650.

* `survey_excerpt_scoring`: which passage of a file answers is judged on its excerpt (the topic still on
  the whole passage). This is what fixes the condition questions inside a file: the reranker compares the
  question with "二、申请主体 年龄在18至35周岁…", not with a page of boilerplate.
* `survey_passages_per_file` 3 → 5: the in-file search reaches the money or date section more often; the
  excerpts keep the extra scoring short.
* `survey_answering_files`: files with an answering sentence come before files without. On the synthetic
  library one more question keeps three files; on the copy of the user's library it is what keeps an e-mail,
  slides and an article "about 创新" out of the eight files, in favour of two policy summaries.
* `survey_excerpts`: the answer model is given the excerpts instead of the passages (1,290 characters of
  context for a survey question instead of 4,840). It now copies one line per file.
* `survey_about_asked` (a file is about the topic also by its score against the whole question): 68 / 39,
  one held-out file fewer; left off.

| | tuning: kept | ≥ 3 files | held out: kept | ≥ 3 files |
|---|---:|---:|---:|---:|
| files first | 65 / 80 | 14 / 22 | 35 / 56 | 5 / 16 |
| + excerpt scoring | 67 / 80 | 15 / 22 | 38 / 56 | 7 / 16 |
| + 5 passages per file | 67 / 80 | 15 / 22 | 41 / 56 | 8 / 16 |
| + answering files first, + 8 question files (new defaults) | 68 / 80 | 16 / 22 | 41 / 56 | 7 / 16 |

### 3. The graph keeps the route (`survey_keep_route`)

A survey answer is not sent back to a search rewrite because its first passage scores under
`relevance_threshold`: the route made its own no-evidence decision, and its passages come file by file,
not best first. A second attempt (after a failed fact check) searches the question as asked, since a
rewrite loses "一般" / "各…".

### 4. One bad line costs that line (`survey_answer_repair`)

On the copy of the user's library, a conditions question was answered with eight cited lines and then
refused: one citation had the file name with a word added, so the fact check failed the whole answer, the
rewrite searched the same files, the model wrote the same answer, and the reply was "资料里没有找到". In a
survey answer (one line per file):

* a cited file name that no evidence block has, but that is close to exactly one (difflib ratio ≥ 0.8, 0.05
  ahead of the next), cites that block's file (`survey_answer.repair_names`; a shortened name already
  resolves);
* the lines the fact check cannot confirm are dropped and the rest kept and numbered again
  (`survey_answer.drop_lines`), when at least one confirmed line is left.

## Answers on Qwen3-8B

`answer_run.py` asks each question through the app's graph (`build_graph`, the shipped prompts and fact
check) on Qwen3-8B, 30 s apart, and scores the reply by rules: a gold file is cited when a citation names
it; an amount is stated when it appears in a line citing its file. The 16 fact guard questions (round one's)
check that fact questions keep their amounts.

| Configuration | Set | Refused | Gold files cited | ≥ 3 files cited | Amounts stated | Kept files unused | Fact guard: amounts | Median s (survey / fact) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| main (round one's defaults) | tuning | 4 / 22 | 38 / 80 | 8 / 22 | 75 / 137 | 14 / 47 | 41 / 47 | 23.8 / 8.25 |
| main (round one's defaults) | held out | 0 / 16 | 21 / 56 | 2 / 16 | 16 / 39 | - | - | 20.0 / - |
| 1 + 3: survey_file_first, survey_keep_route | tuning | 3 / 22 | 57 / 80 | 14 / 22 | 102 / 137 | - | - | 20.4 / - |
| 1 + 2d + 3, without survey_excerpts | tuning | 0 / 22 | 67 / 80 | 15 / 22 | 122 / 137 | - | - | 21.9 / - |
| 1 + 2d + 3, without survey_excerpts | held out | 0 / 16 | 40 / 56 | 7 / 16 | 25 / 39 | - | - | 18.1 / - |
| 1 + 2d + 3 + survey_excerpts, without survey_answer_repair | tuning | 0 / 22 | 67 / 80 | 15 / 22 | 119 / 137 | - | 41 / 47 | 17.9 / 8.55 |
| 1 + 2d + 3 + survey_excerpts + survey_answer_repair | tuning | 0 / 22 | 67 / 80 | 15 / 22 | 119 / 137 | 3 / 68 | - | 18.9 / - |
| 1 + 2d + 3 + survey_excerpts + survey_answer_repair | held out | 0 / 16 | 45 / 56 | 9 / 16 | 25 / 39 | - | - | 19.0 / - |

* Files first and keeping the route (levers 1 and 3) give most of the gain in files: 38 → 57.
* Sentences (lever 2) end the refusals and the amounts lost in long passages: 102 → 119-122 amounts.
  Giving the model the excerpts (`survey_excerpts`) rather than whole passages ties on the tuning set (67
  files; 119 amounts against 122, the two runs differing by one file on four questions) and is ahead on the
  held-out set (45 against 40 files, 9 against 7 questions with three or more), with a quarter of the context
  (1,290 characters instead of 4,840) and 3 s less per answer. It is on.
* Answer repair (lever 4) gives the same numbers on the synthetic library; it is on for the refusal it
  removes on the user's library, below.
* Of the 68 gold files retrieval keeps, the answer leaves 3 uncited (main: 14 of 47).
* Fact questions: the 16 guard answers keep 41 of 47 amounts, as on main.

Questions about programmes the library lacks:

| Configuration | Questions | Refused | Answered with citations |
|---|---:|---:|---:|
| main (round one's defaults) | 8 | 6 | 2 |
| 1 + 2d + 3 + survey_excerpts + survey_answer_repair | 8 | 7 | 1 |


## Regression

The seven prepared sets of `evaluation/suite`, 5,756 questions (`recall_wording_v1/regression.py`). With
every new switch off the run is identical to round one's defaults (0 rankings changed). With the new
defaults:

| | Questions | Rankings changed | More gold | Fewer gold | books | complex | benchmark | CRUD | MultiHop | Du | T2 |
|---|---:|---:|---:|---:|---|---|---|---|---|---|---|
| main | 5,756 | 0 | 0 | 0 | 0.9420 | 0.9524 | 1.0000 | 0.9065 | 0.5747 | 0.9741 | 0.9741 |
| new defaults | 5,756 | 5 | 1 | 2 | 0.9420 | 0.9524 | 1.0000 | 0.9065 | 0.5747 | 0.9741 | 0.9741 |

The survey detector fires on 5 of the 5,756 questions, all web questions with "一般" ("马桶一般多少钱",
"一般车子开几年", "股权转让一般停牌多久", "德国铁元一般吃多久", "中国逃犯一般躲在哪里"): they are survey-like
(several pages answer them), and they are the 5 changed rankings. One keeps one gold passage more, two one
fewer; every one still keeps what it needs, and the unanswerable questions that get nothing back are the same
(3, 14, 271, 22, 14). No other question takes the survey path, so nothing else can change.

## Cost

Live, empty cache, 12 survey questions (`latency.py`):

| | median | slowest | rerank requests | passages scored | rerank tokens (est.) |
|---|---:|---:|---:|---:|---:|
| main | 1.25 s | 2.65 s | 1.7 | 136 | 39,600 |
| new defaults | 1.57 s | 6.17 s | 3.0 | 311 | 52,200 |

The slowest (6.2 s) is one request that waited; the next slowest is 1.9 s. A survey question now scores
each topic separately (1 to 3 requests) plus the question, on up to 24 files × 5 passages, the question on
excerpts. Fact questions do not take this path.

## On the copy of the user's library

A copy of the user's library (in a scratch folder, with its own cache; nothing from it is in this
repository) answered two survey questions about its innovation-programme papers, live, through the app's
graph:

* the amount question: main and the new defaults both cite the programme guide with its three amounts;
  with the new defaults the file list also leaves out an e-mail, slides and an article that mention the
  topic (`survey_answering_files`) in favour of two policy summaries;
* the conditions question: before lever 4 it was refused (one citation with a mistyped file name failed
  the whole answer, see above); with `survey_answer_repair` it is answered from six cited files and passes
  the fact check.

## What is not solved

On the 22 tuning questions, 12 gold files are still lost before the answer and 3 kept ones are not cited
(`losses.py --name final --answers …`, `results/losses_final.json`):

* **6 cut — files that never name the topic.** One guide (a technology-tackling programme, "技术攻关")
  is gold for all four "创新项目" questions S01–S04 and never says 创新; a green-manufacturing plan for
  S15 "技术改造"; an industry-university fund for S21 "实验室". Their topic score is under the files-first
  candidate cut. Fixing this needs the library to say what a programme is about (a programme type per file,
  as the document cards do for document types, or a topic synonym table), not a deeper search: 24 candidate
  files instead of 16 found one of six.
* **2 file_passage, 2 passage, 1 floor, 1 files** — one each, no common cause found.
* **3 kept and not cited** (S12, S14, S19) — Qwen3-8B wrote one line per file and left out a fourth or
  fifth file it was given; asking for "every file" in the prompt is a prompt-side change and was not tried.
* **Held-out questions**: 41 of 56 gold files kept, 7 of 16 questions keep three or more; amounts 25 of
  39 stated. The held-out wording is harder (no "各", fewer listed topics) and the losses are the same kinds.
* **One question about a programme the library lacks** (N07, marathon events) still gets three files about
  events and sports and is answered from them.
* **The detector**: a survey question without "一般 / 通常 / 各 / 分别 / 都" ("哪些企业能申请创新补贴")
  is still answered as a fact question, and "什么企业能申请" is not recognised as a who-may-apply ask.
  Neither is in the tuning set; both are round one's detector.

## Files

* `make_questions.py` — the 16 held-out and 8 negative questions (written before any lever; the library and
  the 22 tuning questions are round one's, rebuilt after the rename in `corpus_spec.py`).
* `losses.py` — the step-by-step loss breakdown (with `--answers`, the answer stage).
* `answer_run.py` — Qwen3-8B answers through the app's graph, paced, scored by rules.
* `latency.py`, `summary.py`, `shared.py`; `results/` holds every number above.
