# Survey questions, round three (survey_recall_v3)

Branch `feat/survey-recall-v3`, from round two's `feat/survey-recall-v2` (`a917cdc`, not yet merged to
main). Models as shipped: Qwen3-8B, bge-m3, bge-reranker-v2-m3; scoring by rules, no judge model.
`python evaluation/survey_recall_v3/summary.py` prints the tables below from `results/*.json`.

Round two left five things open (its "What is not solved"): guides that never use the question's topic
word, survey questions without 一般 / 通常 / 各 / 分别 / 都, kept files the answer leaves out, the held-out
set, and one question about a programme the library lacks that was answered anyway. This round takes them
in that order.

## Result

Answers on Qwen3-8B, scored by rules (a cited file's name, the amounts stated):

| | Round two | Round three |
|---|---:|---:|
| Tuning (22): gold files cited | 67 / 80 | 78 / 80 |
| Tuning: questions citing ≥ 3 gold files | 15 / 22 | 21 / 22 |
| Tuning: amounts stated | 119 / 137 | 132 / 137 |
| Held out, round two's (16): gold files cited | 45 / 56 | 51 / 56 |
| Held out, new (14, written after the levers were chosen) | 12 / 50 | 32 / 50 |
| Survey questions without 一般 / 各 (27) | 7 / 96 | 73 / 96 |
| Negatives (8): refused | 7 / 8 | 8 / 8 |
| Fact guard (16): amounts stated | 41 / 47 | 41 / 47 |
| Regression (5,756): rankings changed, gold kept | | 2, unchanged |
| Retrieval: median / slowest, rerank tokens a survey question | 1.5 s / 20.5 s, 51k | 2.3 s / 15.9 s, 110k |

Switched on (last commit): `survey_wide`, `survey_topic_related`, `survey_file_candidates=24`,
`survey_topic_passages=2`, `survey_missing_files`. Document cards with a 领域 field were tried and not kept.

## 1. Guides that never name the topic (`survey_topic_related`)

Round two lost 6 of the 80 tuning gold files before the reranker ("cut"): the technology-tackling guide
is gold for all four "创新项目" questions and never says 创新, a green-manufacturing plan answers "技术改造",
an industry-university fund "实验室 / 平台". On the held-out set the same cause lost 5 of 15 (会展 under
开拓市场, 首台套 and 绿色制造 under 技改, 产学研 under 实验室, 技术攻关 under 创新).

The reranker does not link the words either (`probe_topics.py`): against "创新" the tackling guide's name
scores 0.006 (rank 42 of the library) and its file entry (name, headings, opening) 0.020 (rank 27);
the industry-university fund against "实验室" 0.002. So scoring a file-level description does not help; the
link has to come from words: a document card written by the model, or a table.

**Document cards (tried, not kept).** Qwen3-8B wrote a card for each of the 135 files with one more field,
领域: "what a policy paper supports, 2 to 4 general words" (`make_cards.py`, the app's card prompt plus one
rule). A card matches a topic when a field contains the topic word or shares a two-character piece with it
that is not common to all programme papers (`probe_cards.py`). Result:

* it finds the tackling guide (its card says 科技创新), so 4 of the 6 tuning losses;
* it misses the green-manufacturing plan ("绿色制造、工业发展、节能减排" does not meet "技术改造"), the
  first-set-equipment opinion and the industry-university fund;
* "创新" matches 24 files that are not gold, because the model writes 科技创新 on most policy cards;
* it costs 77 tokens more per card (936 against 859 on 16 files with both prompts), a new column and a
  re-carding of every file in the library, and the extra field changed the document type on 3 of 16 files.

**A table of related programme words (kept).** `survey.RELATED_TOPICS` maps a topic word to the words of
programmes under it (创新 → 技术攻关, 成果转化, 研发; 技术改造 → 技改, 智能化改造, 绿色制造, 设备更新; 实验室 →
工程技术研究中心, 研发平台, 产学研; 市场 → 参展, 会展, 跨境电商, 出口 …; 21 entries, at most 3 related words
a question). Each related word is searched and scored as one more topic. No model call. Three rules keep
it honest:

* the no-evidence decision is the question's own topic's alone: with related words deciding it, "围棋培训
  机构通常有哪些申报条件" kept 8 training guides through 职业技能;
* a file found only through a related word must be named like a paper that states rules (…指南, …办法,
  …细则, …方案, …意见): on the copy of the user's library, 研发 / 成果转化 brought in business plans;
* `survey_file_candidates` 16 → 24: with related words the 16 candidate places were shared by up to six
  topics, and a real innovation guide was crowded out.

Scoring every topic against every passage of 24 files tripled the rerank tokens (153k a question, median
7.0 s); `survey_topic_passages=2` judges a file's topic on its two passages closest to the question (all
five are still scored against the question): 110k tokens, 2.3 s, and the same or more gold kept on every
set. A variant that scored related words only on the files they brought in kept fewer (held-out 48 → 43)
and was dropped.

The table is written from how policy programmes are named in general, but this library's nine groups were
known when it was written, so the tuning and held-out gains overstate what it does for an unknown library.
On the copy of the user's library it brings the programme guides of a question in; what it does there is
below.

## 2. Survey questions without a generality word (`survey_wide`)

Written before the detector changed (`detector_questions.py`): 28 survey questions without 一般 / 通常 /
各 / 分别 / 都 ("哪些企业能申请创新补贴", "创新项目补贴有多少", "有什么补贴可以申请", "什么企业能申请技术改造补贴")
and 28 look-alike fact questions that must stay fact questions: a place and its programme, a title in 《》,
"这个 / 这份 / 文件里", an article number, a code, a person, the user's own record, a how-to, a definition, a
yes / no, questions not about programmes. Round two's detector: 0 of 28 surveys, 28 of 28 facts right.

The widened detector (`survey.survey_question(..., wide=True)`): a programme word (补贴 / 补助 / 资助 / 扶持 /
奖励 / 津贴 / 政策; 申请 / 申报 only next to a programme word or at the end — "转岗申请免于审批" is not one) with an
ask (an amount, a time, who may apply — now also "什么企业 / 什么样的人" —, or which programmes: "什么补贴",
"有补贴吗", "哪些扶持"), and not pointing at one thing: a title, a code, 这 / 那份 / 该 / 本办法 / 文件里 / 指南里, an
article, a meeting or contract, a past record (拿了 / 去年 / 到账), a how-to or a definition, a place, person or
organisation by jieba's tags (not its "other name" tag, which it also gives 申报), or a place the library's own
file names start with (`survey.place_names`: "青屿市…指南" → 青屿; jieba does not know every town). 28 of 28
surveys and 28 of 28 facts right — on the questions it was written against; the new held-out set below is
the check.

Firing on the regression sets (5,756 questions): 5 with or without the switch, the same 5 web questions as
round two. Without the "申请 next to a programme word" rule it also fired on 2 enterprise-rule questions of
the older sets ("…转岗申请免于…审批的单笔金额上限是多少").

On the synthetic library's own questions it sends 27 of the 31 "generic" one-file questions
("发明专利授权一件补贴多少？") to the survey route; their gold passage is kept by 31 of 31 instead of 30 (the
fact path), and the 62 fact questions keep 0.984.

The 27 cue-less survey questions with gold (every guide of their group): the fact path keeps 6 of 96 gold
files and no question three; the survey route keeps 59 of 96 (11 questions three or more); with the related words 63, with all of round three 70
(15 of 27), and the answers cite 73 of 96 gold files against round two's 7.

## 3. Kept files the answer leaves out (`survey_missing_files`)

Qwen3-8B often stops after three or four files. After the answer, a kept file that it does not cite gets one
line copied from its excerpt with its citation (`survey_answer.missing_lines`), before the fact check, the
way the rejection note and the conflict line are added in code. Offline, on round two's 22 tuning answers
(`missing_sim.py`, the same retrieval):

| Which kept files get a line | Lines added | Of gold files | Gold files cited | Amounts stated |
|---|---:|---:|---:|---:|
| none (round two) | 0 | | 67 | 119 |
| every file with an answering sentence | 31 | 4 | 71 | 125 |
| … named like a rule-stating paper | 18 | 4 | 71 | 125 |
| … and its name carries a topic word, not a collection (kept) | 4 | 3 | 70 | 125 |

The model had good reason to leave most of them out (a start-up measure under a talent question, a
student-training notice under skills training, the policy collections). The kept rule adds 4 lines; 3 are
gold files, the fourth a voucher guide about 创新 under an innovation question. On round two's held-out
answers it adds none. A line is copied from the passage it cites, so the fact check passes it; live, with all of round three's retrieval, it added 1 line on the tuning set (S19, the gold
high-tech guide: tuning 77 → 78 files, 129 → 132 amounts), none on the held-out set, the negatives or
the cue-less set, and 3 lines to 2 of the 14 new questions; the fact check dropped none of them. Two of
the lines repeated a section heading and a clause their excerpt held twice ("二、申报条件 …；三、基本条件
（一）…"); the line now drops a leading heading and a repeated clause (a change of wording only, made
after the runs: the files and amounts are the same).

## 4. The held-out set

Round two's held-out set lost 15 gold files: 5 to the cause of item 1, 3 file_passage, 3 floor, 2 files,
2 passage. With round three's settings: 48 of 56 kept, the losses 3 file_passage, 4 files (the 8-file
limit, now binding: more candidates are about the topic), 1 passage. A 10-file limit kept no more (the
10-passage limit binds too) and was not kept.

A new unseen set (`make_holdout.py`, 14 questions, written after the levers were chosen, with and without
generality words): round two keeps 13 of 50 gold files, round three 28. The detector finds 10 of the 14
(round two 4); the four it misses are open below.

## 5. A programme the library lacks, answered anyway

N07 "各地的马拉松赛事补贴都是几月申报？" was not a threshold problem: jieba tags 马拉松赛 as a person's name,
so round one's detector took it for a question about one named thing, and the fact path, which has no topic
check, kept three files about events. With `survey_wide`, a question about a programme counts only places
and organisations (and the library's places) as "one thing"; N07 is a survey question, its topic "马拉松赛
事" scores under the 0.05 floor, and nothing is kept: 8 of 8 negatives keep nothing. A per-topic relative
threshold was not needed.

## Answers on Qwen3-8B

`answer_run.py` through the app's graph, paced; overrides `survey_wide, survey_topic_related,
survey_file_candidates=24, survey_topic_passages=2, survey_missing_files` (round three) against round
two's defaults. Files `work/answers/*.jsonl` (not committed), totals in `results/answers_*.json`.

| Set | Configuration | Refused | Gold files cited | ≥ 3 files cited | Amounts stated | Median s |
|---|---|---:|---:|---:|---:|---:|
| tuning | round two | 0 / 22 | 67 / 80 | 15 / 22 | 119 / 137 | 18.9 |
| tuning | round three | 0 / 22 | 78 / 80 | 21 / 22 | 132 / 137 | 20.1 |
| tuning | round three without survey_missing_files | 0 / 22 | 77 / 80 | 20 / 22 | 129 / 137 | 21.5 |
| held out | round two | 0 / 16 | 45 / 56 | 9 / 16 | 25 / 39 | 19.0 |
| held out | round three | 0 / 16 | 51 / 56 | 13 / 16 | 30 / 39 | 18.4 |
| new set | round two | 0 / 14 | 12 / 50 | 1 / 14 | 5 / 71 | 8.3 |
| new set | round three | 0 / 14 | 32 / 50 | 7 / 14 | 33 / 71 | 18.3 |
| cue-less | round two | 0 / 27 | 7 / 96 | 0 / 27 | 1 / 77 | 8.3 |
| cue-less | round three | 0 / 27 | 73 / 96 | 15 / 27 | 51 / 77 | 20.0 |

The fact guard states 41 of 47 amounts in both rounds; the one it misses entirely (E2-1, a
multi-document question answered from a summary table) gives the same answer word for word in both.
The negatives: round three refuses 8 of 8 (N07 now too). The new set's and the cue-less set's
round-two medians are short because round two answered most of them on the fact path, from one or
two files.

## On the copy of the user's library

Three questions about one programme kind, on a copy of the user's library (its own cache, run from
the scratchpad; nothing of it is in this repository), answered by both rounds:

* "一般的…能给多少补贴": the same in both rounds: 5 files cited, the programme's main guide among them
  with its three amounts.
* "…一般需要满足什么条件": both cite the main guide; round three cites 6 files to round two's 5, one of
  them a business plan (a related word's file that the rule did not stop: its name ends like a plan,
  but its passage answered the ask).
* "哪些…能申请…补贴" (no generality word): round two took it for a fact question and cited one policy
  collection; round three takes it as a survey and cites 6 files, guides of the programme among them
  but not the main guide, and one business plan.

So on the user's library the wide detector does what it should, and the related words bring in a
little noise (a business plan in 2 of 3 answers), about round two's level for this library.

## Cost and regression

Retrieval of the 16 survey questions, live (`results/latency_v3.json`):

| | median | slowest | rerank requests | passages scored | rerank tokens |
|---|---:|---:|---:|---:|---:|
| round two | 1.53 s | 20.53 s | 2.8 | 263.7 | 51,450 |
| round three | 2.32 s | 15.94 s | 5.9 | 493.8 | 109,707 |

Without `survey_topic_passages` round three was 6.96 s and 153k tokens. A whole answer takes about one
second more at the median (tuning 18.9 → 20.1 s); the answer model, not retrieval, is most of it.
The rerank tokens (bge-reranker-v2-m3) are free on SiliconFlow but count against its rate limit.

The 5,756-question regression (`results/regression_sv3_v3.json`, against round two): 2 rankings
changed (one each in duretrieval and t2retrieval), no question keeps more or less gold, "keeps what it
needs" and "unanswerable left empty" identical on all 7 sets. The detector fires on the same 5 web
questions as round two. Fact questions keep 0.984 of their gold passages (unchanged); the 31 one-file
"generic" questions 1.0 (round two 0.968).

## What is not solved

* **The detector's misses** on the new set: a question without a programme word ("…能领多少钱",
  "…多少钱" with 钱 alone), jieba tagging a word as a person's name (申请专利), and a word between the
  ask and the programme word ("哪些创业补贴"). Widening further costs fact questions; each would need
  its own look-alike set first.
* **The related-word table is hand-written**: it knows how policy programmes are usually named, and
  this library's nine groups were known when it was written. For a library about other programmes it
  adds nothing; doc cards with a field were the general alternative and were too loose with Qwen3-8B.
* **Cost**: 2.1 times the rerank tokens of round two for a survey question (110k), which brings the
  reranker's rate limit nearer on a slow plan.
* **The 8-file and 10-passage limits** now bind on the held-out set (4 files lost); raising them kept
  no more because the answer's passage limit binds too.
* **file_passage losses** (3 on each set): the file is kept but its passage that answers is not among
  the five scored.
* **On the copy of the user's library** a business plan was cited in 2 of 3 answers (round two's noise
  level), and the cue-less question cited the programme's guides but not its main one.
* **The rejection note fired on a survey answer** (round two, S19, 1 of 22): the code-side "资料里没有
  找到…" line should not judge a survey answer by one file's miss. Not touched this round.

## Files

* `detector_questions.py` — the detector's 28 + 28 questions (`detector.jsonl`) and the 27 surveys with gold
  (`wide.jsonl`); written before the detector changed.
* `make_holdout.py` — the 14 unseen questions (`holdout2.jsonl`), written after the levers were chosen.
* `probe_topics.py`, `make_cards.py`, `probe_cards.py` — item 1's probes and the card experiment.
* `missing_sim.py` — item 3 offline.
* `summary.py`, `round3.py`; round two's `losses.py`, `answer_run.py` and `latency.py` take `--into` and
  `--question-file` for this folder. `results/` holds every number above.
