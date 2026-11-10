# 用对话找文件：设计文档

状态：已实现（2026-10-24）。下面是设计稿原文；实现时的调整见文末“实现与设计稿的差异”。

## 0. 目标与边界

产品目标：“文件很多时，通过对话精确找到想要的文件，并给出正确回答”。现有链路只有“召回片段 → 生成回答”，缺两样东西：

1. 按元数据过滤（时间、类型、文件夹、文件名）；
2. 以**文件**为单位排序（同一主题多个文件时，片段互相挤占前 5 名）。

本设计新增一条“找文件”路径（不生成、不做事实校验，目标 2 秒内），一个文件级索引，以及“限定在某文件 / 某文件夹内提问”的能力。普通问答链路的默认行为不变（不传范围时，检索代码路径与现在逐字节一致），以保证 benchmark_v1 / books_v1 不下降。

## 1. 整体结构

```
用户问题
  └─ analyze_query（LLM 规划 + 规则兜底）
        ├─ intent=find_file ──► search_files 节点 ──► END   （不生成、不校验）
        ├─ intent=knowledge ──► retrieve(scope) ─► … 原链路 …
        └─ 其他意图不变

快速搜索窗（Alt+Space）── GET /api/v1/files/search?q=… ──► FileSearchService（不经 LLM）
```

新增模块：

| 模块 | 职责 |
|---|---|
| `project/retrieval/file_filters.py` | 规则抽取过滤条件：时间短语 → 时间区间，类型词 → 扩展名，文件夹、文件名关键词；纯函数，便于单测 |
| `project/retrieval/file_search.py` | `FileSearchService`：元数据过滤 → 内容检索 → 按文件聚合打分 → 片段挑选 |
| `project/retrieval/file_index.py` | 文件级条目（摘要 + 文件名）的生成与写入、补齐 |
| `rag_agent/` | `QueryPlan` 加 `find_file` 意图和 `file_filter` 字段；`search_files` 节点和工具 |
| `api/main.py` | `/files/search`、`/files/{id}/open`，chat 请求加 `scope` |
| `ui/static/` | 文件卡片、范围标签、快速搜索页 `quick.html` |
| `desktop/src-tauri/` | global-shortcut 插件、`quicksearch` 小窗、改快捷键命令 |
| `evaluation/file_search_v1/` | 30 道左右找文件题、运行脚本、报告 |

## 2. 意图识别与过滤条件抽取

### 2.1 规划器

`QueryPlan.intent` 增加 `find_file`，新增可选字段：

```python
class FileFilter(BaseModel):
    time_phrase: str = ""          # 原文时间短语，如“上周”“上个月”“2024年3月”，不让 LLM 算日期
    time_field: Literal["added", "modified", "any"] = "any"   # “加的/放进来的/导入的”→added，“改过的/更新的”→modified
    file_types: list[str] = []     # pdf / word / excel / ppt / markdown / txt / image
    folder: str = ""               # 文件夹名或路径片段
    name_keywords: list[str] = []  # 文件名里可能出现的词，如“报价单”
    topic: str = ""                # 内容主题，如“KV Cache”；可为空（纯元数据查询）
```

提示词里给出几类例子：“那份讲 KV Cache 的 PDF 在哪”“上周加的报价单”“有哪些关于 Agent 评估的资料”“D 盘 论文 文件夹里的东西”。

**取舍：日期由代码算，不由 LLM 算。** Qwen3-8B 不知道“今天”是几号，算相对日期也不稳定；LLM 只照抄时间短语，`file_filters.resolve_time(phrase, now)` 确定性地换算成 `[start, end)`。支持：今天/昨天/前天、本周/上周/这周、本月/上个月、今年/去年、最近 N 天/周/个月、N 天前、具体年/年月/年月日（`2024年3月`、`2024-03`、`3月份`）。认不出的短语忽略并在结果里提示“未识别的时间条件”。

### 2.2 规则兜底与“快速通道”

`_enforce_plan_policy` 增加两条：

- 规划器给出 `find_file` 但问题里没有任何“找文件”线索（在哪/哪份/哪个文件/哪些资料/文件/文档/报告/PDF 等类型词/时间词 + 加的/放进来的）时降级为 `knowledge`，防止把“KV Cache 的原理是什么”误判成找文件。
- 反之，问题命中强线索（“…在哪”“找一下…文件”“有哪些关于…的资料/文档”“上周加的…”）而规划器给了 `knowledge` 时，升级为 `find_file`。

同一套规则（`file_filters.parse_query(text)`）也用于快速搜索窗：不调用 LLM，直接抽取类型词、时间短语、`文件夹:xxx`/`in:xxx`，剩下的部分同时作为文件名关键词和主题。这样快速窗每次按键只有一次 embedding + 一次 rerank 的网络开销。

规划器输出的 `file_filter` 与规则抽取结果合并：规则抽到的类型和时间优先（确定），LLM 补主题和文件名关键词。

### 2.3 时间字段

- `added`：登记表 `files.created_at`（文件第一次被登记的时间，即“放进来的时间”）。
- `modified`：`files.mtime`（磁盘修改时间）。
- `any`（默认）：两者任一落在区间内即可。

## 3. `search_files` 算法

```
filters ─► 1. 元数据过滤（登记表 SQL）─► 候选文件集合 C（可能是“全部”）
query  ─► 2. 内容检索：
             a. 片段：hybrid 检索，限定 doc_id ∈ C（C 很大时全局检索后过滤）
             b. 文件级条目：摘要+文件名的 hybrid 检索，限定 C
             c. 文件名匹配：name_keywords / 原查询 对文件名做 bigram 重叠
          3. 一次 rerank：top 24 片段 + top 12 文件条目 一起送 BGE Reranker，分数可比
          4. 按文件聚合打分，取前 N（默认 8）
          5. 每个文件挑 1~2 个最相关片段（带页码 locator）
```

### 3.1 元数据过滤

`FileRegistry.search_files(types, time_range, time_field, folder, statuses=SEARCHABLE_STATUSES)` 返回 `file_id, name, path, size, mtime, created_at, status`。文件夹匹配：`path_key` 包含规范化后的文件夹片段（大小写不敏感），或等于某个监控文件夹根。

结果为空时：不放宽条件，直接返回“没有符合条件的文件”，并回显识别出的条件（用户能看到是哪个条件过严）。

### 3.2 打分

对文件 f：

```
content(f) = max_rerank(f) + 0.05 · ln(1 + hits(f))         # hits = f 在前 24 片段里的个数
file(f)    = rerank(文件条目 f)                              # 无条目时为 0
name(f)    = 文件名匹配度 ∈ [0, 1]
score(f)   = max(content(f), file(f)) + 0.3 · name(f)
```

- 取 max 而不是平均：长文件的平均分会被无关章节拉低。
- 命中片段数用对数、权重小：只在分数接近时起区分作用，防止长文件凭片段多压过短文件。
- 文件条目与片段取 max：主题类问题（“有哪些关于 Agent 评估的资料”）往往由摘要命中，细节类问题（“讲 KV Cache 的那份”）由片段命中。
- 文件名加分：“报价单”“周报”这类问题用户记得的是文件名。

**没有主题时**（“上周加的 PDF”）：跳过内容检索，按 `name(f)` 再按时间倒序排序。

**仅有主题、无过滤条件且问“有哪些”时**：返回分数 ≥ `0.3 × 最高分` 的文件，最多 N 个，避免把无关文件凑满列表。

权重初值按上面写，用评测集的开发半边调一次后冻结，在报告里写明。

### 3.3 片段级限定检索

`EmbeddedStore.search_hybrid(query, vector, limit, doc_ids: set[str] | None = None)`：

- `doc_ids is None`：现有逻辑，不改一行（保证原评测不变）。
- 候选文件里的片段数 ≤ 20 000：精确扫描这些文件的向量（`vec_distance_cosine` + `WHERE c.doc_id IN (…)`，和现有 `search_in_source` 同一做法），BM25 加同样的 `doc_id IN` 条件。sqlite-vec 暴力扫 2 万个 1024 维向量约 30~60 ms。
- 更多时：全局 KNN 取 `k = 4 × limit` 后过滤，不够再用精确扫描补。

`HybridRetriever.retrieve(..., doc_ids=None)` 透传；对比检索（`_retrieve_comparison`）同样透传。Milvus 后端用 `doc_id in [...]` 表达式过滤（Milvus 原生支持）。

### 3.4 耗时预算（目标 2 秒，不含规划器）

| 步骤 | 预计 |
|---|---|
| 登记表过滤 | < 10 ms |
| 查询 embedding（BGE-M3） | 200–500 ms |
| 片段 + 文件条目 hybrid | 50–150 ms |
| rerank 36 条 | 300–700 ms |
| 聚合 + 片段挑选 | < 10 ms |

对话里另有规划器一次 LLM 调用（Qwen3-8B 约 1–2 s）。快速搜索窗不走规划器。快速窗输入做 250 ms 防抖，并对同一查询文本做 LRU 缓存（embedding 缓存 256 条）。rerank 失败或超时（>1.5 s）时退回 RRF 分数，不报错。

## 4. 文件级索引

### 4.1 摘要怎么来

**默认用抽取式摘要，不调 LLM**：

```
文件名（去扩展名）
标题路径：前 12 个不重复的章节标题，用“ / ”连接
正文开头：去掉目录/版权页后的前 2~3 段，截到 300 字
```

取舍：LLM 摘要更通顺，但每个文件一次 LLM 调用，用户首次添加几百个文件时要多花几分钟和配额，而且限流时会拖慢入库；抽取式摘要对“主题”检索已经够用（章节标题本身就是很好的主题词），也不会编造。预留 `Settings.file_summary_mode = "extractive" | "llm"`，`llm` 模式在入库后台线程里生成（失败回退抽取式），本次只实现抽取式，LLM 模式作为后续项。

### 4.2 存在哪里

单独的表，不混进 `chunks`：

```sql
CREATE TABLE file_entries (
  doc_id TEXT PRIMARY KEY, source TEXT, summary TEXT, version INTEGER, rowid INTEGER UNIQUE
);
CREATE VIRTUAL TABLE file_entries_fts USING fts5(tokens, ...);
CREATE VIRTUAL TABLE file_entries_vec USING vec0(embedding float[1024] distance_metric=cosine);
```

取舍：如果把文件条目写成一种特殊 chunk，普通问答检索会把“摘要”召回进前 5 名，挤掉真正的正文证据，并且回答会引用摘要。分表后普通问答完全不受影响。代价是 `delete_document`/`rename_source`/`reassign_document` 要同步处理这张表（同一事务内）。

### 4.3 写入与补齐

- 入库：`IngestionPipeline.index_document` 在写 chunks 的同一把锁内生成并写入文件条目（多一次 embedding，单条，约 0.2 s）。
- 补齐：`LibraryService` 启动对账时，列出 `status=indexed` 但 `file_entries` 中没有、或 `version` 低于 `FILE_ENTRY_VERSION` 的文件，从 `parsed/<doc_id>.json` 生成摘要（无需重新解析原文件），批量 embedding（每批 32 条）写入。失败的下次启动重试。
- 文件改名：`rename_source` 同步更新 `file_entries.source`，并标记需要重新生成（文件名是条目内容的一部分）。

## 5. API

```
GET  /api/v1/files/search?q=…&limit=8&types=pdf,docx&folder=…&since=…&until=…&time_field=added
POST /api/v1/files/{file_id}/open          # 用系统默认程序打开（仅本机，复用 _local_only）
POST /api/v1/chat, /api/v1/chat/stream     # body 增加 scope
```

`/files/search` 返回：

```json
{
  "query": "上周加的报价单",
  "filters": {"time_range": ["2026-10-12", "2026-10-19"], "time_field": "added", "file_types": [], "folder": "", "name_keywords": ["报价单"], "topic": "", "unrecognized": []},
  "elapsed_ms": 812,
  "files": [
    {"file_id": "…", "name": "…", "path": "…", "ext": ".xlsx", "size": 12345, "mtime": "…", "added_at": "…",
     "status": "indexed", "score": 0.93, "matched_by": ["name", "content"],
     "snippets": [{"text": "…", "locator": "page:12", "heading": "…"}]}
  ]
}
```

`ChatRequest.scope = {"doc_ids": [...], "folder": "D:\\论文"}`（可选）。后端把 folder 解析成该文件夹下的 doc_id 集合（`registry.files_under`），合并进 `State.scope_doc_ids`。范围是请求级参数，不存进会话状态：前端在一个会话里记住当前范围并每次随请求发送，用户点掉范围标签即恢复全库。

对话里 `find_file` 结果通过 SSE `final` 事件的新字段 `files` 返回（同上结构），`answer` 是一句短文本（“找到 3 个文件：…”），便于历史记录和导出可读。限定范围下检索不到时的拒答文案改为“在《xxx》中没有找到相关内容”。

## 6. Agent 图

```
analyze_query ─(find_file)─► search_files ─► END
             ─(knowledge)─► retrieve ─► …（retrieve 读取 state.scope_doc_ids 传给 retriever）
```

- `search_files` 节点调用 `FileSearchService.search(topic or rewritten_query, filters)`，把结果写入 `state.files`，`answer` 写短文本，`messages` 追加一条 AI 消息（让后续追问“第二个文件讲了什么”有上下文：消息里带文件名列表）。
- 同时注册一个 LangChain `StructuredTool`（`search_files`，参数 schema 即 `FileFilter` + query），和现有工具放在同一个 `create_enterprise_tools` 字典里，方便以后改成工具调用式 Agent。
- 范围内问答：`scope_doc_ids` 非空时 `retrieve` 走限定检索；`rewrite_search` 重试同样限定。

## 7. 前端

### 7.1 文件卡片（聊天里）

```
┌──────────────────────────────────────────────┐
│ [PDF] AI Agents 实战.pdf                      │
│ D:\书\AI\AI Agents 实战.pdf · 修改于 2026-09-12 │
│ “…KV Cache 通过缓存已计算的 Key/Value…” 第 87 页 │
│ [打开阅读面板] [在资源管理器中显示] [基于这个文件提问] │
└──────────────────────────────────────────────┘
```

- 图标按扩展名（PDF/Word/Excel/PPT/Markdown/文本/其他），用内联 SVG 或 CSS 色块，不引外部资源。
- “打开阅读面板”：复用现有阅读面板，定位到片段的页码。
- “在资源管理器中显示”：现有 `/files/{id}/reveal`。
- “基于这个文件提问”：在输入框上方加范围标签“仅在：xxx.pdf ✕”，之后的问题都带 `scope.doc_ids`；聚焦输入框。
- 文件状态为 missing/offline 时卡片置灰并标注，可打开阅读面板（索引里有文本），不能在资源管理器显示。

### 7.2 范围选择（普通问答）

输入框左侧加“范围”按钮：全部 / 某个文件（从文件列表中搜索选择）/ 某个文件夹（监控文件夹及其子文件夹）。文件库页面的文件行、文件夹行也加“在此提问”入口。

## 8. 全局快捷键快速搜索（Tauri）

- 依赖 `tauri-plugin-global-shortcut = "2"`，capabilities 加 `global-shortcut:allow-register / unregister`。
- 第二个窗口 `quicksearch`：640×420，无边框、置顶、不在任务栏显示、默认隐藏、居中；内容是后端提供的 `/ui/quick.html`（和主界面同源，复用 token 注入）。
- 按快捷键：显示并聚焦小窗、清空输入；失去焦点或 Esc 隐藏。
- 输入：250 ms 防抖调用 `/files/search`，上下键选择。回车：文件存在时用系统程序打开（`/files/{id}/open`）；Ctrl+回车 或文件不存在：显示主窗口并打开阅读面板（Rust 端 `emit_to("main", "open-reader", {file_id, locator})`，主界面监听）。
- 快捷键设置：存在后端 `preferences.json` 的 `quick_search_shortcut`（默认 `Alt+Space`）。设置页修改时调用 Tauri 命令 `set_quick_search_shortcut(accel)`：先注册新的，成功后注销旧的并保存；失败（被其他程序占用、格式不对）则保留旧的并在设置页提示。启动时 Rust 端从后端读取设置后注册。
- 已知冲突：Windows 的 `Alt+Space` 原本打开窗口系统菜单，注册后被覆盖（PowerToys Run 也是这样）；PowerToys Run 已占用时注册会失败，设置页提示改键。

## 9. 评测

`evaluation/file_search_v1/`：

- 语料：复制的真实数据（3 本书）+ benchmark_v1 的 50 份同模板企业文档（同一制度在总部/分公司、md/docx/pdf 多个版本，天然是“相似文件区分”题）+ `sample_data/generated` 4 份。在评测副本里用 `os.utime` 和改写登记表 `created_at` 设定固定的修改/入库时间，`now` 固定，结果可复现。
- 题目（约 30 道，`questions.jsonl`）：按主题 8、按时间 6、按类型 5、按文件名模糊 6、相似文件区分 6（部分题组合多个条件）。每题标注 `gold_files`（可多个，如“有哪些…”题）和期望过滤条件。
- 指标：Hit@1、Hit@3（任一 gold 在前 k）、“有哪些”题的 Recall@N、过滤条件抽取正确率、耗时 p50/p95（分别统计 `search_files` 本身和含规划器的对话端到端）。
- 对照：只用片段聚合（无文件条目、无文件名分）作为基线，说明各部分的贡献。
- 回归：benchmark_v1 和 books_v1 的 production 模式各跑一次，和改动前的结果对比。

## 10. 测试计划

- `test_file_filters.py`：时间短语换算（固定 now，覆盖跨月/跨年/周一边界）、类型词、文件夹语法、意图规则（正例/反例各若干）。
- `test_file_search.py`：用假 embedding/reranker 和临时 EmbeddedStore + FileRegistry，验证元数据过滤、聚合打分、片段挑选、无主题排序、空结果。
- `test_embedded_store.py` 增加：`doc_ids` 限定检索、文件条目增删改名、`doc_ids=None` 时结果与旧实现一致。
- `test_nodes.py`/`test_graph_integration.py`：`find_file` 路由、`search_files` 节点、`scope_doc_ids` 透传到 retriever。
- `test_api_*`：`/files/search`、`/files/{id}/open`（非本机拒绝）、chat `scope`。
- 补齐：已入库文件缺文件条目时启动对账能补上。

## 11. 已知限制（预期）

- 时间短语只支持常见中文表达；“春节前”“上次开会那天”这类不认识。
- 抽取式摘要对扫描版 PDF（无文本层且未开图片理解）为空，只能靠文件名。
- Milvus 后端没有文件级条目，只用片段聚合 + 文件名。
- 对话里的找文件包含一次规划器 LLM 调用，端到端可能略超 2 秒；快速搜索窗不受影响。
- 全局快捷键只在桌面版可用；浏览器访问时没有快速窗。

## 12. 实现与设计稿的差异

- **文件级条目的写入点**：没有改 `ingestion/pipeline.py` 和 `library/service.py`（当时 PDF 解析会话在改）。改为在 `RAGSystem` 里包一层索引函数 `_index_file`：先 `index_document` 写片段，再写文件级条目；条目失败只记日志，下次启动的补齐会重建。启动补齐在 `start_library` 后的后台线程里跑。
- **正在重新入库的文件**：状态为 indexing/pending/failed、但已有片段（`chunks > 0`）的文件也算候选，否则解析器升级后重新入库期间，这些文件会从找文件结果里消失。
- **精确标识加分**：评测第一轮发现，同模板文件里的“LOG-04-ZX11”这类编号，重排模型几乎打不出差别（0.988 对 0.994），命中片段数加分反而把总部版本排到前面。加了 `CODE_BONUS = 0.2`：片段里原样出现问题中的标识（含字母和数字的串，如 `LOG-04-ZX11`、`GPT-4o`）时加分。
- **对话里的快速通道扩大**：除“在哪/哪份/找一下”之外，“讲/介绍/关于…的书/资料/文件”“那份/那本…文档”也算强信号；另外，30 字以内、不是问句、并且和某个已有文件名高度匹配（≥ 0.75）的消息（如“面试题200问”），直接按找文件处理。这些都不调规划器 LLM。
- **时间基准**：`search_files` 节点用 `FileSearchService.clock` 解析相对时间，评测可以固定“现在”。
- **快速搜索窗打开文件**：回车调用新接口 `POST /files/{id}/open`（`os.startfile`）；原文件不在时自动改为打开阅读面板。
