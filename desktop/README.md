# Windows 桌面客户端

结构：**Tauri 2 外壳**（`desktop/src-tauri`）+ **PyInstaller 打包的 FastAPI 后台**（`packaging/backend.spec`，onedir）。界面直接复用 `project/ui/static`，不做任何构建。

```
寻寻 (Tauri, EnterpriseKB.exe)
 ├─ 启动窗口 splash.html        ← 等后台健康检查通过
 ├─ 主窗口 index.html            ← 注入 window.__RAG_DESKTOP__ = {apiBase, token}
 ├─ 托盘：打开 / 退出；关闭窗口时隐藏到托盘（可关）；单实例
 └─ backend\rag-backend.exe      ← 127.0.0.1:<随机端口>，RAG_APP_TOKEN=<每次启动随机 256 bit>
                                    --parent-pid <外壳 PID>（外壳崩溃时后台自行退出）
```

## 安全模型

- 后台只绑定 `127.0.0.1`，端口每次随机分配。
- 外壳每次启动生成 32 字节随机令牌，通过**环境变量**传给后台（不放在命令行，避免出现在进程列表里），并只通过初始化脚本注入给自己的 webview。
- 后台的 `TokenAuthMiddleware`（`project/api/security.py`）要求除 `GET /api/v1/health/live` 以外的**所有**请求带 `X-App-Token`（或 `Authorization: Bearer`），否则返回 401，包括 `/ui`。网页即使对 localhost 发起跨站请求，也拿不到令牌。
- CORS 只放行 Tauri 的源（`http://tauri.localhost`）和开发用的 `:8000`；自定义请求头会强制预检。
- 外壳的 CSP：`script-src 'self'`，`connect-src` 只允许 `127.0.0.1` 和 IPC。
- 硅基流动 API Key 通过 `keyring` 存入 Windows 凭据管理器（服务名 `EnterpriseKnowledgeBase`，用户名 `siliconflow_api_key`），不写入任何文件；接口只返回打码后的值。

## 数据目录

| 模式 | 位置 |
|---|---|
| 打包后 | `%APPDATA%\EnterpriseKnowledgeBase\` |
| 开发（`python project/app.py`） | 仓库里的 `data/` |
| 任意模式 | `--data-dir <路径>` 或环境变量 `RAG_DATA_DIR` |

目录里包括 `knowledge.sqlite3`（索引和文件表）、`backups/`（[自动备份](../design/DATA_SAFETY.md)）、`parsed/`、`parents/`、`uploads/`、`cache/`（jieba 缓存）、`logs/backend.log`（5 MB × 4 轮转）和 `preferences.json`（所选模型）。卸载程序**不会**删除这个目录。

## 后台命令行

```
rag-backend.exe [--host H] [--port P] [--token T] [--data-dir D] [--parent-pid PID]
```

外壳命令行上的 `--data-dir D` 会传给后台，用来在数据副本上测试。

不传令牌时行为和以前一样（开发模式：`0.0.0.0:8000`，不校验）；传了令牌或运行打包版本时，默认绑定 `127.0.0.1`。

## 构建

一次性准备：

- Rust：`winget install Rustlang.Rustup`（MSVC 工具链）
- C++ 构建工具：`winget install Microsoft.VisualStudio.2022.BuildTools`，并勾选“使用 C++ 的桌面开发”
- Node.js 18+
- `.venv`：`pip install -r requirements.txt -r packaging/requirements-build.txt`

然后执行：

```bash
powershell -ExecutionPolicy Bypass -File packaging/build_desktop.ps1
```

产物为 `desktop/src-tauri/target/release/bundle/nsis/寻寻_<版本>_x64-setup.exe`（发布到 GitHub 时改名为 `Xunxun_<版本>_x64-setup.exe`）。安装模式是当前用户（不需要管理员权限），开始菜单项为“寻寻”。安装时会先静默卸载改名前的“企业知识库”（`packaging/rename/hooks.nsh`）。

开发调试：先构建一次后台，然后在 `desktop/` 下执行 `npx tauri dev`；外壳会自动使用 `packaging/dist/rag-backend/rag-backend.exe`，也可以用环境变量 `RAG_BACKEND_EXE` 指定其他路径。

### 打包要点

- `collect_dynamic_libs("sqlite_vec")`：`vec0.dll` 通过 `load_extension` 加载，静态分析发现不了。构建脚本会检查它是否存在。
- onedir 而不是 onefile：onefile 每次启动都要解压到 `%TEMP%`，既慢，又最容易被杀毒软件误报。
- 排除了 pymilvus/grpc、gradio、langfuse、pandas、PIL、reportlab 以及 jieba 的 `posseg`/`analyse`/`lac_small`；`project/.env` 不会被打进包里。

## 签名和自动更新

自动更新从 1.0.0 起开启：发布版用 `packaging/updater.conf.json`（公钥和 GitHub Release 地址）打包，从 `https://github.com/NG666666158/xunxun/releases/latest/download/latest.json` 检查新版。平时不带它打包时，界面不显示“检查更新”。代码签名还没有做，见 [发布流程](../packaging/README_RELEASE.md)。

## 已知限制

- 只支持 Windows x64；未签名（见上文）。
- 首次启动需要先生成 jieba 词典缓存（约 1 秒），所以首次启动比后续启动慢。
- 退出时直接结束后台进程（TerminateProcess）。SQLite WAL 能保证数据库不损坏，但正在入库的那个文件会在下次启动时重新入库。
- 打包后的后台只包含 embedded（SQLite）存储，`VECTOR_BACKEND=milvus` 不可用。
- 关闭主窗口默认最小化到托盘（后台继续监控文件夹，快速搜索快捷键可用），托盘菜单的“退出”才真正退出；设置页“窗口”一栏可改为关闭即退出（`preferences.json` 的 `close_to_tray`，启动时由外壳读取）。
