"""Soak test: does the backend hold steady over a long run?

Starts the backend on a temporary data folder against the fake model service
(project/tests/smoke/fake_models.py: no key, no network), imports a large synthetic folder
of small mixed files, then leaves it watching that folder for a while with a few file
changes, the way a window left open in the tray would. Every minute it samples the
backend's memory (RSS and private bytes), handle count and thread count, writes them to a
CSV and fails when they grew more than allowed after the warm-up (the import plus a few
quiet minutes).

    .venv/Scripts/python packaging/soak_test.py --backend-exe packaging/dist/rag-backend/rag-backend.exe
    .venv/Scripts/python packaging/soak_test.py --project project --files 500 --idle-minutes 5

Run it with the app's venv (it serves the fake models with FastAPI). psutil is used when it
is installed; on Windows the numbers otherwise come from the Win32 API directly. Nothing
touches the real data folder or the key in Credential Manager: the backend gets its own
data folder and a null keyring, and the run stops if the key it reports is not the fake one.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import random
import secrets
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

REPO = Path(__file__).resolve().parents[1]
SMOKE = REPO / "project" / "tests" / "smoke"
FAKE_KEY = "sk-soak-test-not-a-real-key-7f3a"
FAKE_KEY_MASKED = f"{FAKE_KEY[:5]}…{FAKE_KEY[-4:]}"  # as core/user_settings.mask_key shows it

# --- process numbers ----------------------------------------------------------------


@dataclass
class Sample:
    rss_mb: float
    private_mb: float
    handles: int
    threads: int


def _sample_psutil(pid: int) -> Sample:
    import psutil

    proc = psutil.Process(pid)
    memory = proc.memory_info()
    handles = proc.num_handles() if hasattr(proc, "num_handles") else proc.num_fds()
    private = getattr(memory, "private", 0) or getattr(memory, "uss", 0) or memory.rss
    return Sample(memory.rss / 2**20, private / 2**20, handles, proc.num_threads())


def _sample_win32(pid: int) -> Sample:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t), ("PrivateUsage", ctypes.c_size_t),
        ]

    class THREADENTRY32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ThreadID", wintypes.DWORD),
            ("th32OwnerProcessID", wintypes.DWORD), ("tpBasePri", wintypes.LONG), ("tpDeltaPri", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
        ]

    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    handle = kernel32.OpenProcess(0x1000 | 0x0010, False, pid)  # QUERY_LIMITED_INFORMATION | VM_READ
    if not handle:
        raise ProcessLookupError(pid)
    try:
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo")
        handles = wintypes.DWORD()
        kernel32.GetProcessHandleCount(handle, ctypes.byref(handles))
    finally:
        kernel32.CloseHandle(handle)
    snapshot = kernel32.CreateToolhelp32Snapshot(0x4, 0)  # TH32CS_SNAPTHREAD
    threads = 0
    try:
        entry = THREADENTRY32()
        entry.dwSize = ctypes.sizeof(entry)
        more = kernel32.Thread32First(snapshot, ctypes.byref(entry))
        while more:
            threads += entry.th32OwnerProcessID == pid
            more = kernel32.Thread32Next(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return Sample(counters.WorkingSetSize / 2**20, counters.PrivateUsage / 2**20, handles.value, threads)


def _sample_proc(pid: int) -> Sample:
    status = Path(f"/proc/{pid}/status").read_text()
    fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
    rss = int(fields["VmRSS"].split()[0]) / 1024
    return Sample(rss, rss, len(os.listdir(f"/proc/{pid}/fd")), int(fields["Threads"]))


def sampler():
    try:
        import psutil  # noqa: F401

        return "psutil", _sample_psutil
    except ImportError:
        pass
    if sys.platform == "win32":
        return "win32", _sample_win32
    return "proc", _sample_proc


def child_processes(pid: int) -> list[tuple[int, str]]:
    """Direct children of `pid` as (pid, program name). A venv's Scripts\\python.exe on
    Windows is only a launcher: the interpreter that runs the backend is its child (next to
    a conhost.exe), and that is the one to measure."""
    try:
        import psutil

        return [(child.pid, child.name()) for child in psutil.Process(pid).children()]
    except ImportError:
        pass
    if sys.platform != "win32":
        return []
    import ctypes
    from ctypes import wintypes

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_void_p), ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG), ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snapshot = kernel32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    found = []
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        more = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while more:
            if entry.th32ParentProcessID == pid:
                found.append((entry.th32ProcessID, entry.szExeFile))
            more = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return found


# --- synthetic folder ---------------------------------------------------------------

WORDS = (
    "项目 预算 合同 发票 会议 纪要 报告 计划 采购 付款 客户 供应商 进度 风险 质量 培训 招聘 绩效 审计 税务 "
    "市场 销售 季度 年度 目标 总结 方案 评审 验收 交付 维护 升级 安全 备份 网络 服务器 数据库 接口 测试 上线 "
    "北京 上海 深圳 杭州 成都 武汉 西安 南京 第一 第二 第三 部门 小组 负责人 周一 周三 周五 上午 下午 完成 延期"
).split()
FORMATS = (("txt", 30), ("md", 25), ("docx", 20), ("xlsx", 10), ("pdf", 10), ("pptx", 5))


def _paragraphs(rng: random.Random, serial: int) -> list[str]:
    out = []
    for _ in range(rng.randint(2, 8)):
        words = rng.choices(WORDS, k=rng.randint(12, 40))
        out.append(f"第{serial}号 " + "".join(words) + f"，金额 {rng.randint(1, 99999)} 元，日期 2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}。")
    return out


def _available_formats() -> list[tuple[str, int]]:
    available = []
    for name, weight in FORMATS:
        module = {"docx": "docx", "xlsx": "openpyxl", "pdf": "pymupdf"}.get(name)
        if module:
            try:
                __import__(module)
            except ImportError:
                continue
        available.append((name, weight))
    return available


def write_file(path: Path, kind: str, paragraphs: list[str]) -> None:
    if kind == "txt":
        path.write_text("\n\n".join(paragraphs), encoding="utf-8")
    elif kind == "md":
        path.write_text(f"# {path.stem}\n\n" + "\n\n".join(paragraphs), encoding="utf-8")
    elif kind == "docx":
        import docx

        document = docx.Document()
        document.add_heading(path.stem, level=1)
        for text in paragraphs:
            document.add_paragraph(text)
        document.save(path)
    elif kind == "xlsx":
        import openpyxl

        book = openpyxl.Workbook()
        sheet = book.active
        sheet.append(["序号", "内容", "金额"])
        for number, text in enumerate(paragraphs, start=1):
            sheet.append([number, text[:60], len(text) * 10])
        book.save(path)
    elif kind == "pdf":
        import pymupdf

        document = pymupdf.open()
        page = document.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 545, 800), "\n".join(paragraphs), fontname="china-s", fontsize=10)
        document.save(path)
        document.close()
    elif kind == "pptx":
        sys.path.insert(0, str(SMOKE))
        from office_files import write_deck

        write_deck(path, paragraphs[:4])
    else:
        raise ValueError(kind)


def make_corpus(folder: Path, count: int, seed: int = 7) -> dict[str, int]:
    rng = random.Random(seed)
    formats = _available_formats()
    kinds, weights = zip(*formats)
    made: dict[str, int] = {}
    for serial in range(count):
        kind = rng.choices(kinds, weights)[0]
        sub = folder / f"部门{serial // 200:02d}" / f"组{serial // 40 % 5}"
        sub.mkdir(parents=True, exist_ok=True)
        write_file(sub / f"资料{serial:05d}-{rng.choice(WORDS)}.{kind}", kind, _paragraphs(rng, serial))
        made[kind] = made.get(kind, 0) + 1
    return made


# --- the backend --------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_fake_models() -> int:
    sys.path.insert(0, str(SMOKE))
    import uvicorn
    from fake_models import create_app

    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, name="fake-models", daemon=True).start()
    deadline = time.time() + 30
    while not server.started and time.time() < deadline:
        time.sleep(0.1)
    return port


class Backend:
    def __init__(self, args: argparse.Namespace, data_dir: Path, models_port: int):
        self.port = free_port()
        self.token = secrets.token_hex(16)
        base = f"http://127.0.0.1:{models_port}/v1"
        env = {
            **os.environ,
            "RAG_APP_TOKEN": self.token,
            "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
            "SILICONFLOW_API_KEY": FAKE_KEY,
            "LLM_API_KEY": FAKE_KEY,
            "SILICONFLOW_BASE_URL": base,
            "LLM_BASE_URL": base,
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
        flags = ["--port", str(self.port), "--data-dir", str(data_dir), "--parent-pid", str(os.getpid())]
        if args.backend_exe:
            exe = Path(args.backend_exe).resolve()
            command, cwd = [str(exe), *flags], exe.parent
        else:
            project = Path(args.project).resolve()
            command, cwd = [sys.executable, str(project / "app.py"), *flags], project
        creation = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
        self.log = open(data_dir.parent / "backend.stdout.log", "wb")
        self.process = subprocess.Popen(
            command, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT, creationflags=creation
        )
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.source = not args.backend_exe
        self.measured: int | None = None

    @property
    def pid(self) -> int:
        """The process that runs the backend (see child_processes)."""
        return self.measured or self.process.pid

    def request(self, method: str, path: str, body: dict | None = None, timeout: float = 60) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/v1{path}", data=data, method=method)
        request.add_header("X-App-Token", self.token)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        with self.opener.open(request, timeout=timeout) as answer:
            return json.loads(answer.read() or b"{}")

    def wait_live(self, timeout: float = 180) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise SystemExit(f"backend exited during startup ({self.process.returncode})")
            try:
                self.request("GET", "/health/live", timeout=3)
            except (OSError, urllib.error.URLError):
                time.sleep(0.5)
                continue
            if self.source:
                pythons = [child for child, name in child_processes(self.process.pid) if name.lower().startswith("python")]
                self.measured = pythons[0] if len(pythons) == 1 else None
            return
        raise SystemExit("backend did not start in time")

    def stop(self) -> None:
        if self.measured:
            # The interpreter behind a venv launcher (a PID this script started indirectly).
            try:
                if sys.platform == "win32":
                    import ctypes

                    handle = ctypes.windll.kernel32.OpenProcess(0x0001, False, self.measured)
                    if handle:
                        ctypes.windll.kernel32.TerminateProcess(handle, 1)
                        ctypes.windll.kernel32.CloseHandle(handle)
                else:
                    os.kill(self.measured, 15)
            except OSError:
                pass
        if self.process.poll() is None:
            self.process.terminate()  # this PID only
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.log.close()


# --- the run ------------------------------------------------------------------------


@dataclass
class Run:
    started: float = field(default_factory=time.time)
    rows: list[dict] = field(default_factory=list)
    status: dict = field(default_factory=dict)
    changes: list[str] = field(default_factory=list)

    def elapsed(self) -> float:
        return time.time() - self.started


def change_files(folder: Path, step: int, rng: random.Random) -> str:
    """One change a person might make: add, edit, rename or delete a file."""
    files = sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix in (".txt", ".md"))
    action = ("add", "edit", "rename", "delete")[step % 4]
    if action == "add" or not files:
        target = folder / "新增" / f"新文件{step:04d}.md"
        target.parent.mkdir(exist_ok=True)
        write_file(target, "md", _paragraphs(rng, 90000 + step))
    elif action == "edit":
        target = rng.choice(files)
        with target.open("a", encoding="utf-8") as handle:
            handle.write("\n\n" + "".join(_paragraphs(rng, 80000 + step)))
    elif action == "rename":
        target = rng.choice(files)
        target.rename(target.with_name(f"改名{step:04d}-{target.name}"))
    else:
        target = rng.choice(files)
        target.unlink()
    return f"{action} {target.relative_to(folder)}"


def growth(rows: list[dict], key: str, window: int) -> tuple[float, float, float]:
    """(start, end, end - start): medians of the first and last `window` steady samples."""
    values = [row[key] for row in rows]
    start = statistics.median(values[:window])
    end = statistics.median(values[-window:])
    return start, end, end - start


def slope_per_hour(rows: list[dict], key: str) -> float:
    xs = [row["elapsed_s"] / 3600 for row in rows]
    ys = [row[key] for row in rows]
    if len(xs) < 2:
        return 0.0
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    spread = sum((x - mean_x) ** 2 for x in xs)
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / spread if spread else 0.0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--backend-exe", help="the packaged backend (packaging/dist/rag-backend/rag-backend.exe)")
    source.add_argument("--project", default=str(REPO / "project"), help="run project/app.py from source instead")
    parser.add_argument("--files", type=int, default=3000, help="files in the synthetic folder")
    parser.add_argument("--idle-minutes", type=float, default=30, help="how long to watch after the import")
    parser.add_argument("--warmup-minutes", type=float, default=5, help="quiet minutes after the import before growth counts")
    parser.add_argument("--sample-seconds", type=float, default=60)
    parser.add_argument("--change-seconds", type=float, default=180, help="one file change this often while idle")
    parser.add_argument("--poll-seconds", type=float, default=15, help="the page asks for the library status this often")
    parser.add_argument("--import-timeout-minutes", type=float, default=120)
    parser.add_argument("--max-rss-growth-mb", type=float, default=100)
    parser.add_argument("--max-handle-growth", type=int, default=300)
    parser.add_argument("--max-thread-growth", type=int, default=15)
    parser.add_argument("--out", default="", help="folder for soak.csv and summary.json (default: the work folder)")
    parser.add_argument("--work", default="", help="work folder (default: a new temporary one, removed afterwards)")
    parser.add_argument("--keep", action="store_true", help="keep the work folder")
    args = parser.parse_args(argv)

    work = Path(args.work or tempfile.mkdtemp(prefix="xunxun-soak-")).resolve()
    out = Path(args.out).resolve() if args.out else work
    out.mkdir(parents=True, exist_ok=True)
    try:
        return soak(args, work, out)
    finally:
        if not args.keep and not args.work:
            # The data and the synthetic folder go; soak.csv and summary.json stay.
            for path in (work / "data", work / "corpus") if out == work else (work,):
                shutil.rmtree(path, ignore_errors=True)


def soak(args: argparse.Namespace, work: Path, out: Path) -> int:
    data_dir, corpus = work / "data", work / "corpus"
    data_dir.mkdir(parents=True, exist_ok=True)
    how, sample = sampler()
    print(f"work folder {work}; sampling with {how}")

    began = time.time()
    made = make_corpus(corpus, args.files)
    print(f"made {args.files} files in {time.time() - began:.0f} s: {made}")

    models_port = start_fake_models()
    backend = Backend(args, data_dir, models_port)
    run = Run()
    rng = random.Random(11)
    csv_path = out / "soak.csv"
    columns = ["elapsed_s", "phase", "rss_mb", "private_mb", "handles", "threads", "files_known", "queued", "busy", "event"]
    failures: list[str] = []
    try:
        backend.wait_live()
        print(f"backend pid {backend.pid} on port {backend.port}, live after {run.elapsed():.1f} s")
        key = backend.request("GET", "/settings")["api_key"]
        if key.get("source") != "env" or key.get("masked") != FAKE_KEY_MASKED:
            raise SystemExit(f"backend is not using the fake key (source {key.get('source')!r}); stopping")

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()

            def record(phase: str, event: str = "") -> None:
                numbers = sample(backend.pid)
                counts = run.status.get("counts", {})
                row = {
                    "elapsed_s": round(run.elapsed(), 1), "phase": phase, "rss_mb": round(numbers.rss_mb, 1),
                    "private_mb": round(numbers.private_mb, 1), "handles": numbers.handles, "threads": numbers.threads,
                    "files_known": sum(counts.values()), "queued": run.status.get("queued", 0),
                    "busy": int(bool(run.status.get("busy"))), "event": event,
                }
                run.rows.append(row)
                writer.writerow(row)
                handle.flush()
                print(
                    f"{row['elapsed_s'] / 60:6.1f} min  {phase:7}  rss {row['rss_mb']:7.1f} MB  private {row['private_mb']:7.1f} MB  "
                    f"handles {row['handles']:5}  threads {row['threads']:3}  files {row['files_known']:5}  queued {row['queued']:5}  {event}"
                )

            def poll() -> None:
                if backend.process.poll() is not None:
                    raise SystemExit(f"backend exited ({backend.process.returncode})")
                try:
                    run.status = backend.request("GET", "/library/status")
                except (OSError, urllib.error.URLError) as exc:
                    print(f"status failed: {exc}")

            record("start")
            backend.request("POST", "/watch-roots", {"path": str(corpus)}, timeout=600)
            import_started = time.time()
            next_sample = time.time() + args.sample_seconds
            quiet = 0
            while True:
                time.sleep(2)
                poll()
                known = sum(run.status.get("counts", {}).values())
                quiet = quiet + 1 if not run.status.get("busy") and known >= args.files else 0
                if time.time() >= next_sample:
                    record("import")
                    next_sample += args.sample_seconds
                if quiet >= 3:
                    break
                if time.time() - import_started > args.import_timeout_minutes * 60:
                    failures.append(f"import did not finish within {args.import_timeout_minutes} min")
                    break
            import_seconds = time.time() - import_started
            record("import", f"import done in {import_seconds:.0f} s: {run.status.get('counts')}")

            idle_started = time.time()
            idle_end = idle_started + args.idle_minutes * 60
            warm_until = idle_started + args.warmup_minutes * 60
            next_sample = idle_started + args.sample_seconds
            next_change = idle_started + args.change_seconds
            next_poll = idle_started
            step = 0
            while time.time() < idle_end:
                now = time.time()
                event = ""
                if now >= next_change:
                    event = change_files(corpus, step, rng)
                    run.changes.append(event)
                    step += 1
                    next_change += args.change_seconds
                if now >= next_poll:
                    poll()
                    next_poll += args.poll_seconds
                if now >= next_sample:
                    record("warmup" if now < warm_until else "idle", event)
                    next_sample += args.sample_seconds
                elif event:
                    print(f"{run.elapsed() / 60:6.1f} min  change: {event}")
                time.sleep(1)
            poll()
            record("idle", "end")
    finally:
        alive = backend.process.poll() is None
        backend.stop()

    log_path = data_dir / "logs" / "backend.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    uncaught = log_text.count("Uncaught exception")
    errors = sum(1 for line in log_text.splitlines() if " ERROR " in line or " CRITICAL " in line)
    if not alive:
        failures.append("the backend was not running at the end")
    if uncaught:
        failures.append(f"{uncaught} uncaught exception(s) in backend.log")

    steady = [row for row in run.rows if row["phase"] == "idle"]
    window = max(1, min(3, len(steady) // 3))
    summary: dict = {
        "files": args.files, "formats": made, "sampler": how, "import_seconds": round(import_seconds, 1),
        "final_counts": run.status.get("counts", {}), "changes": run.changes, "backend_log_errors": errors,
        "uncaught_exceptions": uncaught, "samples": len(run.rows), "steady_samples": len(steady),
        "peak_rss_mb": max(row["rss_mb"] for row in run.rows), "limits": {
            "rss_mb": args.max_rss_growth_mb, "handles": args.max_handle_growth, "threads": args.max_thread_growth,
        },
    }
    if len(steady) < 2:
        failures.append("too few samples after the warm-up to judge growth (raise --idle-minutes)")
    else:
        for key, limit in (("rss_mb", args.max_rss_growth_mb), ("private_mb", None), ("handles", args.max_handle_growth), ("threads", args.max_thread_growth)):
            start, end, grew = growth(steady, key, window)
            summary[key] = {"start": start, "end": end, "growth": round(grew, 1), "per_hour": round(slope_per_hour(steady, key), 1)}
            if limit is not None and grew > limit:
                failures.append(f"{key} grew by {grew:.1f} after the warm-up (limit {limit})")
    summary["failures"] = failures
    summary["passed"] = not failures
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nCSV: {csv_path}")
    print("PASSED" if not failures else "FAILED:\n  " + "\n  ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
