"use strict";

/* The library's decisions, kept free of the page so tests can run them under Node: how
   long a batch will take, the ways to read its images, what to say after adding things,
   and the status row's note when the app noticed changes by itself
   (design/LIBRARY_SYNC_AND_IMAGES.md). */
(function (root) {
  const IMAGE = /\.(png|jpe?g|webp|bmp|tiff?)$/i;
  // What the library reads (config.allowed_extensions).
  const EXTENSIONS = ["pdf", "docx", "xlsx", "pptx", "ppt", "md", "txt", "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"];
  const IMAGE_MODES = ["local", "cloud", "name"];
  // Batches this big are confirmed first (files, or seconds of work).
  const LARGE_FILES = 1000;
  const LARGE_SECONDS = 20 * 60;
  // Rows per "显示更多"; the backend answers up to FILE_MAX at a time.
  const FILE_PAGE = 500;
  const FILE_MAX = 5000;

  /** "不到 1 分钟" / "约 7 分钟" / "约 1.5 小时". */
  function duration(seconds) {
    if (seconds == null || Number.isNaN(seconds)) return "";
    if (seconds < 60) return "不到 1 分钟";
    if (seconds < 3600) return `约 ${Math.round(seconds / 60)} 分钟`;
    const hours = seconds / 3600;
    return `约 ${hours < 10 ? String(Math.round(hours * 10) / 10) : Math.round(hours)} 小时`;
  }

  /** The three ways to read a batch's images, for the choice sheet. `estimate` is the
      inspect answer's `estimate.images`; cloud says plainly that images are uploaded. */
  function imageOptions(estimate = {}, { ocrInstalled = true, ocrMB = 0, keyConfigured = true } = {}) {
    const line = (...parts) => parts.filter(Boolean).join(" · ");
    const time = (mode) => duration(estimate[mode]?.seconds);
    const download = !ocrInstalled && ocrMB ? `先下载约 ${ocrMB} MB` : "";
    return [
      { value: "local", icon: "image", label: "本机识别", line: line("识别图中文字，不上传", time("local"), download) },
      keyConfigured
        ? { value: "cloud", icon: "cloud", label: "云端识别", line: line("图片会上传到硅基流动", time("cloud")), warn: true }
        : { value: "cloud", icon: "cloud", label: "云端识别", line: "需要先在设置里填 Key", disabled: true },
      { value: "name", icon: "tag", label: "只记文件名", line: "不识别内容，只能按名字和日期找到" },
    ];
  }

  /** What to do after inspecting a batch: what can be added, what failed, whether to ask
      how to read images, whether it is big enough to confirm, and the time it will take. */
  function plan(inspect) {
    const items = inspect?.items || [];
    const add = items.filter((i) => i.ok && !i.already);
    const already = items.filter((i) => i.ok && i.already);
    const errors = items.filter((i) => !i.ok);
    const totals = inspect?.totals || {};
    const estimate = inspect?.estimate || {};
    const mode = IMAGE_MODES.includes(inspect?.image_mode) ? inspect.image_mode : inspect?.default_image_mode || "local";
    const seconds = (estimate.text_seconds || 0) + (totals.images ? estimate.images?.[mode]?.seconds || 0 : 0);
    return {
      add,
      already,
      errors,
      askImages: !!inspect?.ask_image_mode && add.length > 0 && (totals.images || 0) > 0,
      images: totals.images || 0,
      files: totals.indexable || 0,
      large: (totals.indexable || 0) >= LARGE_FILES || seconds >= LARGE_SECONDS,
      seconds,
    };
  }

  /** One line after adding: what went in, what was there already, what did not and why. */
  function importSummary(result, earlierErrors = []) {
    const results = result?.results || [];
    const folders = results.filter((r) => r.ok && r.kind === "folder" && !r.already).length;
    const files = results.filter((r) => r.ok && r.kind === "file" && !r.already).length;
    const already = results.filter((r) => r.ok && r.already).length;
    const errors = [...earlierErrors, ...results.filter((r) => !r.ok)];
    const parts = [];
    if (folders || files) {
      const what = [folders && `${folders} 个文件夹`, files && `${files} 个文件`].filter(Boolean).join("和 ");
      parts.push(`已添加 ${what}`);
    }
    if (already) parts.push(`${already} 个已经在知识库里`);
    if (errors.length) parts.push(`${errors.length} 个没加上：${errors[0].error || "无法添加"}`);
    return { text: parts.join("，") || "没有要添加的", ok: folders + files > 0, errors };
  }

  /** The status row's note for changes found by the app itself (`changes.counts` from
      /library/status?since=), or null. Losses come first; progress-like news uses 寻寻's
      voice, losses stay plain. `target` is where a click goes. */
  function changeNote(counts = {}, busy = false) {
    const n = (k) => counts[k] || 0;
    if (n("folder_recycled")) return { tone: "warn", text: "有个文件夹在回收站", target: "lost" };
    if (n("folder_missing")) return { tone: "warn", text: "有个文件夹找不到了", target: "lost" };
    const lost = n("missing") + n("offline");
    if (n("recycled") && !lost) return { tone: "warn", text: `${n("recycled")} 个文件在回收站`, target: "lost" };
    if (lost) return { tone: "warn", text: `${lost + n("recycled")} 个文件找不到了`, target: "lost" };
    if (n("new")) return { tone: busy ? "busy" : "note", text: `寻寻发现 ${n("new")} 个新文件${busy ? "，正在读" : ""}`, target: "library" };
    if (n("restored")) return { tone: "note", text: `${n("restored")} 个文件回来了`, target: "library" };
    if (n("folder_moved") || n("folder_restored")) return { tone: "note", text: "寻寻跟上了换位置的文件夹", target: "library" };
    if (n("moved")) return { tone: "note", text: `寻寻跟上了 ${n("moved")} 个换位置的文件`, target: "library" };
    if (n("changed")) return { tone: busy ? "busy" : "note", text: `${n("changed")} 个文件有改动${busy ? "，正在重读" : ""}`, target: "library" };
    return null;
  }

  /** Where a lost file is, in a few words: the "找不到的文件" list. */
  function lostLine(file) {
    const folder = (file?.path || "").replace(/[\\/][^\\/]*$/, "");
    if (file?.status === "recycled") return "在回收站 · 还原后会自动回来";
    if (file?.status === "offline") return `磁盘未连接 · ${folder}`;
    return `原来在 ${folder}`;
  }

  /** The line under a list that does not show everything. `step` is how many more can be
      shown; none past FILE_MAX, where the search box and the filters narrow it down. */
  function moreFiles(total, shown) {
    if (!total || total <= shown) return null;
    const text = `已显示 ${shown} 个，共 ${total.toLocaleString("en-US")} 个文件`;
    return shown >= FILE_MAX ? { text: `${text}。用搜索或筛选缩小范围。`, step: 0 } : { text, step: FILE_PAGE };
  }

  /** While files are being read the counts change every moment, and redrawing a long list takes
      a noticeable moment on the page: short lists follow along, long ones catch up every 15 s
      and once more at the end. */
  function refreshDue({ stale, busy, rows, sinceLoad }) {
    if (!stale) return false;
    if (!busy || rows < 200) return true;
    return sinceLoad >= 15000;
  }

  function isImage(name) {
    return IMAGE.test(name || "");
  }

  root.LibraryUI = {
    duration, imageOptions, plan, importSummary, changeNote, lostLine, isImage, moreFiles, refreshDue,
    EXTENSIONS, IMAGE_MODES, FILE_PAGE, FILE_MAX,
  };
})(typeof window !== "undefined" ? window : globalThis);
