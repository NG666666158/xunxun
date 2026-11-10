"use strict";

/* 设置 › 关于 › 数据安全: the start-up check, automatic backups, the two ways back and the
   diagnostics export. The wording below is plain on purpose: these are errors and
   confirmations, not 寻寻 talking. */
(function (root) {
  /** "今天 10:32" / "昨天 22:05" / "11月3日 10:32". */
  function when(iso, now = new Date()) {
    if (!iso) return "";
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return "";
    const time = `${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
    const day = (d) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
    const days = Math.round((day(now) - day(date)) / 86400000);
    if (days === 0) return `今天 ${time}`;
    if (days === 1) return `昨天 ${time}`;
    return `${date.getMonth() + 1}月${date.getDate()}日 ${time}`;
  }

  function size(bytes) {
    if (!bytes) return "";
    if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
    if (bytes < 1024 ** 3) return `${Math.round(bytes / 1024 / 1024)} MB`;
    return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  }

  const REASONS = { upgrade: "升级前", manual: "手动" };

  /** One backup in the list: "今天 10:32 · 48 MB · 升级前". */
  function backupLine(backup, now) {
    return [when(backup.created_at, now), size(backup.size), REASONS[backup.reason]].filter(Boolean).join(" · ");
  }

  /** What the rows say for a /data-safety answer. */
  function summary(status = {}, now = new Date()) {
    const backups = status.backups || [];
    const damaged = status.state === "damaged";
    return {
      damaged,
      alert: damaged ? status.message || "索引库已损坏。" : status.notice || "",
      check: damaged ? "没有通过" : status.state === "checking" ? "正在检查…" : status.checked_at ? `通过 · ${when(status.checked_at, now)}` : "—",
      backups: status.backup_note || (backups.length ? `${backups.length} 份 · 最新 ${when(backups[0].created_at, now)}` : "还没有备份"),
      pending: status.pending ? "重启后生效" : "",
    };
  }

  /* ---------------- the page ---------------- */

  let deps = null;
  let shown = null;

  function render(status) {
    if (!status) return;
    shown = status;
    const $ = (sel) => document.querySelector(sel);
    const s = summary(status);
    $("#safetyAlert").hidden = !s.alert;
    $("#safetyAlert").textContent = s.alert;
    $("#safetyCheck").textContent = s.check;
    $("#backupState").textContent = s.pending || s.backups;
    $("#backupNow").disabled = s.damaged;
    $("#recoveryBlock").open = s.damaged || $("#recoveryBlock").open;
    $("#backupList").innerHTML = (status.backups || []).map((backup) => `<li><span>${deps.esc(backupLine(backup))}</span>
        <button type="button" class="btn small" data-restore="${deps.esc(backup.name)}">恢复</button></li>`).join("");
    $("#noBackups").hidden = !!(status.backups || []).length;
  }

  async function finishRecovery(result, done) {
    render(result.data_safety);
    deps.onChanged();
    if (!result.restart_required) { deps.toast(done); return; }
    if (deps.restart) {
      if (await deps.confirm({ title: "需要重启寻寻", message: `${done}\n重启后生效。`, confirm: "现在重启", cancel: "稍后" })) {
        await deps.restart();
      }
    } else {
      deps.toast(`${done} 重新打开后生效。`);
    }
  }

  function init(options) {
    deps = options;
    const $ = (sel) => document.querySelector(sel);
    $("#backupNow").addEventListener("click", async (e) => {
      try {
        render(await deps.runDone(e.currentTarget, () => deps.postJson("/data-safety/backup", {})));
        deps.toast("已备份");
      } catch (err) {
        deps.toast(`备份失败：${err.message || err}`);
      }
    });
    $("#backupList").addEventListener("click", async (e) => {
      const button = e.target.closest("[data-restore]");
      if (!button) return;
      const backup = (shown?.backups || []).find((item) => item.name === button.dataset.restore);
      const ok = await deps.confirm({
        title: "恢复这份备份？",
        message: `索引库回到 ${when(backup?.created_at)}。这之后改过的文件会重新读，对话记录不变。`,
        confirm: "恢复",
      });
      if (!ok) return;
      try {
        await finishRecovery(await deps.postJson("/data-safety/restore", { name: button.dataset.restore }), "索引库已恢复。");
      } catch (err) {
        deps.toast(`恢复失败：${err.message || err}`);
      }
    });
    $("#rebuildIndex").addEventListener("click", async () => {
      const ok = await deps.confirm({
        title: "重新读一遍所有文件？",
        message: "索引会清空，文件列表保留。所有文件的文字会再次发到硅基流动，文件多时要很久。",
        confirm: "重新读",
        danger: true,
      });
      if (!ok) return;
      try {
        await finishRecovery(await deps.postJson("/data-safety/rebuild", {}), "索引已清空，文件会重新读。");
      } catch (err) {
        deps.toast(`操作失败：${err.message || err}`);
      }
    });
    $("#exportDiagnostics").addEventListener("click", async (e) => {
      const row = e.currentTarget;
      row.disabled = true;
      try {
        const result = await deps.postJson("/diagnostics/export", {});
        deps.toast(`已导出：${result.path.split(/[\\/]/).pop()}`);
      } catch (err) {
        deps.toast(`导出失败：${err.message || err}`);
      } finally {
        row.disabled = false;
      }
    });
  }

  root.Safety = { when, size, backupLine, summary, render, init };
})(typeof window !== "undefined" ? window : globalThis);
