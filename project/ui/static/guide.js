"use strict";

/* The guided start: where data goes, a key, a model, the first folder. New installs see all
   of it; people upgrading from a version without it only see where data goes.
   Everything else on the page reaches it through `deps` (see Guide.open in app.js), so the
   decisions below run under Node without a page. */
(function (root) {
  const KEY_URL = "https://cloud.siliconflow.cn/account/ak";

  // Measured in evaluation/answer_speed_v1/REPORT.md (strict checking; prices of 2026-10).
  const MODELS = [
    { value: "Qwen/Qwen3-30B-A3B-Instruct-2507", icon: "paw", label: "Qwen3-30B-A3B", line: "推荐 · 约 30 秒一题 · 约 ¥0.006 一题" },
    { value: "deepseek-ai/DeepSeek-V4-Flash", icon: "sparkle", label: "DeepSeek-V4-Flash", line: "回答最好 · 约 20 秒一题 · 约 ¥0.02 一题" },
    { value: "Qwen/Qwen3-8B", icon: "clock", label: "Qwen3-8B", line: "免费 · 约 1 分钟一题" },
  ];

  // Plain words: this is a privacy note, not 寻寻 talking.
  const NOTICE = [
    { icon: "lock", title: "留在这台电脑", line: "原文件、索引库和对话记录。" },
    { icon: "cloud", title: "发到硅基流动", line: "文档文字、文件名和你的问题，用来生成向量、判断文件类型、排序和回答。" },
    { icon: "tag", title: "没有本地模型可选", line: "不想外传的文件，可以设为只按文件名或不收录。" },
  ];

  /** Which steps to show, in order. */
  function steps({ onboardingDone = false, noticeSeen = false, keyConfigured = false, libraryEmpty = true } = {}) {
    const notice = noticeSeen ? [] : ["notice"];
    if (onboardingDone) return notice;
    return [
      ...notice,
      ...(keyConfigured ? [] : ["key", "model"]),
      ...(libraryEmpty ? ["folder"] : []),
    ];
  }

  /** Testing a key: `save` if it should be saved anyway (valid, but the account is empty). */
  function keyOutcome(error) {
    if (!error) return { ok: true, save: true, tone: "ok", text: "可以用，已保存到 Windows 凭据管理器。" };
    if (/余额不足/.test(error)) return { ok: true, save: true, tone: "warn", text: error };
    return { ok: false, save: false, tone: "error", text: error };
  }

  function looksLikeKey(value) {
    return value.length >= 8 && !/\s/.test(value);
  }

  /* ---------------- the sheet ---------------- */

  function open(plan, deps) {
    const { esc, icon, mascot } = deps;
    const scrim = document.createElement("div");
    scrim.className = "guide-scrim";
    scrim.innerHTML = `<div class="guide" role="dialog" aria-modal="true" aria-labelledby="guideTitle">
        <div class="guide-pet">${mascot()}</div>
        <div class="guide-dots" aria-hidden="true">${plan.length > 1 ? plan.map((_, i) => `<i data-dot="${i}"></i>`).join("") : ""}</div>
        <div class="guide-body"></div>
      </div>`;
    document.body.append(scrim);
    document.body.classList.add("guide-open");
    const body = scrim.querySelector(".guide-body");
    let index = 0;
    let finished = false;

    const finish = async () => {
      if (finished) return;
      finished = true;
      document.removeEventListener("keydown", onKey, true);
      scrim.classList.add("closing");
      setTimeout(() => scrim.remove(), 180);
      document.body.classList.remove("guide-open");
      await deps.remember({ onboarding_done: true }).catch(() => {});
      deps.onDone();
    };
    const next = () => {
      index += 1;
      if (index >= plan.length) finish();
      else show();
    };
    const onKey = (e) => {
      if (e.key !== "Tab") return;
      const items = [...scrim.querySelectorAll("button:not([disabled]), input:not([type=radio]), input[type=radio]:checked, a[href]")];
      if (!items.length) return;
      const at = items.indexOf(document.activeElement);
      e.preventDefault();
      items[(at + (e.shiftKey ? -1 : 1) + items.length) % items.length].focus();
    };
    document.addEventListener("keydown", onKey, true);

    const views = {
      notice() {
        body.innerHTML = `<h2 id="guideTitle">先说清楚数据去向</h2>
          <ul class="guide-facts">${NOTICE.map((f) => `<li><span class="guide-fact-icon">${icon(f.icon)}</span><span><strong>${esc(f.title)}</strong><small>${esc(f.line)}</small></span></li>`).join("")}</ul>
          <div class="guide-actions one"><button type="button" class="btn primary" data-go>知道了</button></div>`;
        body.querySelector("[data-go]").addEventListener("click", () => {
          deps.remember({ data_notice_seen: true }).catch(() => {});
          next();
        });
      },
      key() {
        body.innerHTML = `<h2 id="guideTitle">连接硅基流动</h2>
          <p>回答和读资料都靠它。Key 存在 Windows 凭据管理器，不写入文件。</p>
          <a class="guide-link" href="${KEY_URL}" target="_blank" rel="noopener">${icon("key")}<span>打开硅基流动，注册并新建 Key</span>${icon("chevron")}</a>
          <form class="guide-form" autocomplete="off">
            <label class="field"><span>${icon("lock")}</span><input type="password" placeholder="粘贴 Key：sk-…" spellcheck="false" autocomplete="off" aria-label="API Key"></label>
            <button type="submit" class="btn">测试</button>
          </form>
          <p class="guide-result" role="status" hidden></p>
          <div class="guide-actions"><button type="button" class="btn plain" data-skip>稍后再说</button><button type="button" class="btn primary" data-go disabled>继续</button></div>`;
        const input = body.querySelector("input");
        const result = body.querySelector(".guide-result");
        const go = body.querySelector("[data-go]");
        const testButton = body.querySelector("button[type=submit]");
        body.querySelector("form").addEventListener("submit", async (e) => {
          e.preventDefault();
          const value = input.value.trim();
          const say = (outcome) => {
            result.hidden = false;
            result.dataset.tone = outcome.tone;
            result.textContent = outcome.text;
          };
          if (!looksLikeKey(value)) { say({ tone: "error", text: "这不像一个完整的 Key。" }); input.focus(); return; }
          testButton.disabled = true;
          let error = "";
          try {
            await deps.testKey(value);
          } catch (err) {
            error = err.message || String(err);
          }
          const outcome = keyOutcome(error);
          if (outcome.save) {
            try {
              await deps.saveKey(value);
            } catch (err) {
              say({ tone: "error", text: `保存失败：${err.message || err}` });
              testButton.disabled = false;
              return;
            }
          }
          say(outcome);
          testButton.disabled = false;
          go.disabled = !outcome.ok;
          if (outcome.ok) go.focus();
        });
        go.addEventListener("click", next);
        // Without a key the rest cannot do anything yet.
        body.querySelector("[data-skip]").addEventListener("click", finish);
        input.focus();
      },
      model() {
        const current = deps.currentModel();
        const chosen = MODELS.some((m) => m.value === current && m.value !== MODELS[2].value) ? current : MODELS[0].value;
        body.innerHTML = `<h2 id="guideTitle">选一个问答模型</h2>
          <p>之后可以在设置里换。价格以硅基流动官网为准。</p>
          <div class="choices" role="radiogroup" aria-labelledby="guideTitle">${MODELS.map((m) => `<label class="choice">
              <input type="radio" name="guideModel" value="${esc(m.value)}"${m.value === chosen ? " checked" : ""}>
              <span class="choice-icon">${icon(m.icon)}</span>
              <span class="choice-main"><strong>${esc(m.label)}</strong><small>${esc(m.line)}</small></span>
              <span class="choice-mark">${icon("check")}</span>
            </label>`).join("")}</div>
          <p class="guide-result" role="status" hidden></p>
          <div class="guide-actions one"><button type="button" class="btn primary" data-go>继续</button></div>`;
        const go = body.querySelector("[data-go]");
        go.addEventListener("click", async () => {
          go.disabled = true;
          const model = body.querySelector("input:checked").value;
          try {
            await deps.saveModel(model);
            next();
          } catch (err) {
            const result = body.querySelector(".guide-result");
            result.hidden = false;
            result.dataset.tone = "error";
            result.textContent = `保存失败：${err.message || err}`;
            go.disabled = false;
          }
        });
        go.focus();
      },
      folder() {
        const typed = deps.desktop
          ? ""
          : `<form class="guide-form" autocomplete="off">
              <label class="field"><span>${icon("folder")}</span><input type="text" placeholder="文件或文件夹的完整路径" spellcheck="false" aria-label="文件或文件夹路径"></label>
              <button type="submit" class="btn primary">添加</button>
            </form>`;
        const picks = deps.desktop
          ? `<div class="guide-picks">
              <button type="button" class="btn primary" data-pick="folders">${icon("folderPlus")}添加文件夹</button>
              <button type="button" class="btn" data-pick="files">${icon("docPlus")}添加文件</button>
            </div>`
          : "";
        body.innerHTML = `<h2 id="guideTitle">给寻寻第一批资料</h2>
          <p>文件留在原处，只记位置。文件夹里有变化，寻寻会自己跟上。</p>
          ${picks}${typed}
          <p class="guide-result" role="status" hidden></p>
          <div class="guide-actions one"><button type="button" class="btn plain" data-go>稍后再说</button></div>`;
        const go = body.querySelector("[data-go]");
        const added = (ok) => {
          if (!ok) return;
          const result = body.querySelector(".guide-result");
          result.hidden = false;
          result.dataset.tone = "ok";
          result.textContent = "寻寻开始读了，读完的部分马上能问。";
          go.textContent = "开始用";
          go.className = "btn primary";
          go.focus();
        };
        body.querySelectorAll("[data-pick]").forEach((button) => {
          button.addEventListener("click", async () => {
            try {
              added(await (button.dataset.pick === "folders" ? deps.pickFolders() : deps.pickFiles()));
            } catch (err) {
              deps.toast(`无法选择：${err.message || err}`);
            }
          });
        });
        body.querySelector("form")?.addEventListener("submit", async (e) => {
          e.preventDefault();
          const input = e.currentTarget.querySelector("input");
          const path = input.value.trim();
          if (!path) { input.focus(); return; }
          if (await deps.importPaths([path])) { input.value = ""; added(true); }
        });
        go.addEventListener("click", next);
        (body.querySelector("input") || body.querySelector("[data-pick]") || go).focus();
      },
    };

    function show() {
      scrim.querySelectorAll("[data-dot]").forEach((dot, i) => { dot.classList.toggle("on", i === index); });
      views[plan[index]]();
      body.classList.remove("in");
      void body.offsetWidth;
      body.classList.add("in");
    }
    show();
  }

  root.Guide = { steps, keyOutcome, looksLikeKey, open, MODELS, NOTICE, KEY_URL };
})(typeof window !== "undefined" ? window : globalThis);
