"use strict";

/* What the library may read, as the page says it (library/privacy.py, design/PRIVACY.md): the
   choice for a file or folder (读取内容 / 只按文件名 / 不收录), the plain confirmation that
   says what will happen, and the rule rows under 设置 › 数据去向. Kept free of the page so
   tests can run it under Node. Privacy text stays plain: no puppy voice here. */
(function (root) {
  const ACTION_LABEL = { full: "读取内容", name: "只按文件名", exclude: "不收录" };

  function base(path) {
    return String(path || "").split(/[\\/]/).filter(Boolean).pop() || String(path || "");
  }

  /** The choices for a file or folder. `canExclude`: it sits inside a watched folder and is
      not the watched folder itself (that one is removed instead). */
  function options({ kind = "file", canExclude = false } = {}) {
    const list = [
      { value: "full", icon: "cloud", label: "读取内容", line: "文字发到硅基流动建索引，可以针对内容提问" },
      { value: "name", icon: "tag", label: "只按文件名", line: "只在本机按名字和日期找，内容不读取、不上传" },
    ];
    if (canExclude) {
      list.push({
        value: "exclude", icon: "eyeOff", label: "不收录",
        line: kind === "folder" ? "从知识库移除，以后新加的也不收" : "从知识库移除，以后也不收",
      });
    }
    return list;
  }

  /** "name" when every file is kept by name only, "full" when none is, else "mixed". */
  function groupMode(files) {
    const list = files || [];
    const named = list.filter((f) => f.content_mode === "name").length;
    if (!list.length || !named) return "full";
    return named === list.length ? "name" : "mixed";
  }

  /** The confirmation for `action` on a file, a folder or a name pattern, with the dry run's
      counts (`impact.files`, `impact.indexed`). Returns {title, message, confirm, danger}. */
  function confirmText(action, { kind = "file", name = "", impact = {} } = {}) {
    const n = impact.files || 0;
    const indexed = impact.indexed || 0;
    const target = kind === "pattern" ? `名字是 ${name} 的文件` : `「${name}」`;
    const purge = !indexed ? ""
      : kind === "file" || indexed >= n ? "已建的索引会删掉，备份里的文字也一并清除。"
      : `其中 ${indexed} 个已建的索引会删掉，备份里的文字也一并清除。`;
    if (action === "name") {
      const scope = kind === "file" ? "以后只在本机按名字、文件夹和日期找它"
        : kind === "pattern" ? `${n ? `现有 ${n} 个和` : ""}以后新加的，只在本机按名字、文件夹和日期找`
        : `里面 ${n} 个文件和以后新加的，只在本机按名字、文件夹和日期找`;
      return {
        title: `${target}只按文件名收录？`,
        message: [`${scope}，内容不读取、不上传。`, purge].filter(Boolean).join("\n"),
        confirm: "只按文件名",
        danger: false,
      };
    }
    if (action === "full") {
      // Files that will actually be read: those kept by name only until now.
      const reading = impact.changed ?? n;
      const what = kind === "file" ? "它的文字" : `${reading} 个文件的文字`;
      return {
        title: `读取${target}的内容？`,
        message: reading || kind === "file"
          ? `${what}会发到硅基流动建索引；回答问题时，相关段落也会发过去。`
          : "现在没有这样的文件。以后新加的会读取内容。",
        confirm: reading || kind === "file" ? "读取并上传" : "确定",
        danger: false,
      };
    }
    if (action === "include") {
      return {
        title: `重新收录${target}？`,
        message: n
          ? `${n} 个文件会加回知识库，文字会发到硅基流动建索引。`
          : "现在没有这样的文件。以后新加的会照常收录。",
        confirm: "重新收录",
        danger: false,
      };
    }
    const scope = kind === "file" ? "它会从知识库移除，以后也不收。"
      : kind === "pattern" ? (n ? `现有 ${n} 个会从知识库移除，以后新加的也不收。` : "以后新加的这类文件不收录。")
      : `里面 ${n} 个文件会从知识库移除，以后新加的也不收。`;
    return {
      title: `不收录${target}？`,
      message: [scope, purge, "原文件不动。"].filter(Boolean).join("\n"),
      confirm: "不收录",
      danger: true,
    };
  }

  /** What taking a rule back does, for its confirmation: the opposite of the rule. */
  function undoAction(rule) {
    if (rule.action === "exclude") return "include";
    return rule.action === "name" ? "full" : "name";
  }

  /** One rule as a row: {icon, label, line}. Path rules show the folder's own name. */
  function ruleRow(rule) {
    const where = rule.root_id ? `「${base(rule.root_path)}」` : "所有文件夹";
    if (rule.kind === "pattern") {
      return { icon: rule.action === "exclude" ? "eyeOff" : "tag", label: rule.value, line: `${ACTION_LABEL[rule.action]} · ${rule.root_id ? `只在${where}` : where}` };
    }
    const label = rule.value ? base(rule.value) : base(rule.root_path);
    const inside = rule.value && rule.value.includes("\\") ? ` › ${rule.value.split("\\").slice(0, -1).join(" › ")}` : "";
    return {
      icon: rule.action === "exclude" ? "eyeOff" : rule.action === "name" ? "tag" : "cloud",
      label,
      line: `${ACTION_LABEL[rule.action]} · ${base(rule.root_path)}${inside}`,
    };
  }

  /** Rules in the order the settings list shows them: exclusions, then name-only, then the rest. */
  function sortRules(rules) {
    const order = { exclude: 0, name: 1, full: 2 };
    return [...(rules || [])].sort((a, b) => (order[a.action] - order[b.action]) || String(a.created_at).localeCompare(String(b.created_at)));
  }

  root.PrivacyUI = { ACTION_LABEL, options, groupMode, confirmText, undoAction, ruleRow, sortRules, base };
})(typeof window !== "undefined" ? window : globalThis);
