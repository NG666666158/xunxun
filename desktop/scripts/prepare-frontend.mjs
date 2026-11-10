// Assemble the webview assets: the existing web UI (project/ui/static, unchanged) plus the
// shell's splash page. Runs before `tauri dev` / `tauri build` (see tauri.conf.json).
import { cpSync, mkdirSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const desktop = join(dirname(fileURLToPath(import.meta.url)), "..");
const out = join(desktop, "dist");

rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });
cpSync(join(desktop, "..", "project", "ui", "static"), out, { recursive: true });
cpSync(join(desktop, "splash.html"), join(out, "splash.html"));
console.log(`frontend assets -> ${out}`);
