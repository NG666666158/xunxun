# 发布流程

## 每次发布

1. 在 `CHANGELOG.md` 的“未发布”下写好变化。
2. 改版本号。它同时写到 6 个文件，并把“未发布”变成新版本的小节；`project/tests/test_release.py` 和 CI 会检查它们一致：

   ```bash
   .venv/Scripts/python scripts/bump_version.py 1.2.0
   ```

3. 打包：

   ```powershell
   $env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
   $env:CARGO_NET_OFFLINE = "true"   # 依赖都已缓存时；代理下联网要再设 CARGO_HTTP_CHECK_REVOKE=false
   packaging\build_desktop.ps1 -Python .venv\Scripts\python.exe
   ```

   产物在 `desktop\src-tauri\target\release\bundle\nsis\`。`-Config <json>` 可以叠加配置（开启更新、改名），可以写多次。

4. 拿上一版安装包测一次升级。脚本把程序装进单独的文件夹，在数据副本上跑后台，结束时把安装登记恢复原样。正在运行的寻寻不受影响：

   ```bash
   .venv/Scripts/python packaging/test_upgrade.py --old <旧 setup.exe> --new <新 setup.exe> --data <数据副本> --work <短路径>
   ```

   `--work` 要用短路径，否则 `backend\_internal` 下的文件会超过 260 字符。它不能启动外壳（第二个实例只会唤起已运行的那个），外壳的流程要靠单独身份的测试版，见文末。

## 发布到 GitHub（1.4.0 起）

代码在两个仓库：
- **私有仓库** `agentic-rag`：用来开发，里面有含个人资料的评测结果和书籍题目；
- **公开仓库** [`xunxun`](https://github.com/NG666666158/xunxun)：放源码和安装包。

用户从公开仓库的 Releases 下载安装包，自动更新也从那里取 `latest.json`。

1. 打带自动更新的发布版。私钥和密码在 `%USERPROFILE%\.tauri\`，不在任何仓库里，要另外备份：

   ```powershell
   $env:TAURI_SIGNING_PRIVATE_KEY = Get-Content "$env:USERPROFILE\.tauri\xunxun-updater.key" -Raw
   $env:TAURI_SIGNING_PRIVATE_KEY_PASSWORD = Get-Content "$env:USERPROFILE\.tauri\xunxun-updater.password.txt" -Raw
   packaging\build_desktop.ps1 -Python .venv\Scripts\python.exe -Config packaging\updater.conf.json
   ```

2. 把安装包和签名改成英文文件名（GitHub 的附件名不支持中文），再生成 `latest.json`：

   ```powershell
   $v = "1.4.0"; $b = "desktop\src-tauri\target\release\bundle\nsis"
   Copy-Item "$b\寻寻_${v}_x64-setup.exe" "release\Xunxun_${v}_x64-setup.exe"
   Copy-Item "$b\寻寻_${v}_x64-setup.exe.sig" "release\Xunxun_${v}_x64-setup.exe.sig"
   .venv\Scripts\python packaging\make_latest_json.py --setup "release\Xunxun_${v}_x64-setup.exe" --url "https://github.com/NG666666158/xunxun/releases/download/v$v/Xunxun_${v}_x64-setup.exe" --out release\latest.json
   ```

3. 导出公开源码，在公开仓库里提交、打标签。导出脚本会去掉评测结果，并按 `packaging/private_terms.txt` 检查有没有个人资料，有就停下：

   ```bash
   .venv/Scripts/python packaging/public_tree.py --out ../xunxun
   cd ../xunxun && git add -A && git commit -m "寻寻 1.4.0" && git tag v1.4.0 && git push origin main v1.4.0
   ```

4. 建 Release，上传三个文件。说明取自 `CHANGELOG.md` 这一版的小节：

   ```bash
   gh release create v1.4.0 --repo NG666666158/xunxun --title "寻寻 1.4.0" --notes-file <说明> release/Xunxun_1.4.0_x64-setup.exe release/Xunxun_1.4.0_x64-setup.exe.sig release/latest.json
   ```

改名为“寻寻”从 1.4.0 起已启用，见 [rename/README.md](rename/README.md)。

## 需要你决定的事

下面三件事要花钱或对外发布，都还没做。**建议先定托管**：第一批发出去的安装包如果没有开启更新，那些用户以后只能手动升级。

### 代码签名

没有签名时，每个新用户都会看到 SmartScreen 的“已保护你的电脑”，要点“更多信息 → 仍要运行”，部分杀毒软件也会误报。

| 方案 | 费用（2026 年 11 月查到的） | 能不能用 | 说明 |
|---|---|---|---|
| 不签名 | 0 | 能 | 给少数熟人够用，告诉他们怎么点就行 |
| Azure Artifact Signing（原 Trusted Signing） | 约 $9.99/月 | **目前不能** | 个人只限美国、加拿大；组织只限美、加、欧盟、英国 |
| Certum 云签名（OV，个人可申请） | 约 $116/年（经销商 3 年 $348） | 大概能，需确认 | 不用 USB 令牌，最便宜。价格来自经销商博客，未在官网核实 |
| Sectigo OV / 个人验证（IV） | 约 $219/年（经销商） | 大概能，需确认 | 要 USB 令牌或云 HSM，可能另收费 |
| DigiCert OV | 约 $439/年 | 公司 | |
| EV（Sectigo、DigiCert） | 约 $290～685/年 | 一般要公司 | **不再能立刻消除警告**，对普通应用没有多花钱的理由 |
| 国内 CA（沃通、天威诚信） | 要询价 | 未知 | 没查到公开的个人价格 |
| SignPath 开源免费计划 | 0 | 不能 | 只给开源项目，本仓库必须保持私有 |

几点要知道：

- **签名不等于马上没有警告。** 微软已经让 OV 和 EV 走同样的信誉积累，新证书起步时仍可能提示，下载的人多了才消失。
- 2023 年起私钥必须放在硬件或云 HSM 里，不能再用 PFX 文件。2026 年 3 月起单张证书最长约 460 天，要每年续。
- 中国大陆个人能否通过验证，各 CA 不一样，买之前要问清楚。

我的建议：现在只给少数人用就**先不签名**；要公开发布时买 Certum 或 Sectigo 的个人证书，约 ¥800～1600 一年。

接入方法：`tauri.conf.json` 的 `bundle.windows` 填 `certificateThumbprint` 和 `timestampUrl`（云签名用 `signCommand`），并且在 PyInstaller 之后、`tauri build` 之前用 `signtool` 给 `packaging\dist\rag-backend` 里的 exe、pyd、dll 也签上，否则后台照样被拦。

来源：[微软的代码签名选项](https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/code-signing-options)、[DigiCert 说明](https://knowledge.digicert.com/alerts/ev-signed-application-showing-microsoft-defender-smartscreen-warnings)、[Sectigo 指南](https://sslinsights.com/sectigo-code-signing-certificate-guide/)、[SSL Dragon 价格](https://www.ssldragon.com/ssl-certificates/code-signing/)、[Certum 价格来源](https://codenote.net/en/posts/ov-ev-code-signing-certificate-pricing-japan-2026/)。价格多来自经销商，下单前以官网为准。

### 更新托管

更新需要一个不用登录就能下载的地址，放 `latest.json` 和安装包（约 50 MB）。私有 GitHub 仓库的 Release 附件要登录才能下载，不能用；把令牌打进程序也不行。

| 方案 | 费用 | 说明 |
|---|---|---|
| 不开启更新 | 0 | 用户手动下载新安装包覆盖安装，已测过 |
| 单独建一个公开的 GitHub 仓库，只放安装包 | 0 | 代码仍然私有。国内访问 GitHub 不稳定 |
| Cloudflare R2 | 10 GB 内免费，下载不收流量费 | 要绑卡。国内速度一般 |
| 阿里云 OSS | 约 ¥0.25～0.5/GB 下载流量（2024～25 年社区数据，以官网为准）；100 人每月更新一次约 ¥3 | 国内最快。要实名，自定义域名要备案 |
| 自己的服务器 | 已有就是 0 | |

我的建议：用户主要在国内就用**阿里云 OSS**；只给几个人用可以先不开启。

来源：[Cloudflare R2 价格](https://60eb615a.previews.developers.cloudflare.com/r2/pricing/index.md)、[阿里云 OSS 流量费](https://help.aliyun.com/zh/oss/traffic-fees)。

确定后这样开启：

1. 生成更新密钥。**私钥和密码丢了，已发出去的版本就再也收不到更新**，要备份到代码仓库以外的地方：

   ```bash
   cd desktop && npx tauri signer generate -w ~/.tauri/xunxun-updater.key
   ```

2. 复制 `packaging/updater.conf.example.json`，填公钥和地址，保存为 `packaging/updater.conf.json`（可以提交，公钥不是秘密）。
3. 打包时带上它和私钥：

   ```powershell
   $env:TAURI_SIGNING_PRIVATE_KEY = "$env:USERPROFILE\.tauri\xunxun-updater.key"
   $env:TAURI_SIGNING_PRIVATE_KEY_PASSWORD = "<密码>"
   packaging\build_desktop.ps1 -Python .venv\Scripts\python.exe -Config packaging\updater.conf.json
   ```

4. 生成 `latest.json`，和 `*-setup.exe` 一起上传：

   ```bash
   .venv/Scripts/python packaging/make_latest_json.py --setup <setup.exe> --url <setup.exe 的下载地址> --out latest.json
   ```

开启前，界面不显示“检查更新”；开启后在设置 › 关于里。下载的安装包用内置公钥验签，并校验签名里的版本号，防止被人用旧版本冒充新版本（`requireSignedVersion`）。

### 改名为“寻寻”

方案和测试结果见 [rename/README.md](rename/README.md)。已测过，等你确认后再启用。

## 更新流程的测试方法

外壳不能和已运行的寻寻同时跑（同一 `identifier` 只会唤起已有窗口），所以用单独身份的测试版。步骤和结果见 [rename/README.md](rename/README.md) 末尾。
