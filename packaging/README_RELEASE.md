# 发布流程

## 每次发布

安装包改在 GitHub Actions 上打（`.github/workflows/release.yml`）。你推一个标签，它建一个**草稿** Release；你测过草稿再自己点发布。工作流不会发布，也不会改已经发布的 Release。

1. 在 `CHANGELOG.md` 的“未发布”下写好变化。这一节就是 Release 的说明和更新提示里的文字。
2. 改版本号。它同时写到 6 个文件，并把“未发布”变成新版本的小节；`project/tests/test_release.py` 和 CI 会检查它们一致：

   ```bash
   .venv/Scripts/python scripts/bump_version.py 1.2.0
   ```

3. 提交，推到 `main`，等 CI 变绿。
4. 打标签并推送：

   ```bash
   git tag v1.2.0 && git push origin v1.2.0
   ```

   Actions 里的 Release 工作流依次做：
   1. 核对标签：标签不是 `v<文件里的版本号>`、6 个文件不一致、`CHANGELOG.md` 这一节是空的、或者没配更新密钥，几秒内就失败，告诉你怎么改。
   2. 跑 CI 的全部测试（就是 `ci.yml`）。
   3. 在干净的 Windows 机器上打包：`build_desktop.ps1 -BackendOnly` 打后台，`build_desktop.ps1 -SkipBackend -Config packaging/updater.conf.json` 打安装包，用 secrets 里的更新密钥签更新包。依赖只从 `requirements-lock.txt` 和 `package-lock.json` 装，不用缓存。
   4. 配了 SignPath 时做代码签名，见下面的“代码签名”。
   5. 改成英文文件名，生成 `latest.json`，再按程序里更新功能的方式验一遍：签名是不是 `updater.conf.json` 里那把公钥的、签的是不是这个文件、签进去的版本号对不对、`latest.json` 的地址和签名对不对（`packaging/verify_update.py`）。
   6. 建草稿 Release `寻寻 1.2.0`，附三个文件：`Xunxun_1.2.0_x64-setup.exe`、`Xunxun_1.2.0_x64-setup.exe.sig`、`latest.json`。

   失败了改好再推一次：同一个标签要先删掉再打（`git tag -d v1.2.0 && git push origin :refs/tags/v1.2.0`）；只是想重跑，在 Actions 里点 Re-run，或者 Run workflow 时选这个标签（用的是那次提交里的工作流，比本工作流早的标签不能这样跑）。已有草稿会被更新，已发布的 Release 会让工作流直接失败。在分支上 Run workflow 只打包、不建 Release，文件在这次运行的 Artifacts 里，可以拿来试流程。

5. 测草稿。草稿不对外，自动更新也看不到它（更新地址是 `releases/latest/download/latest.json`，只指向已发布的最新版）：

   ```powershell
   gh release download v1.2.0 --repo NG666666158/xunxun --dir release\v1.2.0
   py -3 -m venv $env:TEMP\xv; & $env:TEMP\xv\Scripts\python -m pip install cryptography   # 只第一次
   & $env:TEMP\xv\Scripts\python packaging\verify_update.py --dir release\v1.2.0 --version 1.2.0
   .venv\Scripts\python packaging\test_upgrade.py --old <上一版 setup.exe> --new release\v1.2.0\Xunxun_1.2.0_x64-setup.exe --data <数据副本> --work <短路径>
   ```

   `verify_update.py` 查的就是程序更新时查的那几样。它要用 `cryptography`，单独建一个环境装，别装进打包用的 `.venv`，免得 PyInstaller 把它打进后台。`test_upgrade.py` 把程序装进单独的文件夹，在数据副本上跑后台，结束时把安装登记恢复原样，正在运行的寻寻不受影响。`--work` 要用短路径，否则 `backend\_internal` 下的文件会超过 260 字符。它不能启动外壳（第二个实例只会唤起已运行的那个），要在发布前试真实窗口里的更新，用单独身份的测试版，见文末。

   签过名的安装包，右键 › 属性 › 数字签名 能看到签名。

   后台或导入有改动的版本，再跑一次长时间运行测试（约 45 分钟，可以和上面同时跑）。它要本机打好的后台（`packaginguild_desktop.ps1 -BackendOnly`），或者用 `--project project` 直接跑源码：

   ```bash
   .venv/Scripts/python packaging/soak_test.py --backend-exe packaging/dist/rag-backend/rag-backend.exe --out release/soak
   ```

   脚本在临时数据目录上启动后台，接本机的假模型服务（不联网、不用 Key、不读凭据管理器），导入 3000 个小文件（txt、md、docx、xlsx、pdf、pptx 混合），再挂着监控这个文件夹 30 分钟，每 3 分钟增、改、改名或删一个文件，每 15 秒像界面一样查一次状态，每分钟记一次后台的内存、句柄数和线程数。导入完成后先预热 5 分钟，之后的样本里，开头 3 个和结尾 3 个的中位数相比，内存（RSS）多出 100 MB、句柄多出 300 或线程多出 15 就算失败；后台中途退出、`backend.log` 里有未捕获的异常也算失败，退出码为 1。每分钟的数字在 `--out` 下的 `soak.csv`，结论在 `summary.json`（含每小时的增长斜率）。`--files`、`--idle-minutes` 和三个上限都能改。正在运行的寻寻不受影响。

6. 发布：在 [Releases](https://github.com/NG666666158/xunxun/releases) 打开草稿，确认勾着“Set as the latest release”，点 Publish release。
7. 发布后马上找一台装着上一版的电脑：设置 › 关于 › 检查更新，更新一次。有问题就先把它改回草稿，上一版重新成为最新版，还没更新的人就不会再收到它：

   ```bash
   gh release edit v1.2.0 --repo NG666666158/xunxun --draft=true
   ```

### GitHub 上要配的 secrets

在仓库的 Settings › Secrets and variables › Actions 里加。值只有工作流读得到，日志里会被遮住。

| 名字 | 放在 | 必需 | 内容 |
|---|---|---|---|
| `TAURI_SIGNING_PRIVATE_KEY` | Secrets | 是 | `%USERPROFILE%\.tauri\xunxun-updater.key` 的全部内容（一行文字） |
| `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` | Secrets | 是 | 这把密钥的密码 |
| `SIGNPATH_API_TOKEN` | Secrets | 否 | SignPath 里提交签名用的 API token。没有它就不做代码签名，和现在一样 |
| `SIGNPATH_ORGANIZATION_ID` | Variables | 有 token 时要 | SignPath 的组织 ID |
| `SIGNPATH_PROJECT_SLUG` | Variables | 否 | SignPath 项目的 slug，默认 `xunxun` |
| `SIGNPATH_SIGNING_POLICY_SLUG` | Variables | 否 | 签名策略的 slug，默认 `release-signing` |

用 gh 加，值不会显示出来（也可以在网页上粘贴）：

```powershell
gh secret set TAURI_SIGNING_PRIVATE_KEY --repo NG666666158/xunxun --body (Get-Content "$env:USERPROFILE\.tauri\xunxun-updater.key" -Raw).Trim()
gh secret set TAURI_SIGNING_PRIVATE_KEY_PASSWORD --repo NG666666158/xunxun --body (Get-Content "$env:USERPROFILE\.tauri\xunxun-updater.password.txt" -Raw).Trim()
```

更新密钥必须是 `updater.conf.json` 里公钥对应的那把。错了，工作流在“Check the update the way the app will”这一步失败；要是放过去，已安装的寻寻都会拒绝这个更新。密钥和密码仍要在仓库以外备份：丢了，已发出去的版本就再也收不到更新。

GitHub 账户要开两步验证（Settings › Password and authentication）。能推标签的人就能让工作流用这些 secrets 打包；用 SignPath 时这也是条款要求。

### 在本机打包（备用）

Actions 用不了时（GitHub 故障、secrets 没配好、要排查打包问题），照旧在本机打。产物和工作流的一样，只是没有代码签名：

1. 打带自动更新的发布版。私钥和密码在 `%USERPROFILE%\.tauri\`：

   ```powershell
   $env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
   $env:CARGO_NET_OFFLINE = "true"   # 依赖都已缓存时；要联网就去掉这行，走 Clash 代理（http://127.0.0.1:7892）时再设 CARGO_HTTP_CHECK_REVOKE=false
   $env:TAURI_SIGNING_PRIVATE_KEY = Get-Content "$env:USERPROFILE\.tauri\xunxun-updater.key" -Raw
   $env:TAURI_SIGNING_PRIVATE_KEY_PASSWORD = Get-Content "$env:USERPROFILE\.tauri\xunxun-updater.password.txt" -Raw
   packaging\build_desktop.ps1 -Python .venv\Scripts\python.exe -Config packaging\updater.conf.json
   ```

   产物在 `desktop\src-tauri\target\release\bundle\nsis\`。`-Config <json>` 可以叠加配置，可以写多次；`-BackendOnly` 只打后台。

2. 把安装包和签名改成英文文件名（GitHub 的附件名不支持中文），生成 `latest.json`，再验一遍：

   ```powershell
   $v = "1.2.0"; $b = "desktop\src-tauri\target\release\bundle\nsis"
   New-Item -ItemType Directory release -Force | Out-Null
   Copy-Item "$b\寻寻_${v}_x64-setup.exe" "release\Xunxun_${v}_x64-setup.exe"
   Copy-Item "$b\寻寻_${v}_x64-setup.exe.sig" "release\Xunxun_${v}_x64-setup.exe.sig"
   .venv\Scripts\python packaging\make_latest_json.py --setup "release\Xunxun_${v}_x64-setup.exe" --url "https://github.com/NG666666158/xunxun/releases/download/v$v/Xunxun_${v}_x64-setup.exe" --out release\latest.json
   & $env:TEMP\xv\Scripts\python packaging\verify_update.py --dir release --version $v   # 环境见上面第 5 步
   ```

3. 标签已经推过、工作流失败了的话，先删掉它建的草稿（如果有）。然后建草稿 Release，说明取自 `CHANGELOG.md` 这一版的小节，测过再发布（上面第 5～7 步）：

   ```powershell
   .venv\Scripts\python packaging\release_check.py --tag v$v --notes-out release\notes.md
   gh release create v$v --repo NG666666158/xunxun --draft --verify-tag --title "寻寻 $v" --notes-file release\notes.md release\Xunxun_${v}_x64-setup.exe release\Xunxun_${v}_x64-setup.exe.sig release\latest.json
   ```

### 从私有仓库导出时

公开仓库最早由私有仓库 `agentic-rag` 用 `packaging/public_tree.py` 导出（去掉评测结果，并按 `packaging/private_terms.txt` 查个人资料）。导出会清空本仓库再重写：以后还这样导出的话，`.github/workflows/release.yml`、`packaging/signpath/`、`packaging/release_check.py`、`packaging/verify_update.py`、`docs/code-signing-policy.md` 要先放进私有仓库，否则会被删掉。

改名为“寻寻”从 1.0.0 起已启用，见 [rename/README.md](rename/README.md)。

### 已装的寻寻什么时候发现新版本

启动后约一分钟查一次，之后只要还开着（包括关到托盘），每天查一次；没联网或 GitHub 打不开，一小时后再试，失败只写在控制台里。查到新版本只在设置按钮和“关于”上亮一个小点，点“更新”看说明，确认后才下载安装。“检查更新”按钮照旧可用。调度逻辑在 `project/ui/static/updatecheck.js`。

## 测试版（GitHub 预发布）

先发成预发布，自己装上试过再转正。用户手里的寻寻收不到预发布：

- 已装的版本只读 `releases/latest/download/latest.json`。这个地址指向仓库的“最新版本”（[链接到发布](https://docs.github.com/en/repositories/releasing-projects-on-github/linking-to-releases)）。
- GitHub 的“最新版本”是最近一个非预发布、非草稿的版本，预发布也不能被设为最新。见 [REST 文档](https://docs.github.com/en/rest/releases/releases#get-the-latest-release)：“The latest release is the most recent non-prerelease, non-draft release”，以及 `make_latest` 一项。

所以预发布期间所有人还停在上一个正式版；一旦转正，所有人下次检查时都会收到。

### 1. 发测试版

版本号照常往上加，不加 `-beta`：`bump_version.py` 只认 `x.y.z`，安装包和更新也靠它比较新旧。改版本号、推标签、等工作流建出草稿，都和上面“每次发布”的 1～5 步一样；在本机打包时，把建 Release 的 `--draft` 换成 `--prerelease`，如下。在网页上发布草稿时，勾上“Set as a pre-release”效果相同。`latest.json` 里的下载地址照常写 `releases/download/v<版本>/...`，转正时不用重新生成。


```bash
gh release create v1.0.4 --repo NG666666158/xunxun --prerelease --title "寻寻 1.0.4 测试版" --notes-file <说明> release/Xunxun_1.0.4_x64-setup.exe release/Xunxun_1.0.4_x64-setup.exe.sig release/latest.json
```

确认用户那边的地址还是上一个正式版：

```bash
curl -sL https://github.com/NG666666158/xunxun/releases/latest/download/latest.json
```

### 2. 自己装上试

- **手动安装**：从预发布页面下载 `Xunxun_<版本>_x64-setup.exe`，覆盖安装。和用户手动升级的路径一样。
- **从上一版点“检查更新”**：这才是真正的更新流程（下载、验签、关后台、安装、重启）。已装的版本平时只看正式版。从带 `XUNXUN_UPDATE_ENDPOINT` 的版本起（本改动之后的第一个版本），可以让它这一次改看测试版的 `latest.json`。先在托盘右键“退出”寻寻（第二个实例只会唤起正在运行的那个），再在 PowerShell 里：

  ```powershell
  $exe = (New-Object -ComObject WScript.Shell).CreateShortcut("$env:APPDATA\Microsoft\Windows\Start Menu\Programs\寻寻.lnk").TargetPath
  $env:XUNXUN_UPDATE_ENDPOINT = "https://github.com/NG666666158/xunxun/releases/download/v1.0.4/latest.json"
  & $exe
  ```

  约一分钟后设置按钮上会出现小点（顺便验证了自动检查）；或直接在“设置 → 关于”点“检查更新”。确认后它会下载、验签、安装并重启。重启后不再带这个变量，又回到只看正式版。

  这个变量只改“去哪里找”：安装包仍要用你的私钥签名、版本号仍要比当前高，否则拒绝。不带这个变量的旧版本（1.0.2 及以前），从它们升级只能用 `test_upgrade.py --as-update` 模拟（和更新时的安装方式相同），或在转正后用一台装着旧版的电脑点一次“检查更新”。

### 3. 测试版有问题

不要在同一个版本号下替换安装包：装过 1.0.4 测试版的电脑不会再“更新”到另一个 1.0.4。修好后发下一个号（1.0.5）的测试版。

`CHANGELOG.md` 里把 1.0.4 小节的内容挪回“未发布”、删掉 1.0.4 的标题，再 `bump_version.py 1.0.5`。这样 1.0.5 的说明（也是 `latest.json` 里用户看到的更新说明）包含全部变化。没转正的 1.0.4 预发布可以留着，也可以删掉：

```bash
gh release delete v1.0.4 --repo NG666666158/xunxun --cleanup-tag
```

### 4. 转为正式版

不用重新上传，同一组文件直接转正：

```bash
gh release edit v1.0.4 --repo NG666666158/xunxun --prerelease=false --latest --title "寻寻 1.0.4"
curl -sL https://github.com/NG666666158/xunxun/releases/latest/download/latest.json   # 应该是 1.0.4
```

已装的寻寻在下次启动、或开着时一天之内会查到。

## 需要你决定的事

下面是当初列出的三件事。更新托管（GitHub Releases）和改名已经定了、做了；代码签名还没有。

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
| SignPath Foundation 开源免费计划 | 0 | **可以申请** | 本仓库开源后符合条件。证书发给 SignPath Foundation，属性里的发布者显示它，不是寻寻。工作流已接好，见下 |

几点要知道：

- **签名不等于马上没有警告。** 微软已经让 OV 和 EV 走同样的信誉积累，新证书起步时仍可能提示，下载的人多了才消失。
- 2023 年起私钥必须放在硬件或云 HSM 里，不能再用 PFX 文件。2026 年 3 月起单张证书最长约 460 天，要每年续。
- 中国大陆个人能否通过验证，各 CA 不一样，买之前要问清楚。

我的建议：先申请 SignPath Foundation 的免费签名，批下来之前照旧不签名。买证书的建议（Certum 或 Sectigo 的个人证书，约 ¥800～1600 一年）只在申请不下来时再考虑。

#### SignPath（`release.yml` 里已接好，默认不启用）

工作流在 GitHub 上打包、签名，SignPath 能核对安装包确实是从这个仓库的这次提交打出来的，这是免费计划的前提。没配 `SIGNPATH_API_TOKEN` 时这几步全部跳过。

签哪些文件：

| 文件 | 什么时候签 | 说明 |
|---|---|---|
| `rag-backend.exe` | PyInstaller 之后、`tauri build` 之前 | 后台要先签好才能打进安装包。配置见 `packaging/signpath/backend.xml` |
| `Xunxun_<版本>_x64-setup.exe` | `tauri build` 之后 | 配置见 `packaging/signpath/installer.xml`。签名改变了文件，所以工作流随后用 `tauri signer sign --app-version` 重新生成更新签名，再验一遍 |
| `_internal\` 里的 dll、pyd | 不签 | SignPath Foundation 只签项目自己的程序，Python 和各个库的文件不能用它的证书签（条款写明可以带上未签名的上游 DLL）。用自己买的证书时可以签 |
| 安装包里的 `EnterpriseKB.exe`、卸载程序 | 不签 | SignPath 打不开 NSIS 安装包，只能整体签外层。要签它们得用 `bundle.windows.signCommand`，在 `tauri build` 当中签，SignPath 的免费计划做不到 |

SmartScreen 只看下载下来的安装包，所以外层签好就能消掉警告（新证书仍要积累信誉）。Windows 11 的“智能应用控制”会检查每个程序，开着它的电脑上未签名的 `EnterpriseKB.exe` 仍可能被拦；这项功能大多数电脑上是关着的。

每次发布 SignPath 会收到两个签名请求（后台、安装包），都要你在 SignPath 网页上批准。工作流每个最多等 60 分钟，超时就失败，重跑即可。

启用步骤：

1. GitHub 和 SignPath 账户都开两步验证（条款要求，所有成员都要）。
2. 把 [docs/code-signing-policy.md](../docs/code-signing-policy.md) 定稿（现在是草稿），在 README 和每个 Release 的说明里加上“代码签名政策”的链接（条款要求首页和下载页都有）。
3. 在 [signpath.org/apply](https://signpath.org/apply) 申请。条件以 [signpath.org/terms](https://signpath.org/terms) 为准：OSI 许可证（MIT，满足）、没有闭源组件、已经发布过、仍在维护、有使用说明、隐私说明。
4. 批下来后在 SignPath 里：
   - 项目（slug `xunxun`）链接预定义的 Trusted Build System “GitHub.com”；
   - 建两个 Artifact configuration，slug 分别是 `backend` 和 `installer`，内容取自 `packaging/signpath/` 里的两个文件。先各用一个样本试一下（Update from artifact sample），确认产品名“寻寻”和版本号能对上；
   - 签名策略（slug `release-signing`）要求人工批准；
   - 建一个只有提交权限（Submitter）的 CI 用户，生成 API token。
5. 在 GitHub 加 secret `SIGNPATH_API_TOKEN` 和 variable `SIGNPATH_ORGANIZATION_ID`（slug 不同时再加另外两个 variable，见上面的表）。
6. 推一个新版本的标签试一次。签过名后，把根目录 README 里“安装包目前没有代码签名”那句改掉。

只在推标签时签名；在分支上手动运行工作流不会签。

#### 自己买证书时

`signtool` 要拿得到私钥，而 2023 年起新证书的私钥只能放在 USB 令牌或云 HSM 里。USB 令牌插不到 GitHub 的机器上，Certum 的云签名要在本机登录，所以买来的证书多半只能在本机打包时用：`tauri.conf.json` 的 `bundle.windows` 填 `certificateThumbprint` 和 `timestampUrl`（云签名用 `signCommand`），这样 `tauri build` 会签外壳、卸载程序和安装包，并在签完之后才生成更新签名；另外在 PyInstaller 之后、`tauri build` 之前（`build_desktop.ps1 -BackendOnly` 之后）用 `signtool` 给 `packaging\dist\rag-backend` 里的 exe、pyd、dll 签上，否则后台照样被拦。

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

后台守护（后台意外退出时外壳自动重启它）也这样测。只编译外壳、不打安装包，换一个 `identifier`，后台用 `RAG_BACKEND_EXE` 指过去：

```powershell
'{ "identifier": "com.xunxun.guardtest", "bundle": { "resources": null } }' | Set-Content $env:TEMP\guardtest.json
cd desktop; npx tauri build --no-bundle --config $env:TEMP\guardtest.json
$env:RAG_BACKEND_EXE = "$PWD\..\packaging\dist\rag-backend\rag-backend.exe"
$env:PYTHON_KEYRING_BACKEND = "keyring.backends.null.Keyring"   # 不读真实的 Key
src-tauri\target\release\EnterpriseKB.exe --data-dir <临时目录>
```

然后结束这个测试版外壳下面的 `rag-backend.exe`（按 PID，别按进程名，那样会连正在用的寻寻一起结束）：页面应提示“后台已重启”；10 分钟内再结束一次，应弹出启动窗口，显示退出码和 `<临时目录>\logs\backend.log`。测完删掉 `%LOCALAPPDATA%\com.xunxun.guardtest`。
