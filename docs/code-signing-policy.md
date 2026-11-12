# Code signing policy / 代码签名政策

> **草稿 / DRAFT — not in effect.** No 寻寻 release is code-signed yet. This page is prepared for
> an application to the [SignPath Foundation](https://signpath.org) free code signing program for
> open source projects. It follows the conditions on [signpath.org/terms](https://signpath.org/terms)
> as read on 2026-11-11; check them again before applying, since they may change. Items marked
> **(confirm)** must be checked by the maintainer before this page is published.
>
> 发布者确认后再去掉这段说明，并在根目录 README 和每个 Release 的说明里链接本页。

寻寻 的 Windows 安装包由 GitHub Actions 从本仓库的源码打包，经 SignPath Foundation 免费签名。下面说明签哪些文件、谁能改代码、谁批准签名，以及程序会把哪些数据发到哪里。

Free code signing provided by [SignPath.io](https://about.signpath.io), certificate by [SignPath Foundation](https://signpath.org)

## What is signed

| File | Signed | Notes |
|---|---|---|
| `Xunxun_<version>_x64-setup.exe` (the installer on [Releases](https://github.com/NG666666158/xunxun/releases)) | yes | |
| `rag-backend.exe` (the app's backend, inside the installer) | yes | |
| `EnterpriseKB.exe` (the app window) and the uninstaller, inside the installer | no | Created while the NSIS installer is built; the installer is signed as a whole |
| Python runtime and library files (`.dll`, `.pyd`) in the backend folder | no | Third-party binaries, shipped unsigned as published by their authors |

Every signed file carries the product name **寻寻** and the release's version number.

Only release builds are signed. A release is a tag `v<version>` pushed to this repository. The
workflow [`.github/workflows/release.yml`](../.github/workflows/release.yml) runs on
GitHub-hosted runners and:

1. checks that the tag matches the version in the source files;
2. runs the full test suite ([`ci.yml`](../.github/workflows/ci.yml));
3. builds the backend (PyInstaller) and the installer (Tauri, NSIS) from the source at that
   tag, with dependency versions pinned by the committed lock files;
4. submits `rag-backend.exe` and then the installer to SignPath. Each signing request waits for
   manual approval;
5. puts the signed installer in a draft GitHub release, which the maintainer tests and then
   publishes by hand.

Build scripts and CI configuration are part of this repository and are reviewed like any other
source code.

## Team and roles

**(confirm)** 寻寻 is maintained by one person.

| Role | Who | Responsibility |
|---|---|---|
| Author / committer | [NG666666158](https://github.com/NG666666158) | May change the source code, build scripts and CI configuration without further review |
| Reviewer | [NG666666158](https://github.com/NG666666158) | Reviews every change from anyone else (pull requests) before it is merged |
| Approver | [NG666666158](https://github.com/NG666666158) | Approves each signing request in SignPath after checking that it comes from a release tag of this repository |

All team members use multi-factor authentication for both GitHub and SignPath.

If more people join, this table lists them, and their roles are mirrored in GitHub (repository
collaborators) and SignPath (project members).

## Privacy

**(confirm)** This program will not transfer any information to other networked systems unless
specifically requested by the user or the person installing or operating it.

In detail:

- **Model service.** To read documents and answer questions, 寻寻 sends document text and
  questions to the model service the user connects in the first-run guide with their own API
  key (currently [SiliconFlow](https://cloud.siliconflow.cn/)). That service processes the text
  under its own privacy policy. Folders the user marks as "name only" (只按文件名) or excluded
  (不收录) are not read, and nothing about them is sent. The first-run guide and Settings ›
  数据去向 show what is sent.
- **Updates.** 寻寻 contacts GitHub only when the user clicks Settings › 关于 › 检查更新.
- **Optional components.** The local OCR components are downloaded from PyPI (or a mirror) only
  when the user chooses to install them.
- **Installer.** If WebView2 is missing, the installer downloads Microsoft's WebView2 installer.
- No telemetry, analytics or crash reports are sent. Conversations, the index and settings stay
  on the user's computer; the API key is stored in Windows Credential Manager.

## Reporting a problem

If you believe a signed 寻寻 file is malicious, was not built from this repository, or violates
this policy, please open an issue at
[github.com/NG666666158/xunxun/issues](https://github.com/NG666666158/xunxun/issues) or contact
[SignPath Foundation](https://signpath.org).
