// Desktop shell: starts the packaged Python backend on a random loopback port with a
// per-launch token, waits for it to answer, then shows the UI. The backend dies with us.
// If it dies first, it is started once more the same way (see `guard_backend`).
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Emitter, Manager, RunEvent, WebviewUrl, WebviewWindowBuilder, WindowEvent};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};
use tauri_plugin_updater::UpdaterExt;

const MAIN: &str = "main";
const SPLASH: &str = "splash";
const QUICK: &str = "quick";
const TITLE: &str = "寻寻";
// First launch builds jieba's dictionary cache and imports LangChain; allow for slow disks.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(90);
const LOG_HINT: &str = r"%APPDATA%\EnterpriseKnowledgeBase\logs\backend.log";
const UPDATE_CHECK_TIMEOUT: Duration = Duration::from_secs(30);
// A backend that dies twice within this long is not started a third time.
const RESTART_WINDOW: Duration = Duration::from_secs(10 * 60);
const GUARD_INTERVAL: Duration = Duration::from_millis(500);

/// The running backend process, killed on exit.
struct Backend(Mutex<Option<Child>>);

/// How the backend was started: a restart uses the same program, port and token, so the
/// open windows keep talking to it without reloading.
#[derive(Clone)]
struct BackendLaunch {
    exe: PathBuf,
    port: u16,
    token: String,
}

/// Set when the app quits or hands over to an update or a restart: from then on the
/// backend's exit is expected and must not be answered with a new one.
struct Stopping(AtomicBool);

/// How the running backend was started, kept so a failed update can start it again.
struct Launched(Mutex<Option<BackendLaunch>>);

/// Which guard thread is the current one. A guard that finds a newer number stops, so a
/// backend started again after a failed update is never watched by two of them.
struct GuardEpoch(AtomicU64);

/// The script that tells a webview where the backend runs (set once it is live).
struct DesktopScript(Mutex<Option<String>>);

/// The quick-search shortcut currently registered.
struct QuickShortcut(Mutex<Option<Shortcut>>);

/// Closing the main window hides it to the tray (true) or quits (false).
struct CloseToTray(AtomicBool);

/// The saved choice from the Settings page (`close_to_tray` in preferences.json, which
/// the backend writes); on by default. The page also sends it once it loads.
fn saved_close_to_tray() -> bool {
    let Ok(appdata) = std::env::var("APPDATA") else { return true };
    let path = PathBuf::from(appdata).join("EnterpriseKnowledgeBase").join("preferences.json");
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|prefs| prefs.get("close_to_tray").and_then(|v| v.as_bool()))
        .unwrap_or(true)
}

fn free_port() -> std::io::Result<u16> {
    Ok(TcpListener::bind(("127.0.0.1", 0))?.local_addr()?.port())
}

fn new_token() -> String {
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes).expect("OS random number generator");
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn backend_exe(app: &AppHandle) -> Result<PathBuf, String> {
    if let Ok(path) = std::env::var("RAG_BACKEND_EXE") {
        return Ok(PathBuf::from(path));
    }
    let bundled = app
        .path()
        .resource_dir()
        .map_err(|e| e.to_string())?
        .join("backend")
        .join("rag-backend.exe");
    if bundled.exists() {
        return Ok(bundled);
    }
    #[cfg(debug_assertions)]
    {
        // `tauri dev`: the PyInstaller output inside the repository.
        let dev = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../packaging/dist/rag-backend/rag-backend.exe");
        if dev.exists() {
            return Ok(dev);
        }
    }
    Err(format!("找不到后台程序：{}", bundled.display()))
}

/// `--data-dir <folder>` on the shell's command line goes on to the backend, for tests on a
/// copy of the data (an update's restart passes the same arguments again).
fn data_dir_argument() -> Option<String> {
    let mut args = std::env::args();
    while let Some(arg) = args.next() {
        if arg == "--data-dir" {
            return args.next();
        }
    }
    None
}

/// Where the backend writes its log, for the error messages.
fn log_hint() -> String {
    match data_dir_argument() {
        Some(dir) => Path::new(&dir).join("logs").join("backend.log").display().to_string(),
        None => LOG_HINT.to_string(),
    }
}

fn spawn_backend(exe: &Path, port: u16, token: &str) -> std::io::Result<Child> {
    let mut command = Command::new(exe);
    if let Some(dir) = data_dir_argument() {
        command.arg("--data-dir").arg(dir);
    }
    command
        .arg("--port")
        .arg(port.to_string())
        .arg("--parent-pid")
        .arg(std::process::id().to_string())
        // Via the environment, not argv, so the token does not show in process listings.
        .env("RAG_APP_TOKEN", token)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Some(dir) = exe.parent() {
        command.current_dir(dir);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    command.spawn()
}

/// GET /api/v1/health/live (the one endpoint that needs no token) over a raw socket.
fn is_live(port: u16) -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_millis(500)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(3)));
    let request = format!("GET /api/v1/health/live HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut head = [0u8; 12];
    let mut filled = 0;
    while filled < head.len() {
        match stream.read(&mut head[filled..]) {
            Ok(0) | Err(_) => break,
            Ok(n) => filled += n,
        }
    }
    head.starts_with(b"HTTP/1.1 200")
}

/// The backend's exit status once it has exited; None while it runs or when there is none.
fn backend_exit(app: &AppHandle) -> Option<ExitStatus> {
    app.state::<Backend>()
        .0
        .lock()
        .unwrap()
        .as_mut()
        .and_then(|child| child.try_wait().ok().flatten())
}

fn exit_message(status: ExitStatus) -> String {
    format!("后台程序意外退出（{status}）。\n日志：{}", log_hint())
}

/// Wait until the backend answers on `port`; an error when it exits first or takes too long.
fn wait_until_live(app: &AppHandle, port: u16) -> Result<(), String> {
    let started = Instant::now();
    loop {
        if is_live(port) {
            return Ok(());
        }
        if let Some(status) = backend_exit(app) {
            return Err(exit_message(status));
        }
        if started.elapsed() > STARTUP_TIMEOUT {
            return Err(format!("后台程序启动超时。\n日志：{}", log_hint()));
        }
        std::thread::sleep(Duration::from_millis(150));
    }
}

fn stopping(app: &AppHandle) -> bool {
    app.state::<Stopping>().0.load(Ordering::SeqCst)
}

#[derive(Debug, PartialEq)]
enum AfterExit {
    /// The app is quitting, updating or restarting: the backend was meant to stop.
    Expected,
    Restart,
    /// It already died once within RESTART_WINDOW: show the error instead of looping.
    GiveUp,
}

fn after_exit(stopping: bool, last_restart: Option<Instant>, now: Instant) -> AfterExit {
    if stopping {
        return AfterExit::Expected;
    }
    match last_restart {
        Some(at) if now.saturating_duration_since(at) < RESTART_WINDOW => AfterExit::GiveUp,
        _ => AfterExit::Restart,
    }
}

/// Start the backend again exactly as before. The check against `Stopping` happens under
/// the same lock `stop_backend` takes, so a quit during a restart never leaves one running.
fn restart_backend(app: &AppHandle, launch: &BackendLaunch) -> Result<(), String> {
    let mut child = spawn_backend(&launch.exe, launch.port, &launch.token)
        .map_err(|e| format!("无法重新启动后台程序：{e}\n日志：{}", log_hint()))?;
    {
        let backend = app.state::<Backend>();
        let mut slot = backend.0.lock().unwrap();
        if stopping(app) {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(());
        }
        *slot = Some(child);
    }
    wait_until_live(app, launch.port)
}

/// Runs for as long as the app does. When the backend exits on its own (a crash, or ended
/// from Task Manager) it is started once more with the same port and token, and the
/// windows hear "backend-restarted" so the page can say so and reload what it shows. A
/// second exit within RESTART_WINDOW shows the error with the log's path instead.
fn guard_backend(app: AppHandle, launch: BackendLaunch) {
    let epoch = app.state::<GuardEpoch>().0.fetch_add(1, Ordering::SeqCst) + 1;
    let mut last_restart: Option<Instant> = None;
    loop {
        std::thread::sleep(GUARD_INTERVAL);
        if stopping(&app) || app.state::<GuardEpoch>().0.load(Ordering::SeqCst) != epoch {
            return;
        }
        let Some(status) = backend_exit(&app) else { continue };
        match after_exit(stopping(&app), last_restart, Instant::now()) {
            AfterExit::Expected => return,
            AfterExit::GiveUp => {
                eprintln!("backend exited again ({status}); not restarting");
                show_error(&app, &exit_message(status));
                return;
            }
            AfterExit::Restart => {
                eprintln!("backend exited ({status}); starting it again");
                last_restart = Some(Instant::now());
                match restart_backend(&app, &launch) {
                    Ok(()) if stopping(&app) => return,
                    Ok(()) => {
                        eprintln!("backend restarted on port {}", launch.port);
                        let _ = app.emit("backend-restarted", ());
                    }
                    Err(_) if stopping(&app) => return,
                    Err(message) => {
                        show_error(&app, &message);
                        return;
                    }
                }
            }
        }
    }
}

/// The startup window shows why the backend could not run. After startup it is gone, so it
/// opens again, in front, with the message already in it.
fn show_error(app: &AppHandle, message: &str) {
    let literal = format!("{message:?}"); // a valid JS string literal for our messages
    if let Some(splash) = app.get_webview_window(SPLASH) {
        let _ = splash.eval(&format!("window.showError({literal})"));
        return;
    }
    let built = WebviewWindowBuilder::new(app, SPLASH, WebviewUrl::App("splash.html".into()))
        .title(TITLE)
        .inner_size(420.0, 300.0)
        .resizable(false)
        .maximizable(false)
        .center()
        .always_on_top(true)
        .initialization_script(&format!("window.__SHELL_ERROR__ = {literal};"))
        .build();
    if let Ok(window) = built {
        let _ = window.set_focus();
    }
}

fn launch(app: &AppHandle) -> Result<(), String> {
    let started = Instant::now();
    let exe = backend_exe(app)?;
    let port = free_port().map_err(|e| format!("无法分配端口：{e}"))?;
    let token = new_token();
    let child = spawn_backend(&exe, port, &token).map_err(|e| format!("无法启动后台程序：{e}"))?;
    *app.state::<Backend>().0.lock().unwrap() = Some(child);
    wait_until_live(app, port)?;

    // Only this webview learns where the backend is and the token it must send.
    let script = format!(
        "window.__RAG_DESKTOP__ = Object.freeze({{ apiBase: \"http://127.0.0.1:{port}\", token: \"{token}\" }});"
    );
    *app.state::<DesktopScript>().0.lock().unwrap() = Some(script.clone());
    let window = WebviewWindowBuilder::new(app, MAIN, WebviewUrl::App("index.html".into()))
        .title(TITLE)
        .inner_size(1280.0, 820.0)
        .min_inner_size(880.0, 560.0)
        .center()
        // The native drop handler is on (the default): dropped files and folders reach the
        // page as real paths (webview.onDragDropEvent), so they are added in place, not copied.
        .initialization_script(&script)
        .build()
        .map_err(|e| e.to_string())?;
    if let Some(splash) = app.get_webview_window(SPLASH) {
        let _ = splash.close();
    }
    // Closing the main window hides it to the tray (the backend keeps watching folders and
    // the quick-search shortcut keeps working), or quits when that is switched off. Quitting
    // also covers the hidden quick-search window, which would otherwise keep the app alive.
    let handle = app.clone();
    let hide_target = window.clone();
    window.on_window_event(move |event| match event {
        WindowEvent::CloseRequested { api, .. } if handle.state::<CloseToTray>().0.load(Ordering::Relaxed) => {
            api.prevent_close();
            let _ = hide_target.hide();
        }
        WindowEvent::Destroyed => handle.exit(0),
        _ => {}
    });
    let _ = window.set_focus();
    eprintln!("backend ready on port {port} after {} ms", started.elapsed().as_millis());
    let launched = BackendLaunch { exe, port, token };
    *app.state::<Launched>().0.lock().unwrap() = Some(launched.clone());
    let handle = app.clone();
    std::thread::spawn(move || guard_backend(handle, launched));
    Ok(())
}

/// The backend was stopped for an update that then could not be installed: start it again
/// on the same port with the same token, and watch it again. The windows hear
/// "backend-restarted" and reload what they show.
fn resume_backend(app: &AppHandle) -> Result<(), String> {
    let Some(launched) = app.state::<Launched>().0.lock().unwrap().clone() else {
        return Err("后台程序还没有启动过。".into());
    };
    // Only the update set this. A quit that arrives from here on sets it again and ends
    // the new backend through `stop_backend`, as at any other time.
    app.state::<Stopping>().0.store(false, Ordering::SeqCst);
    restart_backend(app, &launched)?;
    if stopping(app) {
        return Ok(());
    }
    let handle = app.clone();
    std::thread::spawn(move || guard_backend(handle, launched));
    let _ = app.emit("backend-restarted", ());
    Ok(())
}

fn show_main(app: &AppHandle) {
    let window = app
        .get_webview_window(MAIN)
        .or_else(|| app.get_webview_window(SPLASH));
    if let Some(window) = window {
        let _ = window.unminimize();
        let _ = window.show();
        let _ = window.set_focus();
    }
}

/// The small always-on-top search box, created hidden on first use.
fn quick_window(app: &AppHandle) -> Result<tauri::WebviewWindow, String> {
    if let Some(window) = app.get_webview_window(QUICK) {
        return Ok(window);
    }
    let script = app
        .state::<DesktopScript>()
        .0
        .lock()
        .unwrap()
        .clone()
        .ok_or("后台程序尚未就绪")?;
    let window = WebviewWindowBuilder::new(app, QUICK, WebviewUrl::App("quick.html".into()))
        .title("快速搜索")
        .inner_size(640.0, 440.0)
        .resizable(false)
        .decorations(false)
        .always_on_top(true)
        .skip_taskbar(true)
        .visible(false)
        .center()
        .initialization_script(&script)
        .build()
        .map_err(|e| e.to_string())?;
    let handle = window.clone();
    window.on_window_event(move |event| {
        // Clicking anywhere else dismisses it, like a launcher.
        if let WindowEvent::Focused(false) = event {
            let _ = handle.hide();
        }
    });
    Ok(window)
}

fn toggle_quick(app: &AppHandle) {
    let Ok(window) = quick_window(app) else { return };
    if window.is_visible().unwrap_or(false) && window.is_focused().unwrap_or(false) {
        let _ = window.hide();
        return;
    }
    let _ = window.center();
    let _ = window.show();
    let _ = window.set_focus();
    let _ = app.emit_to(QUICK, "quick-show", ());
}

/// Register `accelerator` (e.g. "Alt+Space") for the quick-search window, replacing the
/// previous one only once the new one is registered.
#[tauri::command]
fn set_quick_search_shortcut(app: AppHandle, accelerator: String) -> Result<(), String> {
    let shortcut: Shortcut = accelerator
        .parse()
        .map_err(|e| format!("快捷键格式不正确：{e}"))?;
    let state = app.state::<QuickShortcut>();
    let mut current = state.0.lock().unwrap();
    if current.as_ref() == Some(&shortcut) {
        return Ok(());
    }
    app.global_shortcut()
        .on_shortcut(shortcut, |app, _shortcut, event| {
            if event.state() == ShortcutState::Pressed {
                toggle_quick(app);
            }
        })
        .map_err(|e| format!("快捷键已被其他程序占用或不可用：{e}"))?;
    if let Some(old) = current.replace(shortcut) {
        let _ = app.global_shortcut().unregister(old);
    }
    Ok(())
}

#[tauri::command]
fn set_close_to_tray(app: AppHandle, enabled: bool) {
    app.state::<CloseToTray>().0.store(enabled, Ordering::Relaxed);
}

#[tauri::command]
fn hide_quick(app: AppHandle) {
    if let Some(window) = app.get_webview_window(QUICK) {
        let _ = window.hide();
    }
}

/// From the quick-search window: show the main window's reader at a file (and page), with
/// the matched text to highlight there.
#[tauri::command]
fn open_reader(app: AppHandle, file_id: String, locator: String, quote: Option<String>) {
    hide_quick(app.clone());
    show_main(&app);
    let _ = app.emit_to(
        MAIN,
        "open-reader",
        serde_json::json!({ "file_id": file_id, "locator": locator, "quote": quote.unwrap_or_default() }),
    );
}

/// From the quick-search window: scope the main window's next questions to one file.
#[tauri::command]
fn ask_about_file(app: AppHandle, file_id: String, name: String) {
    hide_quick(app.clone());
    show_main(&app);
    let _ = app.emit_to(MAIN, "ask-file", serde_json::json!({ "file_id": file_id, "name": name }));
}

/// Whether this build knows where updates come from (see packaging/README_RELEASE.md);
/// tauri.conf.json leaves it empty until hosting is decided.
#[tauri::command]
fn update_configured(app: AppHandle) -> bool {
    app.config()
        .plugins
        .0
        .get("updater")
        .and_then(|updater| updater.get("endpoints"))
        .and_then(|endpoints| endpoints.as_array())
        .is_some_and(|endpoints| !endpoints.is_empty())
}

/// Where updates come from: the build's endpoint, or `XUNXUN_UPDATE_ENDPOINT` (an https
/// latest.json) so the owner can try a GitHub pre-release from an installed copy; see "测试版"
/// in packaging/README_RELEASE.md. Either way the download must be a newer version signed with
/// the build's key. The installer's restart does not pass the variable on.
fn updater_builder(app: &AppHandle) -> Result<tauri_plugin_updater::UpdaterBuilder, String> {
    let builder = app.updater_builder();
    match std::env::var("XUNXUN_UPDATE_ENDPOINT").ok().filter(|url| !url.trim().is_empty()) {
        Some(url) => {
            let url = tauri::Url::parse(url.trim()).map_err(|e| format!("XUNXUN_UPDATE_ENDPOINT: {e}"))?;
            builder.endpoints(vec![url]).map_err(|e| e.to_string())
        }
        None => Ok(builder),
    }
}

/// Also run by the page by itself (updatecheck.js), so a check that hangs gives up rather
/// than holding back the next one; the download in install_update has no such limit.
#[tauri::command]
async fn check_update(app: AppHandle) -> Result<Option<serde_json::Value>, String> {
    let updater = updater_builder(&app)?
        .timeout(UPDATE_CHECK_TIMEOUT)
        .build()
        .map_err(|e| e.to_string())?;
    let update = updater.check().await.map_err(|e| e.to_string())?;
    Ok(update.map(|update| serde_json::json!({ "version": update.version, "notes": update.body })))
}

/// Download, verify against the build's public key, then hand over to the installer. The
/// backend stops first so the installer can replace its files; on Windows the installer ends
/// this process. Data stays in %APPDATA%, the installer never touches it.
#[tauri::command]
async fn install_update(app: AppHandle) -> Result<(), String> {
    let updater = updater_builder(&app)?.build().map_err(|e| e.to_string())?;
    let Some(update) = updater.check().await.map_err(|e| e.to_string())? else {
        return Err("已经是最新版本。".into());
    };
    let bytes = update.download(|_, _| {}, || {}).await.map_err(|e| e.to_string())?;
    stop_backend(&app);
    // On Windows a started installer ends this process, so coming back here with an error
    // means nothing was installed: the app stays open, and needs its backend again.
    let Err(error) = update.install(bytes) else { return Ok(()) };
    let handle = app.clone();
    let resumed = tauri::async_runtime::spawn_blocking(move || resume_backend(&handle))
        .await
        .map_err(|e| e.to_string())
        .and_then(|result| result);
    match resumed {
        Ok(()) => Err(format!("更新没有装上（{error}），现在仍是原来的版本。")),
        Err(message) => {
            show_error(&app, &message);
            Err(format!("更新没有装上（{error}），后台也没能重新启动。请退出后重新打开寻寻。"))
        }
    }
}

/// After a recovery was scheduled: the backend applies it on its next start.
#[tauri::command]
fn restart_app(app: AppHandle) {
    stop_backend(&app);
    app.restart();
}

/// Every caller means to end the backend (quit, update, restart, failed start), so the guard
/// is told first and does not start it again.
fn stop_backend(app: &AppHandle) {
    app.state::<Stopping>().0.store(true, Ordering::SeqCst);
    if let Some(mut child) = app.state::<Backend>().0.lock().unwrap().take() {
        let _ = child.kill();
        let _ = child.wait();
    }
}

fn main() {
    tauri::Builder::default()
        // Must be first: a second launch only focuses the running window.
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| show_main(app)))
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(Backend(Mutex::new(None)))
        .manage(Stopping(AtomicBool::new(false)))
        .manage(Launched(Mutex::new(None)))
        .manage(GuardEpoch(AtomicU64::new(0)))
        .manage(DesktopScript(Mutex::new(None)))
        .manage(QuickShortcut(Mutex::new(None)))
        .manage(CloseToTray(AtomicBool::new(saved_close_to_tray())))
        .invoke_handler(tauri::generate_handler![
            set_quick_search_shortcut,
            set_close_to_tray,
            hide_quick,
            open_reader,
            ask_about_file,
            restart_app,
            update_configured,
            check_update,
            install_update
        ])
        .setup(|app| {
            WebviewWindowBuilder::new(app, SPLASH, WebviewUrl::App("splash.html".into()))
                .title(TITLE)
                .inner_size(420.0, 300.0)
                .resizable(false)
                .maximizable(false)
                .center()
                .build()?;

            let open = MenuItem::with_id(app, "open", "打开", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "退出", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&open, &quit])?;
            TrayIconBuilder::with_id("tray")
                .icon(app.default_window_icon().cloned().expect("bundle icon"))
                .tooltip(format!("{TITLE}（单击打开，右键可退出）"))
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "open" => show_main(app),
                    "quit" => app.exit(0),
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        show_main(tray.app_handle());
                    }
                })
                .build(app)?;

            let handle = app.handle().clone();
            std::thread::spawn(move || {
                if let Err(message) = launch(&handle) {
                    stop_backend(&handle);
                    if let Some(splash) = handle.get_webview_window(SPLASH) {
                        let literal = format!("{message:?}"); // a valid JS string literal for our messages
                        let _ = splash.eval(&format!("window.showError({literal})"));
                    }
                }
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("failed to build the desktop shell")
        .run(|app, event| match event {
            // Quitting has begun: the backend's exit from here on is expected.
            RunEvent::ExitRequested { .. } => app.state::<Stopping>().0.store(true, Ordering::SeqCst),
            // Closing the window (or 退出 in the tray) ends the app; take the backend with it.
            RunEvent::Exit => stop_backend(app),
            _ => {}
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_unexpected_exit_restarts() {
        assert_eq!(after_exit(false, None, Instant::now()), AfterExit::Restart);
    }

    #[test]
    fn second_exit_within_ten_minutes_gives_up() {
        let restarted = Instant::now();
        let now = restarted + Duration::from_secs(9 * 60 + 59);
        assert_eq!(after_exit(false, Some(restarted), now), AfterExit::GiveUp);
    }

    #[test]
    fn exit_after_ten_quiet_minutes_restarts_again() {
        let restarted = Instant::now();
        let now = restarted + RESTART_WINDOW + Duration::from_secs(1);
        assert_eq!(after_exit(false, Some(restarted), now), AfterExit::Restart);
    }

    #[test]
    fn never_restarts_while_stopping() {
        let now = Instant::now();
        assert_eq!(after_exit(true, None, now), AfterExit::Expected);
        assert_eq!(after_exit(true, Some(now), now), AfterExit::Expected);
    }
}
