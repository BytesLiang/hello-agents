use std::env;
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use serde::Serialize;
use tauri::{AppHandle, Manager, State};

const DEFAULT_BACKEND_HOST: &str = "127.0.0.1";
const DEFAULT_BACKEND_PORT: u16 = 8000;

struct BackendProcess {
    child: Child,
    command: String,
    log_path: PathBuf,
    port: u16,
}

#[derive(Default)]
struct DesktopBackendState {
    process: Mutex<Option<BackendProcess>>,
}

#[derive(Serialize)]
struct BackendStatus {
    running: bool,
    port: u16,
    command: Option<String>,
    log_path: Option<String>,
    started_by_app: bool,
}

#[tauri::command]
fn start_backend(
    app: AppHandle,
    state: State<'_, DesktopBackendState>,
) -> Result<BackendStatus, String> {
    let mut guard = state
        .process
        .lock()
        .map_err(|_| String::from("Failed to lock desktop backend state."))?;

    if let Some(process) = guard.as_mut() {
        if process
            .child
            .try_wait()
            .map_err(|error| error.to_string())?
            .is_none()
        {
            return Ok(BackendStatus {
                running: true,
                port: process.port,
                command: Some(process.command.clone()),
                log_path: Some(process.log_path.display().to_string()),
                started_by_app: true,
            });
        }
        *guard = None;
    }

    let port = DEFAULT_BACKEND_PORT;
    let host = String::from(DEFAULT_BACKEND_HOST);
    let project_root = resolve_project_root(&app)?;
    let python_command = resolve_python_command(&project_root);
    let log_path = prepare_backend_log_path(&project_root)?;
    let command_display = format!(
        "{} -m hello_agents.apps.knowledge_qa.api --host {} --port {}",
        python_command.display(),
        host,
        port
    );
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|error| {
            format!(
                "Failed to open the desktop backend log file at {}: {}",
                log_path.display(),
                error
            )
        })?;
    let stderr_log = log_file
        .try_clone()
        .map_err(|error| format!("Failed to clone the desktop backend log file: {}", error))?;

    let mut command = Command::new(&python_command);
    command
        .arg("-m")
        .arg("hello_agents.apps.knowledge_qa.api")
        .arg("--host")
        .arg(&host)
        .arg("--port")
        .arg(port.to_string())
        .current_dir(&project_root)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(stderr_log))
        .env("PYTHONPATH", build_python_path(&project_root))
        .env("KNOWLEDGE_QA_API_HOST", &host)
        .env("KNOWLEDGE_QA_API_PORT", port.to_string());

    let child = command.spawn().map_err(|error| {
        format!(
            "Failed to launch the local Python backend with {}: {}",
            python_command.display(),
            error
        )
    })?;

    let mut backend_process = BackendProcess {
        child,
        command: command_display.clone(),
        log_path: log_path.clone(),
        port,
    };

    for _ in 0..10 {
        if let Some(status) = backend_process
            .child
            .try_wait()
            .map_err(|error| error.to_string())?
        {
            return Err(format!(
                "The local Python backend exited immediately with status {}. Check {}.",
                status,
                log_path.display()
            ));
        }
        thread::sleep(Duration::from_millis(100));
    }

    *guard = Some(backend_process);

    Ok(BackendStatus {
        running: true,
        port,
        command: Some(command_display),
        log_path: Some(log_path.display().to_string()),
        started_by_app: true,
    })
}

#[tauri::command]
fn stop_backend(state: State<'_, DesktopBackendState>) -> Result<(), String> {
    let mut guard = state
        .process
        .lock()
        .map_err(|_| String::from("Failed to lock desktop backend state."))?;
    stop_backend_process(&mut guard)
}

#[tauri::command]
fn backend_status(state: State<'_, DesktopBackendState>) -> Result<BackendStatus, String> {
    let mut guard = state
        .process
        .lock()
        .map_err(|_| String::from("Failed to lock desktop backend state."))?;

    if let Some(process) = guard.as_mut() {
        if process
            .child
            .try_wait()
            .map_err(|error| error.to_string())?
            .is_none()
        {
            return Ok(BackendStatus {
                running: true,
                port: process.port,
                command: Some(process.command.clone()),
                log_path: Some(process.log_path.display().to_string()),
                started_by_app: true,
            });
        }
        *guard = None;
    }

    Ok(BackendStatus {
        running: false,
        port: DEFAULT_BACKEND_PORT,
        command: None,
        log_path: None,
        started_by_app: false,
    })
}

fn stop_backend_process(process: &mut Option<BackendProcess>) -> Result<(), String> {
    if let Some(mut running) = process.take() {
        if running
            .child
            .try_wait()
            .map_err(|error| error.to_string())?
            .is_none()
        {
            running.child.kill().map_err(|error| error.to_string())?;
        }
        let _ = running.child.wait();
    }
    Ok(())
}

fn resolve_project_root(_app: &AppHandle) -> Result<PathBuf, String> {
    if let Ok(value) = env::var("KNOWLEDGE_QA_DESKTOP_PROJECT_ROOT") {
        let path = PathBuf::from(value);
        if path.exists() {
            return Ok(path);
        }
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| String::from("Failed to resolve the desktop project root."))
}

fn resolve_python_command(project_root: &Path) -> PathBuf {
    if let Ok(value) = env::var("KNOWLEDGE_QA_DESKTOP_PYTHON") {
        return PathBuf::from(value);
    }

    let venv_python = project_root.join(".venv/bin/python");
    if venv_python.exists() {
        return venv_python;
    }

    PathBuf::from("python3")
}

fn build_python_path(project_root: &Path) -> String {
    let src_path = project_root.join("src");
    if let Some(existing) = env::var_os("PYTHONPATH") {
        let mut paths = vec![src_path.clone()];
        paths.extend(env::split_paths(&existing));
        env::join_paths(paths)
            .unwrap_or_else(|_| src_path.as_os_str().to_os_string())
            .to_string_lossy()
            .into_owned()
    } else {
        src_path.to_string_lossy().into_owned()
    }
}

fn prepare_backend_log_path(project_root: &Path) -> Result<PathBuf, String> {
    let log_dir = project_root.join(".hello_agents/logs");
    fs::create_dir_all(&log_dir).map_err(|error| {
        format!(
            "Failed to create the desktop backend log directory at {}: {}",
            log_dir.display(),
            error
        )
    })?;
    Ok(log_dir.join("knowledge_qa_desktop_backend.log"))
}

fn cleanup_backend(app: &AppHandle) {
    if let Some(state) = app.try_state::<DesktopBackendState>() {
        if let Ok(mut guard) = state.process.lock() {
            let _ = stop_backend_process(&mut guard);
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(DesktopBackendState::default())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            start_backend,
            stop_backend,
            backend_status
        ])
        .build(tauri::generate_context!())
        .expect("error while building knowledge qa desktop application")
        .run(|app, event| {
            if matches!(event, tauri::RunEvent::Exit) {
                cleanup_backend(app);
            }
        });
}
