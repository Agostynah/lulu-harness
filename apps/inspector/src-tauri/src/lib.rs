use std::sync::Mutex;

use tauri::{Manager, RunEvent};
use tauri_plugin_log::{Target, TargetKind};
use tauri_plugin_shell::process::CommandChild;
use tauri_plugin_shell::ShellExt;

// Holds the running `lulu-server` sidecar child so it can be killed on app
// exit -- without this, closing the window would leave an orphaned
// uvicorn process bound to SIDECAR_PORT behind.
struct SidecarHandle(Mutex<Option<CommandChild>>);

const SIDECAR_PORT: &str = "8420";

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .plugin(tauri_plugin_shell::init())
    .manage(SidecarHandle(Mutex::new(None)))
    .setup(|app| {
      // Always on, not just debug builds -- a sidecar spawn failure or
      // an early lulu-server crash is exactly the kind of thing that
      // only ever happens in a packaged build (dev mode runs lulu-server
      // by hand, outside Tauri entirely), so gating this on
      // cfg!(debug_assertions) meant the one build that actually needs
      // it had nowhere for the logs to go. LogDir writes to the
      // platform log directory regardless of build type; stdout is kept
      // too for `cargo tauri dev`.
      app.handle().plugin(
        tauri_plugin_log::Builder::default()
          .level(log::LevelFilter::Info)
          .targets([
            Target::new(TargetKind::Stdout),
            Target::new(TargetKind::LogDir { file_name: None }),
          ])
          .build(),
      )?;

      // v0: root defaults to the user's home directory -- there's no
      // project picker in the UI yet (see ROADMAP.md); this at least
      // makes the packaged app self-contained instead of needing a
      // terminal open to `cd` into a project first.
      let root = app
        .path()
        .home_dir()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| ".".into());

      let (mut rx, child) = app
        .shell()
        .sidecar("lulu-server")
        .expect("failed to create lulu-server sidecar command")
        .args(["--root", &root, "--port", SIDECAR_PORT])
        .spawn()
        .expect("failed to spawn lulu-server sidecar");

      app.state::<SidecarHandle>().0.lock().unwrap().replace(child);

      tauri::async_runtime::spawn(async move {
        use tauri_plugin_shell::process::CommandEvent;
        while let Some(event) = rx.recv().await {
          match event {
            CommandEvent::Stdout(line) => {
              log::info!("[lulu-server] {}", String::from_utf8_lossy(&line));
            }
            CommandEvent::Stderr(line) => {
              log::warn!("[lulu-server] {}", String::from_utf8_lossy(&line));
            }
            CommandEvent::Error(err) => {
              log::error!("[lulu-server] sidecar error: {err}");
            }
            CommandEvent::Terminated(payload) => {
              log::warn!("[lulu-server] exited: {:?}", payload);
            }
            _ => {}
          }
        }
      });

      Ok(())
    })
    .build(tauri::generate_context!())
    .expect("error while building tauri application")
    .run(|app_handle, event| {
      if let RunEvent::Exit = event {
        if let Some(child) = app_handle.state::<SidecarHandle>().0.lock().unwrap().take() {
          let _ = child.kill();
        }
      }
    });
}
