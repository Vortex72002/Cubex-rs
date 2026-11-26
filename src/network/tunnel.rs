use parking_lot::Mutex;
use regex::Regex;
use std::io::{BufRead, BufReader};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::thread;

use crate::network::invite::encode_invite;

#[derive(Debug, Clone, Default)]
pub struct TunnelStatus {
    pub running: bool,
    pub public_address: String,
    pub invite_code: String,
    pub claim_url: String,
    pub status_text: String,
    pub logs: Vec<String>,
}

pub struct PlayitTunnelManager {
    pub binary: PathBuf,
    pub secret_path: PathBuf,
    pub log_path: PathBuf,
    pub status: Arc<Mutex<TunnelStatus>>,
    child: Arc<Mutex<Option<Child>>>,
}

impl PlayitTunnelManager {
    pub fn new(binary: impl AsRef<Path>, secret_path: impl AsRef<Path>, log_path: impl AsRef<Path>) -> Self {
        Self {
            binary: binary.as_ref().to_path_buf(),
            secret_path: secret_path.as_ref().to_path_buf(),
            log_path: log_path.as_ref().to_path_buf(),
            status: Arc::new(Mutex::new(TunnelStatus {
                status_text: "Offline".into(),
                ..Default::default()
            })),
            child: Arc::new(Mutex::new(None)),
        }
    }

    pub fn build_args(&self) -> Vec<String> {
        vec![
            "--secret-path".into(),
            self.secret_path.to_string_lossy().to_string(),
            "--log-path".into(),
            self.log_path.to_string_lossy().to_string(),
        ]
    }

    pub fn is_running(&self) -> bool {
        let mut g = self.child.lock();
        if let Some(ref mut c) = *g {
            match c.try_wait() {
                Ok(None) => return true,
                _ => *g = None,
            }
        }
        false
    }

    pub fn pid(&self) -> Option<u32> {
        self.child.lock().as_ref().map(Child::id)
    }

    pub fn start(&self) -> anyhow::Result<()> {
        if self.is_running() {
            return Ok(());
        }
        if !self.binary.is_file() {
            let mut st = self.status.lock();
            st.running = false;
            st.status_text = format!("Missing binary: {}", self.binary.display());
            anyhow::bail!("playit binary missing at {}", self.binary.display());
        }

        if let Some(p) = self.secret_path.parent() {
            let _ = std::fs::create_dir_all(p);
        }
        if let Some(p) = self.log_path.parent() {
            let _ = std::fs::create_dir_all(p);
        }

        let mut cmd = Command::new(&self.binary);
        cmd.args(self.build_args())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        // Spawn in its own process group for zero-orphan cleanup
        unsafe {
            cmd.pre_exec(|| {
                let _ = libc_setpgid();
                Ok(())
            });
        }

        let mut spawned = cmd.spawn()?;
        let stdout = spawned.stdout.take();
        let stderr = spawned.stderr.take();
        *self.child.lock() = Some(spawned);

        {
            let mut st = self.status.lock();
            st.running = true;
            st.status_text = "Connecting to playit.gg...".into();
        }

        if let Some(out) = stdout {
            let st_arc = self.status.clone();
            thread::spawn(move || {
                let addr_re = Regex::new(r"\b([a-zA-Z0-9][a-zA-Z0-9.\-]*\.(?:ply\.gg|playit\.gg|at\.ply\.gg|joinmc\.link):\d{2,5})\b").unwrap();
                let claim_re = Regex::new(r"(https://playit\.gg/claim/[A-Za-z0-9_\-]+)").unwrap();
                let reader = BufReader::new(out);
                for line in reader.lines().map_while(Result::ok) {
                    let clean = line.trim().to_string();
                    if clean.is_empty() {
                        continue;
                    }
                    let mut st = st_arc.lock();
                    st.logs.push(clean.clone());
                    if st.logs.len() > 200 {
                        st.logs.remove(0);
                    }
                    if let Some(cap) = claim_re.captures(&clean) {
                        st.claim_url = cap[1].to_string();
                        st.status_text = "Claim required — open link in browser".into();
                    }
                    if let Some(cap) = addr_re.captures(&clean) {
                        let addr = cap[1].to_string();
                        if let Some((h, p_s)) = addr.rsplit_once(':') {
                            if let Ok(port) = p_s.parse::<u16>() {
                                if let Ok(code) = encode_invite(h, port) {
                                    st.public_address = addr.clone();
                                    st.invite_code = code.clone();
                                    st.status_text = format!("Online — {addr}");
                                    let _ = std::fs::write(
                                        "/tmp/cubex_wan_info.txt",
                                        format!("Address: {addr}  |  Invite Code: {code}"),
                                    );
                                }
                            }
                        }
                    }
                }
            });
        }

        if let Some(err) = stderr {
            let st_arc = self.status.clone();
            thread::spawn(move || {
                let reader = BufReader::new(err);
                for line in reader.lines().map_while(Result::ok) {
                    let mut st = st_arc.lock();
                    st.logs.push(line);
                    if st.logs.len() > 200 {
                        st.logs.remove(0);
                    }
                }
            });
        }

        Ok(())
    }

    pub fn stop(&self) {
        let child_opt = self.child.lock().take();
        if let Some(mut child) = child_opt {
            let pid = child.id() as i32;
            // 1. Cleanly signal SIGTERM to the entire process group first
            libc_kill_pg(pid, 15);
            let deadline = std::time::Instant::now() + std::time::Duration::from_millis(350);
            let mut exited = false;
            while std::time::Instant::now() < deadline {
                match child.try_wait() {
                    Ok(Some(_)) => {
                        exited = true;
                        break;
                    }
                    Ok(None) => std::thread::sleep(std::time::Duration::from_millis(10)),
                    Err(_) => break,
                }
            }
            // 2. Fallback to SIGKILL on the process group + direct child handle if still alive
            if !exited {
                libc_kill_pg(pid, 9);
                let _ = child.kill();
                let _ = child.wait();
            }
        }
        let mut st = self.status.lock();
        st.running = false;
        st.status_text = "Offline".into();
        st.public_address.clear();
        st.invite_code.clear();
        st.claim_url.clear();
    }
}

impl Drop for PlayitTunnelManager {
    fn drop(&mut self) {
        self.stop();
    }
}

fn libc_setpgid() -> i32 {
    unsafe {
        extern "C" {
            fn setpgid(pid: i32, pgid: i32) -> i32;
        }
        setpgid(0, 0)
    }
}

fn libc_kill_pg(pgid: i32, sig: i32) -> i32 {
    if pgid <= 0 {
        return -1;
    }
    unsafe {
        extern "C" {
            fn kill(pid: i32, sig: i32) -> i32;
        }
        kill(-pgid, sig)
    }
}
