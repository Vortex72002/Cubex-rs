use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::{TcpStream, UdpSocket};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerEntry {
    pub name: String,
    pub host: String,
    pub port: u16,
    pub motd: String,
    #[serde(default)]
    pub featured: bool,
    #[serde(default)]
    pub category: String,
    #[serde(skip)]
    pub players_online: u32,
    #[serde(skip)]
    pub players_max: u32,
    #[serde(skip)]
    pub ping_ms: Option<u32>,
    #[serde(skip)]
    pub online: bool,
    #[serde(skip)]
    pub is_lan: bool,
}

impl ServerEntry {
    pub fn signal_bars(&self) -> u8 {
        if !self.online {
            return 0;
        }
        match self.ping_ms {
            None => 0,
            Some(ms) if ms < 80 => 5,
            Some(ms) if ms < 150 => 4,
            Some(ms) if ms < 250 => 3,
            Some(ms) if ms < 400 => 2,
            _ => 1,
        }
    }

    pub fn icon_seed(&self) -> String {
        format!("{}:{}:{}", self.name, self.host, self.port)
    }
}

pub fn featured_servers() -> Vec<ServerEntry> {
    vec![
        ServerEntry {
            name: "Mineclonia Survival Realm".into(),
            host: "mineclonia.org".into(),
            port: 30000,
            motd: "§2§lMineclonia Official §r§7- §eSurvival & Exploration\n§bVillages • Nether • End Dragon • 100% Free".into(),
            featured: true,
            category: "Survival".into(),
            players_online: 0,
            players_max: 60,
            ping_ms: None,
            online: false,
            is_lan: false,
        },
        ServerEntry {
            name: "Clamity Anarchy & PvP".into(),
            host: "clamity.net".into(),
            port: 30000,
            motd: "§c§lClamity PvP §r§8| §6Mineclonia Combat Arena\n§7Clans • Crystal PvP • Custom Kits & Events".into(),
            featured: true,
            category: "PvP".into(),
            players_online: 0,
            players_max: 100,
            ping_ms: None,
            online: false,
            is_lan: false,
        },
        ServerEntry {
            name: "Skyblock Islands Hub".into(),
            host: "skyblock.luanti.org".into(),
            port: 30000,
            motd: "§b§lSkyblock Horizons §r§7- §aFloating Island Survival\n§eCo-op Islands • Economy • Daily Challenges".into(),
            featured: true,
            category: "Skyblock".into(),
            players_online: 0,
            players_max: 80,
            ping_ms: None,
            online: false,
            is_lan: false,
        },
    ]
}

pub fn load_servers(path: &Path) -> Vec<ServerEntry> {
    let mut all = Vec::new();
    if let Ok(raw) = std::fs::read_to_string(path) {
        if let Ok(custom) = serde_json::from_str::<Vec<ServerEntry>>(&raw) {
            all.extend(custom);
        }
    }
    all.extend(featured_servers());
    all
}

pub fn save_custom_servers(path: &Path, entries: &[ServerEntry]) {
    let custom: Vec<&ServerEntry> = entries
        .iter()
        .filter(|s| !s.featured && !s.is_lan)
        .collect();
    if let Some(p) = path.parent() {
        let _ = std::fs::create_dir_all(p);
    }
    if let Ok(json) = serde_json::to_string_pretty(&custom) {
        let _ = std::fs::write(path, json);
    }
}

pub fn get_local_ip() -> String {
    let Ok(sock) = std::net::UdpSocket::bind("0.0.0.0:0") else {
        return "127.0.0.1".into();
    };
    if sock.connect("8.8.8.8:80").is_ok() {
        if let Ok(addr) = sock.local_addr() {
            return addr.ip().to_string();
        }
    }
    "127.0.0.1".into()
}

/// Async tokio UDP (Luanti peer-init) + TCP fallback server status probe.
pub async fn ping_server(host: &str, port: u16, timeout_dur: Duration) -> (bool, Option<u32>) {
    let addr = format!("{host}:{port}");
    let t0 = Instant::now();
    if let Ok(sock) = UdpSocket::bind("0.0.0.0:0").await {
        let probe = [0x4f, 0x45, 0x74, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00];
        if sock.send_to(&probe, &addr).await.is_ok() {
            let mut buf = [0u8; 512];
            if let Ok(Ok((n, _))) = tokio::time::timeout(timeout_dur, sock.recv_from(&mut buf)).await {
                if n > 0 {
                    let ms = t0.elapsed().as_millis().max(1) as u32;
                    return (true, Some(ms));
                }
            }
        }
    }

    let t1 = Instant::now();
    if let Ok(Ok(_stream)) = tokio::time::timeout(timeout_dur, TcpStream::connect(&addr)).await {
        let ms = t1.elapsed().as_millis().max(1) as u32;
        return (true, Some(ms));
    }

    (false, None)
}

pub async fn scan_lan_servers() -> Vec<ServerEntry> {
    crate::network::lan::discover_lan_servers(Duration::from_millis(300)).await
}

/// Concurrent server ping + LAN discovery helper that never blocks the UI thread.
#[derive(Clone, Default)]
pub struct AsyncServerRefresh {
    pub in_progress: Arc<Mutex<bool>>,
    pub pending_result: Arc<Mutex<Option<Vec<ServerEntry>>>>,
}

impl AsyncServerRefresh {
    pub fn trigger(&self, rt: &tokio::runtime::Runtime, current: Vec<ServerEntry>) {
        {
            let mut busy = self.in_progress.lock();
            if *busy {
                return;
            }
            *busy = true;
        }
        let in_progress = self.in_progress.clone();
        let pending_result = self.pending_result.clone();
        rt.spawn(async move {
            let mut base: Vec<ServerEntry> = current.into_iter().filter(|s| !s.is_lan).collect();
            let mut set = tokio::task::JoinSet::new();
            for (idx, s) in base.iter().enumerate() {
                let h = s.host.clone();
                let p = s.port;
                set.spawn(async move {
                    let (on, ms) = ping_server(&h, p, Duration::from_millis(600)).await;
                    (idx, on, ms)
                });
            }
            let lan_fut = crate::network::lan::discover_lan_servers(Duration::from_millis(350));
            let lan = lan_fut.await;
            while let Some(Ok((idx, on, ms))) = set.join_next().await {
                if let Some(entry) = base.get_mut(idx) {
                    entry.online = on;
                    entry.ping_ms = ms;
                }
            }
            base.extend(lan);
            *pending_result.lock() = Some(base);
            *in_progress.lock() = false;
        });
    }

    pub fn poll(&self) -> Option<Vec<ServerEntry>> {
        self.pending_result.lock().take()
    }
}

/// Generate a crisp deterministic 16x16 pixel-art server icon RGBA buffer (upscalable 4x nearest to 64x64).
pub fn generate_server_icon_rgba16(seed_text: &str) -> Vec<u8> {
    let mut hash = 2166136261u32;
    for b in seed_text.bytes() {
        hash ^= b as u32;
        hash = hash.wrapping_mul(16777619);
    }
    const THEMES: [([u8; 3], [u8; 3]); 5] = [
        ([86, 176, 62], [134, 96, 67]),   // Grass & dirt
        ([62, 168, 212], [96, 106, 118]), // Diamond & stone
        ([186, 72, 62], [92, 42, 42]),    // Crimson nether
        ([224, 176, 54], [118, 114, 106]), // Gold & stone
        ([148, 86, 212], [56, 44, 72]),   // End obsidian
    ];
    let (top_c, body_c) = THEMES[(hash as usize) % THEMES.len()];
    let mut out = Vec::with_capacity(16 * 16 * 4);
    for y in 0..16u32 {
        for x in 0..16u32 {
            hash = hash.wrapping_mul(1664525).wrapping_add(1013904223);
            let jitter = ((hash >> 16) % 25) as i16 - 12;
            let is_top = y < 5 || (y == 5 && ((hash >> 24) & 1) == 0);
            let col = if is_top { top_c } else { body_c };
            let mut r = (col[0] as i16 + jitter).clamp(0, 255) as u8;
            let mut g = (col[1] as i16 + jitter).clamp(0, 255) as u8;
            let mut b = (col[2] as i16 + jitter).clamp(0, 255) as u8;
            if x == 0 || y == 0 || x == 15 || y == 15 {
                r /= 2;
                g /= 2;
                b /= 2;
            }
            out.extend_from_slice(&[r, g, b, 255]);
        }
    }
    out
}
