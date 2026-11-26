use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine as _;
use glam::IVec3;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::{broadcast, watch};

use crate::network::protocol::{NetPacket, RemotePlayerState, VoxelDelta};
use crate::registry::{EngineRegistries, CONTENT_AIR};
use crate::world::{world_to_chunk, WorldStore};

const WS_GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

/// Latest touch/gamepad-style input sent from a phone cloud-streaming browser client.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StreamInputState {
    #[serde(default)]
    pub move_x: f32,
    #[serde(default)]
    pub move_y: f32,
    #[serde(default)]
    pub look_dx: f32,
    #[serde(default)]
    pub look_dy: f32,
    #[serde(default)]
    pub jump: bool,
    #[serde(default)]
    pub sneak: bool,
    #[serde(default)]
    pub sprint: bool,
    #[serde(default)]
    pub attack_held: bool,
    #[serde(default)]
    pub attack_just: bool,
    #[serde(default)]
    pub place_just: bool,
    #[serde(default)]
    pub selected_slot: Option<usize>,
}

/// Compact HUD telemetry streamed to the phone overlay (~10 Hz).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StreamHudTelemetry {
    pub hp: i32,
    pub hunger: i32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub selected_slot: usize,
    pub held_item: String,
    pub hotbar: Vec<String>,
}

pub struct StreamClientEntry {
    pub id: u64,
    pub player_name: RwLock<String>,
    pub input: Mutex<StreamInputState>,
    pub frame_tx: watch::Sender<Arc<Vec<u8>>>,
    pub hud_tx: watch::Sender<String>,
}

/// Coordinates PC-to-Phone WebSocket cloud streaming clients on port 8090.
#[derive(Default)]
pub struct StreamHub {
    next_id: AtomicU64,
    clients: RwLock<HashMap<u64, Arc<StreamClientEntry>>>,
}

impl StreamHub {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(
        &self,
        default_name: String,
    ) -> (
        Arc<StreamClientEntry>,
        watch::Receiver<Arc<Vec<u8>>>,
        watch::Receiver<String>,
    ) {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        let (frame_tx, frame_rx) = watch::channel(Arc::new(Vec::new()));
        let (hud_tx, hud_rx) = watch::channel(String::new());
        let entry = Arc::new(StreamClientEntry {
            id,
            player_name: RwLock::new(default_name),
            input: Mutex::new(StreamInputState::default()),
            frame_tx,
            hud_tx,
        });
        self.clients.write().insert(id, entry.clone());
        (entry, frame_rx, hud_rx)
    }

    pub fn unregister(&self, id: u64) {
        self.clients.write().remove(&id);
    }

    pub fn has_clients(&self) -> bool {
        !self.clients.read().is_empty()
    }

    pub fn list_clients(&self) -> Vec<(u64, String)> {
        self.clients
            .read()
            .values()
            .map(|c| (c.id, c.player_name.read().clone()))
            .collect()
    }

    /// Reads accumulated inputs for a connected phone client and clears one-shot impulses.
    pub fn take_input(&self, id: u64) -> Option<StreamInputState> {
        let clients = self.clients.read();
        let entry = clients.get(&id)?;
        let mut inp = entry.input.lock();
        let snap = inp.clone();
        inp.look_dx = 0.0;
        inp.look_dy = 0.0;
        inp.attack_just = false;
        inp.place_just = false;
        inp.selected_slot = None;
        Some(snap)
    }

    pub fn publish_frame(&self, id: u64, jpeg_bytes: Vec<u8>) {
        if let Some(entry) = self.clients.read().get(&id) {
            let _ = entry.frame_tx.send(Arc::new(jpeg_bytes));
        }
    }

    pub fn publish_hud(&self, id: u64, hud: &StreamHudTelemetry) {
        if let Some(entry) = self.clients.read().get(&id) {
            if let Ok(json) = serde_json::to_string(&serde_json::json!({ "StreamHud": hud })) {
                let _ = entry.hud_tx.send(json);
            }
        }
    }
}

/// Handle an incoming HTTP/1.1 or RFC 6455 WebSocket connection on the Cubex LAN port (8090).
#[allow(clippy::too_many_arguments)]
pub async fn handle_http_or_ws(
    mut stream: TcpStream,
    world: Arc<WorldStore>,
    registries: Arc<RwLock<EngineRegistries>>,
    tx: broadcast::Sender<NetPacket>,
    remote_players: Arc<RwLock<HashMap<String, RemotePlayerState>>>,
    chunk_deltas: Arc<RwLock<HashMap<IVec3, Vec<VoxelDelta>>>>,
    stream_hub: Arc<StreamHub>,
    spawn_pos: Arc<RwLock<[f32; 3]>>,
    stop: Arc<AtomicBool>,
) {
    let mut raw = Vec::with_capacity(2048);
    let mut tmp = [0u8; 1024];
    let header_end = loop {
        if raw.len() > 16 * 1024 {
            return;
        }
        let Ok(Ok(n)) =
            tokio::time::timeout(std::time::Duration::from_secs(3), stream.read(&mut tmp)).await
        else {
            return;
        };
        if n == 0 {
            return;
        }
        raw.extend_from_slice(&tmp[..n]);
        if let Some(pos) = find_header_end(&raw) {
            break pos;
        }
    };

    let req_text = String::from_utf8_lossy(&raw[..header_end]);
    let mut lines = req_text.lines();
    let first_line = lines.next().unwrap_or("");
    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or("GET");
    let path = parts.next().unwrap_or("/");

    let mut upgrade_ws = false;
    let mut ws_key: Option<String> = None;
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            let kl = k.trim().to_ascii_lowercase();
            let vl = v.trim();
            if kl == "upgrade" && vl.eq_ignore_ascii_case("websocket") {
                upgrade_ws = true;
            } else if kl == "sec-websocket-key" {
                ws_key = Some(vl.to_string());
            }
        }
    }

    if upgrade_ws {
        if let Some(key) = ws_key {
            let accept = compute_ws_accept(&key);
            let resp = format!(
                "HTTP/1.1 101 Switching Protocols\r\n\
                 Upgrade: websocket\r\n\
                 Connection: Upgrade\r\n\
                 Sec-WebSocket-Accept: {accept}\r\n\r\n"
            );
            if stream.write_all(resp.as_bytes()).await.is_err() {
                return;
            }
            let _ = stream.flush().await;
            handle_ws_session(
                stream,
                world,
                registries,
                tx,
                remote_players,
                chunk_deltas,
                stream_hub,
                spawn_pos,
                stop,
            )
            .await;
            return;
        }
    }

    if path.starts_with("/api/status") {
        let names: Vec<String> = remote_players.read().keys().cloned().collect();
        let wt = world.world_type.read().as_str().to_string();
        let sp = *spawn_pos.read();
        let body = serde_json::json!({
            "server": "Cubex-RS",
            "seed": world.seed,
            "world_type": wt,
            "spawn_pos": sp,
            "players_online": names.len(),
            "players": names,
        })
        .to_string();
        write_http_response(&mut stream, method, "application/json; charset=utf-8", body.as_bytes()).await;
        return;
    }

    if path == "/" || path.starts_with("/?") || path == "/index.html" {
        write_http_response(&mut stream, method, "text/html; charset=utf-8", MOBILE_CLIENT_HTML.as_bytes()).await;
        return;
    }

    let body = b"Cubex-RS LAN Server";
    let hdr = format!(
        "HTTP/1.1 404 Not Found\r\n\
         Content-Type: text/plain; charset=utf-8\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(hdr.as_bytes()).await;
    if !method.eq_ignore_ascii_case("HEAD") {
        let _ = stream.write_all(body).await;
    }
    let _ = stream.flush().await;
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
        return Some(i + 4);
    }
    buf.windows(2).position(|w| w == b"\n\n").map(|i| i + 2)
}

pub fn compute_ws_accept(client_key: &str) -> String {
    let mut hasher = Sha1::new();
    hasher.update(client_key.trim().as_bytes());
    hasher.update(WS_GUID.as_bytes());
    B64.encode(hasher.finalize())
}

async fn write_http_response(
    stream: &mut TcpStream,
    method: &str,
    content_type: &str,
    body: &[u8],
) {
    let hdr = format!(
        "HTTP/1.1 200 OK\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Cache-Control: no-store\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Connection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(hdr.as_bytes()).await;
    if !method.eq_ignore_ascii_case("HEAD") {
        let _ = stream.write_all(body).await;
    }
    let _ = stream.flush().await;
}

async fn write_ws_frame<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    opcode: u8,
    payload: &[u8],
) -> std::io::Result<()> {
    let len = payload.len();
    let mut hdr = Vec::with_capacity(10);
    hdr.push(0x80 | (opcode & 0x0f));
    if len < 126 {
        hdr.push(len as u8);
    } else if len <= 65535 {
        hdr.push(126);
        hdr.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        hdr.push(127);
        hdr.extend_from_slice(&(len as u64).to_be_bytes());
    }
    writer.write_all(&hdr).await?;
    writer.write_all(payload).await?;
    writer.flush().await
}

async fn read_ws_frame<R: AsyncReadExt + Unpin>(reader: &mut R) -> std::io::Result<Option<(u8, Vec<u8>)>> {
    let mut head = [0u8; 2];
    if reader.read_exact(&mut head).await.is_err() {
        return Ok(None);
    }
    let opcode = head[0] & 0x0f;
    let masked = (head[1] & 0x80) != 0;
    let mut len = (head[1] & 0x7f) as u64;
    if len == 126 {
        let mut ext = [0u8; 2];
        reader.read_exact(&mut ext).await?;
        len = u16::from_be_bytes(ext) as u64;
    } else if len == 127 {
        let mut ext = [0u8; 8];
        reader.read_exact(&mut ext).await?;
        len = u64::from_be_bytes(ext);
    }
    if len > 4 * 1024 * 1024 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "WebSocket frame too large",
        ));
    }
    let mask_key = if masked {
        let mut mk = [0u8; 4];
        reader.read_exact(&mut mk).await?;
        Some(mk)
    } else {
        None
    };
    let mut data = vec![0u8; len as usize];
    reader.read_exact(&mut data).await?;
    if let Some(mk) = mask_key {
        for (i, b) in data.iter_mut().enumerate() {
            *b ^= mk[i % 4];
        }
    }
    Ok(Some((opcode, data)))
}

#[allow(clippy::too_many_arguments)]
async fn handle_ws_session(
    stream: TcpStream,
    world: Arc<WorldStore>,
    registries: Arc<RwLock<EngineRegistries>>,
    tx: broadcast::Sender<NetPacket>,
    remote_players: Arc<RwLock<HashMap<String, RemotePlayerState>>>,
    chunk_deltas: Arc<RwLock<HashMap<IVec3, Vec<VoxelDelta>>>>,
    stream_hub: Arc<StreamHub>,
    spawn_pos: Arc<RwLock<[f32; 3]>>,
    stop: Arc<AtomicBool>,
) {
    let (mut reader, mut writer) = stream.into_split();
    let mut rx = tx.subscribe();
    let (direct_tx, mut direct_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let (stream_entry, mut frame_rx, mut hud_rx) =
        stream_hub.register("PhonePlayer".into());
    // Mark initial empty watch values as seen so we only push real rendered frames/HUD
    frame_rx.borrow_and_update();
    hud_rx.borrow_and_update();

    let welcome = NetPacket::Welcome {
        seed: world.seed,
        motd: "§aCubex-RS LAN World".into(),
        world_type: world.world_type.read().as_str().to_string(),
        spawn_pos: Some(*spawn_pos.read()),
    };
    if let Ok(js) = serde_json::to_string(&welcome) {
        let _ = direct_tx.send(js);
    }
    for ps in remote_players.read().values().cloned().collect::<Vec<_>>() {
        let pkt = NetPacket::PlayerMove {
            player_name: ps.player_name,
            x: ps.x,
            y: ps.y,
            z: ps.z,
            yaw: ps.yaw,
            pitch: ps.pitch,
            swinging: ps.swinging,
            sneaking: ps.sneaking,
            sprinting: ps.sprinting,
            held_item: ps.held_item,
        };
        if let Ok(js) = serde_json::to_string(&pkt) {
            let _ = direct_tx.send(js);
        }
    }

    let writer_task = tokio::spawn(async move {
        loop {
            tokio::select! {
                res = rx.recv() => {
                    match res {
                        Ok(pkt) => {
                            if let Ok(js) = serde_json::to_string(&pkt) {
                                if write_ws_frame(&mut writer, 0x1, js.as_bytes()).await.is_err() {
                                    break;
                                }
                            }
                        }
                        Err(broadcast::error::RecvError::Lagged(_)) => continue,
                        Err(broadcast::error::RecvError::Closed) => break,
                    }
                }
                Some(raw_msg) = direct_rx.recv() => {
                    if raw_msg == "__PONG__" {
                        if write_ws_frame(&mut writer, 0xA, &[]).await.is_err() {
                            break;
                        }
                    } else if write_ws_frame(&mut writer, 0x1, raw_msg.as_bytes()).await.is_err() {
                        break;
                    }
                }
                Ok(()) = frame_rx.changed() => {
                    let frame = frame_rx.borrow_and_update().clone();
                    if !frame.is_empty() && write_ws_frame(&mut writer, 0x2, &frame).await.is_err() {
                        break;
                    }
                }
                Ok(()) = hud_rx.changed() => {
                    let hud_json = hud_rx.borrow_and_update().clone();
                    if !hud_json.is_empty() && write_ws_frame(&mut writer, 0x1, hud_json.as_bytes()).await.is_err() {
                        break;
                    }
                }
            }
        }
    });

    let mut client_name: Option<String> = None;
    while !stop.load(Ordering::Relaxed) {
        let Ok(Some((opcode, data))) = read_ws_frame(&mut reader).await else {
            break;
        };
        match opcode {
            0x8 => break,
            0x9 => {
                let _ = direct_tx.send("__PONG__".into());
            }
            0x1 | 0x0 => {
                let Ok(txt) = std::str::from_utf8(&data) else {
                    continue;
                };
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(txt) {
                    if let Some(wc) = val.get("WebChunk") {
                        let cx = wc.get("cx").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
                        let cz = wc.get("cz").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
                        let col_json = build_web_chunk_column_json(cx, cz, &world, &registries);
                        let _ = direct_tx.send(col_json);
                        continue;
                    }
                    if let Some(si_val) = val.get("StreamInput") {
                        if let Ok(si) = serde_json::from_value::<StreamInputState>(si_val.clone()) {
                            let mut dst = stream_entry.input.lock();
                            dst.move_x = si.move_x.clamp(-1.0, 1.0);
                            dst.move_y = si.move_y.clamp(-1.0, 1.0);
                            dst.look_dx += si.look_dx;
                            dst.look_dy += si.look_dy;
                            dst.jump = si.jump;
                            dst.sneak = si.sneak;
                            dst.sprint = si.sprint;
                            dst.attack_held = si.attack_held;
                            dst.attack_just |= si.attack_just;
                            dst.place_just |= si.place_just;
                            if si.selected_slot.is_some() {
                                dst.selected_slot = si.selected_slot;
                            }
                            continue;
                        }
                    }
                }
                if let Ok(pkt) = serde_json::from_str::<NetPacket>(txt) {
                    match &pkt {
                        NetPacket::Hello { player_name } => {
                            let clean_name = if player_name.trim().is_empty() {
                                format!("Phone_{}", stream_entry.id)
                            } else {
                                player_name.trim().to_string()
                            };
                            client_name = Some(clean_name.clone());
                            *stream_entry.player_name.write() = clean_name.clone();
                            let sp = *spawn_pos.read();
                            remote_players.write().entry(player_name.clone()).or_insert_with(|| {
                                RemotePlayerState {
                                    player_name: player_name.clone(),
                                    x: sp[0],
                                    y: sp[1],
                                    z: sp[2],
                                    yaw: 0.0,
                                    pitch: 0.0,
                                    swinging: false,
                                    sneaking: false,
                                    sprinting: false,
                                    held_item: String::new(),
                                }
                            });
                        }
                        NetPacket::PlayerMove {
                            player_name,
                            x,
                            y,
                            z,
                            yaw,
                            pitch,
                            swinging,
                            sneaking,
                            sprinting,
                            held_item,
                        } => {
                            client_name.get_or_insert_with(|| player_name.clone());
                            remote_players.write().insert(
                                player_name.clone(),
                                RemotePlayerState {
                                    player_name: player_name.clone(),
                                    x: *x,
                                    y: *y,
                                    z: *z,
                                    yaw: *yaw,
                                    pitch: *pitch,
                                    swinging: *swinging,
                                    sneaking: *sneaking,
                                    sprinting: *sprinting,
                                    held_item: held_item.clone(),
                                },
                            );
                        }
                        NetPacket::BlockEdit {
                            x,
                            y,
                            z,
                            node_name,
                            param1,
                            param2,
                        } => {
                            let wpos = IVec3::new(*x, *y, *z);
                            let id = if node_name == "air" {
                                CONTENT_AIR
                            } else {
                                registries.read().get_content_id(node_name)
                            };
                            world.set_node(wpos, id, *param1, *param2);
                            world.mark_priority_dirty_with_neighbors(wpos);
                            let (cpos, lx, ly, lz) = world_to_chunk(wpos);
                            chunk_deltas.write().entry(cpos).or_default().push(VoxelDelta {
                                lx: lx as u8,
                                ly: ly as u8,
                                lz: lz as u8,
                                node_name: node_name.clone(),
                                param1: *param1,
                                param2: *param2,
                            });
                        }
                        _ => {}
                    }
                    let _ = tx.send(pkt);
                }
            }
            _ => {}
        }
    }

    stream_hub.unregister(stream_entry.id);
    if let Some(name) = client_name {
        remote_players.write().remove(&name);
        let _ = tx.send(NetPacket::PlayerLeave { player_name: name });
    }
    writer_task.abort();
}

fn color_for_node_name(name: &str) -> &'static str {
    if name.contains("water") {
        "#3b74d6"
    } else if name.contains("lava") {
        "#e05d20"
    } else if name.contains("grass") {
        "#589e3c"
    } else if name.contains("leaves") || name.contains("needles") {
        "#3f7c2b"
    } else if name.contains("tree") || name.contains("wood") || name.contains("plank") {
        "#8f673f"
    } else if name.contains("dirt") {
        "#7c5838"
    } else if name.contains("sand") {
        "#d9cf9a"
    } else if name.contains("snow") || name.contains("ice") {
        "#e4eef5"
    } else if name.contains("cobble") || name.contains("stone") {
        "#7d8085"
    } else {
        "#9aa0a6"
    }
}

fn build_web_chunk_column_json(
    cx: i32,
    cz: i32,
    world: &Arc<WorldStore>,
    registries: &Arc<RwLock<EngineRegistries>>,
) -> String {
    let reg = registries.read();
    let mut blocks: Vec<(i32, i32, i32, &'static str, String)> = Vec::new();
    for cy in 0..=8 {
        let cpos = IVec3::new(cx, cy, cz);
        let Some(ch_arc) = world.get_chunk(cpos) else {
            continue;
        };
        let ch = ch_arc.read();
        for lz in 0..16i32 {
            for lx in 0..16i32 {
                for ly in 0..16i32 {
                    let id = ch.get(lx as usize, ly as usize, lz as usize).0;
                    if id == CONTENT_AIR {
                        continue;
                    }
                    let wx = cx * 16 + lx;
                    let wy = cy * 16 + ly;
                    let wz = cz * 16 + lz;
                    let above_id = if ly < 15 {
                        ch.get(lx as usize, (ly + 1) as usize, lz as usize).0
                    } else {
                        world.get_node(IVec3::new(wx, wy + 1, wz)).0
                    };
                    if above_id != CONTENT_AIR {
                        continue;
                    }
                    let name = reg.get_name_from_content_id(id).to_string();
                    blocks.push((wx, wy, wz, color_for_node_name(&name), name));
                }
            }
        }
    }
    serde_json::json!({
        "WebChunkData": {
            "cx": cx,
            "cz": cz,
            "blocks": blocks,
        }
    })
    .to_string()
}

const MOBILE_CLIENT_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no,viewport-fit=cover">
<title>Cubex-RS LAN Mobile Client</title>
<style>
  *{box-sizing:border-box;user-select:none;-webkit-user-select:none;touch-action:none}
  html,body{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#090b0f;color:#fff;font-family:monospace}
  canvas{display:block;width:100%;height:100%;object-fit:cover;background:#090b0f}
  #crosshair{position:fixed;left:50%;top:50%;width:16px;height:16px;transform:translate(-50%,-50%);pointer-events:none;z-index:5}
  #crosshair::before,#crosshair::after{content:'';position:absolute;background:rgba(255,255,255,.9);box-shadow:0 0 2px #000}
  #crosshair::before{left:7px;top:0;width:2px;height:16px}
  #crosshair::after{left:0;top:7px;width:16px;height:2px}
  #hud{position:fixed;top:8px;left:8px;right:8px;display:flex;justify-content:space-between;align-items:center;pointer-events:none;font-size:12px;z-index:6}
  .pill{background:rgba(12,14,18,.68);border:1px solid rgba(255,255,255,.22);padding:5px 9px;border-radius:4px;pointer-events:auto;text-shadow:1px 1px 0 #000}
  #statsBar{position:fixed;bottom:56px;left:50%;transform:translateX(-50%);display:flex;gap:14px;font-size:13px;pointer-events:none;text-shadow:1px 1px 0 #000;z-index:6}
  #hotbar{position:fixed;bottom:10px;left:50%;transform:translateX(-50%);display:flex;gap:3px;background:rgba(16,18,24,.78);padding:4px;border:2px solid #484c56;border-radius:4px;z-index:8}
  .hslot{width:34px;height:34px;border:2px solid #666;background:rgba(40,44,52,.7);display:flex;align-items:center;justify-content:center;font-size:10px;color:#eee;overflow:hidden;text-align:center;line-height:1}
  .hslot.sel{border-color:#fff;background:rgba(100,140,85,.78)}
  #joinBox{position:fixed;inset:0;display:flex;align-items:center;justify-content:center;background:rgba(8,10,14,.88);z-index:20}
  #joinCard{background:#23262d;border:2px solid #555;padding:18px;width:min(90vw,340px);text-align:center}
  input{width:100%;padding:10px;margin:10px 0;background:#111;color:#fff;border:1px solid #666;font-family:monospace;font-size:15px;touch-action:auto;user-select:text;-webkit-user-select:text}
  button{background:#5a8f49;color:#fff;border:2px solid #2b2b2b;padding:10px 16px;font-family:monospace;font-size:15px;font-weight:bold;cursor:pointer}
  #joyBase{position:fixed;left:18px;bottom:60px;width:118px;height:118px;border-radius:50%;background:rgba(20,20,24,.38);border:2px solid rgba(255,255,255,.42);z-index:7}
  #joyKnob{position:absolute;left:37px;top:37px;width:40px;height:40px;border-radius:50%;background:rgba(235,235,235,.78)}
  #actions{position:fixed;right:14px;bottom:60px;display:grid;grid-template-columns:56px 56px;gap:8px;z-index:7}
  .actBtn{width:56px;height:52px;border-radius:8px;background:rgba(28,32,40,.66);border:2px solid rgba(255,255,255,.42);color:#fff;font-family:monospace;font-weight:bold;font-size:12px;display:flex;align-items:center;justify-content:center}
  .actBtn.on,.actBtn:active{background:rgba(90,165,80,.85)}
  #chatLog{position:fixed;left:10px;top:40px;max-width:65vw;font-size:12px;pointer-events:none;text-shadow:1px 1px 0 #000;z-index:6}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="crosshair"></div>
<div id="hud">
  <div class="pill" id="statusPill">Connecting...</div>
  <div class="pill" id="coordsPill">0, 0, 0</div>
</div>
<div id="chatLog"></div>
<div id="statsBar">
  <span id="hpText" style="color:#ff5e5e">HP 20/20</span>
  <span id="foodText" style="color:#e0a44b">Food 20/20</span>
</div>
<div id="hotbar"></div>
<div id="joyBase"><div id="joyKnob"></div></div>
<div id="actions">
  <div class="actBtn" id="btnAtk">ATK</div>
  <div class="actBtn" id="btnJmp">JMP</div>
  <div class="actBtn" id="btnUse">USE</div>
  <div class="actBtn" id="btnSnk">SNK</div>
  <div class="actBtn" id="btnChat" style="grid-column:span 2;height:36px">CHAT</div>
</div>
<div id="joinBox">
  <div id="joinCard">
    <h3 style="margin:0 0 6px 0;color:#7fe066">CUBEX-RS CLOUD STREAM</h3>
    <div style="font-size:12px;color:#bbb">PC GPU Rendered &bull; Port 8090</div>
    <input id="nameIn" value="PhonePlayer" maxlength="16" placeholder="Player Name">
    <button id="joinBtn" style="width:100%">JOIN STREAM</button>
  </div>
</div>
<script>
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const statusPill = document.getElementById('statusPill');
const coordsPill = document.getElementById('coordsPill');
const hpText = document.getElementById('hpText');
const foodText = document.getElementById('foodText');
const hotbarEl = document.getElementById('hotbar');
const chatLog = document.getElementById('chatLog');
const joinBox = document.getElementById('joinBox');
const joyBase = document.getElementById('joyBase');
const joyKnob = document.getElementById('joyKnob');

let ws = null, myName = 'PhonePlayer';
let joyVec = {x: 0, y: 0}, joyFinger = null, lookFinger = null, lookLast = null;
let lookAccX = 0, lookAccY = 0;
let jmpHeld = false, snkHeld = false, atkHeld = false, atkJust = false, useJust = false;
let pendingSlot = null, selectedSlot = 0;
let chatLines = [], frameCount = 0, fpsLast = performance.now(), curFps = 0, decoding = false;

for (let i = 0; i < 9; i++) {
  const d = document.createElement('div');
  d.className = 'hslot' + (i === 0 ? ' sel' : '');
  d.textContent = String(i + 1);
  d.onclick = () => { selectedSlot = i; pendingSlot = i; renderHotbar([]); };
  hotbarEl.appendChild(d);
}
function renderHotbar(names) {
  const slots = hotbarEl.children;
  for (let i = 0; i < 9; i++) {
    slots[i].className = 'hslot' + (i === selectedSlot ? ' sel' : '');
    if (names && names[i] !== undefined) {
      const short = (names[i] || '').split(':').pop().replace(/_/g, ' ').slice(0, 6);
      slots[i].textContent = short || String(i + 1);
    }
  }
}

function resize(){ canvas.width = innerWidth; canvas.height = innerHeight; }
addEventListener('resize', resize); resize();

function addChat(t){
  chatLines.push(t.replace(/§./g, ''));
  if(chatLines.length > 5) chatLines.shift();
  chatLog.innerHTML = chatLines.map(s => '<div>'+s.replace(/</g,'&lt;')+'</div>').join('');
}

function connectWs(){
  const proto = location.protocol === 'https:' ? 'wss://' : 'ws://';
  ws = new WebSocket(proto + location.host + '/ws');
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => {
    statusPill.textContent = 'Streaming: ' + myName;
    ws.send(JSON.stringify({Hello: {player_name: myName}}));
  };
  ws.onclose = () => { statusPill.textContent = 'Disconnected'; };
  ws.onmessage = async (ev) => {
    if (typeof ev.data !== 'string') {
      if (decoding) return;
      decoding = true;
      try {
        const blob = new Blob([ev.data], {type: 'image/jpeg'});
        const bmp = await createImageBitmap(blob);
        ctx.drawImage(bmp, 0, 0, canvas.width, canvas.height);
        bmp.close();
        frameCount++;
        const now = performance.now();
        if (now - fpsLast >= 1000) {
          curFps = Math.round((frameCount * 1000) / (now - fpsLast));
          frameCount = 0; fpsLast = now;
          statusPill.textContent = myName + ' · ' + curFps + ' FPS';
        }
      } catch (_) {}
      decoding = false;
      return;
    }
    try {
      const m = JSON.parse(ev.data);
      if (m.Welcome) {
        addChat('Joined ' + (m.Welcome.motd || 'Cubex-RS Cloud'));
      } else if (m.Chat) {
        addChat('<' + m.Chat.sender + '> ' + m.Chat.message);
      } else if (m.StreamHud) {
        const h = m.StreamHud;
        coordsPill.textContent = h.x.toFixed(1)+', '+h.y.toFixed(1)+', '+h.z.toFixed(1);
        hpText.textContent = 'HP ' + h.hp + '/20';
        foodText.textContent = 'Food ' + h.hunger + '/20';
        selectedSlot = h.selected_slot || 0;
        renderHotbar(h.hotbar);
      }
    } catch (_) {}
  };
}

document.getElementById('joinBtn').onclick = () => {
  myName = (document.getElementById('nameIn').value || 'PhonePlayer').trim();
  joinBox.style.display = 'none';
  connectWs();
};

joyBase.addEventListener('touchstart', e => {
  const t = e.changedTouches[0]; joyFinger = t.identifier; updateJoy(t); e.preventDefault();
}, {passive:false});
joyBase.addEventListener('touchmove', e => {
  for (const t of e.changedTouches) if (t.identifier === joyFinger) updateJoy(t);
  e.preventDefault();
}, {passive:false});
const resetJoy = e => {
  for (const t of e.changedTouches) if (t.identifier === joyFinger) {
    joyFinger = null; joyVec = {x:0, y:0};
    joyKnob.style.left = '37px'; joyKnob.style.top = '37px';
  }
};
joyBase.addEventListener('touchend', resetJoy);
joyBase.addEventListener('touchcancel', resetJoy);

function updateJoy(t){
  const r = joyBase.getBoundingClientRect();
  let dx = t.clientX - (r.left + 59), dy = t.clientY - (r.top + 59);
  const d = Math.hypot(dx, dy), max = 42;
  if (d > max) { dx *= max/d; dy *= max/d; }
  joyKnob.style.left = (37 + dx) + 'px';
  joyKnob.style.top = (37 + dy) + 'px';
  joyVec.x = dx / max; joyVec.y = -dy / max;
}

canvas.addEventListener('touchstart', e => {
  const t = e.changedTouches[0];
  if (lookFinger === null) { lookFinger = t.identifier; lookLast = {x:t.clientX, y:t.clientY}; }
}, {passive:true});
canvas.addEventListener('touchmove', e => {
  for (const t of e.changedTouches) if (t.identifier === lookFinger && lookLast) {
    lookAccX += (t.clientX - lookLast.x) * 0.0055;
    lookAccY -= (t.clientY - lookLast.y) * 0.0055;
    lookLast = {x:t.clientX, y:t.clientY};
  }
}, {passive:true});
const endLook = e => { for (const t of e.changedTouches) if (t.identifier === lookFinger) { lookFinger = null; lookLast = null; } };
canvas.addEventListener('touchend', endLook); canvas.addEventListener('touchcancel', endLook);

document.getElementById('btnJmp').ontouchstart = e => { jmpHeld = true; e.preventDefault(); };
document.getElementById('btnJmp').ontouchend = () => { jmpHeld = false; };
document.getElementById('btnSnk').onclick = () => {
  snkHeld = !snkHeld;
  document.getElementById('btnSnk').classList.toggle('on', snkHeld);
};
document.getElementById('btnAtk').ontouchstart = e => { atkHeld = true; atkJust = true; e.preventDefault(); };
document.getElementById('btnAtk').ontouchend = () => { atkHeld = false; };
document.getElementById('btnAtk').onclick = () => { atkJust = true; };
document.getElementById('btnUse').ontouchstart = e => { useJust = true; e.preventDefault(); };
document.getElementById('btnUse').onclick = () => { useJust = true; };
document.getElementById('btnChat').onclick = () => {
  const msg = prompt('Send chat message:');
  if (msg && ws && ws.readyState === 1) ws.send(JSON.stringify({Chat:{sender:myName,message:msg}}));
};

setInterval(() => {
  if (!ws || ws.readyState !== 1) return;
  const pkt = {
    StreamInput: {
      move_x: joyVec.x,
      move_y: joyVec.y,
      look_dx: lookAccX,
      look_dy: lookAccY,
      jump: jmpHeld,
      sneak: snkHeld,
      sprint: Math.abs(joyVec.y) > 0.85,
      attack_held: atkHeld,
      attack_just: atkJust,
      place_just: useJust,
      selected_slot: pendingSlot
    }
  };
  lookAccX = 0; lookAccY = 0; atkJust = false; useJust = false; pendingSlot = null;
  ws.send(JSON.stringify(pkt));
}, 33);
</script>
</body>
</html>"##;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::protocol::start_server_with_handle;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    #[tokio::test]
    async fn http_browser_on_port_8090_gets_valid_http_and_ws_not_cannot_parse_response() {
        let tmp = std::env::temp_dir().join("cubex_rs_web_client_test");
        let _ = std::fs::remove_dir_all(&tmp);
        let world = Arc::new(WorldStore::new(tmp, 8090));
        let reg = Arc::new(RwLock::new(EngineRegistries::new()));
        let spawn = Arc::new(RwLock::new([0.5, 65.0, 0.5]));
        let handle = start_server_with_handle(0, world, reg, spawn)
            .await
            .expect("start_server_with_handle");
        let port = handle.port;
        let mut srv_rx = handle.subscribe();

        // 1. Phone browser GET / receives proper HTTP/1.1 200 OK HTML (never raw JSON)
        let mut s = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        s.write_all(b"GET / HTTP/1.1\r\nHost: localhost:8090\r\n\r\n")
            .await
            .unwrap();
        let mut resp = String::new();
        s.read_to_string(&mut resp).await.unwrap();
        assert!(resp.starts_with("HTTP/1.1 200 OK\r\n"), "got: {resp}");
        assert!(resp.contains("Content-Length:"));
        assert!(resp.contains("<title>Cubex-RS LAN Mobile Client</title>"));

        // 2. GET /api/status returns valid JSON with 200 OK
        let mut s2 = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        s2.write_all(b"GET /api/status HTTP/1.1\r\nHost: localhost:8090\r\n\r\n")
            .await
            .unwrap();
        let mut status_resp = String::new();
        s2.read_to_string(&mut status_resp).await.unwrap();
        assert!(status_resp.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(status_resp.contains("\"seed\":8090"));

        // 3. WebSocket upgrade + masked client PlayerMove syncs into server broadcast
        let mut ws = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        let key = "dGhlIHNhbXBsZSBub25jZQ==";
        ws.write_all(
            format!(
                "GET /ws HTTP/1.1\r\nHost: localhost:8090\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
            )
            .as_bytes(),
        )
        .await
        .unwrap();
        let mut br = BufReader::new(ws);
        let mut hdrs = String::new();
        loop {
            let mut line = String::new();
            br.read_line(&mut line).await.unwrap();
            hdrs.push_str(&line);
            if line == "\r\n" || line.is_empty() {
                break;
            }
        }
        assert!(hdrs.starts_with("HTTP/1.1 101 Switching Protocols\r\n"));
        assert!(hdrs.contains(&compute_ws_accept(key)));

        // Read Welcome frame from server
        let first = read_ws_frame(&mut br).await.unwrap().expect("Welcome frame");
        assert_eq!(first.0, 0x1);
        assert!(String::from_utf8_lossy(&first.1).contains("\"Welcome\""));

        // Send masked WebSocket PlayerMove frame from phone
        let move_msg = serde_json::to_string(&NetPacket::PlayerMove {
            player_name: "PhoneUser".into(),
            x: 3.5,
            y: 66.0,
            z: 4.5,
            yaw: 0.5,
            pitch: 0.0,
            swinging: false,
            sneaking: false,
            sprinting: false,
            held_item: String::new(),
        })
        .unwrap();
        let mask = [0x12u8, 0x34, 0x56, 0x78];
        let payload = move_msg.as_bytes();
        let mut frame = vec![0x81u8];
        if payload.len() < 126 {
            frame.push(0x80 | (payload.len() as u8));
        } else {
            frame.push(0x80 | 126);
            frame.extend_from_slice(&(payload.len() as u16).to_be_bytes());
        }
        frame.extend_from_slice(&mask);
        for (i, &b) in payload.iter().enumerate() {
            frame.push(b ^ mask[i % 4]);
        }
        br.get_mut().write_all(&frame).await.unwrap();

        let ev = tokio::time::timeout(std::time::Duration::from_secs(2), srv_rx.recv())
            .await
            .expect("broadcast timeout")
            .unwrap();
        match ev {
            NetPacket::PlayerMove { player_name, x, z, .. } => {
                assert_eq!(player_name, "PhoneUser");
                assert!((x - 3.5).abs() < 0.01 && (z - 4.5).abs() < 0.01);
            }
            other => panic!("Expected PlayerMove, got {other:?}"),
        }

        // 4. Cloud-streaming: phone sends StreamInput -> PC StreamHub receives it,
        //    and PC publishes a binary JPEG frame -> phone receives opcode 0x2 WS frame!
        let clients = handle.stream_hub.list_clients();
        assert_eq!(clients.len(), 1);
        let client_id = clients[0].0;

        let inp_msg = serde_json::json!({
            "StreamInput": {
                "move_x": 0.5,
                "move_y": 1.0,
                "look_dx": 0.25,
                "look_dy": -0.1,
                "jump": true,
                "attack_just": true
            }
        })
        .to_string();
        let inp_bytes = inp_msg.as_bytes();
        let mut inp_frame = vec![0x81u8];
        if inp_bytes.len() < 126 {
            inp_frame.push(0x80 | (inp_bytes.len() as u8));
        } else {
            inp_frame.push(0x80 | 126);
            inp_frame.extend_from_slice(&(inp_bytes.len() as u16).to_be_bytes());
        }
        inp_frame.extend_from_slice(&mask);
        for (i, &b) in inp_bytes.iter().enumerate() {
            inp_frame.push(b ^ mask[i % 4]);
        }
        br.get_mut().write_all(&inp_frame).await.unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let taken = handle.stream_hub.take_input(client_id).expect("take_input");
        assert!((taken.move_y - 1.0).abs() < 0.01);
        assert!((taken.look_dx - 0.25).abs() < 0.01);
        assert!(taken.jump && taken.attack_just);

        let fake_jpeg = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x11, 0x22, 0xFF, 0xD9];
        handle.stream_hub.publish_frame(client_id, fake_jpeg.clone());
        let bin_frame = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                let (op, data) = read_ws_frame(&mut br).await.unwrap().expect("ws frame");
                if op == 0x2 {
                    return data;
                }
            }
        })
        .await
        .expect("binary JPEG frame timeout");
        assert_eq!(bin_frame, fake_jpeg);

        handle.stop();
    }
}
