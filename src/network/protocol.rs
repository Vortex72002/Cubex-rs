use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine as _;
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use glam::IVec3;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, mpsc};

use crate::config::MinetestConfig;
use crate::lua_api::LuaRuntime;
use crate::network::lan::{LanBeaconInfo, LanBroadcaster};
use crate::paths::EnginePaths;
use crate::registry::EngineRegistries;
use crate::world::mapgen::{generate_chunk, MapgenPalette};
use crate::world::{world_to_chunk, Chunk, WorldStore, CHUNK_VOLUME};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoxelDelta {
    pub lx: u8,
    pub ly: u8,
    pub lz: u8,
    pub node_name: String,
    pub param1: u8,
    pub param2: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemotePlayerState {
    pub player_name: String,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub swinging: bool,
    pub sneaking: bool,
    pub sprinting: bool,
    pub held_item: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetPacket {
    Hello { player_name: String },
    Welcome {
        seed: u64,
        motd: String,
        #[serde(default)]
        world_type: String,
        #[serde(default)]
        spawn_pos: Option<[f32; 3]>,
    },
    PlayerPos { player_name: String, x: f32, y: f32, z: f32, yaw: f32 },
    BlockSet { x: i32, y: i32, z: i32, node_name: String },
    Chat { sender: String, message: String },

    // Extended 1:1 multiplayer sync packets
    PlayerMove {
        player_name: String,
        x: f32,
        y: f32,
        z: f32,
        yaw: f32,
        pitch: f32,
        swinging: bool,
        sneaking: bool,
        sprinting: bool,
        held_item: String,
    },
    PlayerLeave {
        player_name: String,
    },
    BlockEdit {
        x: i32,
        y: i32,
        z: i32,
        node_name: String,
        param1: u8,
        param2: u8,
    },
    ChunkRequest {
        cx: i32,
        cy: i32,
        cz: i32,
    },
    ChunkDelta {
        cx: i32,
        cy: i32,
        cz: i32,
        edits: Vec<VoxelDelta>,
    },
    ChunkData {
        cx: i32,
        cy: i32,
        cz: i32,
        compressed_b64: String,
    },
    TimeWeather {
        time_of_day: f32,
        weather: String,
    },
    PlayerPunch {
        attacker: String,
        target: String,
        damage: i32,
        kb_x: f32,
        kb_y: f32,
        kb_z: f32,
    },
}

#[derive(Serialize, Deserialize)]
struct WireChunkPayload {
    palette: Vec<String>,
    node_indices: Vec<u16>,
    param1: Vec<u8>,
    param2: Vec<u8>,
}

pub fn encode_chunk_data(chunk: &Chunk, reg: &EngineRegistries) -> Option<String> {
    let mut palette = Vec::<String>::new();
    let mut name_to_idx = HashMap::<String, u16>::new();
    let mut indices = Vec::with_capacity(CHUNK_VOLUME);
    for &id in &chunk.nodes {
        let name = reg.get_name_from_content_id(id).to_string();
        let idx = if let Some(&i) = name_to_idx.get(&name) {
            i
        } else {
            let i = palette.len() as u16;
            palette.push(name.clone());
            name_to_idx.insert(name, i);
            i
        };
        indices.push(idx);
    }
    let payload = WireChunkPayload {
        palette,
        node_indices: indices,
        param1: chunk.param1.to_vec(),
        param2: chunk.param2.to_vec(),
    };
    let json = serde_json::to_vec(&payload).ok()?;
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::fast());
    enc.write_all(&json).ok()?;
    let compressed = enc.finish().ok()?;
    Some(B64.encode(&compressed))
}

pub fn decode_chunk_data(cx: i32, cy: i32, cz: i32, b64: &str, reg: &EngineRegistries) -> Option<Chunk> {
    let raw = B64.decode(b64).ok()?;
    let mut dec = ZlibDecoder::new(&raw[..]);
    let mut json = Vec::new();
    dec.read_to_end(&mut json).ok()?;
    let payload: WireChunkPayload = serde_json::from_slice(&json).ok()?;
    if payload.node_indices.len() != CHUNK_VOLUME
        || payload.param1.len() != CHUNK_VOLUME
        || payload.param2.len() != CHUNK_VOLUME
    {
        return None;
    }
    let mut ch = Chunk::new(IVec3::new(cx, cy, cz));
    for i in 0..CHUNK_VOLUME {
        let p_idx = payload.node_indices[i] as usize;
        let id = payload
            .palette
            .get(p_idx)
            .map(|n| reg.get_content_id(n))
            .unwrap_or(crate::registry::CONTENT_AIR);
        ch.nodes[i] = id;
        ch.param1[i] = payload.param1[i];
        ch.param2[i] = payload.param2[i];
    }
    ch.dirty_mesh = true;
    Some(ch)
}

fn handle_server_chat_command(
    sender: &str,
    msg: &str,
    players: &RwLock<HashMap<String, RemotePlayerState>>,
    seed: u64,
    tx: &broadcast::Sender<NetPacket>,
) -> bool {
    let trimmed = msg.trim();
    if !trimmed.starts_with('/') {
        return false;
    }
    let mut parts = trimmed[1..].split_whitespace();
    let cmd = parts.next().unwrap_or("");
    let reply = match cmd {
        "help" => "§eCommands: /help, /list, /seed, /time <day|night|0..1>, /weather <clear|rain|thunder>, /say <msg>".into(),
        "list" => {
            let names: Vec<String> = players.read().keys().cloned().collect();
            format!("§aOnline ({}): {}", names.len(), names.join(", "))
        }
        "seed" => format!("§bWorld Seed: {seed}"),
        "time" => {
            let arg = parts.next().unwrap_or("day");
            let tod = match arg {
                "day" | "noon" => 0.5,
                "night" | "midnight" => 0.0,
                "sunrise" => 0.25,
                "sunset" => 0.75,
                other => other.parse::<f32>().unwrap_or(0.5).clamp(0.0, 1.0),
            };
            let _ = tx.send(NetPacket::TimeWeather {
                time_of_day: tod,
                weather: "unchanged".into(),
            });
            format!("§e[Server] Time set to {tod:.2} by {sender}")
        }
        "weather" => {
            let w = parts.next().unwrap_or("clear").to_string();
            let _ = tx.send(NetPacket::TimeWeather {
                time_of_day: -1.0,
                weather: w.clone(),
            });
            format!("§b[Server] Weather set to {w} by {sender}")
        }
        "say" => {
            let rest: Vec<&str> = parts.collect();
            format!("§d[{sender}] {}", rest.join(" "))
        }
        other => format!("§cUnknown command: /{other}. Try /help"),
    };
    let _ = tx.send(NetPacket::Chat {
        sender: "Server".into(),
        message: reply,
    });
    true
}

/// Handle controlling a live in-process LAN/WAN multiplayer server synchronized with the host's world.
pub struct LocalServerHandle {
    pub port: u16,
    pub tx: broadcast::Sender<NetPacket>,
    pub remote_players: Arc<RwLock<HashMap<String, RemotePlayerState>>>,
    pub chunk_deltas: Arc<RwLock<HashMap<IVec3, Vec<VoxelDelta>>>>,
    pub stream_hub: Arc<crate::network::web_client::StreamHub>,
    stop: Arc<AtomicBool>,
    task: tokio::task::JoinHandle<()>,
}

impl LocalServerHandle {
    pub fn subscribe(&self) -> broadcast::Receiver<NetPacket> {
        self.tx.subscribe()
    }

    pub fn record_host_block_edit(&self, wpos: IVec3, node_name: String, param1: u8, param2: u8) {
        let (cpos, lx, ly, lz) = world_to_chunk(wpos);
        self.chunk_deltas.write().entry(cpos).or_default().push(VoxelDelta {
            lx: lx as u8,
            ly: ly as u8,
            lz: lz as u8,
            node_name: node_name.clone(),
            param1,
            param2,
        });
        let _ = self.tx.send(NetPacket::BlockEdit {
            x: wpos.x,
            y: wpos.y,
            z: wpos.z,
            node_name,
            param1,
            param2,
        });
    }

    pub fn update_host_player(&self, state: RemotePlayerState) {
        let pkt = NetPacket::PlayerMove {
            player_name: state.player_name.clone(),
            x: state.x,
            y: state.y,
            z: state.z,
            yaw: state.yaw,
            pitch: state.pitch,
            swinging: state.swinging,
            sneaking: state.sneaking,
            sprinting: state.sprinting,
            held_item: state.held_item.clone(),
        };
        self.remote_players.write().insert(state.player_name.clone(), state);
        let _ = self.tx.send(pkt);
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
        self.task.abort();
    }
}

impl Drop for LocalServerHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

pub async fn start_server_with_handle(
    port: u16,
    world: Arc<WorldStore>,
    registries: Arc<RwLock<EngineRegistries>>,
    spawn_pos: Arc<RwLock<[f32; 3]>>,
) -> anyhow::Result<LocalServerHandle> {
    let listener = TcpListener::bind(("0.0.0.0", port)).await?;
    let bound_port = listener.local_addr()?.port();
    let (tx, _) = broadcast::channel::<NetPacket>(512);
    let tx_srv = tx.clone();
    let remote_players = Arc::new(RwLock::new(HashMap::<String, RemotePlayerState>::new()));
    let chunk_deltas = Arc::new(RwLock::new(HashMap::<IVec3, Vec<VoxelDelta>>::new()));
    let stream_hub = Arc::new(crate::network::web_client::StreamHub::new());
    let stop = Arc::new(AtomicBool::new(false));

    let rp_srv = remote_players.clone();
    let cd_srv = chunk_deltas.clone();
    let sh_srv = stream_hub.clone();
    let stop_srv = stop.clone();

    let task = tokio::spawn(async move {
        loop {
            if stop_srv.load(Ordering::Relaxed) {
                break;
            }
            let Ok((stream, _addr)) = listener.accept().await else {
                break;
            };
            let tx_c = tx_srv.clone();
            let mut rx_c = tx_srv.subscribe();
            let world_c = world.clone();
            let reg_c = registries.clone();
            let rp_c = rp_srv.clone();
            let cd_c = cd_srv.clone();
            let sh_c = sh_srv.clone();
            let sp_c = spawn_pos.clone();
            let stop_c = stop_srv.clone();

            tokio::spawn(async move {
                let mut peek_buf = [0u8; 8];
                let is_http = match tokio::time::timeout(
                    std::time::Duration::from_millis(200),
                    stream.peek(&mut peek_buf),
                )
                .await
                {
                    Ok(Ok(n)) if n > 0 => {
                        let head = &peek_buf[..n];
                        head.starts_with(b"GET ")
                            || head.starts_with(b"HEAD ")
                            || head.starts_with(b"POST ")
                            || head.starts_with(b"OPTIONS")
                            || head.starts_with(b"PUT ")
                    }
                    _ => false,
                };
                if is_http {
                    crate::network::web_client::handle_http_or_ws(
                        stream, world_c, reg_c, tx_c, rp_c, cd_c, sh_c, sp_c, stop_c,
                    )
                    .await;
                    return;
                }

                let (reader, mut writer) = stream.into_split();
                let mut lines = BufReader::new(reader).lines();
                let (direct_tx, mut direct_rx) = mpsc::unbounded_channel::<NetPacket>();
                let mut client_name: Option<String> = None;

                let welcome = NetPacket::Welcome {
                    seed: world_c.seed,
                    motd: "§aCubex-RS Mineclonia Server".into(),
                    world_type: world_c.world_type.read().as_str().to_string(),
                    spawn_pos: Some(*sp_c.read()),
                };
                if let Ok(js) = serde_json::to_string(&welcome) {
                    let _ = writer.write_all(format!("{js}\n").as_bytes()).await;
                }

                // Send current chunk deltas to late-joining peer
                {
                    let snap: Vec<(IVec3, Vec<VoxelDelta>)> = cd_c
                        .read()
                        .iter()
                        .map(|(&k, v)| (k, v.clone()))
                        .collect();
                    for (cpos, edits) in snap {
                        if !edits.is_empty() {
                            let _ = direct_tx.send(NetPacket::ChunkDelta {
                                cx: cpos.x,
                                cy: cpos.y,
                                cz: cpos.z,
                                edits,
                            });
                        }
                    }
                }

                // Send currently active players (including Host!) immediately so joiners see everyone in 3D
                {
                    let existing: Vec<RemotePlayerState> = rp_c.read().values().cloned().collect();
                    for ps in existing {
                        let _ = direct_tx.send(NetPacket::PlayerMove {
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
                        });
                    }
                }

                let writer_task = tokio::spawn(async move {
                    loop {
                        tokio::select! {
                            res = rx_c.recv() => {
                                match res {
                                    Ok(pkt) => {
                                        if let Ok(js) = serde_json::to_string(&pkt) {
                                            if writer.write_all(format!("{js}\n").as_bytes()).await.is_err() {
                                                break;
                                            }
                                        }
                                    }
                                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                                    Err(broadcast::error::RecvError::Closed) => break,
                                }
                            }
                            Some(pkt) = direct_rx.recv() => {
                                if let Ok(js) = serde_json::to_string(&pkt) {
                                    if writer.write_all(format!("{js}\n").as_bytes()).await.is_err() {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                });

                while !stop_c.load(Ordering::Relaxed) {
                    let Ok(Some(line)) = lines.next_line().await else {
                        break;
                    };
                    if let Ok(pkt) = serde_json::from_str::<NetPacket>(&line) {
                        match &pkt {
                            NetPacket::Hello { player_name } => {
                                client_name = Some(player_name.clone());
                                rp_c.write().entry(player_name.clone()).or_insert_with(|| RemotePlayerState {
                                    player_name: player_name.clone(),
                                    x: 0.5,
                                    y: 24.0,
                                    z: 0.5,
                                    yaw: 0.0,
                                    pitch: 0.0,
                                    swinging: false,
                                    sneaking: false,
                                    sprinting: false,
                                    held_item: String::new(),
                                });
                            }
                            NetPacket::PlayerPos { player_name, x, y, z, yaw } => {
                                client_name.get_or_insert_with(|| player_name.clone());
                                let mut g = rp_c.write();
                                let entry = g.entry(player_name.clone()).or_insert_with(|| RemotePlayerState {
                                    player_name: player_name.clone(),
                                    x: *x,
                                    y: *y,
                                    z: *z,
                                    yaw: *yaw,
                                    pitch: 0.0,
                                    swinging: false,
                                    sneaking: false,
                                    sprinting: false,
                                    held_item: String::new(),
                                });
                                entry.x = *x;
                                entry.y = *y;
                                entry.z = *z;
                                entry.yaw = *yaw;
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
                                rp_c.write().insert(
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
                            NetPacket::BlockSet { x, y, z, node_name } => {
                                let wpos = IVec3::new(*x, *y, *z);
                                let id = reg_c.read().get_content_id(node_name);
                                world_c.set_node(wpos, id, 15, 0);
                                world_c.mark_priority_dirty_with_neighbors(wpos);
                                let (cpos, lx, ly, lz) = world_to_chunk(wpos);
                                cd_c.write().entry(cpos).or_default().push(VoxelDelta {
                                    lx: lx as u8,
                                    ly: ly as u8,
                                    lz: lz as u8,
                                    node_name: node_name.clone(),
                                    param1: 15,
                                    param2: 0,
                                });
                            }
                            NetPacket::BlockEdit { x, y, z, node_name, param1, param2 } => {
                                let wpos = IVec3::new(*x, *y, *z);
                                let id = reg_c.read().get_content_id(node_name);
                                world_c.set_node(wpos, id, *param1, *param2);
                                world_c.mark_priority_dirty_with_neighbors(wpos);
                                let (cpos, lx, ly, lz) = world_to_chunk(wpos);
                                cd_c.write().entry(cpos).or_default().push(VoxelDelta {
                                    lx: lx as u8,
                                    ly: ly as u8,
                                    lz: lz as u8,
                                    node_name: node_name.clone(),
                                    param1: *param1,
                                    param2: *param2,
                                });
                            }
                            NetPacket::ChunkRequest { cx, cy, cz } => {
                                let cpos = IVec3::new(*cx, *cy, *cz);
                                if let Some(ch_arc) = world_c.get_chunk(cpos) {
                                    let ch = ch_arc.read().clone();
                                    let reg = reg_c.read();
                                    if let Some(b64) = encode_chunk_data(&ch, &reg) {
                                        let _ = direct_tx.send(NetPacket::ChunkData {
                                            cx: *cx,
                                            cy: *cy,
                                            cz: *cz,
                                            compressed_b64: b64,
                                        });
                                    }
                                }
                                continue;
                            }
                            NetPacket::Chat { sender, message } => {
                                if handle_server_chat_command(sender, message, &rp_c, world_c.seed, &tx_c) {
                                    continue;
                                }
                            }
                            _ => {}
                        }
                        let _ = tx_c.send(pkt);
                    }
                }

                if let Some(name) = client_name {
                    rp_c.write().remove(&name);
                    let _ = tx_c.send(NetPacket::PlayerLeave { player_name: name });
                }
                writer_task.abort();
            });
        }
    });

    Ok(LocalServerHandle {
        port: bound_port,
        tx,
        remote_players,
        chunk_deltas,
        stream_hub,
        stop,
        task,
    })
}

pub async fn start_server(
    port: u16,
    world: Arc<WorldStore>,
    registries: Arc<RwLock<EngineRegistries>>,
) -> anyhow::Result<(u16, broadcast::Sender<NetPacket>)> {
    let spawn = Arc::new(RwLock::new([0.5, 24.0, 0.5]));
    let handle = start_server_with_handle(port, world, registries, spawn).await?;
    let port = handle.port;
    let tx = handle.tx.clone();
    std::mem::forget(handle);
    Ok((port, tx))
}

pub async fn connect_client(host: &str, port: u16, player_name: &str) -> anyhow::Result<NetPacket> {
    let stream = TcpStream::connect(format!("{host}:{port}")).await?;
    let (reader, mut writer) = stream.into_split();
    let hello = NetPacket::Hello {
        player_name: player_name.to_string(),
    };
    writer
        .write_all(format!("{}\n", serde_json::to_string(&hello)?).as_bytes())
        .await?;
    let mut lines = BufReader::new(reader).lines();
    if let Some(line) = lines.next_line().await? {
        let pkt: NetPacket = serde_json::from_str(&line)?;
        return Ok(pkt);
    }
    anyhow::bail!("Server closed connection before Welcome")
}

/// Active non-blocking multiplayer client session synchronized over background tokio tasks.
pub struct NetClientSession {
    pub player_name: String,
    out_tx: mpsc::UnboundedSender<NetPacket>,
    in_rx: parking_lot::Mutex<mpsc::UnboundedReceiver<NetPacket>>,
    pub remote_players: Arc<RwLock<HashMap<String, RemotePlayerState>>>,
    alive: Arc<AtomicBool>,
}

impl NetClientSession {
    pub async fn connect(host: &str, port: u16, player_name: &str) -> anyhow::Result<(NetPacket, Self)> {
        let stream = TcpStream::connect(format!("{host}:{port}")).await?;
        let (reader, mut writer) = stream.into_split();
        let hello = NetPacket::Hello {
            player_name: player_name.to_string(),
        };
        writer
            .write_all(format!("{}\n", serde_json::to_string(&hello)?).as_bytes())
            .await?;

        let mut lines = BufReader::new(reader).lines();
        let Some(first) = lines.next_line().await? else {
            anyhow::bail!("Server closed connection before Welcome");
        };
        let welcome: NetPacket = serde_json::from_str(&first)?;

        let (out_tx, mut out_rx) = mpsc::unbounded_channel::<NetPacket>();
        let (in_tx, in_rx) = mpsc::unbounded_channel::<NetPacket>();
        let remote_players = Arc::new(RwLock::new(HashMap::<String, RemotePlayerState>::new()));
        let alive = Arc::new(AtomicBool::new(true));

        let alive_w = alive.clone();
        tokio::spawn(async move {
            while let Some(pkt) = out_rx.recv().await {
                if !alive_w.load(Ordering::Relaxed) {
                    break;
                }
                if let Ok(js) = serde_json::to_string(&pkt) {
                    if writer.write_all(format!("{js}\n").as_bytes()).await.is_err() {
                        break;
                    }
                }
            }
            alive_w.store(false, Ordering::Relaxed);
        });

        let alive_r = alive.clone();
        let rp_r = remote_players.clone();
        let self_name = player_name.to_string();
        tokio::spawn(async move {
            while let Ok(Some(line)) = lines.next_line().await {
                if !alive_r.load(Ordering::Relaxed) {
                    break;
                }
                if let Ok(pkt) = serde_json::from_str::<NetPacket>(&line) {
                    match &pkt {
                        NetPacket::PlayerPos { player_name, x, y, z, yaw } if player_name != &self_name => {
                            let mut g = rp_r.write();
                            let e = g.entry(player_name.clone()).or_insert_with(|| RemotePlayerState {
                                player_name: player_name.clone(),
                                x: *x,
                                y: *y,
                                z: *z,
                                yaw: *yaw,
                                pitch: 0.0,
                                swinging: false,
                                sneaking: false,
                                sprinting: false,
                                held_item: String::new(),
                            });
                            e.x = *x;
                            e.y = *y;
                            e.z = *z;
                            e.yaw = *yaw;
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
                        } if player_name != &self_name => {
                            rp_r.write().insert(
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
                        NetPacket::PlayerLeave { player_name } => {
                            rp_r.write().remove(player_name);
                        }
                        _ => {}
                    }
                    let _ = in_tx.send(pkt);
                }
            }
            alive_r.store(false, Ordering::Relaxed);
        });

        Ok((
            welcome,
            Self {
                player_name: player_name.to_string(),
                out_tx,
                in_rx: parking_lot::Mutex::new(in_rx),
                remote_players,
                alive,
            },
        ))
    }

    pub fn is_connected(&self) -> bool {
        self.alive.load(Ordering::Relaxed)
    }

    pub fn send(&self, pkt: NetPacket) {
        let _ = self.out_tx.send(pkt);
    }

    pub fn drain_packets(&self) -> Vec<NetPacket> {
        let mut rx = self.in_rx.lock();
        let mut out = Vec::new();
        while let Ok(pkt) = rx.try_recv() {
            out.push(pkt);
        }
        out
    }

    pub fn disconnect(&self) {
        self.alive.store(false, Ordering::Relaxed);
    }
}

impl Drop for NetClientSession {
    fn drop(&mut self) {
        self.disconnect();
    }
}

pub fn run_dedicated_server(world_dir: &Path, port: u16) -> anyhow::Result<()> {
    let paths = EnginePaths::resolve();
    paths.ensure_dirs();
    let registries = Arc::new(RwLock::new(EngineRegistries::new()));
    let world = Arc::new(WorldStore::new(world_dir, 1337));
    let config = Arc::new(RwLock::new(MinetestConfig::load(&paths.conf_file)));

    let rt_lua = LuaRuntime::boot(paths, registries.clone(), world.clone(), config)?;
    println!(
        "[Dedicated] Mineclonia loaded: {} mods ({} nodes, {} recipes)",
        rt_lua.load_report.mods_loaded,
        registries.read().node_count(),
        registries.read().crafts.len()
    );

    let palette = MapgenPalette::from_registries(&registries.read());
    for cz in -2..=2 {
        for cx in -2..=2 {
            for cy in -1..=1 {
                let ch = generate_chunk(IVec3::new(cx, cy, cz), world.seed, &palette, &registries.read());
                world.insert_chunk(ch);
            }
        }
    }

    let tokio_rt = tokio::runtime::Runtime::new()?;
    tokio_rt.block_on(async move {
        let (bound_port, _) = start_server(port, world, registries).await?;
        let _beacon = LanBroadcaster::spawn(LanBeaconInfo {
            name: "Dedicated Cubex Server".into(),
            port: bound_port,
            motd: "§aCubex-RS Dedicated Server\n§7Mineclonia Native Rust".into(),
            players_online: 0,
            players_max: 32,
        });
        println!("[Dedicated] Listening on 0.0.0.0:{bound_port} (LAN broadcast active, Ctrl+C to stop)");
        tokio::signal::ctrl_c().await?;
        Ok::<(), anyhow::Error>(())
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{DrawType, NodeDef};

    #[test]
    fn chunk_compression_and_session_sync_roundtrip() {
        let mut reg = EngineRegistries::new();
        let stone = reg.register_node(NodeDef {
            name: "mcl_core:stone".into(),
            drawtype: DrawType::Normal,
            ..Default::default()
        });
        let mut ch = Chunk::new(IVec3::new(1, 0, -1));
        ch.set(2, 3, 4, stone, 15, 7);
        let b64 = encode_chunk_data(&ch, &reg).expect("encode chunk");
        let dec = decode_chunk_data(1, 0, -1, &b64, &reg).expect("decode chunk");
        assert_eq!(dec.get(2, 3, 4), (stone, 15, 7));

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let tmp = std::env::temp_dir().join("cubex_rs_proto_session");
            let world = Arc::new(WorldStore::new(tmp, 999));
            let reg_arc = Arc::new(RwLock::new(reg));
            let (port, _) = start_server(0, world.clone(), reg_arc.clone()).await.unwrap();
            let (welcome, client) = NetClientSession::connect("127.0.0.1", port, "Alice")
                .await
                .unwrap();
            assert!(matches!(welcome, NetPacket::Welcome { seed: 999, .. }));

            client.send(NetPacket::BlockEdit {
                x: 4,
                y: 5,
                z: 6,
                node_name: "mcl_core:stone".into(),
                param1: 15,
                param2: 3,
            });
            client.send(NetPacket::Chat {
                sender: "Alice".into(),
                message: "/seed".into(),
            });

            let mut got_seed_reply = false;
            for _ in 0..20 {
                for pkt in client.drain_packets() {
                    if let NetPacket::Chat { message, .. } = pkt {
                        if message.contains("999") {
                            got_seed_reply = true;
                        }
                    }
                }
                if got_seed_reply && world.get_node(IVec3::new(4, 5, 6)) == (stone, 15, 3) {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(25)).await;
            }
            assert!(got_seed_reply);
            assert_eq!(world.get_node(IVec3::new(4, 5, 6)), (stone, 15, 3));
        });
    }

    #[test]
    fn lan_local_server_handle_syncs_same_world_and_stops_cleanly() {
        let mut reg = EngineRegistries::new();
        let stone = reg.register_node(NodeDef {
            name: "mcl_core:stone".into(),
            drawtype: DrawType::Normal,
            ..Default::default()
        });
        let cobble = reg.register_node(NodeDef {
            name: "mcl_core:cobble".into(),
            drawtype: DrawType::Normal,
            ..Default::default()
        });
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let tmp = std::env::temp_dir().join("cubex_rs_lan_8090_sync");
            let world = Arc::new(WorldStore::new(tmp, 8090));
            let reg_arc = Arc::new(RwLock::new(reg));
            let spawn = Arc::new(RwLock::new([12.5, 68.0, -4.5]));
            // Bind port 8090 if available, else ephemeral 0 so parallel/busy hosts never flake
            let handle = match start_server_with_handle(
                crate::network::DEFAULT_LAN_PORT,
                world.clone(),
                reg_arc.clone(),
                spawn.clone(),
            )
            .await
            {
                Ok(h) => h,
                Err(_) => start_server_with_handle(0, world.clone(), reg_arc.clone(), spawn)
                    .await
                    .unwrap(),
            };

            handle.update_host_player(RemotePlayerState {
                player_name: "HostPlayer".into(),
                x: 12.5,
                y: 68.0,
                z: -4.5,
                yaw: 0.25,
                pitch: 0.0,
                swinging: false,
                sneaking: false,
                sprinting: false,
                held_item: "mcl_core:cobble".into(),
            });

            let (welcome, client) = NetClientSession::connect("127.0.0.1", handle.port, "LanGuest")
                .await
                .unwrap();
            match welcome {
                NetPacket::Welcome {
                    seed, spawn_pos, ..
                } => {
                    assert_eq!(seed, 8090);
                    assert_eq!(spawn_pos, Some([12.5, 68.0, -4.5]));
                }
                other => panic!("Unexpected Welcome packet: {other:?}"),
            }

            // Host edits world -> Guest receives BlockEdit; Guest moves & edits -> Host updates world & sees remote player
            handle.record_host_block_edit(IVec3::new(1, 2, 3), "mcl_core:cobble".into(), 15, 0);
            client.send(NetPacket::BlockEdit {
                x: 5,
                y: 6,
                z: 7,
                node_name: "mcl_core:stone".into(),
                param1: 15,
                param2: 0,
            });
            client.send(NetPacket::PlayerMove {
                player_name: "LanGuest".into(),
                x: 14.0,
                y: 68.0,
                z: -4.0,
                yaw: 1.0,
                pitch: 0.1,
                swinging: true,
                sneaking: false,
                sprinting: true,
                held_item: "mcl_core:stone".into(),
            });

            let mut client_got_host_edit = false;
            for _ in 0..30 {
                for pkt in client.drain_packets() {
                    if let NetPacket::BlockEdit {
                        x: 1,
                        y: 2,
                        z: 3,
                        node_name,
                        ..
                    } = pkt
                    {
                        if node_name == "mcl_core:cobble" {
                            client_got_host_edit = true;
                        }
                    }
                }
                if client_got_host_edit
                    && world.get_node(IVec3::new(5, 6, 7)).0 == stone
                    && handle.remote_players.read().contains_key("LanGuest")
                {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
            assert!(client_got_host_edit, "guest must receive host BlockEdit");
            assert_eq!(world.get_node(IVec3::new(5, 6, 7)).0, stone);
            let _ = cobble;
            assert!(client.remote_players.read().contains_key("HostPlayer"));
            assert!(handle.remote_players.read().contains_key("LanGuest"));

            client.disconnect();
            handle.stop();
        });
    }
}
