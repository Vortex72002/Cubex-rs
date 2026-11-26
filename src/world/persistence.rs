use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use glam::{IVec3, Vec3};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::lua_api::userdata::ObjectState;
use crate::registry::{EngineRegistries, CONTENT_UNKNOWN};
use crate::world::{
    Chunk, Difficulty, DimensionId, Gamerules, ItemStackData, MetaDataState, NodeTimerState,
    WorldGenType, WorldStore, CHUNK_VOLUME,
};

const MAGIC_CHUNKS: &[u8; 4] = b"CBXC";
const MAGIC_PLAYERS: &[u8; 4] = b"CBXP";
const MAGIC_META: &[u8; 4] = b"CBXM";
const PERSIST_VERSION: u16 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedItemStack {
    pub name: String,
    pub count: u16,
    pub wear: u16,
    pub meta: PersistedMeta,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PersistedMeta {
    pub fields: HashMap<String, String>,
    pub inventories: HashMap<String, Vec<PersistedItemStack>>,
}

impl PersistedMeta {
    pub fn from_state(st: &MetaDataState) -> Self {
        let mut inventories = HashMap::with_capacity(st.inventories.len());
        for (lname, vec) in &st.inventories {
            let items = vec
                .iter()
                .map(|it| PersistedItemStack {
                    name: it.name.clone(),
                    count: it.count,
                    wear: it.wear,
                    meta: Self::from_state(&it.meta.read()),
                })
                .collect();
            inventories.insert(lname.clone(), items);
        }
        Self {
            fields: st.fields.clone(),
            inventories,
        }
    }

    pub fn to_state(&self) -> MetaDataState {
        let mut inventories = HashMap::with_capacity(self.inventories.len());
        for (lname, vec) in &self.inventories {
            let items = vec
                .iter()
                .map(|it| ItemStackData {
                    name: it.name.clone(),
                    count: it.count,
                    wear: it.wear,
                    meta: Arc::new(RwLock::new(it.meta.to_state())),
                })
                .collect();
            inventories.insert(lname.clone(), items);
        }
        MetaDataState {
            fields: self.fields.clone(),
            inventories,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedPlayer {
    pub name: String,
    pub pos: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub hp: i32,
    pub hp_max: i32,
    pub hunger: i32,
    pub xp_level: i32,
    pub xp_progress: f32,
    pub breath: u16,
    pub wield_index: usize,
    pub gamemode: String,
    #[serde(default)]
    pub dimension: DimensionId,
    pub meta: PersistedMeta,
}

impl PersistedPlayer {
    pub fn from_object(obj: &ObjectState, gamemode: &str) -> Self {
        Self {
            name: obj.name.clone(),
            pos: [obj.pos.x, obj.pos.y, obj.pos.z],
            yaw: obj.yaw,
            pitch: obj.pitch,
            hp: obj.hp,
            hp_max: obj.hp_max,
            hunger: obj.hunger,
            xp_level: obj.xp_level,
            xp_progress: obj.xp_progress,
            breath: obj.breath,
            wield_index: obj.wield_index,
            gamemode: gamemode.to_string(),
            dimension: obj.dimension,
            meta: PersistedMeta::from_state(&obj.meta.read()),
        }
    }

    pub fn apply_to_object(&self, obj: &mut ObjectState) {
        obj.pos = Vec3::new(self.pos[0], self.pos[1], self.pos[2]);
        obj.yaw = self.yaw;
        obj.pitch = self.pitch;
        obj.hp_max = self.hp_max.max(1);
        obj.hp = self.hp.clamp(0, obj.hp_max);
        obj.hunger = self.hunger;
        obj.xp_level = self.xp_level;
        obj.xp_progress = self.xp_progress;
        obj.breath = self.breath;
        obj.wield_index = self.wield_index.max(1);
        obj.dimension = self.dimension;
        let mut restored = self.meta.to_state();
        let main = restored
            .inventories
            .entry("main".into())
            .or_insert_with(|| vec![ItemStackData::default(); 36]);
        if main.len() < 36 {
            main.resize_with(36, ItemStackData::default);
        }
        *obj.meta.write() = restored;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedNodeMetaEntry {
    pos: [i32; 3],
    meta: PersistedMeta,
    timer: Option<(f64, f64, bool)>,
}

#[derive(Debug, Clone)]
pub struct SavedWorldState {
    pub seed: u64,
    pub world_type: WorldGenType,
    pub gamemode: String,
    pub difficulty: Difficulty,
    pub gamerules: Gamerules,
    pub player: Option<PersistedPlayer>,
    pub chunks_loaded: usize,
    pub meta_loaded: usize,
}

fn write_framed_zlib(path: &Path, magic: &[u8; 4], payload: &[u8]) -> anyhow::Result<()> {
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::fast());
    enc.write_all(payload)?;
    let compressed = enc.finish()?;

    let mut framed = Vec::with_capacity(6 + compressed.len());
    framed.extend_from_slice(magic);
    framed.extend_from_slice(&PERSIST_VERSION.to_le_bytes());
    framed.extend_from_slice(&compressed);

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("bin.tmp");
    std::fs::write(&tmp, &framed)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn read_framed_zlib(path: &Path, expected_magic: &[u8; 4]) -> anyhow::Result<Vec<u8>> {
    let raw = std::fs::read(path)?;
    if raw.len() < 6 || &raw[0..4] != expected_magic {
        anyhow::bail!("Invalid binary header in {}", path.display());
    }
    let ver = u16::from_le_bytes([raw[4], raw[5]]);
    if ver != PERSIST_VERSION {
        anyhow::bail!("Unsupported persistence version {ver} in {}", path.display());
    }
    let mut dec = ZlibDecoder::new(&raw[6..]);
    let mut out = Vec::new();
    dec.read_to_end(&mut out)?;
    Ok(out)
}

fn write_world_mt(world: &WorldStore) -> anyhow::Result<()> {
    let wt = *world.world_type.read();
    let diff = *world.difficulty.read();
    let gr = world.gamerules.read().clone();
    let gm = world.gamemode.read().clone();
    let creative = gm.eq_ignore_ascii_case("creative");
    let content = format!(
        "gameid = mineclonia\nbackend = cubex\ncreative_mode = {}\ngamemode = {}\nseed = {}\nworld_type = {}\ndifficulty = {}\nkeep_inventory = {}\ndo_daylight_cycle = {}\npvp = {}\nfall_damage = {}\nload_mod_cubex_bridge = true\n",
        if creative { "true" } else { "false" },
        gm,
        world.seed,
        wt.as_str(),
        diff.as_str(),
        gr.keep_inventory,
        gr.do_daylight_cycle,
        gr.pvp,
        gr.fall_damage,
    );
    std::fs::write(world.world_path.join("world.mt"), content)?;
    std::fs::write(
        world.world_path.join("map_meta.txt"),
        format!("mg_name = v7\nseed = {}\n[end_of_params]\n", world.seed),
    )?;
    Ok(())
}

fn resolve_load_bin(world: &WorldStore, filename: &str) -> Option<PathBuf> {
    let dim_path = world.dimension_dir().join(filename);
    if dim_path.is_file() {
        return Some(dim_path);
    }
    if world.dimension == DimensionId::Overworld {
        let legacy = world.world_path.join(filename);
        if legacy.is_file() {
            return Some(legacy);
        }
    }
    None
}

fn save_chunks_bin(world: &WorldStore, reg: &EngineRegistries) -> anyhow::Result<()> {
    let chunk_arcs: Vec<_> = world.chunks.read().values().cloned().collect();
    let mut modified_chunks = Vec::new();
    for c_arc in chunk_arcs {
        let g = c_arc.read();
        if g.modified {
            modified_chunks.push(g.clone());
        }
    }

    let mut palette_names: Vec<String> = Vec::new();
    let mut id_to_pal: HashMap<u16, u16> = HashMap::new();
    for ch in &modified_chunks {
        for &id in &ch.nodes {
            if !id_to_pal.contains_key(&id) {
                let pal_idx = palette_names.len() as u16;
                palette_names.push(reg.get_name_from_content_id(id).to_string());
                id_to_pal.insert(id, pal_idx);
            }
        }
    }

    let mut payload = Vec::with_capacity(128 + modified_chunks.len() * (12 + CHUNK_VOLUME * 4));
    payload.extend_from_slice(&(palette_names.len() as u32).to_le_bytes());
    for nm in &palette_names {
        let b = nm.as_bytes();
        payload.extend_from_slice(&(b.len() as u16).to_le_bytes());
        payload.extend_from_slice(b);
    }
    payload.extend_from_slice(&(modified_chunks.len() as u32).to_le_bytes());
    for ch in &modified_chunks {
        payload.extend_from_slice(&ch.pos.x.to_le_bytes());
        payload.extend_from_slice(&ch.pos.y.to_le_bytes());
        payload.extend_from_slice(&ch.pos.z.to_le_bytes());
        for &id in &ch.nodes {
            let pal = id_to_pal.get(&id).copied().unwrap_or(0);
            payload.extend_from_slice(&pal.to_le_bytes());
        }
        payload.extend_from_slice(&ch.param1);
        payload.extend_from_slice(&ch.param2);
    }

    write_framed_zlib(
        &world.dimension_dir().join("chunks.bin"),
        MAGIC_CHUNKS,
        &payload,
    )
}

fn load_chunks_bin(world: &WorldStore, reg: &EngineRegistries) -> anyhow::Result<usize> {
    let Some(path) = resolve_load_bin(world, "chunks.bin") else {
        return Ok(0);
    };
    let data = read_framed_zlib(&path, MAGIC_CHUNKS)?;
    let mut cur = 0usize;
    let read_u32 = |d: &[u8], c: &mut usize| -> anyhow::Result<u32> {
        if *c + 4 > d.len() {
            anyhow::bail!("Unexpected EOF in chunks.bin");
        }
        let v = u32::from_le_bytes([d[*c], d[*c + 1], d[*c + 2], d[*c + 3]]);
        *c += 4;
        Ok(v)
    };
    let read_i32 = |d: &[u8], c: &mut usize| -> anyhow::Result<i32> {
        if *c + 4 > d.len() {
            anyhow::bail!("Unexpected EOF in chunks.bin");
        }
        let v = i32::from_le_bytes([d[*c], d[*c + 1], d[*c + 2], d[*c + 3]]);
        *c += 4;
        Ok(v)
    };
    let read_u16 = |d: &[u8], c: &mut usize| -> anyhow::Result<u16> {
        if *c + 2 > d.len() {
            anyhow::bail!("Unexpected EOF in chunks.bin");
        }
        let v = u16::from_le_bytes([d[*c], d[*c + 1]]);
        *c += 2;
        Ok(v)
    };

    let pal_count = read_u32(&data, &mut cur)? as usize;
    let mut pal_ids = Vec::with_capacity(pal_count);
    for _ in 0..pal_count {
        let slen = read_u16(&data, &mut cur)? as usize;
        if cur + slen > data.len() {
            anyhow::bail!("Truncated palette string in chunks.bin");
        }
        let name = std::str::from_utf8(&data[cur..cur + slen])?;
        cur += slen;
        let id = reg.get_content_id(name);
        pal_ids.push(if id == CONTENT_UNKNOWN && name == "air" {
            crate::registry::CONTENT_AIR
        } else {
            id
        });
    }

    let chunk_count = read_u32(&data, &mut cur)? as usize;
    for _ in 0..chunk_count {
        let cx = read_i32(&data, &mut cur)?;
        let cy = read_i32(&data, &mut cur)?;
        let cz = read_i32(&data, &mut cur)?;
        let mut ch = Chunk::new(IVec3::new(cx, cy, cz));
        for i in 0..CHUNK_VOLUME {
            let pal_idx = read_u16(&data, &mut cur)? as usize;
            ch.nodes[i] = pal_ids.get(pal_idx).copied().unwrap_or(CONTENT_UNKNOWN);
        }
        if cur + CHUNK_VOLUME * 2 > data.len() {
            anyhow::bail!("Truncated chunk lighting/param2 in chunks.bin");
        }
        ch.param1.copy_from_slice(&data[cur..cur + CHUNK_VOLUME]);
        cur += CHUNK_VOLUME;
        ch.param2.copy_from_slice(&data[cur..cur + CHUNK_VOLUME]);
        cur += CHUNK_VOLUME;
        ch.modified = true;
        ch.dirty_mesh = true;
        world.insert_chunk(ch);
    }

    Ok(chunk_count)
}

fn save_players_bin(world: &WorldStore, players_in: &[&ObjectState]) -> anyhow::Result<()> {
    let gm = world.gamemode.read().clone();
    let list: Vec<PersistedPlayer> = players_in
        .iter()
        .map(|p| PersistedPlayer::from_object(p, &gm))
        .collect();
    let payload = serde_json::to_vec(&list)?;
    write_framed_zlib(&world.world_path.join("players.bin"), MAGIC_PLAYERS, &payload)
}

fn load_players_bin(world: &WorldStore) -> anyhow::Result<Vec<PersistedPlayer>> {
    let path = world.world_path.join("players.bin");
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let data = read_framed_zlib(&path, MAGIC_PLAYERS)?;
    let list: Vec<PersistedPlayer> = serde_json::from_slice(&data)?;
    Ok(list)
}

fn save_nodemeta_bin(world: &WorldStore) -> anyhow::Result<()> {
    let meta_map = world.node_meta.read();
    let timers = world.node_timers.read();
    let mut entries = Vec::with_capacity(meta_map.len());
    for (&pos, m_arc) in meta_map.iter() {
        let pm = PersistedMeta::from_state(&m_arc.read());
        let timer = timers
            .get(&pos)
            .map(|t| (t.timeout, t.elapsed, t.started));
        entries.push(PersistedNodeMetaEntry {
            pos: [pos.x, pos.y, pos.z],
            meta: pm,
            timer,
        });
    }
    let payload = serde_json::to_vec(&entries)?;
    write_framed_zlib(
        &world.dimension_dir().join("nodemeta.bin"),
        MAGIC_META,
        &payload,
    )
}

fn load_nodemeta_bin(world: &WorldStore) -> anyhow::Result<usize> {
    let Some(path) = resolve_load_bin(world, "nodemeta.bin") else {
        return Ok(0);
    };
    let data = read_framed_zlib(&path, MAGIC_META)?;
    let entries: Vec<PersistedNodeMetaEntry> = serde_json::from_slice(&data)?;
    let count = entries.len();
    let mut meta_map = world.node_meta.write();
    let mut timers = world.node_timers.write();
    for e in entries {
        let pos = IVec3::new(e.pos[0], e.pos[1], e.pos[2]);
        meta_map.insert(pos, Arc::new(RwLock::new(e.meta.to_state())));
        if let Some((timeout, elapsed, started)) = e.timer {
            timers.insert(
                pos,
                NodeTimerState {
                    timeout,
                    elapsed,
                    started,
                },
            );
        }
    }
    Ok(count)
}

/// Persist full world state (`world.mt`, `chunks.bin`, `players.bin`, `nodemeta.bin`).
pub fn save_world(
    world: &WorldStore,
    reg: &EngineRegistries,
    player: Option<&ObjectState>,
) -> anyhow::Result<()> {
    let players: Vec<&ObjectState> = player.into_iter().collect();
    save_world_players(world, reg, &players)
}

/// Persist full world state alongside one or more local players (Singleplayer or Co-op).
pub fn save_world_players(
    world: &WorldStore,
    reg: &EngineRegistries,
    players: &[&ObjectState],
) -> anyhow::Result<()> {
    std::fs::create_dir_all(&world.world_path)?;
    std::fs::create_dir_all(world.dimension_dir())?;
    if world.dimension == DimensionId::Overworld {
        write_world_mt(world)?;
    }
    save_chunks_bin(world, reg)?;
    if world.dimension == DimensionId::Overworld {
        save_players_bin(world, players)?;
    }
    save_nodemeta_bin(world)?;
    Ok(())
}

/// Persist modified chunks and node metadata for a single dimension (`world/overworld/` or `world/nether/`).
pub fn save_dimension_chunks(world: &WorldStore, reg: &EngineRegistries) -> anyhow::Result<()> {
    std::fs::create_dir_all(world.dimension_dir())?;
    save_chunks_bin(world, reg)?;
    save_nodemeta_bin(world)?;
    Ok(())
}

/// Load modified chunks and node metadata for a single dimension, returning `(chunks_loaded, meta_loaded)`.
pub fn load_dimension_chunks(
    world: &WorldStore,
    reg: &EngineRegistries,
) -> anyhow::Result<(usize, usize)> {
    let chunks_loaded = load_chunks_bin(world, reg)?;
    let meta_loaded = load_nodemeta_bin(world)?;
    Ok((chunks_loaded, meta_loaded))
}

/// Load a specific player's saved state from `players.bin` by exact name (never falling back to Player 1).
pub fn load_player_by_name(
    world: &WorldStore,
    target_player: &mut ObjectState,
) -> Option<PersistedPlayer> {
    let players = load_players_bin(world).ok()?;
    let matched = players.into_iter().find(|p| p.name == target_player.name)?;
    matched.apply_to_object(target_player);
    Some(matched)
}

/// Load persisted world metadata, modified chunks, player state, and node inventories/metadata.
pub fn load_world(
    world: &WorldStore,
    reg: &EngineRegistries,
    target_player: Option<&mut ObjectState>,
) -> anyhow::Result<SavedWorldState> {
    let mut world_type = *world.world_type.read();
    let mut difficulty = *world.difficulty.read();
    let mut gamerules = world.gamerules.read().clone();
    let mut gamemode = world.gamemode.read().clone();
    let mut seed = world.seed;

    if let Ok(mt) = std::fs::read_to_string(world.world_path.join("world.mt")) {
        for line in mt.lines() {
            if let Some((k, v)) = line.split_once('=') {
                let key = k.trim();
                let val = v.trim();
                match key {
                    "creative_mode" if val == "true" => gamemode = "Creative".into(),
                    "gamemode" if !val.is_empty() => gamemode = val.to_string(),
                    "world_type" => world_type = WorldGenType::from_str(val),
                    "difficulty" => difficulty = Difficulty::from_str(val),
                    "keep_inventory" => gamerules.keep_inventory = val != "false",
                    "do_daylight_cycle" => gamerules.do_daylight_cycle = val != "false",
                    "pvp" => gamerules.pvp = val != "false",
                    "fall_damage" => gamerules.fall_damage = val != "false",
                    "seed" => {
                        if let Ok(s) = val.parse::<u64>() {
                            seed = s;
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    *world.world_type.write() = world_type;
    *world.difficulty.write() = difficulty;
    *world.gamerules.write() = gamerules.clone();
    *world.gamemode.write() = gamemode.clone();

    let chunks_loaded = load_chunks_bin(world, reg)?;
    let meta_loaded = load_nodemeta_bin(world)?;
    let players = load_players_bin(world)?;

    let matched_player = if let Some(tp) = target_player {
        let p = players
            .iter()
            .find(|p| p.name == tp.name)
            .or_else(|| players.first())
            .cloned();
        if let Some(ref found) = p {
            found.apply_to_object(tp);
        }
        p
    } else {
        players.into_iter().next()
    };

    Ok(SavedWorldState {
        seed,
        world_type,
        gamemode,
        difficulty,
        gamerules,
        player: matched_player,
        chunks_loaded,
        meta_loaded,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::NodeDef;

    #[test]
    fn world_save_load_roundtrip() {
        let dir = std::env::temp_dir().join("cubex_rs_persist_roundtrip");
        let _ = std::fs::remove_dir_all(&dir);

        let mut reg = EngineRegistries::new();
        let stone = reg.register_node(NodeDef {
            name: "mcl_core:stone".into(),
            ..Default::default()
        });
        let chest = reg.register_node(NodeDef {
            name: "mcl_chests:chest".into(),
            ..Default::default()
        });

        let world = WorldStore::new_with_options(
            &dir,
            987654,
            WorldGenType::Amplified,
            Difficulty::Hard,
            Gamerules {
                keep_inventory: true,
                do_daylight_cycle: false,
                pvp: false,
                fall_damage: true,
            },
            "Creative",
        );
        world.set_node(IVec3::new(20, 14, -2), stone, 15, 0);
        world.set_node(IVec3::new(4, 14, -2), chest, 15, 2);
        {
            let m = world.get_node_meta(IVec3::new(4, 14, -2));
            let mut mg = m.write();
            mg.fields.insert("infotext".into(), "Loot Chest".into());
            let mut list = vec![ItemStackData::default(); 27];
            list[0] = ItemStackData::from_spec("mcl_core:stone 32");
            mg.inventories.insert("main".into(), list);
        }
        world.node_timers.write().insert(
            IVec3::new(4, 14, -2),
            NodeTimerState {
                timeout: 10.0,
                elapsed: 4.5,
                started: true,
            },
        );

        let mut player = ObjectState::new_player("Hero");
        player.pos = Vec3::new(12.5, 42.0, -7.25);
        player.yaw = 1.25;
        player.pitch = -0.35;
        player.hp = 17;
        player.hunger = 14;
        player.xp_level = 9;
        player.xp_progress = 0.6;
        player.meta.write().inventories.get_mut("main").unwrap()[2] =
            ItemStackData::from_spec("mcl_tools:pick_diamond 1 512");

        save_world(&world, &reg, Some(&player)).expect("save world");

        let loaded_world = WorldStore::new(&dir, 987654);
        let mut loaded_player = ObjectState::new_player("Hero");
        let saved = load_world(&loaded_world, &reg, Some(&mut loaded_player)).expect("load world");

        assert_eq!(saved.seed, 987654);
        assert_eq!(saved.world_type, WorldGenType::Amplified);
        assert_eq!(saved.difficulty, Difficulty::Hard);
        assert_eq!(saved.gamemode, "Creative");
        assert!(saved.gamerules.keep_inventory);
        assert!(!saved.gamerules.do_daylight_cycle);
        assert_eq!(saved.chunks_loaded, 2);
        assert_eq!(saved.meta_loaded, 1);

        assert_eq!(loaded_world.get_node(IVec3::new(20, 14, -2)).0, stone);
        assert_eq!(loaded_world.get_node(IVec3::new(4, 14, -2)), (chest, 15, 2));
        let lm = loaded_world.get_node_meta(IVec3::new(4, 14, -2));
        assert_eq!(lm.read().fields.get("infotext").map(String::as_str), Some("Loot Chest"));
        assert_eq!(lm.read().inventories["main"][0].name, "mcl_core:stone");
        assert_eq!(lm.read().inventories["main"][0].count, 32);
        let lt = loaded_world.node_timers.read().get(&IVec3::new(4, 14, -2)).copied().unwrap();
        assert!((lt.elapsed - 4.5).abs() < 1e-6);

        assert!((loaded_player.pos - Vec3::new(12.5, 42.0, -7.25)).length() < 1e-5);
        assert!((loaded_player.yaw - 1.25).abs() < 1e-5);
        assert_eq!(loaded_player.hp, 17);
        assert_eq!(loaded_player.hunger, 14);
        assert_eq!(loaded_player.xp_level, 9);
        let slot2 = &loaded_player.meta.read().inventories["main"][2];
        assert_eq!(slot2.name, "mcl_tools:pick_diamond");
        assert_eq!(slot2.wear, 512);
    }

    #[test]
    fn survival_empty_inventory_roundtrip() {
        let dir = std::env::temp_dir().join("cubex_rs_persist_survival_empty");
        let _ = std::fs::remove_dir_all(&dir);

        let reg = EngineRegistries::new();
        let world = WorldStore::new_with_options(
            &dir,
            4242,
            WorldGenType::Default,
            Difficulty::Normal,
            Gamerules::default(),
            "Survival",
        );

        let player = ObjectState::new_player("Survivor");
        assert_eq!(player.meta.read().inventories["main"].len(), 36);
        assert!(player.meta.read().inventories["main"].iter().all(|s| s.is_empty()));

        save_world(&world, &reg, Some(&player)).expect("save survival world");

        let reloaded_world = WorldStore::new(&dir, 4242);
        let mut reloaded_player = ObjectState::new_player("Survivor");
        let saved = load_world(&reloaded_world, &reg, Some(&mut reloaded_player))
            .expect("load survival world");
        assert_eq!(saved.gamemode, "Survival");
        {
            let m = reloaded_player.meta.read();
            let main = &m.inventories["main"];
            assert_eq!(main.len(), 36);
            assert!(
                main.iter().all(|s| s.is_empty()),
                "Survival save/load must preserve empty inventory without injecting debug items"
            );
        }

        // Verify populate_creative_starter only populates when explicitly called
        reloaded_player.populate_creative_starter();
        let crea_main = &reloaded_player.meta.read().inventories["main"];
        assert!(!crea_main[0].is_empty());
        assert_eq!(crea_main[0].name, "mcl_core:dirt_with_grass");
        assert_eq!(crea_main[6].name, "mcl_tools:pick_diamond");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn two_player_persistence_roundtrip_keeps_inventories_and_positions_distinct() {
        let dir = std::env::temp_dir().join("cubex_rs_persist_two_players");
        let _ = std::fs::remove_dir_all(&dir);

        let mut reg = EngineRegistries::new();
        let stone = reg.register_node(NodeDef {
            name: "mcl_core:stone".into(),
            ..Default::default()
        });
        let _ = stone;

        let world = WorldStore::new_with_options(
            &dir,
            777,
            WorldGenType::Default,
            Difficulty::Normal,
            Gamerules::default(),
            "Survival",
        );

        let mut p1 = ObjectState::new_player("Vortex");
        p1.pos = Vec3::new(10.0, 65.0, -4.0);
        p1.meta.write().inventories.get_mut("main").unwrap()[0] =
            ItemStackData::from_spec("mcl_core:stone 12");

        let mut p2 = ObjectState::new_player("Player2");
        p2.pos = Vec3::new(14.5, 66.0, 2.5);
        p2.meta.write().inventories.get_mut("main").unwrap()[1] =
            ItemStackData::from_spec("mcl_core:stone 3");

        save_world_players(&world, &reg, &[&p1, &p2]).expect("save two players");

        let reloaded_world = WorldStore::new(&dir, 777);
        let mut r_p1 = ObjectState::new_player("Vortex");
        let _ = load_world(&reloaded_world, &reg, Some(&mut r_p1)).expect("load p1");
        let mut r_p2 = ObjectState::new_player("Player2");
        let loaded_p2 = load_player_by_name(&reloaded_world, &mut r_p2).expect("load p2 by name");

        assert!((loaded_p2.pos[0] - 14.5).abs() < 1e-5);
        assert_eq!(r_p1.meta.read().inventories["main"][0].count, 12);
        assert!(r_p1.meta.read().inventories["main"][1].is_empty());
        assert!(r_p2.meta.read().inventories["main"][0].is_empty());
        assert_eq!(r_p2.meta.read().inventories["main"][1].count, 3);

        let mut r_p3 = ObjectState::new_player("Player3");
        assert!(load_player_by_name(&reloaded_world, &mut r_p3).is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn multi_dimension_persistence_roundtrip() {
        let dir = std::env::temp_dir().join("cubex_rs_multi_dim_persist");
        let _ = std::fs::remove_dir_all(&dir);

        let mut reg = EngineRegistries::new();
        let stone = reg.register_node(NodeDef {
            name: "mcl_core:stone".into(),
            ..Default::default()
        });
        let netherrack = reg.register_node(NodeDef {
            name: "mcl_nether:netherrack".into(),
            ..Default::default()
        });

        let ow = WorldStore::new_for_dimension(
            &dir,
            5151,
            DimensionId::Overworld,
            WorldGenType::Default,
            Difficulty::Normal,
            Gamerules::default(),
            "Survival",
        );
        let nw = WorldStore::new_for_dimension(
            &dir,
            5151,
            DimensionId::Nether,
            WorldGenType::Default,
            Difficulty::Normal,
            Gamerules::default(),
            "Survival",
        );

        let shared_pos = IVec3::new(12, 48, -6);
        ow.set_node(shared_pos, stone, 15, 0);
        nw.set_node(shared_pos, netherrack, 0, 0);

        let mut p1 = ObjectState::new_player("PortalTraveler");
        p1.pos = Vec3::new(12.5, 50.0, -5.5);
        p1.dimension = DimensionId::Nether;

        save_world(&ow, &reg, Some(&p1)).expect("save overworld + player");
        save_dimension_chunks(&nw, &reg).expect("save nether chunks");

        assert!(dir.join("overworld/chunks.bin").is_file());
        assert!(dir.join("nether/chunks.bin").is_file());

        let ow2 = WorldStore::new_for_dimension(
            &dir,
            5151,
            DimensionId::Overworld,
            WorldGenType::Default,
            Difficulty::Normal,
            Gamerules::default(),
            "Survival",
        );
        let nw2 = WorldStore::new_for_dimension(
            &dir,
            5151,
            DimensionId::Nether,
            WorldGenType::Default,
            Difficulty::Normal,
            Gamerules::default(),
            "Survival",
        );

        let mut loaded_p1 = ObjectState::new_player("PortalTraveler");
        let saved = load_world(&ow2, &reg, Some(&mut loaded_p1)).expect("load overworld");
        let (n_chunks, _) = load_dimension_chunks(&nw2, &reg).expect("load nether");

        assert_eq!(saved.chunks_loaded, 1);
        assert_eq!(n_chunks, 1);
        assert_eq!(loaded_p1.dimension, DimensionId::Nether);
        assert_eq!(saved.player.unwrap().dimension, DimensionId::Nether);
        assert_eq!(ow2.get_node(shared_pos).0, stone);
        assert_eq!(nw2.get_node(shared_pos).0, netherrack);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
