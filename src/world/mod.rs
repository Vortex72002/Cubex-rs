pub mod biome;
pub mod dimension;
pub mod light;
pub mod mapgen;
pub mod mobs;
pub mod nether;
pub mod persistence;
pub mod portal;
pub mod schematic;
pub mod structures;

pub use biome::{biome_name_at, sample_biome, GenBiome, SEA_LEVEL};
pub use dimension::{scale_portal_coords, DimensionId};
pub use mapgen::chunk_y_range_for;
pub use mobs::{
    GuardianLaserBeam, MeleeNativeResult, MobAiState, MobManager, MobParticleEvent,
    MobParticleKind, MobPlayerDamage, MobPlayerInput, MobProjectile, MobTickEvents, NativeMob,
    NativeMobKind, ProjectileKind, MAX_ACTIVE_MOBS_PER_DIM,
};
pub use nether::{
    nether_biome_name_at, NetherBiome, NetherPalette, NETHER_CHUNK_Y_RANGE, NETHER_LAVA_LEVEL,
};
pub use portal::{PortalAxis, PortalFrame};
pub use structures::{
    ensure_structure_chest_loot, locate_nearest_structure, populate_structure_chests_for_chunk,
    structure_chest_at, LootTableKind, StructureAnchor, StructureKind, StructurePalette,
};

use glam::IVec3;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::registry::{CONTENT_AIR, CONTENT_IGNORE};

pub const CHUNK_SIZE: usize = 16;
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE; // 4096

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum WorldGenType {
    #[default]
    Default,
    Superflat,
    Amplified,
    Skyblock,
}

impl WorldGenType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Superflat => "superflat",
            Self::Amplified => "amplified",
            Self::Skyblock => "skyblock",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Default => "Default",
            Self::Superflat => "Superflat",
            Self::Amplified => "Amplified",
            Self::Skyblock => "Skyblock",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "superflat" | "flat" => Self::Superflat,
            "amplified" => Self::Amplified,
            "skyblock" => Self::Skyblock,
            _ => Self::Default,
        }
    }

    pub fn cycle(self) -> Self {
        match self {
            Self::Default => Self::Superflat,
            Self::Superflat => Self::Amplified,
            Self::Amplified => Self::Skyblock,
            Self::Skyblock => Self::Default,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Difficulty {
    Peaceful,
    Easy,
    #[default]
    Normal,
    Hard,
}

impl Difficulty {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Peaceful => "peaceful",
            Self::Easy => "easy",
            Self::Normal => "normal",
            Self::Hard => "hard",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Peaceful => "Peaceful",
            Self::Easy => "Easy",
            Self::Normal => "Normal",
            Self::Hard => "Hard",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "peaceful" => Self::Peaceful,
            "easy" => Self::Easy,
            "hard" => Self::Hard,
            _ => Self::Normal,
        }
    }

    pub fn cycle(self) -> Self {
        match self {
            Self::Peaceful => Self::Easy,
            Self::Easy => Self::Normal,
            Self::Normal => Self::Hard,
            Self::Hard => Self::Peaceful,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gamerules {
    pub keep_inventory: bool,
    pub do_daylight_cycle: bool,
    pub pvp: bool,
    pub fall_damage: bool,
}

impl Default for Gamerules {
    fn default() -> Self {
        Self {
            keep_inventory: true,
            do_daylight_cycle: true,
            pvp: true,
            fall_damage: true,
        }
    }
}

/// Unified metadata & inventory container for nodes, mod_storage, players, and itemstacks.
#[derive(Debug, Clone, Default)]
pub struct MetaDataState {
    pub fields: HashMap<String, String>,
    pub inventories: HashMap<String, Vec<ItemStackData>>,
}

pub type MetaStore = MetaDataState;

#[derive(Debug, Clone)]
pub struct ItemStackData {
    pub name: String,
    pub count: u16,
    pub wear: u16,
    pub meta: Arc<RwLock<MetaDataState>>,
}

impl Default for ItemStackData {
    fn default() -> Self {
        Self {
            name: String::new(),
            count: 0,
            wear: 0,
            meta: Arc::new(RwLock::new(MetaDataState::default())),
        }
    }
}

impl ItemStackData {
    pub fn from_spec(spec: &str) -> Self {
        let trimmed = spec.trim();
        if trimmed.is_empty() {
            return Self::default();
        }
        let mut parts = trimmed.split_whitespace();
        let name = parts.next().unwrap_or("").to_string();
        let count = parts.next().and_then(|s| s.parse::<u16>().ok()).unwrap_or(1);
        let wear = parts.next().and_then(|s| s.parse::<u16>().ok()).unwrap_or(0);
        if name.is_empty() || count == 0 {
            Self::default()
        } else {
            Self {
                name,
                count,
                wear,
                meta: Arc::new(RwLock::new(MetaDataState::default())),
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.name.is_empty() || self.count == 0
    }

    pub fn to_itemstring(&self) -> String {
        if self.is_empty() {
            String::new()
        } else if self.wear > 0 {
            format!("{} {} {}", self.name, self.count, self.wear)
        } else if self.count > 1 {
            format!("{} {}", self.name, self.count)
        } else {
            self.name.clone()
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NodeTimerState {
    pub timeout: f64,
    pub elapsed: f64,
    pub started: bool,
}

#[inline(always)]
pub const fn local_index(x: usize, y: usize, z: usize) -> usize {
    (y << 8) | (z << 4) | x
}

#[inline(always)]
pub fn world_to_chunk(pos: IVec3) -> (IVec3, usize, usize, usize) {
    let cx = pos.x.div_euclid(16);
    let cy = pos.y.div_euclid(16);
    let cz = pos.z.div_euclid(16);
    let lx = pos.x.rem_euclid(16) as usize;
    let ly = pos.y.rem_euclid(16) as usize;
    let lz = pos.z.rem_euclid(16) as usize;
    (IVec3::new(cx, cy, cz), lx, ly, lz)
}

#[derive(Clone)]
pub struct Chunk {
    pub pos: IVec3,
    pub nodes: [u16; CHUNK_VOLUME],
    pub param1: [u8; CHUNK_VOLUME], // lower 4 bits = daylight, upper 4 bits = artificial light
    pub param2: [u8; CHUNK_VOLUME],
    pub dirty_mesh: bool,
    pub modified: bool,
    pub meta: HashMap<usize, HashMap<String, String>>,
    pub timers: HashMap<usize, (f32, f32)>,
}

impl Chunk {
    pub fn new(pos: IVec3) -> Self {
        Self {
            pos,
            nodes: [CONTENT_AIR; CHUNK_VOLUME],
            param1: [0x0F; CHUNK_VOLUME],
            param2: [0; CHUNK_VOLUME],
            dirty_mesh: true,
            modified: false,
            meta: HashMap::new(),
            timers: HashMap::new(),
        }
    }

    #[inline(always)]
    pub fn get(&self, x: usize, y: usize, z: usize) -> (u16, u8, u8) {
        let i = local_index(x, y, z);
        (self.nodes[i], self.param1[i], self.param2[i])
    }

    #[inline(always)]
    pub fn set(&mut self, x: usize, y: usize, z: usize, id: u16, p1: u8, p2: u8) {
        let i = local_index(x, y, z);
        self.nodes[i] = id;
        self.param1[i] = p1;
        self.param2[i] = p2;
        self.dirty_mesh = true;
        self.modified = true;
    }
}

/// Zero-lock snapshot of a 16x16x16 chunk plus its 1-voxel boundary slices for parallel rayon meshing.
pub struct ChunkSnapshot {
    pub pos: IVec3,
    pub nodes: Vec<u16>,  // 18 x 18 x 18 padded array
    pub param1: Vec<u8>,  // 18 x 18 x 18 padded array
    pub param2: Vec<u8>,  // 18 x 18 x 18 padded array
}

impl ChunkSnapshot {
    #[inline(always)]
    pub const fn padded_idx(px: usize, py: usize, pz: usize) -> usize {
        py * 324 + pz * 18 + px
    }

    #[inline(always)]
    pub fn get(&self, lx: isize, ly: isize, lz: isize) -> (u16, u8, u8) {
        let idx = Self::padded_idx((lx + 1) as usize, (ly + 1) as usize, (lz + 1) as usize);
        (self.nodes[idx], self.param1[idx], self.param2[idx])
    }
}

#[derive(Debug, Clone)]
pub struct WorldInfo {
    pub name: String,
    pub path: PathBuf,
    pub gamemode: String,
    pub seed: u64,
    pub world_type: WorldGenType,
    pub difficulty: Difficulty,
    pub gamerules: Gamerules,
}

impl Default for WorldInfo {
    fn default() -> Self {
        Self {
            name: "Cubex_World".into(),
            path: PathBuf::new(),
            gamemode: "Survival".into(),
            seed: 1337,
            world_type: WorldGenType::Default,
            difficulty: Difficulty::Normal,
            gamerules: Gamerules::default(),
        }
    }
}

pub struct WorldStore {
    pub chunks: RwLock<HashMap<IVec3, Arc<RwLock<Chunk>>>>,
    pub dirty_queue: RwLock<HashSet<IVec3>>,
    pub priority_dirty_queue: RwLock<HashSet<IVec3>>,
    pub node_meta: RwLock<HashMap<IVec3, Arc<RwLock<MetaStore>>>>,
    pub node_timers: RwLock<HashMap<IVec3, NodeTimerState>>,
    pub world_path: PathBuf,
    pub seed: u64,
    pub dimension: DimensionId,
    pub world_type: RwLock<WorldGenType>,
    pub difficulty: RwLock<Difficulty>,
    pub gamerules: RwLock<Gamerules>,
    pub gamemode: RwLock<String>,
}

impl WorldStore {
    pub fn new(world_path: impl AsRef<Path>, seed: u64) -> Self {
        Self::new_with_options(
            world_path,
            seed,
            WorldGenType::Default,
            Difficulty::Normal,
            Gamerules::default(),
            "Survival",
        )
    }

    pub fn new_with_options(
        world_path: impl AsRef<Path>,
        seed: u64,
        world_type: WorldGenType,
        difficulty: Difficulty,
        gamerules: Gamerules,
        gamemode: impl Into<String>,
    ) -> Self {
        Self::new_for_dimension(
            world_path,
            seed,
            DimensionId::Overworld,
            world_type,
            difficulty,
            gamerules,
            gamemode,
        )
    }

    pub fn new_for_dimension(
        world_path: impl AsRef<Path>,
        seed: u64,
        dimension: DimensionId,
        world_type: WorldGenType,
        difficulty: Difficulty,
        gamerules: Gamerules,
        gamemode: impl Into<String>,
    ) -> Self {
        let wp = world_path.as_ref().to_path_buf();
        let _ = std::fs::create_dir_all(&wp);
        Self {
            chunks: RwLock::new(HashMap::with_capacity(1024)),
            dirty_queue: RwLock::new(HashSet::new()),
            priority_dirty_queue: RwLock::new(HashSet::new()),
            node_meta: RwLock::new(HashMap::new()),
            node_timers: RwLock::new(HashMap::new()),
            world_path: wp,
            seed,
            dimension,
            world_type: RwLock::new(world_type),
            difficulty: RwLock::new(difficulty),
            gamerules: RwLock::new(gamerules),
            gamemode: RwLock::new(gamemode.into()),
        }
    }

    pub fn has_skylight(&self) -> bool {
        self.dimension.has_skylight()
    }

    pub fn dimension_dir(&self) -> PathBuf {
        self.world_path.join(self.dimension.subdir())
    }

    pub fn get_node_meta(&self, pos: IVec3) -> Arc<RwLock<MetaStore>> {
        if let Some(m) = self.node_meta.read().get(&pos) {
            return m.clone();
        }
        self.node_meta
            .write()
            .entry(pos)
            .or_insert_with(|| Arc::new(RwLock::new(MetaStore::default())))
            .clone()
    }

    pub fn peek_node_meta(&self, pos: IVec3) -> Option<Arc<RwLock<MetaStore>>> {
        self.node_meta.read().get(&pos).cloned()
    }

    pub fn remove_node_meta(&self, pos: IVec3) {
        self.node_meta.write().remove(&pos);
        self.node_timers.write().remove(&pos);
    }

    pub fn get_chunk(&self, cpos: IVec3) -> Option<Arc<RwLock<Chunk>>> {
        self.chunks.read().get(&cpos).cloned()
    }

    pub fn insert_chunk(&self, chunk: Chunk) {
        let pos = chunk.pos;
        self.chunks.write().insert(pos, Arc::new(RwLock::new(chunk)));
        let mut dq = self.dirty_queue.write();
        dq.insert(pos);
        // Mark immediate neighbors dirty so chunk border faces update seamlessly
        for delta in [
            IVec3::X,
            IVec3::NEG_X,
            IVec3::Y,
            IVec3::NEG_Y,
            IVec3::Z,
            IVec3::NEG_Z,
        ] {
            let np = pos + delta;
            if self.chunks.read().contains_key(&np) {
                dq.insert(np);
            }
        }
    }

    pub fn mark_dirty_with_neighbors(&self, cpos: IVec3, lx: usize, ly: usize, lz: usize) {
        let mut dq = self.dirty_queue.write();
        dq.insert(cpos);
        if lx == 0 {
            dq.insert(cpos - IVec3::X);
        } else if lx == 15 {
            dq.insert(cpos + IVec3::X);
        }
        if ly == 0 {
            dq.insert(cpos - IVec3::Y);
        } else if ly == 15 {
            dq.insert(cpos + IVec3::Y);
        }
        if lz == 0 {
            dq.insert(cpos - IVec3::Z);
        } else if lz == 15 {
            dq.insert(cpos + IVec3::Z);
        }
    }

    /// Queue immediate same-frame remesh for player block break/place edits and any affected borders.
    pub fn mark_priority_dirty_with_neighbors(&self, pos: IVec3) {
        let (cpos, lx, ly, lz) = world_to_chunk(pos);
        let mut pq = self.priority_dirty_queue.write();
        pq.insert(cpos);
        if lx == 0 {
            pq.insert(cpos - IVec3::X);
        } else if lx == 15 {
            pq.insert(cpos + IVec3::X);
        }
        if ly == 0 {
            pq.insert(cpos - IVec3::Y);
        } else if ly == 15 {
            pq.insert(cpos + IVec3::Y);
        }
        if lz == 0 {
            pq.insert(cpos - IVec3::Z);
        } else if lz == 15 {
            pq.insert(cpos + IVec3::Z);
        }
    }

    pub fn take_priority_dirty_chunks(&self) -> Vec<IVec3> {
        let mut pq = self.priority_dirty_queue.write();
        if pq.is_empty() {
            return Vec::new();
        }
        let out: Vec<IVec3> = pq.drain().collect();
        let mut dq = self.dirty_queue.write();
        for cpos in &out {
            dq.remove(cpos);
        }
        out
    }

    pub fn get_node(&self, pos: IVec3) -> (u16, u8, u8) {
        let (cpos, lx, ly, lz) = world_to_chunk(pos);
        if let Some(ch) = self.get_chunk(cpos) {
            ch.read().get(lx, ly, lz)
        } else {
            (CONTENT_IGNORE, 0, 0)
        }
    }

    pub fn set_node(&self, pos: IVec3, id: u16, p1: u8, p2: u8) {
        let (cpos, lx, ly, lz) = world_to_chunk(pos);
        let chunk = {
            let existing = self.chunks.read().get(&cpos).cloned();
            if let Some(c) = existing {
                c
            } else {
                let mut map = self.chunks.write();
                map.entry(cpos)
                    .or_insert_with(|| {
                        let mut ch = Chunk::new(cpos);
                        if !self.has_skylight() {
                            ch.param1.fill(0);
                        }
                        Arc::new(RwLock::new(ch))
                    })
                    .clone()
            }
        };
        chunk.write().set(lx, ly, lz, id, p1, p2);
        self.mark_dirty_with_neighbors(cpos, lx, ly, lz);
    }

    pub fn swap_node(&self, pos: IVec3, id: u16, p2: u8) {
        let (cpos, lx, ly, lz) = world_to_chunk(pos);
        if let Some(ch) = self.get_chunk(cpos) {
            let mut g = ch.write();
            let i = local_index(lx, ly, lz);
            g.nodes[i] = id;
            g.param2[i] = p2;
            g.dirty_mesh = true;
            g.modified = true;
            drop(g);
            self.mark_dirty_with_neighbors(cpos, lx, ly, lz);
        }
    }

    pub fn find_nodes_in_area(&self, minp: IVec3, maxp: IVec3, target_ids: &HashSet<u16>) -> Vec<IVec3> {
        let mut out = Vec::new();
        let lo = minp.min(maxp);
        let hi = minp.max(maxp);
        for y in lo.y..=hi.y {
            for z in lo.z..=hi.z {
                for x in lo.x..=hi.x {
                    let p = IVec3::new(x, y, z);
                    let (id, _, _) = self.get_node(p);
                    if target_ids.contains(&id) {
                        out.push(p);
                    }
                }
            }
        }
        out
    }

    pub fn take_dirty_chunks(&self, max_count: usize) -> Vec<IVec3> {
        let mut dq = self.dirty_queue.write();
        let mut out = Vec::with_capacity(max_count.min(dq.len()));
        for &p in dq.iter().take(max_count) {
            out.push(p);
        }
        for p in &out {
            dq.remove(p);
        }
        out
    }

    pub fn snapshot_chunk(&self, cpos: IVec3) -> Option<ChunkSnapshot> {
        let map = self.chunks.read();
        let center = map.get(&cpos)?.clone();
        // Grab up to 26 neighbor chunk arcs quickly, or 6 face neighbors + center (sufficient for face culling + AO)
        let mut neigh = HashMap::with_capacity(27);
        for dy in -1..=1 {
            for dz in -1..=1 {
                for dx in -1..=1 {
                    let np = cpos + IVec3::new(dx, dy, dz);
                    if let Some(c) = map.get(&np) {
                        neigh.insert((dx, dy, dz), c.clone());
                    }
                }
            }
        }
        drop(map);

        let total = 18 * 18 * 18;
        let default_p1 = if self.has_skylight() { 0x0Fu8 } else { 0x00u8 };
        let mut nodes = vec![CONTENT_AIR; total];
        let mut param1 = vec![default_p1; total];
        let mut param2 = vec![0u8; total];

        // Fill interior 16x16x16 fast
        {
            let cg = center.read();
            for y in 0..16usize {
                for z in 0..16usize {
                    let src_start = local_index(0, y, z);
                    let dst_start = ChunkSnapshot::padded_idx(1, y + 1, z + 1);
                    nodes[dst_start..dst_start + 16].copy_from_slice(&cg.nodes[src_start..src_start + 16]);
                    param1[dst_start..dst_start + 16].copy_from_slice(&cg.param1[src_start..src_start + 16]);
                    param2[dst_start..dst_start + 16].copy_from_slice(&cg.param2[src_start..src_start + 16]);
                }
            }
        }

        // Fill 1-voxel border shell from neighboring chunks
        for pz in 0..18isize {
            for py in 0..18isize {
                for px in 0..18isize {
                    if (1..=16).contains(&px) && (1..=16).contains(&py) && (1..=16).contains(&pz) {
                        continue;
                    }
                    let lx = px - 1;
                    let ly = py - 1;
                    let lz = pz - 1;
                    let dx = lx.div_euclid(16) as i32;
                    let dy = ly.div_euclid(16) as i32;
                    let dz = lz.div_euclid(16) as i32;
                    if let Some(nc) = neigh.get(&(dx, dy, dz)) {
                        let g = nc.read();
                        let (id, p1, p2) = g.get(
                            lx.rem_euclid(16) as usize,
                            ly.rem_euclid(16) as usize,
                            lz.rem_euclid(16) as usize,
                        );
                        let pid = ChunkSnapshot::padded_idx(px as usize, py as usize, pz as usize);
                        nodes[pid] = id;
                        param1[pid] = p1;
                        param2[pid] = p2;
                    }
                }
            }
        }

        Some(ChunkSnapshot {
            pos: cpos,
            nodes,
            param1,
            param2,
        })
    }
}

pub fn list_worlds(worlds_dir: &Path) -> Vec<WorldInfo> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(worlds_dir) else {
        return out;
    };
    for ent in entries.flatten() {
        let path = ent.path();
        if !path.is_dir() {
            continue;
        }
        let name = ent.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        let mut gamemode = "Survival".to_string();
        let mut seed = 1337u64;
        let mut world_type = WorldGenType::Default;
        let mut difficulty = Difficulty::Normal;
        let mut gamerules = Gamerules::default();

        if let Ok(mt) = std::fs::read_to_string(path.join("world.mt")) {
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
        if let Ok(meta) = std::fs::read_to_string(path.join("map_meta.txt")) {
            for line in meta.lines() {
                if let Some((k, v)) = line.split_once('=') {
                    if k.trim() == "seed" {
                        seed = v.trim().parse::<u64>().unwrap_or(seed);
                    }
                }
            }
        }
        out.push(WorldInfo {
            name,
            path,
            gamemode,
            seed,
            world_type,
            difficulty,
            gamerules,
        });
    }
    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    out
}

pub fn create_world(worlds_dir: &Path, name: &str, creative: bool, seed: u64) -> anyhow::Result<WorldInfo> {
    create_world_ex(
        worlds_dir,
        name,
        if creative { "Creative" } else { "Survival" },
        seed,
        WorldGenType::Default,
        Difficulty::Normal,
        Gamerules::default(),
    )
}

/// Sanitize a user-provided world name into a filesystem-safe folder name.
pub fn sanitize_world_folder_name(name: &str) -> String {
    let safe: String = name
        .trim()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if safe.is_empty() {
        "Cubex_World".into()
    } else {
        safe
    }
}

/// Choose a unique default world name ("New World", "New World 2", ...) not yet present in `worlds_dir`.
pub fn next_default_world_name(worlds_dir: &Path) -> String {
    for idx in 1u32.. {
        let candidate = if idx == 1 {
            "New World".to_string()
        } else {
            format!("New World {idx}")
        };
        let safe = sanitize_world_folder_name(&candidate);
        if !worlds_dir.join(&safe).exists() && !worlds_dir.join(&candidate).exists() {
            return candidate;
        }
    }
    "New World".to_string()
}

pub fn create_world_ex(
    worlds_dir: &Path,
    name: &str,
    gamemode: &str,
    seed: u64,
    world_type: WorldGenType,
    difficulty: Difficulty,
    gamerules: Gamerules,
) -> anyhow::Result<WorldInfo> {
    std::fs::create_dir_all(worlds_dir)?;
    let base_safe = sanitize_world_folder_name(name);
    let mut suffix = 0u32;
    let (final_safe, path) = loop {
        let candidate = if suffix == 0 {
            base_safe.clone()
        } else {
            format!("{base_safe}_{}", suffix + 1)
        };
        let candidate_path = worlds_dir.join(&candidate);
        match std::fs::create_dir(&candidate_path) {
            Ok(()) => break (candidate, candidate_path),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                suffix = suffix.saturating_add(1);
            }
            Err(e) => return Err(e.into()),
        }
    };
    let creative = gamemode.eq_ignore_ascii_case("creative");
    let gm_label = if creative {
        "Creative"
    } else if gamemode.eq_ignore_ascii_case("spectator") {
        "Spectator"
    } else {
        "Survival"
    };
    std::fs::write(
        path.join("world.mt"),
        format!(
            "gameid = mineclonia\nbackend = sqlite3\ncreative_mode = {}\ngamemode = {}\nseed = {seed}\nworld_type = {}\ndifficulty = {}\nkeep_inventory = {}\ndo_daylight_cycle = {}\npvp = {}\nfall_damage = {}\nload_mod_cubex_bridge = true\n",
            if creative { "true" } else { "false" },
            gm_label,
            world_type.as_str(),
            difficulty.as_str(),
            gamerules.keep_inventory,
            gamerules.do_daylight_cycle,
            gamerules.pvp,
            gamerules.fall_damage,
        ),
    )?;
    std::fs::write(
        path.join("map_meta.txt"),
        format!("mg_name = v7\nseed = {seed}\n[end_of_params]\n"),
    )?;
    Ok(WorldInfo {
        name: final_safe,
        path,
        gamemode: gm_label.into(),
        seed,
        world_type,
        difficulty,
        gamerules,
    })
}

pub fn delete_world(world_path: &Path) -> anyhow::Result<()> {
    if world_path.exists() && world_path.is_dir() {
        std::fs::remove_dir_all(world_path)?;
    }
    Ok(())
}

#[cfg(test)]
mod world_creation_tests {
    use super::*;

    fn temp_worlds_dir(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "cubex_world_test_{tag}_{}_{}",
            std::process::id(),
            nanos
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn create_world_ex_never_reuses_existing_folder_or_data() {
        let dir = temp_worlds_dir("dedup");
        let w1 = create_world_ex(
            &dir,
            "My Adventure",
            "Survival",
            111,
            WorldGenType::Default,
            Difficulty::Normal,
            Gamerules::default(),
        )
        .unwrap();
        // Simulate saved chunk & player state inside first world
        std::fs::write(w1.path.join("chunks.bin"), b"OLD_CHUNKS").unwrap();
        std::fs::write(w1.path.join("players.bin"), b"OLD_PLAYERS").unwrap();

        let w2 = create_world_ex(
            &dir,
            "My Adventure",
            "Survival",
            222,
            WorldGenType::Default,
            Difficulty::Normal,
            Gamerules::default(),
        )
        .unwrap();
        let w3 = create_world_ex(
            &dir,
            "My Adventure",
            "Survival",
            333,
            WorldGenType::Default,
            Difficulty::Normal,
            Gamerules::default(),
        )
        .unwrap();

        assert_eq!(w1.name, "My_Adventure");
        assert_eq!(w2.name, "My_Adventure_2");
        assert_eq!(w3.name, "My_Adventure_3");
        assert_ne!(w1.path, w2.path);
        assert_ne!(w2.path, w3.path);
        assert!(!w2.path.join("chunks.bin").exists());
        assert!(!w2.path.join("players.bin").exists());
        assert!(!w3.path.join("chunks.bin").exists());
        assert!(!w3.path.join("players.bin").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn next_default_world_name_increments() {
        let dir = temp_worlds_dir("next_name");
        assert_eq!(next_default_world_name(&dir), "New World");
        create_world_ex(
            &dir,
            "New World",
            "Survival",
            1,
            WorldGenType::Default,
            Difficulty::Normal,
            Gamerules::default(),
        )
        .unwrap();
        assert_eq!(next_default_world_name(&dir), "New World 2");
        create_world_ex(
            &dir,
            "New World 2",
            "Survival",
            2,
            WorldGenType::Default,
            Difficulty::Normal,
            Gamerules::default(),
        )
        .unwrap();
        assert_eq!(next_default_world_name(&dir), "New World 3");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
