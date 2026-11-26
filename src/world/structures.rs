use glam::{IVec3, Vec3};

use crate::lua_api::userdata::ItemStackData;
use crate::registry::{EngineRegistries, CONTENT_AIR, CONTENT_UNKNOWN};
use crate::world::biome::{sample_biome, GenBiome, SEA_LEVEL};
use crate::world::dimension::DimensionId;
use crate::world::mapgen::terrain_height_for;
use crate::world::{local_index, Chunk, WorldGenType, WorldStore, CHUNK_SIZE};

const FORTRESS_CELL: i32 = 288;
const BASTION_CELL: i32 = 320;
const MONUMENT_CELL: i32 = 320;
const TEMPLE_CELL: i32 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StructureKind {
    NetherFortress,
    BastionRemnant,
    OceanMonument,
    DesertTemple,
}

impl StructureKind {
    pub const ALL: [StructureKind; 4] = [
        Self::NetherFortress,
        Self::BastionRemnant,
        Self::OceanMonument,
        Self::DesertTemple,
    ];

    pub fn dimension(self) -> DimensionId {
        match self {
            Self::NetherFortress | Self::BastionRemnant => DimensionId::Nether,
            Self::OceanMonument | Self::DesertTemple => DimensionId::Overworld,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::NetherFortress => "Nether Fortress",
            Self::BastionRemnant => "Bastion Remnant",
            Self::OceanMonument => "Ocean Monument",
            Self::DesertTemple => "Desert Temple",
        }
    }

    pub fn loot_table(self) -> LootTableKind {
        match self {
            Self::NetherFortress => LootTableKind::NetherFortress,
            Self::BastionRemnant => LootTableKind::BastionRemnant,
            Self::OceanMonument => LootTableKind::OceanMonument,
            Self::DesertTemple => LootTableKind::DesertTemple,
        }
    }

    pub fn cell_size(self) -> i32 {
        match self {
            Self::NetherFortress => FORTRESS_CELL,
            Self::BastionRemnant => BASTION_CELL,
            Self::OceanMonument => MONUMENT_CELL,
            Self::DesertTemple => TEMPLE_CELL,
        }
    }

    pub fn from_command_arg(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "fortress" | "nether_fortress" | "netherfortress" => Some(Self::NetherFortress),
            "bastion" | "bastion_remnant" | "bastionremnant" => Some(Self::BastionRemnant),
            "monument" | "ocean_monument" | "oceanmonument" => Some(Self::OceanMonument),
            "temple" | "desert_temple" | "deserttemple" | "pyramid" => Some(Self::DesertTemple),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LootTableKind {
    NetherFortress,
    BastionRemnant,
    OceanMonument,
    DesertTemple,
}

impl LootTableKind {
    pub const ALL: [LootTableKind; 4] = [
        Self::NetherFortress,
        Self::BastionRemnant,
        Self::OceanMonument,
        Self::DesertTemple,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::NetherFortress => "nether_fortress",
            Self::BastionRemnant => "bastion_remnant",
            Self::OceanMonument => "ocean_monument",
            Self::DesertTemple => "desert_temple",
        }
    }

    pub fn chest_title(self) -> &'static str {
        match self {
            Self::NetherFortress => "Nether Fortress Chest",
            Self::BastionRemnant => "Bastion Treasure Chest",
            Self::OceanMonument => "Monument Vault Chest",
            Self::DesertTemple => "Desert Temple Chest",
        }
    }
}

#[derive(Debug, Clone)]
pub struct StructureAnchor {
    pub kind: StructureKind,
    pub origin: IVec3,
    pub bbox_min: IVec3,
    pub bbox_max: IVec3,
    pub teleport_pos: Vec3,
    pub seed_hash: u64,
    pub chests: Vec<(IVec3, LootTableKind)>,
    pub spawners: Vec<(IVec3, &'static str)>,
}

#[derive(Debug, Clone, Default)]
pub struct StructurePalette {
    pub chest: u16,
    pub spawner: u16,
    // Nether Fortress
    pub nether_brick: u16,
    pub chiseled_nether_brick: u16,
    pub cracked_nether_brick: u16,
    pub nether_brick_fence: u16,
    pub soul_sand: u16,
    pub nether_wart: u16,
    pub lava_source: u16,
    pub glowstone: u16,
    // Bastion Remnant
    pub blackstone: u16,
    pub blackstone_polished: u16,
    pub blackstone_brick_polished: u16,
    pub blackstone_brick_cracked: u16,
    pub blackstone_chiseled: u16,
    pub blackstone_gilded: u16,
    pub gold_block: u16,
    pub crying_obsidian: u16,
    pub obsidian: u16,
    pub basalt: u16,
    // Ocean Monument
    pub prismarine: u16,
    pub prismarine_brick: u16,
    pub prismarine_dark: u16,
    pub sea_lantern: u16,
    pub sponge: u16,
    pub sponge_wet: u16,
    pub water_source: u16,
    // Desert Temple
    pub sandstone: u16,
    pub sandstone_smooth: u16,
    pub sandstone_carved: u16,
    pub sand: u16,
    pub terracotta_orange: u16,
    pub terracotta_blue: u16,
    pub stone: u16,
}

impl StructurePalette {
    pub fn from_registries(reg: &EngineRegistries) -> Self {
        let resolve = |names: &[&str], fallback: u16| -> u16 {
            for &nm in names {
                let id = reg.get_content_id(nm);
                if id != CONTENT_UNKNOWN {
                    return id;
                }
            }
            fallback
        };
        let fallback_solid = reg
            .nodes_by_id
            .iter()
            .flatten()
            .find(|d| d.drawtype.is_solid_cube() && d.name != "unknown")
            .map(|d| d.id)
            .unwrap_or(1);

        let stone = resolve(&["mcl_core:stone", "default:stone"], fallback_solid);
        let chest = resolve(
            &["mcl_chests:chest", "mcl_chests:chest_small", "default:chest"],
            stone,
        );
        let spawner = resolve(
            &["mcl_mobspawners:spawner", "mcl_core:obsidian"],
            stone,
        );
        let lava_source = resolve(
            &["mcl_nether:nether_lava_source", "mcl_core:lava_source"],
            CONTENT_AIR,
        );
        let glowstone = resolve(&["mcl_nether:glowstone"], stone);
        let nether_brick = resolve(
            &["mcl_nether:nether_brick", "mcl_nether:netherrack"],
            stone,
        );
        let chiseled_nether_brick =
            resolve(&["mcl_nether:chiseled_nether_brick"], nether_brick);
        let cracked_nether_brick =
            resolve(&["mcl_nether:cracked_nether_brick"], nether_brick);
        let nether_brick_fence = resolve(
            &[
                "mcl_fences:nether_brick_fence",
                "mcl_nether:nether_brick",
            ],
            nether_brick,
        );
        let soul_sand = resolve(&["mcl_nether:soul_sand"], nether_brick);
        let nether_wart = resolve(
            &["mcl_nether:nether_wart", "mcl_nether:nether_wart_2"],
            CONTENT_AIR,
        );

        let obsidian = resolve(&["mcl_core:obsidian"], stone);
        let blackstone = resolve(&["mcl_blackstone:blackstone"], stone);
        let blackstone_polished =
            resolve(&["mcl_blackstone:blackstone_polished"], blackstone);
        let blackstone_brick_polished = resolve(
            &["mcl_blackstone:blackstone_brick_polished"],
            blackstone_polished,
        );
        let blackstone_brick_cracked = resolve(
            &["mcl_blackstone:blackstone_brick_polished_cracked"],
            blackstone_brick_polished,
        );
        let blackstone_chiseled = resolve(
            &["mcl_blackstone:blackstone_chiseled_polished"],
            blackstone_polished,
        );
        let blackstone_gilded = resolve(
            &["mcl_blackstone:blackstone_gilded"],
            blackstone_brick_polished,
        );
        let gold_block = resolve(&["mcl_core:goldblock"], glowstone);
        let crying_obsidian = resolve(&["mcl_core:crying_obsidian"], obsidian);
        let basalt = resolve(&["mcl_blackstone:basalt"], blackstone);

        let water_source = resolve(
            &["mcl_core:water_source", "default:water_source"],
            CONTENT_AIR,
        );
        let prismarine = resolve(&["mcl_ocean:prismarine"], stone);
        let prismarine_brick =
            resolve(&["mcl_ocean:prismarine_brick"], prismarine);
        let prismarine_dark =
            resolve(&["mcl_ocean:prismarine_dark"], prismarine);
        let sea_lantern = resolve(&["mcl_ocean:sea_lantern"], glowstone);
        let sponge = resolve(&["mcl_sponges:sponge"], gold_block);
        let sponge_wet = resolve(&["mcl_sponges:sponge_wet"], sponge);

        let sand = resolve(&["mcl_core:sand", "default:sand"], stone);
        let sandstone = resolve(&["mcl_core:sandstone"], sand);
        let sandstone_smooth =
            resolve(&["mcl_core:sandstonesmooth", "mcl_core:sandstonesmooth2"], sandstone);
        let sandstone_carved = resolve(&["mcl_core:sandstonecarved"], sandstone);
        let clay_base = resolve(&["mcl_colorblocks:hardened_clay"], sandstone);
        let terracotta_orange = resolve(
            &[
                "mcl_colorblocks:hardened_clay_orange",
                "mcl_colorblocks:hardened_clay_terracotta",
            ],
            clay_base,
        );
        let terracotta_blue = resolve(
            &[
                "mcl_colorblocks:hardened_clay_blue",
                "mcl_colorblocks:hardened_clay_light_blue",
            ],
            clay_base,
        );

        Self {
            chest,
            spawner,
            nether_brick,
            chiseled_nether_brick,
            cracked_nether_brick,
            nether_brick_fence,
            soul_sand,
            nether_wart,
            lava_source,
            glowstone,
            blackstone,
            blackstone_polished,
            blackstone_brick_polished,
            blackstone_brick_cracked,
            blackstone_chiseled,
            blackstone_gilded,
            gold_block,
            crying_obsidian,
            obsidian,
            basalt,
            prismarine,
            prismarine_brick,
            prismarine_dark,
            sea_lantern,
            sponge,
            sponge_wet,
            water_source,
            sandstone,
            sandstone_smooth,
            sandstone_carved,
            sand,
            terracotta_orange,
            terracotta_blue,
            stone,
        }
    }
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[inline]
fn region_hash(rx: i32, rz: i32, seed: u64, salt: u64) -> u64 {
    let mixed = ((rx as i64 as u64) << 32) ^ (rz as i64 as u32 as u64) ^ seed ^ salt;
    splitmix64(mixed)
}

#[inline]
fn pos_hash(pos: IVec3, seed: u64) -> u64 {
    let p = ((pos.x as i64 as u64).wrapping_mul(73856093))
        ^ ((pos.y as i64 as u64).wrapping_mul(19349663))
        ^ ((pos.z as i64 as u64).wrapping_mul(83492791))
        ^ seed;
    splitmix64(p)
}

struct FastRng(u64);

impl FastRng {
    fn new(seed: u64) -> Self {
        Self(splitmix64(seed | 1))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = splitmix64(self.0);
        self.0
    }

    fn range_usize(&mut self, lo: usize, hi_incl: usize) -> usize {
        if hi_incl <= lo {
            return lo;
        }
        lo + (self.next_u64() as usize) % (hi_incl - lo + 1)
    }

    fn range_u16(&mut self, lo: u16, hi_incl: u16) -> u16 {
        if hi_incl <= lo {
            return lo;
        }
        lo + (self.next_u64() as u16) % (hi_incl - lo + 1)
    }
}

/// Compute the deterministic structure anchor inside grid cell `(rx, rz)`, if that cell hosts one.
pub fn structure_in_cell(
    kind: StructureKind,
    rx: i32,
    rz: i32,
    seed: u64,
    world_type: WorldGenType,
) -> Option<StructureAnchor> {
    match kind {
        StructureKind::NetherFortress => {
            let h = region_hash(rx, rz, seed, 0xF027_4E55);
            let ax = rx * FORTRESS_CELL + 64 + (h % 140) as i32;
            let az = rz * FORTRESS_CELL + 64 + ((h >> 16) % 140) as i32;
            let ay = 52 + ((h >> 32) % 10) as i32;
            let origin = IVec3::new(ax, ay, az);
            Some(StructureAnchor {
                kind,
                origin,
                bbox_min: origin + IVec3::new(-32, -36, -32),
                bbox_max: origin + IVec3::new(32, 18, 32),
                teleport_pos: origin.as_vec3() + Vec3::new(0.5, 1.15, 8.5),
                seed_hash: h,
                chests: vec![
                    (origin + IVec3::new(3, 7, -23), LootTableKind::NetherFortress),
                    (origin + IVec3::new(20, 1, -3), LootTableKind::NetherFortress),
                    (origin + IVec3::new(-24, 1, 2), LootTableKind::NetherFortress),
                ],
                spawners: vec![(origin + IVec3::new(0, 7, -22), "mobs_mc:blaze")],
            })
        }
        StructureKind::BastionRemnant => {
            let h = region_hash(rx, rz, seed, 0xBA57_104E);
            // Shifted half-cell away from Fortress anchors so Nether structures never collide
            let ax = rx * BASTION_CELL + 180 + (h % 110) as i32;
            let az = rz * BASTION_CELL + 180 + ((h >> 16) % 110) as i32;
            let ay = 38 + ((h >> 32) % 10) as i32;
            let origin = IVec3::new(ax, ay, az);
            Some(StructureAnchor {
                kind,
                origin,
                bbox_min: origin + IVec3::new(-20, -6, -20),
                bbox_max: origin + IVec3::new(20, 24, 20),
                teleport_pos: origin.as_vec3() + Vec3::new(0.5, 11.15, 14.5),
                seed_hash: h,
                chests: vec![
                    (origin + IVec3::new(-3, 1, -3), LootTableKind::BastionRemnant),
                    (origin + IVec3::new(12, 11, 0), LootTableKind::BastionRemnant),
                    (origin + IVec3::new(-14, 11, -12), LootTableKind::BastionRemnant),
                ],
                spawners: vec![(origin + IVec3::new(0, 6, 0), "mobs_mc:magma_cube")],
            })
        }
        StructureKind::OceanMonument => {
            if !matches!(world_type, WorldGenType::Default | WorldGenType::Amplified) {
                return None;
            }
            let h = region_hash(rx, rz, seed, 0x0C3A_4107);
            let mut best_x = rx * MONUMENT_CELL + 160;
            let mut best_z = rz * MONUMENT_CELL + 160;
            let mut best_h = i32::MAX;
            for gz in 0..5 {
                for gx in 0..5 {
                    let cx = rx * MONUMENT_CELL + 48 + gx * 56;
                    let cz = rz * MONUMENT_CELL + 48 + gz * 56;
                    let sh = terrain_height_for(cx, cz, seed, world_type);
                    if sh < best_h {
                        best_h = sh;
                        best_x = cx;
                        best_z = cz;
                    }
                }
            }
            let origin = IVec3::new(best_x, 38, best_z);
            Some(StructureAnchor {
                kind,
                origin,
                bbox_min: origin + IVec3::new(-26, -4, -26),
                bbox_max: origin + IVec3::new(26, 58, 26),
                teleport_pos: origin.as_vec3() + Vec3::new(0.5, 11.0, 23.5),
                seed_hash: h,
                chests: vec![
                    (origin + IVec3::new(0, 2, 5), LootTableKind::OceanMonument),
                    (origin + IVec3::new(8, 9, 0), LootTableKind::OceanMonument),
                ],
                spawners: Vec::new(),
            })
        }
        StructureKind::DesertTemple => {
            if !matches!(world_type, WorldGenType::Default | WorldGenType::Amplified) {
                return None;
            }
            let h = region_hash(rx, rz, seed, 0xDE5E_771E);
            let mut candidate: Option<(i32, i32, i32)> = None;
            let off_x = ((h % 3) as i32) * 12;
            let off_z = (((h >> 8) % 3) as i32) * 12;
            for gz in 0..5 {
                for gx in 0..5 {
                    let cx = rx * TEMPLE_CELL + 40 + off_x + gx * 42;
                    let cz = rz * TEMPLE_CELL + 40 + off_z + gz * 42;
                    if sample_biome(cx, cz, seed) == GenBiome::Desert {
                        let surf = terrain_height_for(cx, cz, seed, world_type);
                        if surf > SEA_LEVEL + 2 {
                            candidate = Some((cx, surf, cz));
                            break;
                        }
                    }
                }
                if candidate.is_some() {
                    break;
                }
            }
            let (ax, ay, az) = candidate?;
            let origin = IVec3::new(ax, ay, az);
            Some(StructureAnchor {
                kind,
                origin,
                bbox_min: origin + IVec3::new(-12, -12, -12),
                bbox_max: origin + IVec3::new(12, 16, 12),
                teleport_pos: origin.as_vec3() + Vec3::new(0.5, 1.15, 13.5),
                seed_hash: h,
                chests: vec![
                    (origin + IVec3::new(0, -9, -2), LootTableKind::DesertTemple),
                    (origin + IVec3::new(0, -9, 2), LootTableKind::DesertTemple),
                    (origin + IVec3::new(-2, -9, 0), LootTableKind::DesertTemple),
                    (origin + IVec3::new(2, -9, 0), LootTableKind::DesertTemple),
                ],
                spawners: Vec::new(),
            })
        }
    }
}

/// Locate the closest deterministic structure of `kind` to horizontal world coordinates `near_xz`.
pub fn locate_nearest_structure(
    kind: StructureKind,
    seed: u64,
    world_type: WorldGenType,
    near_xz: (i32, i32),
) -> Option<StructureAnchor> {
    let cell = kind.cell_size();
    let crx = near_xz.0.div_euclid(cell);
    let crz = near_xz.1.div_euclid(cell);

    let mut best: Option<(i64, StructureAnchor)> = None;
    for ring in 0i32..=14 {
        for dz in -ring..=ring {
            for dx in -ring..=ring {
                if ring > 0 && dx.abs() != ring && dz.abs() != ring {
                    continue;
                }
                if let Some(anchor) =
                    structure_in_cell(kind, crx + dx, crz + dz, seed, world_type)
                {
                    let ddx = (anchor.origin.x - near_xz.0) as i64;
                    let ddz = (anchor.origin.z - near_xz.1) as i64;
                    let d2 = ddx * ddx + ddz * ddz;
                    if best.as_ref().is_none_or(|(bd2, _)| d2 < *bd2) {
                        best = Some((d2, anchor));
                    }
                }
            }
        }
        if ring >= 1 && best.is_some() {
            break;
        }
    }
    best.map(|(_, a)| a)
}

/// Collect all structure anchors whose bounding box overlaps chunk `cpos`.
pub fn structures_overlapping_chunk(
    cpos: IVec3,
    dim: DimensionId,
    seed: u64,
    world_type: WorldGenType,
) -> Vec<StructureAnchor> {
    let wmin = cpos * 16;
    let wmax = wmin + IVec3::splat(15);
    let mut out = Vec::new();
    let kinds: &[StructureKind] = if dim.is_nether() {
        &[StructureKind::NetherFortress, StructureKind::BastionRemnant]
    } else {
        &[StructureKind::OceanMonument, StructureKind::DesertTemple]
    };

    const PAD: i32 = 40;
    for &kind in kinds {
        let cell = kind.cell_size();
        let min_rx = (wmin.x - PAD).div_euclid(cell);
        let max_rx = (wmax.x + PAD).div_euclid(cell);
        let min_rz = (wmin.z - PAD).div_euclid(cell);
        let max_rz = (wmax.z + PAD).div_euclid(cell);
        for rz in min_rz..=max_rz {
            for rx in min_rx..=max_rx {
                if let Some(a) = structure_in_cell(kind, rx, rz, seed, world_type) {
                    if a.bbox_min.x <= wmax.x
                        && a.bbox_max.x >= wmin.x
                        && a.bbox_min.y <= wmax.y
                        && a.bbox_max.y >= wmin.y
                        && a.bbox_min.z <= wmax.z
                        && a.bbox_max.z >= wmin.z
                    {
                        out.push(a);
                    }
                }
            }
        }
    }
    out
}

/// Identify if world coordinate `pos` is a generated structure loot chest.
pub fn structure_chest_at(
    pos: IVec3,
    dim: DimensionId,
    seed: u64,
    world_type: WorldGenType,
) -> Option<LootTableKind> {
    let (cpos, _, _, _) = crate::world::world_to_chunk(pos);
    for a in structures_overlapping_chunk(cpos, dim, seed, world_type) {
        for &(c_pos, table) in &a.chests {
            if c_pos == pos {
                return Some(table);
            }
        }
    }
    None
}

// =========================================================================
// Chunk Voxel Stamping Helpers
// =========================================================================

struct ChunkWriteCtx<'a> {
    chunk: &'a mut Chunk,
    ox: i32,
    oy: i32,
    oz: i32,
}

impl<'a> ChunkWriteCtx<'a> {
    fn new(chunk: &'a mut Chunk) -> Self {
        let o = chunk.pos * 16;
        Self {
            chunk,
            ox: o.x,
            oy: o.y,
            oz: o.z,
        }
    }

    #[inline]
    fn contains(&self, wx: i32, wy: i32, wz: i32) -> bool {
        wx >= self.ox
            && wx < self.ox + CHUNK_SIZE as i32
            && wy >= self.oy
            && wy < self.oy + CHUNK_SIZE as i32
            && wz >= self.oz
            && wz < self.oz + CHUNK_SIZE as i32
    }

    #[inline]
    fn set(&mut self, wx: i32, wy: i32, wz: i32, id: u16, p2: u8) {
        if !self.contains(wx, wy, wz) {
            return;
        }
        let lx = (wx - self.ox) as usize;
        let ly = (wy - self.oy) as usize;
        let lz = (wz - self.oz) as usize;
        let idx = local_index(lx, ly, lz);
        self.chunk.nodes[idx] = id;
        self.chunk.param2[idx] = p2;
    }

    fn fill_box(&mut self, min: IVec3, max: IVec3, id: u16, p2: u8) {
        let x0 = min.x.max(self.ox);
        let x1 = max.x.min(self.ox + 15);
        let y0 = min.y.max(self.oy);
        let y1 = max.y.min(self.oy + 15);
        let z0 = min.z.max(self.oz);
        let z1 = max.z.min(self.oz + 15);
        if x0 > x1 || y0 > y1 || z0 > z1 {
            return;
        }
        for wy in y0..=y1 {
            let ly = (wy - self.oy) as usize;
            for wz in z0..=z1 {
                let lz = (wz - self.oz) as usize;
                for wx in x0..=x1 {
                    let lx = (wx - self.ox) as usize;
                    let idx = local_index(lx, ly, lz);
                    self.chunk.nodes[idx] = id;
                    self.chunk.param2[idx] = p2;
                }
            }
        }
    }
}

/// Stamp Nether Fortresses and Bastion Remnants intersecting `chunk`.
pub fn stamp_nether_structures_into_chunk(chunk: &mut Chunk, seed: u64, pal: &StructurePalette) {
    let anchors =
        structures_overlapping_chunk(chunk.pos, DimensionId::Nether, seed, WorldGenType::Default);
    if anchors.is_empty() {
        return;
    }
    let mut ctx = ChunkWriteCtx::new(chunk);
    for a in &anchors {
        match a.kind {
            StructureKind::NetherFortress => stamp_nether_fortress(&mut ctx, a.origin, pal),
            StructureKind::BastionRemnant => {
                stamp_bastion_remnant(&mut ctx, a.origin, a.seed_hash, pal);
            }
            _ => {}
        }
    }
}

/// Stamp Ocean Monuments and Desert Temples intersecting `chunk`.
pub fn stamp_overworld_structures_into_chunk(
    chunk: &mut Chunk,
    seed: u64,
    world_type: WorldGenType,
    pal: &StructurePalette,
) {
    if !matches!(world_type, WorldGenType::Default | WorldGenType::Amplified) {
        return;
    }
    let anchors =
        structures_overlapping_chunk(chunk.pos, DimensionId::Overworld, seed, world_type);
    if anchors.is_empty() {
        return;
    }
    let mut ctx = ChunkWriteCtx::new(chunk);
    for a in &anchors {
        match a.kind {
            StructureKind::OceanMonument => stamp_ocean_monument(&mut ctx, a.origin, pal),
            StructureKind::DesertTemple => stamp_desert_temple(&mut ctx, a.origin, pal),
            _ => {}
        }
    }
}

fn stamp_nether_fortress(ctx: &mut ChunkWriteCtx<'_>, o: IVec3, pal: &StructurePalette) {
    let nb = pal.nether_brick;
    let cnb = pal.chiseled_nether_brick;
    let crnb = pal.cracked_nether_brick;
    let fence = pal.nether_brick_fence;

    // 1. Deep 3x3 Nether Brick support columns down to y = 18
    for (px, pz) in [(-18, 0), (0, 0), (18, 0), (0, -18), (0, 18), (0, -22)] {
        let min_y = 18.max(ctx.oy);
        let max_y = (o.y - 1).min(ctx.oy + 15);
        if min_y <= max_y {
            ctx.fill_box(
                IVec3::new(o.x + px - 1, min_y, o.z + pz - 1),
                IVec3::new(o.x + px + 1, max_y, o.z + pz + 1),
                nb,
                0,
            );
        }
    }

    // 2. Main N-S and E-W bridge viaducts (`-31..=31`, width 5, clear airspace above)
    // E-W bridge
    ctx.fill_box(o + IVec3::new(-31, 0, -2), o + IVec3::new(31, 0, 2), nb, 0);
    ctx.fill_box(o + IVec3::new(-31, 1, -2), o + IVec3::new(31, 5, 2), CONTENT_AIR, 0);
    ctx.fill_box(o + IVec3::new(-31, -1, -1), o + IVec3::new(31, -1, 1), nb, 0);
    // N-S bridge
    ctx.fill_box(o + IVec3::new(-2, 0, -31), o + IVec3::new(2, 0, 31), nb, 0);
    ctx.fill_box(o + IVec3::new(-2, 1, -31), o + IVec3::new(2, 5, 31), CONTENT_AIR, 0);
    ctx.fill_box(o + IVec3::new(-1, -1, -31), o + IVec3::new(1, -1, 31), nb, 0);

    // Brick fence parapets along bridges (outside central crossroads)
    for d in -30i32..=30 {
        if d.abs() >= 6 {
            ctx.set(o.x + d, o.y + 1, o.z - 2, fence, 0);
            ctx.set(o.x + d, o.y + 1, o.z + 2, fence, 0);
            ctx.set(o.x - 2, o.y + 1, o.z + d, fence, 0);
            ctx.set(o.x + 2, o.y + 1, o.z + d, fence, 0);
        }
        // Rhythmic bridge archways every 12 blocks
        if d.abs() >= 10 && (d.abs() - 10) % 12 == 0 {
            for dy in 1..=4 {
                ctx.set(o.x + d, o.y + dy, o.z - 2, nb, 0);
                ctx.set(o.x + d, o.y + dy, o.z + 2, nb, 0);
                ctx.set(o.x - 2, o.y + dy, o.z + d, nb, 0);
                ctx.set(o.x + 2, o.y + dy, o.z + d, nb, 0);
            }
            ctx.fill_box(o + IVec3::new(d, 4, -2), o + IVec3::new(d, 4, 2), nb, 0);
            ctx.fill_box(o + IVec3::new(-2, 4, d), o + IVec3::new(2, 4, d), nb, 0);
        }
    }

    // 3. Covered Central Crossroads Hall (`[-5..=5, 0..=8, -5..=5]`)
    ctx.fill_box(o + IVec3::new(-5, 0, -5), o + IVec3::new(5, 0, 5), cnb, 0);
    ctx.fill_box(o + IVec3::new(-4, 1, -4), o + IVec3::new(4, 7, 4), CONTENT_AIR, 0);
    // Corner pillars + perimeter walls
    for (sx, sz) in [(-5, -5), (5, -5), (-5, 5), (5, 5)] {
        ctx.fill_box(o + IVec3::new(sx, 1, sz), o + IVec3::new(sx, 8, sz), nb, 0);
    }
    ctx.fill_box(o + IVec3::new(-5, 7, -5), o + IVec3::new(5, 7, 5), nb, 0);
    ctx.fill_box(o + IVec3::new(-5, 8, -5), o + IVec3::new(5, 8, 5), nb, 0);
    // Cardinal fence arch openings & glowing chandelier
    for s in [-4, -3, 3, 4] {
        ctx.fill_box(o + IVec3::new(s, 1, -5), o + IVec3::new(s, 6, -5), nb, 0);
        ctx.fill_box(o + IVec3::new(s, 1, 5), o + IVec3::new(s, 6, 5), nb, 0);
        ctx.fill_box(o + IVec3::new(-5, 1, s), o + IVec3::new(-5, 6, s), nb, 0);
        ctx.fill_box(o + IVec3::new(5, 1, s), o + IVec3::new(5, 6, s), nb, 0);
    }
    ctx.fill_box(o + IVec3::new(-2, 5, -5), o + IVec3::new(2, 6, -5), nb, 0);
    ctx.fill_box(o + IVec3::new(-2, 5, 5), o + IVec3::new(2, 6, 5), nb, 0);
    ctx.fill_box(o + IVec3::new(-5, 5, -2), o + IVec3::new(-5, 6, 2), nb, 0);
    ctx.fill_box(o + IVec3::new(5, 5, -2), o + IVec3::new(5, 6, 2), nb, 0);
    ctx.set(o.x, o.y + 6, o.z, pal.glowstone, 0);

    // 4. Blaze Spawner Balcony Room (`o + (0, 6, -22)`) + stepped staircase (`z = -12..=-17`)
    for step in 0..6 {
        let z = -12 - step;
        let dy = 1 + step;
        ctx.fill_box(
            o + IVec3::new(-1, 0, z),
            o + IVec3::new(1, dy, z),
            crnb,
            0,
        );
        ctx.fill_box(
            o + IVec3::new(-1, dy + 1, z),
            o + IVec3::new(1, dy + 4, z),
            CONTENT_AIR,
            0,
        );
        ctx.set(o.x - 2, o.y + dy + 1, o.z + z, fence, 0);
        ctx.set(o.x + 2, o.y + dy + 1, o.z + z, fence, 0);
    }
    // Balcony platform `[-5..=5, 6, -27..=-17]`
    ctx.fill_box(o + IVec3::new(-5, 6, -27), o + IVec3::new(5, 6, -17), nb, 0);
    ctx.fill_box(o + IVec3::new(-4, 7, -26), o + IVec3::new(4, 12, -18), CONTENT_AIR, 0);
    // Fence parapet around balcony except stair entry (`x=-1..=1, z=-17`)
    for z in -27..=-17 {
        ctx.set(o.x - 5, o.y + 7, o.z + z, fence, 0);
        ctx.set(o.x + 5, o.y + 7, o.z + z, fence, 0);
    }
    for x in -5..=5 {
        ctx.set(o.x + x, o.y + 7, o.z - 27, fence, 0);
        if x.abs() >= 2 {
            ctx.set(o.x + x, o.y + 7, o.z - 17, fence, 0);
        }
    }
    // Central spawner pedestal + Blaze Spawner + glowing lava sconces + Chest #1
    ctx.fill_box(o + IVec3::new(-1, 6, -23), o + IVec3::new(1, 6, -21), cnb, 0);
    ctx.set(o.x, o.y + 7, o.z - 22, pal.spawner, 0);
    ctx.set(o.x - 4, o.y + 7, o.z - 22, pal.glowstone, 0);
    ctx.set(o.x + 4, o.y + 7, o.z - 22, pal.glowstone, 0);
    ctx.set(o.x + 3, o.y + 7, o.z - 23, pal.chest, 2);

    // 5. Nether Wart Soul-Sand Garden Room (`o + (16, 0, 0)`)
    ctx.fill_box(o + IVec3::new(11, 0, -5), o + IVec3::new(22, 0, 5), nb, 0);
    ctx.fill_box(o + IVec3::new(12, 1, -4), o + IVec3::new(21, 6, 4), CONTENT_AIR, 0);
    ctx.fill_box(o + IVec3::new(11, 7, -5), o + IVec3::new(22, 7, 5), nb, 0);
    for z in [-5, 5] {
        ctx.fill_box(o + IVec3::new(11, 1, z), o + IVec3::new(22, 6, z), nb, 0);
    }
    for dy in 1..=6 {
        ctx.set(o.x + 22, o.y + dy, o.z - 3, nb, 0);
        ctx.set(o.x + 22, o.y + dy, o.z + 3, nb, 0);
    }
    // Two soul-sand beds with Nether Wart!
    ctx.fill_box(o + IVec3::new(13, 0, -4), o + IVec3::new(19, 0, -3), pal.soul_sand, 0);
    ctx.fill_box(o + IVec3::new(13, 0, 3), o + IVec3::new(19, 0, 4), pal.soul_sand, 0);
    if pal.nether_wart != CONTENT_AIR {
        for wx in 13..=19 {
            for wz in [-4, -3, 3, 4] {
                ctx.set(o.x + wx, o.y + 1, o.z + wz, pal.nether_wart, 0);
            }
        }
    }
    // Brick staircase steps in garden center + glowstone + Chest #2
    for s in 0..3 {
        ctx.fill_box(
            o + IVec3::new(18 + s, 1, -1),
            o + IVec3::new(18 + s, 1 + s, 1),
            nb,
            0,
        );
    }
    ctx.set(o.x + 16, o.y + 6, o.z, pal.glowstone, 0);
    ctx.set(o.x + 20, o.y + 1, o.z - 3, pal.chest, 3);

    // 6. West Corridor Treasure Pavilion + Chest #3
    ctx.fill_box(o + IVec3::new(-27, 0, -4), o + IVec3::new(-20, 0, 4), cnb, 0);
    ctx.fill_box(o + IVec3::new(-26, 1, -3), o + IVec3::new(-21, 5, 3), CONTENT_AIR, 0);
    ctx.fill_box(o + IVec3::new(-27, 6, -4), o + IVec3::new(-20, 6, 4), nb, 0);
    for (px, pz) in [(-27, -4), (-27, 4), (-20, -4), (-20, 4)] {
        ctx.fill_box(o + IVec3::new(px, 1, pz), o + IVec3::new(px, 6, pz), nb, 0);
    }
    ctx.set(o.x - 24, o.y + 5, o.z, pal.glowstone, 0);
    ctx.set(o.x - 24, o.y + 1, o.z + 2, pal.chest, 0);
}

fn stamp_bastion_remnant(
    ctx: &mut ChunkWriteCtx<'_>,
    o: IVec3,
    seed_hash: u64,
    pal: &StructurePalette,
) {
    let bs = pal.blackstone;
    let bsp = pal.blackstone_polished;
    let bsbp = pal.blackstone_brick_polished;
    let bsc = pal.blackstone_brick_cracked;
    let bsg = pal.blackstone_gilded;
    let bsch = pal.blackstone_chiseled;

    // 1. Solid blackstone base foundation (`-18..=18`, `dy = -5..=-3`) + lava moat (`dy = -2`)
    ctx.fill_box(o + IVec3::new(-18, -5, -18), o + IVec3::new(18, -3, 18), bs, 0);
    ctx.fill_box(o + IVec3::new(-14, -2, -14), o + IVec3::new(14, -2, 14), pal.lava_source, 0);
    // Clear interior fortress courtyard air (`dy = -1..=20`)
    ctx.fill_box(o + IVec3::new(-14, -1, -14), o + IVec3::new(14, 20, 14), CONTENT_AIR, 0);

    // 2. Ruined outer Blackstone rampart walls (`[-18..=18, 0..=16, -18..=18]`, thickness 3)
    for dz in -18i32..=18 {
        for dx in -18i32..=18 {
            let on_rim = dx.abs() >= 15 || dz.abs() >= 15;
            if !on_rim {
                continue;
            }
            let wx = o.x + dx;
            let wz = o.z + dz;
            // Ruined weathered top height 12..=17
            let corner = dx.abs() >= 14 && dz.abs() >= 14;
            let h_mod = splitmix64(
                seed_hash ^ ((dx as i64 as u64) << 16) ^ (dz as i64 as u32 as u64),
            );
            let wall_h = if corner {
                16 + (h_mod % 3) as i32
            } else if (dx.abs() + dz.abs()) % 5 == 0 {
                12 + (h_mod % 3) as i32
            } else {
                14 + (h_mod % 2) as i32
            };
            // South front gate opening (`dz >= 15 && dx.abs() <= 2`)
            if dz >= 15 && dx.abs() <= 2 {
                ctx.fill_box(IVec3::new(wx, o.y - 2, wz), IVec3::new(wx, o.y, wz), bsp, 0);
                ctx.fill_box(IVec3::new(wx, o.y + 1, wz), IVec3::new(wx, o.y + 6, wz), CONTENT_AIR, 0);
                ctx.fill_box(IVec3::new(wx, o.y + 7, wz), IVec3::new(wx, o.y + wall_h.min(13), wz), bsbp, 0);
                continue;
            }
            for dy in -2..=wall_h {
                let wy = o.y + dy;
                if !ctx.contains(wx, wy, wz) {
                    continue;
                }
                let blk = if dy == 9 || dy == 10 {
                    bsp
                } else if dy == 12 && corner {
                    bsg
                } else if (h_mod ^ (dy as u64 * 17)) % 19 == 0 {
                    pal.crying_obsidian
                } else if (h_mod ^ (dy as u64 * 11)) % 7 == 0 {
                    bsc
                } else if dy % 4 == 0 {
                    bsbp
                } else {
                    bs
                };
                ctx.set(wx, wy, wz, blk, 0);
            }
        }
    }

    // 3. Upper perimeter rampart walkway (`dy = 10`, inner overhang ring `11..=15`)
    for dz in -15i32..=15 {
        for dx in -15i32..=15 {
            if dx.abs() >= 11 || dz.abs() >= 11 {
                ctx.set(o.x + dx, o.y + 10, o.z + dz, bsp, 0);
                ctx.fill_box(
                    o + IVec3::new(dx, 11, dz),
                    o + IVec3::new(dx, 15, dz),
                    CONTENT_AIR,
                    0,
                );
            }
        }
    }

    // 4. Bridges spanning across lava moat into the Central Treasure Keep (`dy = 0`)
    ctx.fill_box(o + IVec3::new(-2, 0, -14), o + IVec3::new(2, 0, 14), bsbp, 0);
    ctx.fill_box(o + IVec3::new(-14, 0, -2), o + IVec3::new(14, 0, 2), bsbp, 0);

    // 5. Central Treasure Keep (`[-6..=6, 0..=11, -6..=6]`) with Gilded Blackstone pillars,
    // Gold Block hoard, Magma Cube Spawner, and Treasure Chests!
    ctx.fill_box(o + IVec3::new(-6, 0, -6), o + IVec3::new(6, 0, 6), bsbp, 0);
    ctx.fill_box(o + IVec3::new(-5, 1, -5), o + IVec3::new(5, 9, 5), CONTENT_AIR, 0);
    // Corner gilded columns & chiseled trim
    for (sx, sz) in [(-6, -6), (6, -6), (-6, 6), (6, 6)] {
        ctx.fill_box(o + IVec3::new(sx, 1, sz), o + IVec3::new(sx, 11, sz), bsg, 0);
    }
    ctx.fill_box(o + IVec3::new(-6, 5, -6), o + IVec3::new(6, 5, -6), bsch, 0);
    ctx.fill_box(o + IVec3::new(-6, 5, 6), o + IVec3::new(6, 5, 6), bsch, 0);
    // Gold block pile inside keep (`[0..=2, 1..=2, -2..=0]`)
    ctx.fill_box(o + IVec3::new(1, 1, -2), o + IVec3::new(2, 2, -1), pal.gold_block, 0);
    ctx.set(o.x + 1, o.y + 3, o.z - 1, pal.gold_block, 0);
    ctx.set(o.x - 1, o.y + 1, o.z, bsg, 0);
    // Spawner pedestal at `o + (0, 6, 0)` + Magma Cube spawner
    ctx.fill_box(o + IVec3::new(-1, 5, -1), o + IVec3::new(1, 5, 1), bsp, 0);
    ctx.set(o.x, o.y + 6, o.z, pal.spawner, 0);
    ctx.set(o.x - 4, o.y + 1, o.z - 4, pal.glowstone, 0);
    ctx.set(o.x + 4, o.y + 1, o.z + 4, pal.glowstone, 0);

    // 3 Bastion Treasure Chests
    ctx.set(o.x - 3, o.y + 1, o.z - 3, pal.chest, 0);
    ctx.set(o.x + 12, o.y + 11, o.z, pal.chest, 3);
    ctx.set(o.x - 14, o.y + 11, o.z - 12, pal.chest, 2);
}

fn stamp_ocean_monument(ctx: &mut ChunkWriteCtx<'_>, o: IVec3, pal: &StructurePalette) {
    let pr = pal.prismarine;
    let prb = pal.prismarine_brick;
    let prd = pal.prismarine_dark;
    let lan = pal.sea_lantern;
    let wat = pal.water_source;

    // 1. Deep underwater basin excavation around monument (`r = 25`, `y = 36..=92`)
    for dz in -25i32..=25 {
        for dx in -25i32..=25 {
            if dx.abs() + dz.abs() > 38 {
                continue;
            }
            let wx = o.x + dx;
            let wz = o.z + dz;
            if wx < ctx.ox || wx > ctx.ox + 15 || wz < ctx.oz || wz > ctx.oz + 15 {
                continue;
            }
            let floor_y = (o.y - 2).max(34);
            ctx.fill_box(
                IVec3::new(wx, floor_y, wz),
                IVec3::new(wx, o.y - 1, wz),
                pal.stone,
                0,
            );
            let min_w = o.y.max(ctx.oy);
            let max_w = SEA_LEVEL.min(ctx.oy + 15);
            if min_w <= max_w {
                ctx.fill_box(IVec3::new(wx, min_w, wz), IVec3::new(wx, max_w, wz), wat, 0);
            }
            let min_a = (SEA_LEVEL + 1).max(ctx.oy);
            let max_a = 95.min(ctx.oy + 15);
            if min_a <= max_a {
                ctx.fill_box(
                    IVec3::new(wx, min_a, wz),
                    IVec3::new(wx, max_a, wz),
                    CONTENT_AIR,
                    0,
                );
            }
        }
    }

    // 2. Monument Foundation Slab (`[-19..=19, -1..=1, -19..=19]`) + corner legs
    ctx.fill_box(o + IVec3::new(-19, -1, -19), o + IVec3::new(19, 0, 19), prd, 0);
    ctx.fill_box(o + IVec3::new(-19, 1, -19), o + IVec3::new(19, 1, 19), pr, 0);

    // 3. Stepped Prismarine Pyramid Exterior + flooded interior chambers
    // Tier 1: `[-17..=17, 2..=7, -17..=17]`
    // Tier 2: `[-13..=13, 8..=13, -13..=13]`
    // Tier 3: `[-9..=9, 14..=18, -9..=9]`
    // Crown:  `[-5..=5, 19..=20, -5..=5]`
    let tiers = [
        (17i32, 2i32, 7i32),
        (13, 8, 13),
        (9, 14, 18),
        (5, 19, 20),
    ];
    for (half, y0, y1) in tiers {
        ctx.fill_box(
            o + IVec3::new(-half, y0, -half),
            o + IVec3::new(half, y1, half),
            prb,
            0,
        );
        // Carve interior water chamber inside each tier
        if half > 5 {
            ctx.fill_box(
                o + IVec3::new(-half + 2, y0, -half + 2),
                o + IVec3::new(half - 2, y1 - 1, half - 2),
                wat,
                0,
            );
        }
        // Dark prismarine corners + glowing Sea Lantern beacons
        for (sx, sz) in [(-half, -half), (half, -half), (-half, half), (half, half)] {
            ctx.fill_box(o + IVec3::new(sx, y0, sz), o + IVec3::new(sx, y1, sz), prd, 0);
            ctx.set(o.x + sx, o.y + y1, o.z + sz, lan, 0);
        }
        // Horizontal prismarine trim along upper edge
        for s in -half..=half {
            if s.abs() % 4 == 0 {
                ctx.set(o.x + s, o.y + y0 + 2, o.z + half, pr, 0);
                ctx.set(o.x + s, o.y + y0 + 2, o.z - half, pr, 0);
                ctx.set(o.x + half, o.y + y0 + 2, o.z + s, pr, 0);
                ctx.set(o.x - half, o.y + y0 + 2, o.z + s, pr, 0);
            }
        }
    }
    ctx.set(o.x, o.y + 20, o.z, lan, 0);

    // 4. Grand Front Entrance Archway (`+Z` side: `z = 14..=19`)
    ctx.fill_box(o + IVec3::new(-2, 2, 14), o + IVec3::new(2, 6, 19), wat, 0);
    ctx.fill_box(o + IVec3::new(-3, 2, 18), o + IVec3::new(-3, 6, 19), prd, 0);
    ctx.fill_box(o + IVec3::new(3, 2, 18), o + IVec3::new(3, 6, 19), prd, 0);
    ctx.fill_box(o + IVec3::new(-3, 7, 17), o + IVec3::new(3, 7, 19), prd, 0);
    ctx.set(o.x - 3, o.y + 7, o.z + 19, lan, 0);
    ctx.set(o.x + 3, o.y + 7, o.z + 19, lan, 0);
    ctx.set(o.x, o.y + 7, o.z + 19, lan, 0);

    // 5. Central Dark-Prismarine Gold Vault (`[-4..=4, 1..=7, -4..=4]`) + 8 Gold Blocks
    ctx.fill_box(o + IVec3::new(-4, 2, -4), o + IVec3::new(4, 7, 4), prd, 0);
    ctx.fill_box(o + IVec3::new(-3, 2, -3), o + IVec3::new(3, 6, 3), wat, 0);
    ctx.fill_box(o + IVec3::new(-1, 2, -1), o + IVec3::new(0, 3, 0), pal.gold_block, 0);
    // Open vault doorway on `+Z` + glowing Sea Lanterns inside vault
    ctx.fill_box(o + IVec3::new(-1, 2, 4), o + IVec3::new(1, 5, 4), wat, 0);
    ctx.set(o.x - 3, o.y + 5, o.z - 3, lan, 0);
    ctx.set(o.x + 3, o.y + 5, o.z - 3, lan, 0);
    ctx.set(o.x - 3, o.y + 5, o.z + 3, lan, 0);
    ctx.set(o.x + 3, o.y + 5, o.z + 3, lan, 0);

    // 6. Upper Sponge Room (`[4..=11, 8..=13, -4..=4]`) filled with wet/dry sponges!
    ctx.fill_box(o + IVec3::new(4, 8, -4), o + IVec3::new(11, 8, 4), prd, 0);
    ctx.fill_box(o + IVec3::new(5, 9, -3), o + IVec3::new(10, 12, 3), wat, 0);
    for (sx, sy, sz, dry) in [
        (6, 12, -2, false),
        (7, 12, -2, true),
        (8, 12, -2, false),
        (6, 12, 1, true),
        (7, 12, 1, false),
        (8, 12, 1, true),
        (9, 11, 0, false),
        (9, 12, 0, true),
        (10, 12, -1, false),
        (10, 12, 2, false),
    ] {
        ctx.set(
            o.x + sx,
            o.y + sy,
            o.z + sz,
            if dry { pal.sponge } else { pal.sponge_wet },
            0,
        );
    }
    ctx.set(o.x + 10, o.y + 11, o.z + 3, lan, 0);

    // 2 Submerged Monument Vault Chests
    ctx.set(o.x, o.y + 2, o.z + 5, pal.chest, 0);
    ctx.set(o.x + 8, o.y + 9, o.z, pal.chest, 3);
}

fn stamp_desert_temple(ctx: &mut ChunkWriteCtx<'_>, o: IVec3, pal: &StructurePalette) {
    let ss = pal.sandstone;
    let ssm = pal.sandstone_smooth;
    let ssc = pal.sandstone_carved;
    let to = pal.terracotta_orange;
    let tb = pal.terracotta_blue;

    // Clear entrance courtyard apron in front (`+Z`: `z = 11..=14`) above ground so doors aren't buried
    ctx.fill_box(o + IVec3::new(-3, 1, 11), o + IVec3::new(3, 6, 14), CONTENT_AIR, 0);

    // 1. Solid sandstone foundation skirt (`[-10..=10, -5..=0, -10..=10]`)
    ctx.fill_box(o + IVec3::new(-10, -5, -10), o + IVec3::new(10, 0, 10), ss, 0);

    // 2. Stepped Pyramid Shell (`dy = 1..=9`) + hollow central sanctuary + skyline oculus
    for dy in 1..=9 {
        let half = (10 - (dy - 1)).max(2);
        ctx.fill_box(o + IVec3::new(-half, dy, -half), o + IVec3::new(half, dy, half), ss, 0);
        if half > 2 {
            let wall = if dy % 3 == 0 { ssc } else { ssm };
            ctx.fill_box(
                o + IVec3::new(-half + 1, dy, -half + 1),
                o + IVec3::new(half - 1, dy, half - 1),
                CONTENT_AIR,
                0,
            );
            // Smooth corner trim
            for (sx, sz) in [(-half, -half), (half, -half), (-half, half), (half, half)] {
                ctx.set(o.x + sx, o.y + dy, o.z + sz, wall, 0);
            }
        }
    }
    // Roof cap & central 3x3 skylight oculus (`dy = 9..=10`)
    ctx.fill_box(o + IVec3::new(-3, 9, -3), o + IVec3::new(3, 9, 3), ssm, 0);
    ctx.fill_box(o + IVec3::new(-1, 9, -1), o + IVec3::new(1, 10, 1), CONTENT_AIR, 0);

    // 3. Open Front Doorway (`+Z` face: `x = -1..=1, dy = 1..=4, z = 6..=10`)
    ctx.fill_box(o + IVec3::new(-1, 1, 6), o + IVec3::new(1, 4, 10), CONTENT_AIR, 0);
    ctx.fill_box(o + IVec3::new(-2, 5, 9), o + IVec3::new(2, 5, 10), ssc, 0);
    ctx.set(o.x - 2, o.y + 4, o.z + 10, to, 0);
    ctx.set(o.x + 2, o.y + 4, o.z + 10, to, 0);

    // 4. Twin Front Flanking Towers (`x in [-10..=-6]` and `[6..=10]`, `z in [6..=10]`, `dy = 1..=13`)
    for side_x in [-8i32, 8i32] {
        ctx.fill_box(
            o + IVec3::new(side_x - 2, 1, 6),
            o + IVec3::new(side_x + 2, 12, 10),
            ss,
            0,
        );
        ctx.fill_box(
            o + IVec3::new(side_x - 1, 2, 7),
            o + IVec3::new(side_x + 1, 11, 9),
            CONTENT_AIR,
            0,
        );
        // Crown trim + crenellated corner merlons (`dy = 12..=13`)
        ctx.fill_box(
            o + IVec3::new(side_x - 2, 12, 6),
            o + IVec3::new(side_x + 2, 12, 10),
            ssm,
            0,
        );
        for (cx, cz) in [
            (side_x - 2, 6),
            (side_x + 2, 6),
            (side_x - 2, 10),
            (side_x + 2, 10),
        ] {
            ctx.set(o.x + cx, o.y + 13, o.z + cz, ssc, 0);
        }
        // Iconic Orange Terracotta Ankh/Pillar pattern on front `+Z` face (`z = 10`)
        ctx.fill_box(
            o + IVec3::new(side_x, 3, 10),
            o + IVec3::new(side_x, 8, 10),
            to,
            0,
        );
        ctx.fill_box(
            o + IVec3::new(side_x - 1, 6, 10),
            o + IVec3::new(side_x + 1, 6, 10),
            to,
            0,
        );
        ctx.set(o.x + side_x, o.y + 9, o.z + 10, to, 0);
    }

    // 5. Sanctuary Floor Terracotta Diamond Pattern (`dy = 0`)
    for dz in -4i32..=4 {
        for dx in -4i32..=4 {
            let manhattan = dx.abs() + dz.abs();
            let blk = if dx == 0 && dz == 0 {
                tb
            } else if manhattan == 3 || (dx.abs() == 3 && dz.abs() <= 1) || (dz.abs() == 3 && dx.abs() <= 1) {
                to
            } else {
                ssm
            };
            ctx.set(o.x + dx, o.y, o.z + dz, blk, 0);
        }
    }

    // 6. Hidden Lower Treasure Room (`[-3..=3, -10..=-1, -3..=3]`)
    ctx.fill_box(o + IVec3::new(-4, -11, -4), o + IVec3::new(4, -1, 4), ssm, 0);
    // Vertical access drop shaft directly below center blue terracotta tile (`dy = -9..=-1`)
    ctx.fill_box(o + IVec3::new(0, -9, 0), o + IVec3::new(0, -1, 0), CONTENT_AIR, 0);
    // Hollow vault room (`dy = -9..=-6`)
    ctx.fill_box(o + IVec3::new(-2, -9, -2), o + IVec3::new(2, -6, 2), CONTENT_AIR, 0);
    // Carved corner pillars
    for (px, pz) in [(-2, -2), (2, -2), (-2, 2), (2, 2)] {
        ctx.fill_box(o + IVec3::new(px, -9, pz), o + IVec3::new(px, -6, pz), ssc, 0);
    }
    // Vault chest alcoves & 4 Temple Treasure Chests
    ctx.set(o.x, o.y - 9, o.z - 2, pal.chest, 0);
    ctx.set(o.x, o.y - 9, o.z + 2, pal.chest, 2);
    ctx.set(o.x - 2, o.y - 9, o.z, pal.chest, 1);
    ctx.set(o.x + 2, o.y - 9, o.z, pal.chest, 3);
}

// =========================================================================
// Deterministic 27-Slot Structure Loot Table Generator
// =========================================================================

struct LootEntry<'a> {
    candidates: &'a [&'a str],
    min_count: u16,
    max_count: u16,
    weight: u32,
    enchantment_label: Option<&'a str>,
}

pub fn loot_pool_candidates(kind: LootTableKind) -> Vec<&'static [&'static str]> {
    loot_pool(kind).iter().map(|e| e.candidates).collect()
}

fn loot_pool(kind: LootTableKind) -> &'static [LootEntry<'static>] {
    match kind {
        LootTableKind::NetherFortress => &[
            LootEntry { candidates: &["mcl_core:diamond"], min_count: 1, max_count: 3, weight: 14, enchantment_label: None },
            LootEntry { candidates: &["mcl_core:gold_ingot"], min_count: 2, max_count: 6, weight: 20, enchantment_label: None },
            LootEntry { candidates: &["mcl_mobitems:blaze_rod"], min_count: 1, max_count: 3, weight: 16, enchantment_label: None },
            LootEntry { candidates: &["mcl_core:obsidian"], min_count: 2, max_count: 5, weight: 16, enchantment_label: None },
            LootEntry { candidates: &["mcl_nether:nether_wart"], min_count: 3, max_count: 7, weight: 18, enchantment_label: None },
            LootEntry { candidates: &["mcl_core:apple_gold"], min_count: 1, max_count: 2, weight: 10, enchantment_label: None },
            LootEntry { candidates: &["mcl_tools:sword_gold"], min_count: 1, max_count: 1, weight: 10, enchantment_label: Some("Fire Aspect I, Unbreaking II") },
            LootEntry { candidates: &["mcl_armor:chestplate_gold"], min_count: 1, max_count: 1, weight: 8, enchantment_label: Some("Protection II, Unbreaking II") },
            LootEntry { candidates: &["mcl_tools:sword_diamond"], min_count: 1, max_count: 1, weight: 6, enchantment_label: Some("Sharpness II") },
        ],
        LootTableKind::BastionRemnant => &[
            LootEntry { candidates: &["mcl_core:gold_ingot"], min_count: 4, max_count: 9, weight: 22, enchantment_label: None },
            LootEntry { candidates: &["mcl_core:goldblock"], min_count: 1, max_count: 2, weight: 12, enchantment_label: None },
            LootEntry { candidates: &["mcl_core:diamond"], min_count: 1, max_count: 4, weight: 12, enchantment_label: None },
            LootEntry { candidates: &["mcl_core:obsidian"], min_count: 3, max_count: 6, weight: 16, enchantment_label: None },
            LootEntry { candidates: &["mcl_core:crying_obsidian"], min_count: 1, max_count: 3, weight: 12, enchantment_label: None },
            LootEntry { candidates: &["mcl_throwing:ender_pearl"], min_count: 1, max_count: 3, weight: 12, enchantment_label: None },
            LootEntry { candidates: &["mcl_core:apple_gold_enchanted", "mcl_core:apple_gold"], min_count: 1, max_count: 1, weight: 7, enchantment_label: None },
            LootEntry { candidates: &["mcl_tools:pick_diamond"], min_count: 1, max_count: 1, weight: 9, enchantment_label: Some("Efficiency III, Unbreaking III") },
            LootEntry { candidates: &["mcl_armor:helmet_diamond"], min_count: 1, max_count: 1, weight: 7, enchantment_label: Some("Protection III") },
            LootEntry { candidates: &["mcl_armor:boots_gold"], min_count: 1, max_count: 1, weight: 10, enchantment_label: Some("Soul Speed II, Feather Falling II") },
        ],
        LootTableKind::OceanMonument => &[
            LootEntry { candidates: &["mcl_core:diamond"], min_count: 2, max_count: 4, weight: 16, enchantment_label: None },
            LootEntry { candidates: &["mcl_core:gold_ingot"], min_count: 4, max_count: 8, weight: 20, enchantment_label: None },
            LootEntry { candidates: &["mcl_ocean:sea_lantern"], min_count: 2, max_count: 6, weight: 16, enchantment_label: None },
            LootEntry { candidates: &["mcl_sponges:sponge", "mcl_sponges:sponge_wet"], min_count: 2, max_count: 4, weight: 16, enchantment_label: None },
            LootEntry { candidates: &["mcl_ocean:prismarine_shard"], min_count: 3, max_count: 8, weight: 14, enchantment_label: None },
            LootEntry { candidates: &["mcl_core:apple_gold"], min_count: 1, max_count: 2, weight: 10, enchantment_label: None },
            LootEntry { candidates: &["mcl_throwing:ender_pearl"], min_count: 1, max_count: 3, weight: 10, enchantment_label: None },
            LootEntry { candidates: &["mcl_tools:pick_diamond"], min_count: 1, max_count: 1, weight: 8, enchantment_label: Some("Efficiency IV, Fortune II") },
            LootEntry { candidates: &["mcl_armor:helmet_diamond"], min_count: 1, max_count: 1, weight: 8, enchantment_label: Some("Respiration III, Aqua Affinity") },
        ],
        LootTableKind::DesertTemple => &[
            LootEntry { candidates: &["mcl_core:diamond"], min_count: 1, max_count: 3, weight: 16, enchantment_label: None },
            LootEntry { candidates: &["mcl_core:gold_ingot"], min_count: 2, max_count: 7, weight: 20, enchantment_label: None },
            LootEntry { candidates: &["mcl_core:apple_gold"], min_count: 1, max_count: 2, weight: 12, enchantment_label: None },
            LootEntry { candidates: &["mcl_core:apple_gold_enchanted", "mcl_core:apple_gold"], min_count: 1, max_count: 1, weight: 7, enchantment_label: None },
            LootEntry { candidates: &["mcl_mobitems:gunpowder"], min_count: 3, max_count: 8, weight: 16, enchantment_label: None },
            LootEntry { candidates: &["mcl_mobitems:bone"], min_count: 3, max_count: 6, weight: 16, enchantment_label: None },
            LootEntry { candidates: &["mcl_throwing:ender_pearl"], min_count: 1, max_count: 2, weight: 10, enchantment_label: None },
            LootEntry { candidates: &["mcl_core:obsidian"], min_count: 2, max_count: 4, weight: 12, enchantment_label: None },
            LootEntry { candidates: &["mcl_tools:sword_diamond"], min_count: 1, max_count: 1, weight: 8, enchantment_label: Some("Sharpness III, Unbreaking II") },
            LootEntry { candidates: &["mcl_armor:chestplate_diamond"], min_count: 1, max_count: 1, weight: 7, enchantment_label: Some("Protection III, Unbreaking II") },
        ],
    }
}

fn resolve_registered_item<'a>(reg: &EngineRegistries, candidates: &[&'a str]) -> Option<String> {
    for &c in candidates {
        let resolved = reg.resolve_alias(c);
        if reg.get_item_def(resolved).is_some() {
            return Some(resolved.to_string());
        }
    }
    None
}

fn friendly_item_display_name(item_id: &str, reg: &EngineRegistries) -> String {
    if let Some(def) = reg.get_item_def(item_id) {
        if !def.description.trim().is_empty() {
            return def.description.trim().to_string();
        }
    }
    let short = item_id.rsplit(':').next().unwrap_or(item_id);
    short
        .split('_')
        .filter(|s| !s.is_empty())
        .map(|w| {
            let mut ch = w.chars();
            match ch.next() {
                Some(first) => format!("{}{}", first.to_ascii_uppercase(), ch.as_str()),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Deterministically populate a structure chest's 27-slot `"main"` inventory (idempotent).
pub fn populate_structure_chest(
    world: &WorldStore,
    pos: IVec3,
    kind: LootTableKind,
    seed: u64,
    reg: &EngineRegistries,
) -> bool {
    let meta = world.get_node_meta(pos);
    {
        let g = meta.read();
        if g.fields.get("cubex_loot_init").map(String::as_str) == Some("1") {
            return false;
        }
    }
    let mut g = meta.write();
    if g.fields.get("cubex_loot_init").map(String::as_str) == Some("1") {
        return false;
    }

    let inv = g
        .inventories
        .entry("main".into())
        .or_insert_with(|| vec![ItemStackData::default(); 27]);
    if inv.len() < 27 {
        inv.resize(27, ItemStackData::default());
    }

    let pool = loot_pool(kind);
    let total_weight: u32 = pool.iter().map(|e| e.weight).sum::<u32>().max(1);
    let mut rng = FastRng::new(pos_hash(pos, seed ^ 0xC4E5_7001));

    // Guarantee at least one signature enchanted/diamond/gold item + 6..=10 scattered slots
    let num_stacks = rng.range_usize(7, 10);
    let mut used_slots = [false; 27];
    let mut placed = 0usize;

    for attempt in 0..48 {
        if placed >= num_stacks {
            break;
        }
        let slot_idx = if attempt == 0 {
            13
        } else {
            (rng.next_u64() as usize) % 27
        };
        if used_slots[slot_idx] {
            continue;
        }
        used_slots[slot_idx] = true;

        let entry = if attempt == 0 {
            // Pick a guaranteed high-tier item first
            pool.iter()
                .find(|e| e.enchantment_label.is_some())
                .unwrap_or(&pool[0])
        } else {
            let mut roll = (rng.next_u64() as u32) % total_weight;
            let mut picked = &pool[0];
            for e in pool {
                if roll < e.weight {
                    picked = e;
                    break;
                }
                roll -= e.weight;
            }
            picked
        };

        let item_id = resolve_registered_item(reg, entry.candidates)
            .unwrap_or_else(|| entry.candidates[0].to_string());
        let count = rng.range_u16(entry.min_count, entry.max_count);
        let stack = ItemStackData {
            name: item_id.clone(),
            count,
            wear: 0,
            ..Default::default()
        };
        if let Some(ench) = entry.enchantment_label {
            let base_label = friendly_item_display_name(&item_id, reg);
            let mut sm = stack.meta.write();
            sm.fields.insert("cubex_enchanted".into(), "1".into());
            sm.fields
                .insert("mcl_enchanting:enchantments".into(), ench.into());
            sm.fields
                .insert("description".into(), format!("{base_label}\n§b{ench}\n§8{item_id}"));
        }
        inv[slot_idx] = stack;
        placed += 1;
    }

    g.fields.insert("cubex_loot_init".into(), "1".into());
    g.fields
        .insert("cubex_loot_table".into(), kind.as_str().into());
    g.fields.insert("infotext".into(), kind.chest_title().into());
    true
}

/// Populate all structure chests belonging to chunk `cpos` right after chunk insertion.
pub fn populate_structure_chests_for_chunk(
    world: &WorldStore,
    cpos: IVec3,
    seed: u64,
    world_type: WorldGenType,
    reg: &EngineRegistries,
) {
    let anchors = structures_overlapping_chunk(cpos, world.dimension, seed, world_type);
    if anchors.is_empty() {
        return;
    }
    let wmin = cpos * 16;
    let wmax = wmin + IVec3::splat(15);
    for a in anchors {
        for (chest_pos, table) in a.chests {
            if chest_pos.x >= wmin.x
                && chest_pos.x <= wmax.x
                && chest_pos.y >= wmin.y
                && chest_pos.y <= wmax.y
                && chest_pos.z >= wmin.z
                && chest_pos.z <= wmax.z
            {
                let _ = populate_structure_chest(world, chest_pos, table, seed, reg);
            }
        }
    }
}

/// Ensure chest at `pos` has a 27-slot `"main"` inventory (populating structure loot if applicable).
pub fn ensure_structure_chest_loot(
    world: &WorldStore,
    pos: IVec3,
    seed: u64,
    world_type: WorldGenType,
    reg: &EngineRegistries,
) {
    if let Some(table) = structure_chest_at(pos, world.dimension, seed, world_type) {
        let _ = populate_structure_chest(world, pos, table, seed, reg);
    }
    let meta = world.get_node_meta(pos);
    let mut g = meta.write();
    let inv = g
        .inventories
        .entry("main".into())
        .or_insert_with(|| vec![ItemStackData::default(); 27]);
    if inv.len() < 27 {
        inv.resize(27, ItemStackData::default());
    }
    g.fields
        .entry("infotext".into())
        .or_insert_with(|| "Chest".into());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{DrawType, ItemDef, NodeDef};
    use crate::world::nether::{generate_nether_chunk, NetherPalette};

    fn make_structure_test_reg() -> EngineRegistries {
        let mut reg = EngineRegistries::new();
        for (name, dt) in [
            ("mcl_core:stone", DrawType::Normal),
            ("mcl_core:dirt", DrawType::Normal),
            ("mcl_core:dirt_with_grass", DrawType::Normal),
            ("mcl_core:sand", DrawType::Normal),
            ("mcl_core:sandstone", DrawType::Normal),
            ("mcl_core:sandstonesmooth", DrawType::Normal),
            ("mcl_core:sandstonecarved", DrawType::Normal),
            ("mcl_colorblocks:hardened_clay", DrawType::Normal),
            ("mcl_colorblocks:hardened_clay_orange", DrawType::Normal),
            ("mcl_colorblocks:hardened_clay_blue", DrawType::Normal),
            ("mcl_core:water_source", DrawType::Liquid),
            ("mcl_core:goldblock", DrawType::Normal),
            ("mcl_core:obsidian", DrawType::Normal),
            ("mcl_core:crying_obsidian", DrawType::Normal),
            ("mcl_ocean:prismarine", DrawType::Normal),
            ("mcl_ocean:prismarine_brick", DrawType::Normal),
            ("mcl_ocean:prismarine_dark", DrawType::Normal),
            ("mcl_ocean:sea_lantern", DrawType::Normal),
            ("mcl_sponges:sponge", DrawType::Normal),
            ("mcl_sponges:sponge_wet", DrawType::Normal),
            ("mcl_nether:netherrack", DrawType::Normal),
            ("mcl_nether:nether_brick", DrawType::Normal),
            ("mcl_nether:chiseled_nether_brick", DrawType::Normal),
            ("mcl_nether:cracked_nether_brick", DrawType::Normal),
            ("mcl_fences:nether_brick_fence", DrawType::Fencelike),
            ("mcl_nether:soul_sand", DrawType::Normal),
            ("mcl_nether:nether_wart", DrawType::Plantlike),
            ("mcl_nether:nether_lava_source", DrawType::Liquid),
            ("mcl_nether:glowstone", DrawType::Normal),
            ("mcl_blackstone:blackstone", DrawType::Normal),
            ("mcl_blackstone:blackstone_polished", DrawType::Normal),
            ("mcl_blackstone:blackstone_brick_polished", DrawType::Normal),
            ("mcl_blackstone:blackstone_brick_polished_cracked", DrawType::Normal),
            ("mcl_blackstone:blackstone_chiseled_polished", DrawType::Normal),
            ("mcl_blackstone:blackstone_gilded", DrawType::Normal),
            ("mcl_blackstone:basalt", DrawType::Normal),
            ("mcl_mobspawners:spawner", DrawType::Allfaces),
            ("mcl_chests:chest", DrawType::Mesh),
        ] {
            reg.register_node(NodeDef {
                name: name.into(),
                drawtype: dt,
                walkable: dt != DrawType::Liquid && dt != DrawType::Plantlike,
                ..Default::default()
            });
        }
        for item in [
            "mcl_core:diamond",
            "mcl_core:gold_ingot",
            "mcl_core:goldblock",
            "mcl_core:obsidian",
            "mcl_core:crying_obsidian",
            "mcl_core:apple_gold",
            "mcl_core:apple_gold_enchanted",
            "mcl_mobitems:blaze_rod",
            "mcl_mobitems:gunpowder",
            "mcl_mobitems:bone",
            "mcl_throwing:ender_pearl",
            "mcl_ocean:sea_lantern",
            "mcl_ocean:prismarine_shard",
            "mcl_sponges:sponge",
            "mcl_tools:sword_diamond",
            "mcl_tools:sword_gold",
            "mcl_tools:pick_diamond",
            "mcl_armor:helmet_diamond",
            "mcl_armor:chestplate_diamond",
            "mcl_armor:chestplate_gold",
            "mcl_armor:boots_gold",
        ] {
            reg.register_item(ItemDef {
                name: item.into(),
                description: item.into(),
                stack_max: if item.contains("sword")
                    || item.contains("pick")
                    || item.contains("helmet")
                    || item.contains("chestplate")
                    || item.contains("boots")
                {
                    1
                } else {
                    64
                },
                ..Default::default()
            });
        }
        reg
    }

    #[test]
    fn deterministic_nether_fortress_and_bastion_generation() {
        let reg = make_structure_test_reg();
        let pal = NetherPalette::from_registries(&reg);
        let seed = 8842_991u64;

        let f1 = locate_nearest_structure(
            StructureKind::NetherFortress,
            seed,
            WorldGenType::Default,
            (0, 0),
        )
        .expect("Nether Fortress must exist near spawn");
        let f2 = locate_nearest_structure(
            StructureKind::NetherFortress,
            seed,
            WorldGenType::Default,
            (0, 0),
        )
        .expect("Deterministic second locate must match");
        assert_eq!(f1.origin, f2.origin);
        assert_eq!(f1.chests.len(), 3);
        assert_eq!(f1.spawners.len(), 1);

        let b1 = locate_nearest_structure(
            StructureKind::BastionRemnant,
            seed,
            WorldGenType::Default,
            (0, 0),
        )
        .expect("Bastion Remnant must exist near spawn");
        assert_eq!(b1.chests.len(), 3);
        assert_ne!(f1.origin, b1.origin);

        let dir = std::env::temp_dir().join("cubex_rs_nether_struct_test");
        let _ = std::fs::remove_dir_all(&dir);
        let nw = WorldStore::new_for_dimension(
            &dir,
            seed,
            DimensionId::Nether,
            WorldGenType::Default,
            crate::world::Difficulty::Normal,
            crate::world::Gamerules::default(),
            "Survival",
        );

        let min_c = IVec3::new(
            f1.bbox_min.x.div_euclid(16),
            f1.bbox_min.y.div_euclid(16).max(0),
            f1.bbox_min.z.div_euclid(16),
        );
        let max_c = IVec3::new(
            f1.bbox_max.x.div_euclid(16),
            f1.bbox_max.y.div_euclid(16).min(7),
            f1.bbox_max.z.div_euclid(16),
        );
        for cz in min_c.z..=max_c.z {
            for cy in min_c.y..=max_c.y {
                for cx in min_c.x..=max_c.x {
                    let cpos = IVec3::new(cx, cy, cz);
                    let ch = generate_nether_chunk(cpos, seed, &pal, &reg);
                    nw.insert_chunk(ch);
                    populate_structure_chests_for_chunk(&nw, cpos, seed, WorldGenType::Default, &reg);
                }
            }
        }

        // Verify Nether Brick bridge deck, Blaze spawner, Nether Wart garden, and loot chest
        assert_eq!(
            nw.get_node(f1.origin).0,
            pal.structures.chiseled_nether_brick
        );
        assert_eq!(nw.get_node(f1.spawners[0].0).0, pal.structures.spawner);
        assert_eq!(
            nw.get_node(f1.origin + IVec3::new(15, 0, -4)).0,
            pal.structures.soul_sand
        );
        assert_eq!(
            nw.get_node(f1.origin + IVec3::new(15, 1, -4)).0,
            pal.structures.nether_wart
        );
        assert_eq!(nw.get_node(f1.chests[0].0).0, pal.structures.chest);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn deterministic_ocean_monument_and_desert_temple_generation() {
        let reg = make_structure_test_reg();
        let spal = StructurePalette::from_registries(&reg);
        let seed = 1337_2026u64;

        let mon1 = locate_nearest_structure(
            StructureKind::OceanMonument,
            seed,
            WorldGenType::Default,
            (0, 0),
        )
        .expect("Ocean Monument must be located");
        let mon2 = locate_nearest_structure(
            StructureKind::OceanMonument,
            seed,
            WorldGenType::Default,
            (0, 0),
        )
        .unwrap();
        assert_eq!(mon1.origin, mon2.origin);
        assert_eq!(mon1.origin.y, 38);

        let temple = locate_nearest_structure(
            StructureKind::DesertTemple,
            seed,
            WorldGenType::Default,
            (0, 0),
        )
        .expect("Desert Temple must be located");
        assert_eq!(temple.chests.len(), 4);

        let dir = std::env::temp_dir().join("cubex_rs_ow_struct_test");
        let _ = std::fs::remove_dir_all(&dir);
        let ow = WorldStore::new(&dir, seed);

        for anchor in [&mon1, &temple] {
            let min_c = IVec3::new(
                anchor.bbox_min.x.div_euclid(16),
                0,
                anchor.bbox_min.z.div_euclid(16),
            );
            let max_c = IVec3::new(
                anchor.bbox_max.x.div_euclid(16),
                8,
                anchor.bbox_max.z.div_euclid(16),
            );
            for cz in min_c.z..=max_c.z {
                for cy in min_c.y..=max_c.y {
                    for cx in min_c.x..=max_c.x {
                        let cpos = IVec3::new(cx, cy, cz);
                        let mut ch = Chunk::new(cpos);
                        stamp_overworld_structures_into_chunk(
                            &mut ch,
                            seed,
                            WorldGenType::Default,
                            &spal,
                        );
                        ow.insert_chunk(ch);
                        populate_structure_chests_for_chunk(
                            &ow,
                            cpos,
                            seed,
                            WorldGenType::Default,
                            &reg,
                        );
                    }
                }
            }
        }

        // Monument assertions: underwater crown sea_lantern, central gold block vault, sponge, and submerged chest
        assert_eq!(ow.get_node(mon1.origin + IVec3::new(0, 20, 0)).0, spal.sea_lantern);
        assert_eq!(ow.get_node(mon1.origin + IVec3::new(0, 2, 0)).0, spal.gold_block);
        assert_eq!(ow.get_node(mon1.origin + IVec3::new(0, 12, 23)).0, spal.water_source);
        assert_eq!(ow.get_node(mon1.chests[0].0).0, spal.chest);

        // Desert Temple assertions: blue center terracotta + 4 treasure chests below
        assert_eq!(ow.get_node(temple.origin).0, spal.terracotta_blue);
        for (cpos, _) in &temple.chests {
            assert_eq!(ow.get_node(*cpos).0, spal.chest);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn loot_chest_population_is_deterministic_and_idempotent() {
        let reg = make_structure_test_reg();
        let dir = std::env::temp_dir().join("cubex_rs_loot_chest_test");
        let _ = std::fs::remove_dir_all(&dir);
        let world = WorldStore::new(&dir, 99123);
        let cpos = IVec3::new(12, 50, -8);

        for table in [
            LootTableKind::NetherFortress,
            LootTableKind::BastionRemnant,
            LootTableKind::OceanMonument,
            LootTableKind::DesertTemple,
        ] {
            let test_pos = cpos + IVec3::new(table as i32, 0, 0);
            assert!(populate_structure_chest(&world, test_pos, table, 99123, &reg));
            let meta = world.get_node_meta(test_pos);
            let non_empty: Vec<ItemStackData> = meta
                .read()
                .inventories
                .get("main")
                .expect("27-slot chest inventory")
                .iter()
                .filter(|s| !s.is_empty())
                .cloned()
                .collect();
            assert_eq!(meta.read().inventories["main"].len(), 27);
            assert!(
                non_empty.len() >= 6,
                "Expected populated loot chest for {table:?}, got {}",
                non_empty.len()
            );
            assert!(
                non_empty
                    .iter()
                    .any(|s| s.meta.read().fields.get("cubex_enchanted").map(String::as_str) == Some("1")),
                "Expected at least one enchanted item tooltip metadata in {table:?}"
            );

            // Idempotency: clearing slot 13 must NOT get refilled by re-invoking populate
            meta.write().inventories.get_mut("main").unwrap()[13] = ItemStackData::default();
            assert!(!populate_structure_chest(&world, test_pos, table, 99123, &reg));
            assert!(meta.read().inventories["main"][13].is_empty());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
