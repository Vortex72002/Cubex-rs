use glam::IVec3;
use std::ops::RangeInclusive;
use std::path::Path;

use crate::paths::EnginePaths;
use crate::registry::{EngineRegistries, CONTENT_AIR, CONTENT_UNKNOWN};
pub use crate::world::biome::{
    biome_name_at, blended_biome_profile, fbm_2d, sample_biome, BiomeTerrainProfile, GenBiome,
    SEA_LEVEL,
};
use crate::world::light::propagate_chunk_light;
use crate::world::schematic::{
    rotate_param2_along_y, PlaceFlags, Schematic, SchematicRotation,
};
use crate::world::structures::{stamp_overworld_structures_into_chunk, StructurePalette};
use crate::world::{Chunk, WorldGenType, CHUNK_SIZE};

const TREE_CELL: i32 = 9;
const TREE_SEARCH_RADIUS: i32 = 10;

/// Chunk Y range covered for each `WorldGenType`.
/// - `Default` / `Amplified`: `0..=8` (`Y = 0..=143`, true bottom floor at `Y = 0` and open sky above terrain around `SEA_LEVEL = 62`)
/// - `Superflat` / `Skyblock`: `-1..=2` (`Y = -16..=47`, preserving `Y = 0` / `Y = 12`)
pub fn chunk_y_range_for(world_type: WorldGenType) -> RangeInclusive<i32> {
    match world_type {
        WorldGenType::Default | WorldGenType::Amplified => 0..=8,
        WorldGenType::Superflat | WorldGenType::Skyblock => -1..=2,
    }
}

/// Cached Mineclonia content IDs & tree `.mts` schematics for fast parallel mapgen.
#[derive(Debug, Clone, Default)]
pub struct MapgenPalette {
    pub bedrock: u16,
    pub stone: u16,
    pub deepslate: u16,
    pub dirt: u16,
    pub grass: u16,
    pub dirt_with_grass_snow: u16,
    pub podzol: u16,
    pub clay: u16,
    pub sand: u16,
    pub sandstone: u16,
    pub gravel: u16,
    pub water: u16,
    pub lava: u16,
    pub oak_log: u16,
    pub oak_leaves: u16,
    pub birch_log: u16,
    pub birch_leaves: u16,
    pub spruce_log: u16,
    pub spruce_leaves: u16,
    pub coal_ore: u16,
    pub iron_ore: u16,
    pub gold_ore: u16,
    pub diamond_ore: u16,
    pub tallgrass: u16,
    pub fern: u16,
    pub dandelion: u16,
    pub poppy: u16,
    pub cactus: u16,
    pub deadbush: u16,
    pub snow: u16,
    pub snow_layer: u16,
    pub chest: u16,
    pub structures: StructurePalette,
    pub oak_trees: Vec<Schematic>,
    pub swamp_oak_trees: Vec<Schematic>,
    pub birch_trees: Vec<Schematic>,
    pub spruce_trees: Vec<Schematic>,
    pub jungle_trees: Vec<Schematic>,
    pub acacia_trees: Vec<Schematic>,
    pub dark_oak_trees: Vec<Schematic>,
    pub crimson_fungi: Vec<Schematic>,
    pub warped_fungi: Vec<Schematic>,
}

fn load_schems(dir: &Path, filenames: &[&str]) -> Vec<Schematic> {
    let mut out = Vec::new();
    for name in filenames {
        let p = dir.join(name);
        if p.is_file() {
            if let Ok(s) = Schematic::load_from_file(&p) {
                out.push(s);
            }
        }
    }
    out
}

impl MapgenPalette {
    pub fn from_registries(reg: &EngineRegistries) -> Self {
        let paths = EnginePaths::resolve();
        let game_dir = if paths.game_dir.join("mods").is_dir() {
            Some(paths.game_dir.as_path())
        } else {
            None
        };
        Self::from_registries_with_game_dir(reg, game_dir)
    }

    pub fn from_registries_with_game_dir(
        reg: &EngineRegistries,
        game_dir: Option<&Path>,
    ) -> Self {
        let resolve_first = |names: &[&str]| -> u16 {
            for n in names {
                let id = reg.get_content_id(n);
                if id != CONTENT_UNKNOWN {
                    return id;
                }
            }
            reg.nodes_by_id
                .iter()
                .flatten()
                .find(|d| d.drawtype.is_solid_cube() && d.name != "unknown")
                .map(|d| d.id)
                .unwrap_or(1)
        };

        let stone = resolve_first(&["mcl_core:stone", "default:stone"]);
        let dirt = resolve_first(&["mcl_core:dirt", "default:dirt"]);
        let grass = resolve_first(&["mcl_core:dirt_with_grass", "default:dirt_with_grass"]);
        let sand = resolve_first(&["mcl_core:sand", "default:sand"]);
        let water = resolve_first(&["mcl_core:water_source", "default:water_source"]);
        let oak_log = resolve_first(&["mcl_trees:tree_oak", "mcl_core:tree", "mcl_core:dirt"]);
        let oak_leaves = resolve_first(&[
            "mcl_trees:leaves_oak",
            "mcl_core:leaves",
            "mcl_core:dirt_with_grass",
        ]);
        let birch_log = resolve_first(&["mcl_trees:tree_birch", "mcl_trees:tree_oak", "mcl_core:tree"]);
        let birch_leaves = resolve_first(&[
            "mcl_trees:leaves_birch",
            "mcl_trees:leaves_oak",
            "mcl_core:leaves",
        ]);
        let spruce_log = resolve_first(&["mcl_trees:tree_spruce", "mcl_trees:tree_oak", "mcl_core:tree"]);
        let spruce_leaves = resolve_first(&[
            "mcl_trees:leaves_spruce",
            "mcl_trees:leaves_oak",
            "mcl_core:leaves",
        ]);
        let snow = resolve_first(&[
            "mcl_core:snowblock",
            "mcl_core:snow",
            "mcl_core:dirt_with_grass",
        ]);
        let dirt_with_grass_snow = resolve_first(&[
            "mcl_core:dirt_with_grass_snow",
            "mcl_core:snowblock",
            "mcl_core:dirt_with_grass",
        ]);

        let mut oak_trees = Vec::new();
        let mut swamp_oak_trees = Vec::new();
        let mut birch_trees = Vec::new();
        let mut spruce_trees = Vec::new();
        let mut jungle_trees = Vec::new();
        let mut acacia_trees = Vec::new();
        let mut dark_oak_trees = Vec::new();
        let mut crimson_fungi = Vec::new();
        let mut warped_fungi = Vec::new();

        if let Some(gd) = game_dir {
            let core_schem = gd.join("mods/ITEMS/mcl_core/schematics");
            if core_schem.is_dir() {
                oak_trees = load_schems(
                    &core_schem,
                    &[
                        "mcl_core_oak_classic.mts",
                        "mcl_core_oak_balloon.mts",
                        "mcl_core_oak_v6.mts",
                    ],
                );
                swamp_oak_trees = load_schems(
                    &core_schem,
                    &["mcl_core_oak_swamp.mts", "mcl_core_oak_classic.mts"],
                );
                birch_trees = load_schems(
                    &core_schem,
                    &["mcl_core_birch.mts", "mcl_core_birch_tall.mts"],
                );
                spruce_trees = load_schems(
                    &core_schem,
                    &[
                        "mcl_core_spruce_1.mts",
                        "mcl_core_spruce_2.mts",
                        "mcl_core_spruce_3.mts",
                        "mcl_core_spruce_4.mts",
                        "mcl_core_spruce_tall.mts",
                    ],
                );
                jungle_trees = load_schems(
                    &core_schem,
                    &[
                        "mcl_core_jungle_tree.mts",
                        "mcl_core_jungle_tree_2.mts",
                        "mcl_core_jungle_tree_3.mts",
                    ],
                );
                acacia_trees = load_schems(
                    &core_schem,
                    &[
                        "mcl_core_acacia_1.mts",
                        "mcl_core_acacia_2.mts",
                        "mcl_core_acacia_3.mts",
                    ],
                );
                dark_oak_trees = load_schems(&core_schem, &["mcl_core_dark_oak.mts"]);
            }
            let crimson_schem = gd.join("mods/ITEMS/mcl_crimson/schematics");
            if crimson_schem.is_dir() {
                crimson_fungi = load_schems(
                    &crimson_schem,
                    &[
                        "crimson_fungus_1.mts",
                        "crimson_fungus_2.mts",
                        "crimson_fungus_3.mts",
                    ],
                );
                warped_fungi = load_schems(
                    &crimson_schem,
                    &[
                        "warped_fungus_1.mts",
                        "warped_fungus_2.mts",
                        "warped_fungus_3.mts",
                    ],
                );
            }
        }

        Self {
            bedrock: resolve_first(&["mcl_core:bedrock", "mcl_core:stone"]),
            stone,
            deepslate: resolve_first(&["mcl_deepslate:deepslate", "mcl_core:stone"]),
            dirt,
            grass,
            dirt_with_grass_snow,
            podzol: resolve_first(&["mcl_core:podzol", "mcl_core:dirt_with_grass"]),
            clay: resolve_first(&["mcl_core:clay", "mcl_core:sand"]),
            sand,
            sandstone: resolve_first(&["mcl_core:sandstone", "mcl_core:sand"]),
            gravel: resolve_first(&["mcl_core:gravel", "mcl_core:stone"]),
            water,
            lava: reg.get_content_id("mcl_core:lava_source"),
            oak_log,
            oak_leaves,
            birch_log,
            birch_leaves,
            spruce_log,
            spruce_leaves,
            coal_ore: resolve_first(&[
                "mcl_core:stone_with_coal",
                "mcl_core:coal_ore",
                "mcl_core:stone",
            ]),
            iron_ore: resolve_first(&[
                "mcl_core:stone_with_iron",
                "mcl_core:iron_ore",
                "mcl_core:stone",
            ]),
            gold_ore: resolve_first(&[
                "mcl_core:stone_with_gold",
                "mcl_core:gold_ore",
                "mcl_core:stone",
            ]),
            diamond_ore: resolve_first(&[
                "mcl_core:stone_with_diamond",
                "mcl_core:diamond_ore",
                "mcl_core:stone",
            ]),
            tallgrass: reg.get_content_id("mcl_flowers:tallgrass"),
            fern: reg.get_content_id("mcl_flowers:fern"),
            dandelion: reg.get_content_id("mcl_flowers:dandelion"),
            poppy: reg.get_content_id("mcl_flowers:poppy"),
            cactus: reg.get_content_id("mcl_core:cactus"),
            deadbush: reg.get_content_id("mcl_core:deadbush"),
            snow,
            snow_layer: reg.get_content_id("mcl_core:snow"),
            chest: reg.get_content_id("mcl_chests:chest"),
            structures: StructurePalette::from_registries(reg),
            oak_trees,
            swamp_oak_trees,
            birch_trees,
            spruce_trees,
            jungle_trees,
            acacia_trees,
            dark_oak_trees,
            crimson_fungi,
            warped_fungi,
        }
    }

    fn resolve_schem_node(&self, name: &str, reg: &EngineRegistries) -> u16 {
        let id = reg.get_content_id(name);
        if id != CONTENT_UNKNOWN {
            return id;
        }
        if name.contains("birch") {
            return if name.contains("leaves") {
                self.birch_leaves
            } else {
                self.birch_log
            };
        }
        if name.contains("spruce") {
            return if name.contains("leaves") {
                self.spruce_leaves
            } else {
                self.spruce_log
            };
        }
        if name.contains("tree_") || name.ends_with(":tree") {
            return self.oak_log;
        }
        if name.contains("leaves") {
            return self.oak_leaves;
        }
        CONTENT_UNKNOWN
    }
}

#[inline]
fn hash3d(x: i32, y: i32, z: i32, seed: u64) -> u32 {
    let mut n = (x as i64).wrapping_mul(374761393)
        ^ (y as i64).wrapping_mul(2147483647)
        ^ (z as i64).wrapping_mul(668265263)
        ^ (seed as i64);
    n = (n ^ (n >> 13)).wrapping_mul(1274126177);
    (n ^ (n >> 16)) as u32
}

pub fn terrain_height(wx: i32, wz: i32, seed: u64) -> i32 {
    terrain_height_for(wx, wz, seed, WorldGenType::Default)
}

pub fn terrain_height_for(wx: i32, wz: i32, seed: u64, world_type: WorldGenType) -> i32 {
    match world_type {
        WorldGenType::Superflat => 0,
        WorldGenType::Skyblock => 12,
        WorldGenType::Amplified | WorldGenType::Default => {
            let prof = blended_biome_profile(wx, wz, seed);
            let cont = ((fbm_2d(wx as f32 * 0.0045, wz as f32 * 0.0045, seed, 4) - 0.5) * 2.0)
                .clamp(-1.0, 1.0);
            let hills = (fbm_2d(wx as f32 * 0.018, wz as f32 * 0.018, seed ^ 0xA5A5, 3) - 0.5) * 2.0;
            let ridge =
                1.0 - (fbm_2d(wx as f32 * 0.011, wz as f32 * 0.011, seed ^ 0x5A5A, 2) - 0.5).abs() * 2.0;

            if world_type == WorldGenType::Amplified {
                let h = prof.base_height
                    + 4.0
                    + cont * (prof.continental_amp * 1.45)
                    + hills * (prof.hill_amp * 2.0)
                    + ridge * (prof.ridge_amp * 1.7);
                (h.round() as i32).clamp(40, 132)
            } else {
                let h = prof.base_height
                    + cont * prof.continental_amp
                    + hills * prof.hill_amp
                    + ridge * prof.ridge_amp;
                (h.round() as i32).clamp(46, 108)
            }
        }
    }
}

/// Find a dry player spawn block `(x, surf_y + 1, z)` near `(0, 0)` cleanly above `SEA_LEVEL`.
pub fn find_spawn_pos(seed: u64, world_type: WorldGenType) -> IVec3 {
    match world_type {
        WorldGenType::Superflat => IVec3::new(0, 1, 0),
        WorldGenType::Skyblock => IVec3::new(0, 13, 0),
        WorldGenType::Default | WorldGenType::Amplified => {
            let h0 = terrain_height_for(0, 0, seed, world_type);
            if h0 > SEA_LEVEL + 1 {
                return IVec3::new(0, h0 + 1, 0);
            }
            for radius in (8i32..=160).step_by(8) {
                for dz in (-radius..=radius).step_by(8) {
                    for dx in (-radius..=radius).step_by(8) {
                        if dx.abs() != radius && dz.abs() != radius {
                            continue;
                        }
                        let h = terrain_height_for(dx, dz, seed, world_type);
                        if h > SEA_LEVEL + 1 && sample_biome(dx, dz, seed) != GenBiome::Swamp {
                            return IVec3::new(dx, h + 1, dz);
                        }
                    }
                }
            }
            IVec3::new(0, h0.max(SEA_LEVEL) + 1, 0)
        }
    }
}

fn pick_schem(list: &[Schematic], roll: u32) -> Option<&Schematic> {
    if list.is_empty() {
        None
    } else {
        Some(&list[(roll as usize) % list.len()])
    }
}

fn select_tree_schematic(
    biome: GenBiome,
    roll: u32,
    palette: &MapgenPalette,
) -> Option<&Schematic> {
    match biome {
        GenBiome::Plains => {
            if (roll >> 8) % 6 == 0 {
                pick_schem(&palette.birch_trees, roll).or_else(|| pick_schem(&palette.oak_trees, roll))
            } else {
                pick_schem(&palette.oak_trees, roll)
            }
        }
        GenBiome::Forest => {
            if (roll >> 8) % 5 == 0 {
                pick_schem(&palette.birch_trees, roll).or_else(|| pick_schem(&palette.oak_trees, roll))
            } else {
                pick_schem(&palette.oak_trees, roll)
            }
        }
        GenBiome::BirchForest => {
            pick_schem(&palette.birch_trees, roll).or_else(|| pick_schem(&palette.oak_trees, roll))
        }
        GenBiome::Taiga | GenBiome::SnowyTundra => {
            pick_schem(&palette.spruce_trees, roll).or_else(|| pick_schem(&palette.oak_trees, roll))
        }
        GenBiome::Swamp => {
            pick_schem(&palette.swamp_oak_trees, roll).or_else(|| pick_schem(&palette.oak_trees, roll))
        }
        GenBiome::Desert => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn stamp_schematic_into_chunk(
    chunk: &mut Chunk,
    schem: &Schematic,
    place_pos: IVec3,
    rot: SchematicRotation,
    biome_leaf_idx: u8,
    seed_mix: u64,
    palette: &MapgenPalette,
    registries: &EngineRegistries,
) {
    let base_x = chunk.pos.x * 16;
    let base_y = chunk.pos.y * 16;
    let base_z = chunk.pos.z * 16;
    let rot_steps = rot.resolve(0);

    schem.visit_rotated_nodes(
        place_pos,
        rot,
        PlaceFlags {
            place_center_x: true,
            place_center_y: false,
            place_center_z: true,
            force_placement: false,
        },
        false,
        None,
        seed_mix,
        |wpos, name, raw_p2, force| {
            if name == "air" && !force {
                return;
            }
            let lx = wpos.x - base_x;
            let ly = wpos.y - base_y;
            let lz = wpos.z - base_z;
            if !(0..16).contains(&lx) || !(0..16).contains(&ly) || !(0..16).contains(&lz) {
                return;
            }
            let idx = crate::world::local_index(lx as usize, ly as usize, lz as usize);
            if !force && chunk.nodes[idx] != CONTENT_AIR {
                return;
            }
            let id = palette.resolve_schem_node(name, registries);
            if id == CONTENT_UNKNOWN {
                return;
            }
            let is_tree_trunk = name.contains("tree_") || name.ends_with(":tree") || name.contains("stem");
            let p2 = if is_tree_trunk {
                // Keep vertical trunks param2 = 0; rotate horizontal branch logs
                if raw_p2 < 4 {
                    0
                } else {
                    rotate_param2_along_y("facedir", raw_p2, rot_steps)
                }
            } else if name.contains("leaves") {
                // Preserve leaf distance band in bits 5..7 while encoding biome foliage index in bits 0..4
                (raw_p2 & 0xE0) | (biome_leaf_idx & 0x1F)
            } else if name.contains("vine") {
                rotate_param2_along_y("wallmounted", raw_p2, rot_steps)
            } else {
                raw_p2
            };
            chunk.nodes[idx] = id;
            chunk.param2[idx] = p2;
        },
    );
}

fn stamp_fallback_tree(
    chunk: &mut Chunk,
    root: IVec3,
    biome: GenBiome,
    seed: u64,
    palette: &MapgenPalette,
) {
    let base_x = chunk.pos.x * 16;
    let base_y = chunk.pos.y * 16;
    let base_z = chunk.pos.z * 16;
    let (log_id, leaf_id, is_spruce) = match biome {
        GenBiome::BirchForest => (palette.birch_log, palette.birch_leaves, false),
        GenBiome::Taiga | GenBiome::SnowyTundra => (palette.spruce_log, palette.spruce_leaves, true),
        _ => (palette.oak_log, palette.oak_leaves, false),
    };
    let leaf_p2 = 64 | (biome.palette_index() & 0x1F);
    let trunk_h = if is_spruce {
        6 + (hash3d(root.x, 1, root.z, seed) % 3) as i32
    } else {
        4 + (hash3d(root.x, 1, root.z, seed) % 2) as i32
    };

    for ty in 0..trunk_h {
        let wy = root.y + ty;
        let lx = root.x - base_x;
        let ly = wy - base_y;
        let lz = root.z - base_z;
        if (0..16).contains(&lx) && (0..16).contains(&ly) && (0..16).contains(&lz) {
            let idx = crate::world::local_index(lx as usize, ly as usize, lz as usize);
            chunk.nodes[idx] = log_id;
            chunk.param2[idx] = 0;
        }
    }

    if is_spruce {
        for dy in 2..=(trunk_h + 1) {
            let rad: i32 = if dy >= trunk_h + 1 {
                0
            } else if dy >= trunk_h - 1 || dy == 2 {
                1
            } else {
                2
            };
            let wy = root.y + dy;
            let ly = wy - base_y;
            if !(0..16).contains(&ly) {
                continue;
            }
            for dz in -rad..=rad {
                for dx in -rad..=rad {
                    if dx == 0 && dz == 0 && dy < trunk_h {
                        continue;
                    }
                    if rad == 2 && dx.abs() == 2 && dz.abs() == 2 {
                        continue;
                    }
                    let lx = root.x + dx - base_x;
                    let lz = root.z + dz - base_z;
                    if (0..16).contains(&lx) && (0..16).contains(&lz) {
                        let idx = crate::world::local_index(lx as usize, ly as usize, lz as usize);
                        if chunk.nodes[idx] == CONTENT_AIR {
                            chunk.nodes[idx] = leaf_id;
                            chunk.param2[idx] = leaf_p2;
                        }
                    }
                }
            }
        }
    } else {
        for dy in (trunk_h - 2)..=(trunk_h + 1) {
            let rad: i32 = if dy >= trunk_h { 1 } else { 2 };
            let wy = root.y + dy;
            let ly = wy - base_y;
            if !(0..16).contains(&ly) {
                continue;
            }
            for dz in -rad..=rad {
                for dx in -rad..=rad {
                    if dx == 0 && dz == 0 && dy < trunk_h {
                        continue;
                    }
                    if rad == 2 && dx.abs() == 2 && dz.abs() == 2 && dy == trunk_h - 1 {
                        continue;
                    }
                    let lx = root.x + dx - base_x;
                    let lz = root.z + dz - base_z;
                    if (0..16).contains(&lx) && (0..16).contains(&lz) {
                        let idx = crate::world::local_index(lx as usize, ly as usize, lz as usize);
                        if chunk.nodes[idx] == CONTENT_AIR {
                            chunk.nodes[idx] = leaf_id;
                            chunk.param2[idx] = leaf_p2;
                        }
                    }
                }
            }
        }
    }
}

const LAVA_POOL_CELL: i32 = 44;
const LAVA_POOL_SEARCH: i32 = 5;

fn stamp_surface_lava_pools_into_chunk(
    chunk: &mut Chunk,
    seed: u64,
    world_type: WorldGenType,
    palette: &MapgenPalette,
) {
    if palette.lava == CONTENT_UNKNOWN || palette.lava == CONTENT_AIR {
        return;
    }
    let base_x = chunk.pos.x * 16;
    let base_y = chunk.pos.y * 16;
    let base_z = chunk.pos.z * 16;
    if base_y + 15 < SEA_LEVEL || base_y > 84 {
        return;
    }

    let min_cx = (base_x - LAVA_POOL_SEARCH).div_euclid(LAVA_POOL_CELL);
    let max_cx = (base_x + 15 + LAVA_POOL_SEARCH).div_euclid(LAVA_POOL_CELL);
    let min_cz = (base_z - LAVA_POOL_SEARCH).div_euclid(LAVA_POOL_CELL);
    let max_cz = (base_z + 15 + LAVA_POOL_SEARCH).div_euclid(LAVA_POOL_CELL);

    for cell_z in min_cz..=max_cz {
        for cell_x in min_cx..=max_cx {
            let phash = hash3d(cell_x, 0, cell_z, seed ^ 0x1A7A_9001);
            if ((phash >> 10) % 100) >= 65 {
                continue;
            }
            let cx_pool = cell_x * LAVA_POOL_CELL + 8 + ((phash % 29) as i32);
            let cz_pool = cell_z * LAVA_POOL_CELL + 8 + (((phash >> 5) % 29) as i32);
            if cx_pool * cx_pool + cz_pool * cz_pool < 14 * 14 {
                continue;
            }

            let center_y = terrain_height_for(cx_pool, cz_pool, seed, world_type);
            if center_y < SEA_LEVEL + 2 || center_y > 78 {
                continue;
            }

            let mut h_min = center_y;
            let mut h_max = center_y;
            for (ox, oz) in [(3, 0), (-3, 0), (0, 3), (0, -3)] {
                let h = terrain_height_for(cx_pool + ox, cz_pool + oz, seed, world_type);
                h_min = h_min.min(h);
                h_max = h_max.max(h);
            }
            if h_max - h_min > 4 {
                continue;
            }

            if center_y + 4 < base_y || center_y - 2 > base_y + 15 {
                continue;
            }

            for dz in -4..=4 {
                let wz = cz_pool + dz;
                let lz = wz - base_z;
                if !(0..16).contains(&lz) {
                    continue;
                }
                for dx in -4..=4 {
                    let wx = cx_pool + dx;
                    let lx = wx - base_x;
                    if !(0..16).contains(&lx) {
                        continue;
                    }
                    let nx = dx as f32 / 2.6;
                    let nz = dz as f32 / 2.2;
                    let r2 = nx * nx + nz * nz;
                    if r2 > 2.1 {
                        continue;
                    }
                    for dy in -2..=4 {
                        let wy = center_y + dy;
                        let ly = wy - base_y;
                        if !(0..16).contains(&ly) {
                            continue;
                        }
                        let idx = crate::world::local_index(lx as usize, ly as usize, lz as usize);
                        if r2 <= 1.0 {
                            if dy == -2 {
                                chunk.nodes[idx] = palette.stone;
                                chunk.param2[idx] = 0;
                            } else if dy == -1 {
                                chunk.nodes[idx] = palette.lava;
                                chunk.param2[idx] = 0;
                            } else {
                                chunk.nodes[idx] = CONTENT_AIR;
                                chunk.param2[idx] = 0;
                            }
                        } else if dy == -2 && chunk.nodes[idx] == CONTENT_AIR {
                            chunk.nodes[idx] = palette.stone;
                            chunk.param2[idx] = 0;
                        } else if dy == -1 {
                            chunk.nodes[idx] = palette.stone;
                            chunk.param2[idx] = 0;
                        } else if dy >= 0 && chunk.nodes[idx] != CONTENT_AIR {
                            chunk.nodes[idx] = CONTENT_AIR;
                            chunk.param2[idx] = 0;
                        }
                    }
                }
            }
        }
    }
}

/// Generate a 16x16x16 terrain chunk using Mineclonia node IDs (Default world type).
pub fn generate_chunk(
    cpos: IVec3,
    seed: u64,
    palette: &MapgenPalette,
    registries: &EngineRegistries,
) -> Chunk {
    generate_chunk_for_world(cpos, seed, WorldGenType::Default, palette, registries)
}

/// Generate a 16x16x16 terrain chunk for any `WorldGenType`.
pub fn generate_chunk_for_world(
    cpos: IVec3,
    seed: u64,
    world_type: WorldGenType,
    palette: &MapgenPalette,
    registries: &EngineRegistries,
) -> Chunk {
    let mut chunk = Chunk::new(cpos);
    let base_x = cpos.x * 16;
    let base_y = cpos.y * 16;
    let base_z = cpos.z * 16;

    match world_type {
        WorldGenType::Superflat => {
            for lz in 0..CHUNK_SIZE {
                for lx in 0..CHUNK_SIZE {
                    for ly in 0..CHUNK_SIZE {
                        let wy = base_y + ly as i32;
                        let id = if wy == -4 {
                            palette.bedrock
                        } else if (-3..=-1).contains(&wy) {
                            palette.dirt
                        } else if wy == 0 {
                            palette.grass
                        } else {
                            CONTENT_AIR
                        };
                        chunk.nodes[crate::world::local_index(lx, ly, lz)] = id;
                    }
                }
            }
            propagate_chunk_light(&mut chunk, registries, cpos.y >= 0);
            chunk.dirty_mesh = true;
            chunk.modified = false;
            return chunk;
        }
        WorldGenType::Skyblock => {
            for lz in 0..CHUNK_SIZE {
                let wz = base_z + lz as i32;
                for lx in 0..CHUNK_SIZE {
                    let wx = base_x + lx as i32;
                    for ly in 0..CHUNK_SIZE {
                        let wy = base_y + ly as i32;
                        let in_main =
                            (-3..=3).contains(&wx) && (-3..=3).contains(&wz) && (10..=12).contains(&wy);
                        let id = if wx == 0 && wy == 10 && wz == 0 {
                            palette.bedrock
                        } else if in_main {
                            if wy == 12 {
                                palette.grass
                            } else if wy == 11 {
                                palette.dirt
                            } else {
                                palette.stone
                            }
                        } else {
                            CONTENT_AIR
                        };
                        chunk.nodes[crate::world::local_index(lx, ly, lz)] = id;
                    }
                }
            }
            if let Some(oak) = palette.oak_trees.first() {
                stamp_schematic_into_chunk(
                    &mut chunk,
                    oak,
                    IVec3::new(2, 13, 2),
                    SchematicRotation::Rot0,
                    GenBiome::Plains.palette_index(),
                    seed ^ 0x517B,
                    palette,
                    registries,
                );
            } else {
                stamp_fallback_tree(
                    &mut chunk,
                    IVec3::new(2, 13, 2),
                    GenBiome::Plains,
                    seed,
                    palette,
                );
            }
            propagate_chunk_light(&mut chunk, registries, cpos.y >= 0);
            chunk.dirty_mesh = true;
            chunk.modified = false;
            return chunk;
        }
        WorldGenType::Default | WorldGenType::Amplified => {}
    }

    let snow_line = if world_type == WorldGenType::Amplified {
        90
    } else {
        82
    };
    let tree_max_y = if world_type == WorldGenType::Amplified {
        94
    } else {
        84
    };

    for lz in 0..CHUNK_SIZE {
        let wz = base_z + lz as i32;
        for lx in 0..CHUNK_SIZE {
            let wx = base_x + lx as i32;
            let surface_y = terrain_height_for(wx, wz, seed, world_type);
            let biome = sample_biome(wx, wz, seed);
            let p2_idx = biome.palette_index();
            let moisture = fbm_2d(wx as f32 * 0.014, wz as f32 * 0.014, seed ^ 0x1234, 2);

            for ly in 0..CHUNK_SIZE {
                let wy = base_y + ly as i32;
                let idx = crate::world::local_index(lx, ly, lz);
                let mut p2 = 0u8;

                let id = if wy <= 0 {
                    palette.bedrock
                } else if wy > surface_y {
                    if wy <= SEA_LEVEL {
                        palette.water
                    } else {
                        CONTENT_AIR
                    }
                } else if wy == surface_y {
                    if surface_y < SEA_LEVEL - 1 {
                        if biome == GenBiome::Swamp && hash3d(wx, wy, wz, seed ^ 0xC1A7) % 4 == 0 {
                            palette.clay
                        } else if hash3d(wx, wy, wz, seed ^ 0x6AA5) % 5 == 0 {
                            palette.gravel
                        } else {
                            palette.sand
                        }
                    } else if surface_y <= SEA_LEVEL + 1 {
                        if biome == GenBiome::Swamp {
                            if surface_y <= SEA_LEVEL {
                                if hash3d(wx, wy, wz, seed ^ 0xC1A7) % 3 == 0 {
                                    palette.clay
                                } else {
                                    palette.sand
                                }
                            } else {
                                p2 = p2_idx;
                                palette.grass
                            }
                        } else if biome == GenBiome::SnowyTundra && surface_y >= SEA_LEVEL {
                            palette.dirt_with_grass_snow
                        } else {
                            palette.sand
                        }
                    } else {
                        match biome {
                            GenBiome::Desert => palette.sand,
                            GenBiome::SnowyTundra => {
                                if hash3d(wx, 4, wz, seed ^ 0x5A01) % 5 == 0 {
                                    palette.snow
                                } else {
                                    palette.dirt_with_grass_snow
                                }
                            }
                            GenBiome::Taiga => {
                                if surface_y >= snow_line {
                                    palette.dirt_with_grass_snow
                                } else if hash3d(wx, 3, wz, seed ^ 0xB0D2) % 3 == 0 {
                                    palette.podzol
                                } else {
                                    p2 = p2_idx;
                                    palette.grass
                                }
                            }
                            GenBiome::Plains
                            | GenBiome::Forest
                            | GenBiome::BirchForest
                            | GenBiome::Swamp => {
                                p2 = p2_idx;
                                palette.grass
                            }
                        }
                    }
                } else if wy >= surface_y - 3 {
                    if biome == GenBiome::Desert {
                        palette.sandstone
                    } else if surface_y <= SEA_LEVEL + 1 && biome != GenBiome::Swamp {
                        palette.sand
                    } else {
                        palette.dirt
                    }
                } else {
                    let r = hash3d(wx, wy, wz, seed) % 1000;
                    if wy < 28 && r < 4 {
                        palette.diamond_ore
                    } else if wy < 38 && r < 10 {
                        palette.gold_ore
                    } else if r < 24 {
                        palette.iron_ore
                    } else if r < 45 {
                        palette.coal_ore
                    } else if wy >= 28
                        && wy < surface_y - 4
                        && palette.gravel != CONTENT_UNKNOWN
                        && hash3d(
                            wx.div_euclid(3),
                            wy.div_euclid(2),
                            wz.div_euclid(3),
                            seed ^ 0x6AA5_4321,
                        ) % 70
                            == 0
                    {
                        palette.gravel
                    } else if wy < 26 {
                        palette.deepslate
                    } else {
                        palette.stone
                    }
                };

                chunk.nodes[idx] = id;
                chunk.param2[idx] = p2;
            }

            // Surface decorations & snow cover
            if surface_y > SEA_LEVEL {
                if biome == GenBiome::Desert {
                    if surface_y > SEA_LEVEL + 1 {
                        let d_roll = hash3d(wx, 2, wz, seed ^ 0xD357);
                        if d_roll % 97 == 0 && palette.cactus != CONTENT_UNKNOWN {
                            let ch_h = 2 + (d_roll >> 8) as i32 % 2;
                            for dy in 1..=ch_h {
                                let wy = surface_y + dy;
                                if wy >= base_y && wy < base_y + 16 {
                                    let ly = (wy - base_y) as usize;
                                    let idx = crate::world::local_index(lx, ly, lz);
                                    if chunk.nodes[idx] == CONTENT_AIR {
                                        chunk.nodes[idx] = palette.cactus;
                                    }
                                }
                            }
                        } else if d_roll % 43 == 0 && palette.deadbush != CONTENT_UNKNOWN {
                            let wy = surface_y + 1;
                            if wy >= base_y && wy < base_y + 16 {
                                let ly = (wy - base_y) as usize;
                                let idx = crate::world::local_index(lx, ly, lz);
                                if chunk.nodes[idx] == CONTENT_AIR {
                                    chunk.nodes[idx] = palette.deadbush;
                                }
                            }
                        }
                    }
                } else if (biome == GenBiome::SnowyTundra
                    || (biome == GenBiome::Taiga && surface_y >= snow_line))
                    && surface_y + 1 >= base_y
                    && surface_y + 1 < base_y + 16
                {
                    let ly = (surface_y + 1 - base_y) as usize;
                    let idx = crate::world::local_index(lx, ly, lz);
                    if chunk.nodes[idx] == CONTENT_AIR && palette.snow_layer != CONTENT_UNKNOWN {
                        chunk.nodes[idx] = palette.snow_layer;
                    }
                } else if surface_y > SEA_LEVEL + 1
                    && surface_y < snow_line
                    && moisture > 0.28
                    && surface_y + 1 >= base_y
                    && surface_y + 1 < base_y + 16
                {
                    let deco_roll = hash3d(wx, 2, wz, seed ^ 0x9999) % 36;
                    let ly = (surface_y + 1 - base_y) as usize;
                    let idx = crate::world::local_index(lx, ly, lz);
                    if chunk.nodes[idx] == CONTENT_AIR {
                        if deco_roll < 4 && palette.tallgrass != CONTENT_UNKNOWN {
                            chunk.nodes[idx] = palette.tallgrass;
                            chunk.param2[idx] = p2_idx;
                        } else if deco_roll == 4
                            && matches!(biome, GenBiome::Forest | GenBiome::Taiga | GenBiome::Swamp)
                            && palette.fern != CONTENT_UNKNOWN
                        {
                            chunk.nodes[idx] = palette.fern;
                            chunk.param2[idx] = p2_idx;
                        } else if deco_roll == 5 && palette.dandelion != CONTENT_UNKNOWN {
                            chunk.nodes[idx] = palette.dandelion;
                        } else if deco_roll == 6 && palette.poppy != CONTENT_UNKNOWN {
                            chunk.nodes[idx] = palette.poppy;
                        }
                    }
                }
            }
        }
    }

    // Deterministic jittered-grid cross-chunk tree placement (never fuses adjacent trunks,
    // searches neighboring cells >= 10 blocks outside chunk bounds across full schematic Y range).
    if base_y < tree_max_y + 24 && base_y + 16 > SEA_LEVEL {
        let min_cell_x = (base_x - TREE_SEARCH_RADIUS).div_euclid(TREE_CELL);
        let max_cell_x = (base_x + 15 + TREE_SEARCH_RADIUS).div_euclid(TREE_CELL);
        let min_cell_z = (base_z - TREE_SEARCH_RADIUS).div_euclid(TREE_CELL);
        let max_cell_z = (base_z + 15 + TREE_SEARCH_RADIUS).div_euclid(TREE_CELL);

        for cell_z in min_cell_z..=max_cell_z {
            for cell_x in min_cell_x..=max_cell_x {
                let chash = hash3d(cell_x, 0, cell_z, seed ^ 0x77EE_C311);
                // Jitter offset inside [2..=6] of 9x9 cell -> minimum trunk separation >= 5 blocks!
                let wx = cell_x * TREE_CELL + 2 + (chash % 5) as i32;
                let wz = cell_z * TREE_CELL + 2 + ((chash >> 4) % 5) as i32;

                let biome = sample_biome(wx, wz, seed);
                let prob_pct = match biome {
                    GenBiome::Forest => 82,
                    GenBiome::BirchForest => 80,
                    GenBiome::Taiga => 65,
                    GenBiome::Swamp => 50,
                    GenBiome::Plains => 15,
                    GenBiome::SnowyTundra => 12,
                    GenBiome::Desert => 0,
                };
                if prob_pct == 0 || ((chash >> 10) % 100) >= prob_pct {
                    continue;
                }

                let surface_y = terrain_height_for(wx, wz, seed, world_type);
                let min_ground = if biome == GenBiome::Swamp {
                    SEA_LEVEL
                } else {
                    SEA_LEVEL + 2
                };
                if surface_y < min_ground || surface_y >= tree_max_y {
                    continue;
                }

                // Reject steep cliffs so trees sit solidly on terrain
                let h_px = terrain_height_for(wx + 1, wz, seed, world_type);
                let h_nx = terrain_height_for(wx - 1, wz, seed, world_type);
                let h_pz = terrain_height_for(wx, wz + 1, seed, world_type);
                let h_nz = terrain_height_for(wx, wz - 1, seed, world_type);
                let h_min = h_px.min(h_nx).min(h_pz).min(h_nz).min(surface_y);
                let h_max = h_px.max(h_nx).max(h_pz).max(h_nz).max(surface_y);
                if h_max - h_min > 2 {
                    continue;
                }

                let root = IVec3::new(wx, surface_y + 1, wz);
                let variant_roll = hash3d(wx, 1, wz, seed ^ 0x3333);
                let rot_roll = ((variant_roll >> 4) % 4) as u8;
                let rot = match rot_roll {
                    1 => SchematicRotation::Rot90,
                    2 => SchematicRotation::Rot180,
                    3 => SchematicRotation::Rot270,
                    _ => SchematicRotation::Rot0,
                };

                if let Some(schem) = select_tree_schematic(biome, variant_roll, palette) {
                    if root.y + schem.size.y <= base_y || root.y >= base_y + 16 {
                        continue;
                    }
                    let seed_mix = ((wx as i64 as u64) << 32) ^ (wz as i64 as u64) ^ seed;
                    stamp_schematic_into_chunk(
                        &mut chunk,
                        schem,
                        root,
                        rot,
                        biome.palette_index(),
                        seed_mix,
                        palette,
                        registries,
                    );
                } else {
                    if root.y + 10 <= base_y || root.y >= base_y + 16 {
                        continue;
                    }
                    stamp_fallback_tree(&mut chunk, root, biome, seed, palette);
                }
            }
        }
    }

    stamp_surface_lava_pools_into_chunk(&mut chunk, seed, world_type, palette);
    stamp_overworld_structures_into_chunk(&mut chunk, seed, world_type, &palette.structures);

    let top_has_air = (0..CHUNK_SIZE)
        .any(|z| (0..CHUNK_SIZE).any(|x| chunk.nodes[crate::world::local_index(x, 15, z)] == CONTENT_AIR));
    let is_sky_chunk = cpos.y >= *chunk_y_range_for(world_type).end() || top_has_air;
    propagate_chunk_light(&mut chunk, registries, is_sky_chunk);
    chunk.dirty_mesh = true;
    chunk.modified = false;
    chunk
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{DrawType, NodeDef};
    use crate::world::light::get_sky_light;
    use std::collections::HashSet;

    fn make_full_test_registry() -> EngineRegistries {
        let mut reg = EngineRegistries::new();
        for name in [
            "mcl_core:stone",
            "mcl_core:dirt",
            "mcl_core:dirt_with_grass",
            "mcl_core:dirt_with_grass_snow",
            "mcl_core:snowblock",
            "mcl_core:podzol",
            "mcl_core:clay",
            "mcl_core:sand",
            "mcl_core:sandstone",
            "mcl_core:gravel",
            "mcl_core:bedrock",
        ] {
            reg.register_node(NodeDef {
                name: name.into(),
                drawtype: DrawType::Normal,
                ..Default::default()
            });
        }
        reg.register_node(NodeDef {
            name: "mcl_core:water_source".into(),
            drawtype: DrawType::Liquid,
            ..Default::default()
        });
        for name in [
            "mcl_trees:tree_oak",
            "mcl_trees:tree_birch",
            "mcl_trees:tree_spruce",
        ] {
            reg.register_node(NodeDef {
                name: name.into(),
                drawtype: DrawType::Normal,
                paramtype2: "facedir".into(),
                ..Default::default()
            });
        }
        for name in [
            "mcl_trees:leaves_oak",
            "mcl_trees:leaves_birch",
            "mcl_trees:leaves_spruce",
        ] {
            reg.register_node(NodeDef {
                name: name.into(),
                drawtype: DrawType::AllfacesOptional,
                paramtype: "light".into(),
                ..Default::default()
            });
        }
        reg
    }

    #[test]
    fn world_gen_types_produce_expected_terrain() {
        let reg = make_full_test_registry();
        let palette = MapgenPalette::from_registries_with_game_dir(&reg, None);
        let bedrock = reg.get_content_id("mcl_core:bedrock");
        let stone = reg.get_content_id("mcl_core:stone");
        let dirt = reg.get_content_id("mcl_core:dirt");
        let grass = reg.get_content_id("mcl_core:dirt_with_grass");
        let log = reg.get_content_id("mcl_trees:tree_oak");

        // Superflat: bedrock at y=-4, dirt at -3..-1, grass at y=0, air above
        let flat_neg = generate_chunk_for_world(
            IVec3::new(0, -1, 0),
            1337,
            WorldGenType::Superflat,
            &palette,
            &reg,
        );
        assert_eq!(flat_neg.get(5, 12, 5).0, bedrock);
        assert_eq!(flat_neg.get(5, 15, 5).0, dirt);
        let flat_zero = generate_chunk_for_world(
            IVec3::new(0, 0, 0),
            1337,
            WorldGenType::Superflat,
            &palette,
            &reg,
        );
        assert_eq!(flat_zero.get(5, 0, 5).0, grass);
        assert_eq!(flat_zero.get(5, 1, 5).0, CONTENT_AIR);

        // Skyblock: starter island at (0, 10..=12, 0) + tree at (2, 13, 2)
        let sky = generate_chunk_for_world(
            IVec3::new(0, 0, 0),
            1337,
            WorldGenType::Skyblock,
            &palette,
            &reg,
        );
        assert_eq!(sky.get(0, 10, 0).0, bedrock);
        assert_eq!(sky.get(1, 10, 1).0, stone);
        assert_eq!(sky.get(1, 11, 1).0, dirt);
        assert_eq!(sky.get(1, 12, 1).0, grass);
        assert_eq!(sky.get(2, 13, 2).0, log);
        assert_eq!(sky.get(2, 13, 2).2, 0);
        assert_eq!(sky.get(14, 12, 14).0, CONTENT_AIR);
    }

    #[test]
    fn sea_level_62_biomes_jittered_trees_and_cross_chunk_continuity() {
        assert_eq!(SEA_LEVEL, 62);
        assert_eq!(chunk_y_range_for(WorldGenType::Default), 0..=8);
        assert_eq!(chunk_y_range_for(WorldGenType::Amplified), 0..=8);
        assert_eq!(chunk_y_range_for(WorldGenType::Superflat), -1..=2);
        assert_eq!(chunk_y_range_for(WorldGenType::Skyblock), -1..=2);

        let reg = make_full_test_registry();
        let palette = MapgenPalette::from_registries_with_game_dir(&reg, None);
        let bedrock = reg.get_content_id("mcl_core:bedrock");
        let oak_log = reg.get_content_id("mcl_trees:tree_oak");
        let birch_log = reg.get_content_id("mcl_trees:tree_birch");
        let spruce_log = reg.get_content_id("mcl_trees:tree_spruce");
        let oak_leaves = reg.get_content_id("mcl_trees:leaves_oak");
        let birch_leaves = reg.get_content_id("mcl_trees:leaves_birch");
        let spruce_leaves = reg.get_content_id("mcl_trees:leaves_spruce");

        // All 7 biomes must be reachable around spawn
        let mut seen = HashSet::new();
        for z in (-1200..=1200).step_by(32) {
            for x in (-1200..=1200).step_by(32) {
                let b = sample_biome(x, z, 4242);
                assert_eq!(b.name(), biome_name_at(x, z, 4242));
                seen.insert(b);
            }
        }
        for b in GenBiome::ALL {
            assert!(seen.contains(&b), "Missing biome {b:?}");
        }

        // Smooth biome height blending: no sheer vertical cliff jumps across adjacent columns
        let mut max_step_delta = 0i32;
        for z in (-256..=256).step_by(2) {
            for x in (-256..=256).step_by(2) {
                let h0 = terrain_height_for(x, z, 4242, WorldGenType::Default);
                let hx = terrain_height_for(x + 1, z, 4242, WorldGenType::Default);
                let hz = terrain_height_for(x, z + 1, 4242, WorldGenType::Default);
                max_step_delta = max_step_delta.max((h0 - hx).abs()).max((h0 - hz).abs());
            }
        }
        assert!(
            max_step_delta <= 4,
            "Terrain step across adjacent blocks must remain smooth, got delta={max_step_delta}"
        );

        // Verify trees in generated chunks never fuse, bedrock is ONLY at wy == 0 (never Y=15/16),
        // and sky above surface + trees is completely open air (no ceiling!)
        let mut trunk_positions = Vec::new();
        let mut leaf_count = 0usize;
        for cz in -2..=2 {
            for cx in -2..=2 {
                for cy in chunk_y_range_for(WorldGenType::Default) {
                    let ch = generate_chunk_for_world(
                        IVec3::new(cx, cy, cz),
                        4242,
                        WorldGenType::Default,
                        &palette,
                        &reg,
                    );
                    for ly in 0..16 {
                        let wy = cy * 16 + ly as i32;
                        for lz in 0..16 {
                            for lx in 0..16 {
                                let wx = cx * 16 + lx as i32;
                                let wz = cz * 16 + lz as i32;
                                let (id, p1, p2) = ch.get(lx, ly, lz);
                                if wy == 0 {
                                    assert_eq!(id, bedrock, "Bottom floor Y=0 must be bedrock");
                                } else {
                                    assert_ne!(
                                        id, bedrock,
                                        "Bedrock must NEVER spawn above world floor (found at {wx},{wy},{wz})"
                                    );
                                }
                                // Top of world range above all surface terrain & trees is open sky air
                                if cy == *chunk_y_range_for(WorldGenType::Default).end() && wy >= 130 {
                                    assert_eq!(id, CONTENT_AIR, "Sky ceiling must remain open air");
                                }
                                if id == oak_log || id == birch_log || id == spruce_log {
                                    assert_eq!(p2, 0, "Trunk param2 must be 0");
                                    trunk_positions.push((wx, wy, wz));
                                } else if id == oak_leaves || id == birch_leaves || id == spruce_leaves {
                                    leaf_count += 1;
                                    assert!(
                                        get_sky_light(p1) >= 10,
                                        "Leaf at ({wx},{wy},{wz}) too dark: sky={}",
                                        get_sky_light(p1)
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(leaf_count > 20, "Expected trees to spawn across terrain");
        // No two distinct tree columns should be adjacent (Chebyshev distance >= 5 unless same X,Z trunk)
        for &(x1, _, z1) in &trunk_positions {
            for &(x2, _, z2) in &trunk_positions {
                if (x1, z1) != (x2, z2) {
                    let dist = (x1 - x2).abs().max((z1 - z2).abs());
                    assert!(
                        dist >= 5,
                        "Fused trunks detected at ({x1},{z1}) and ({x2},{z2}) dist={dist}"
                    );
                }
            }
        }

        // Verify Desert produces sand/sandstone and ZERO trees, and dry spawn works
        let spawn = find_spawn_pos(1337, WorldGenType::Default);
        assert!(spawn.y > SEA_LEVEL + 1, "Spawn {spawn:?} must be dry above sea level");
        let paths = EnginePaths::resolve();
        if paths.game_dir.join("mods/ITEMS/mcl_core/schematics").is_dir() {
            let real_pal = MapgenPalette::from_registries(&reg);
            assert!(!real_pal.oak_trees.is_empty());
            assert!(!real_pal.birch_trees.is_empty());
            assert!(!real_pal.spruce_trees.is_empty());
            assert!(!real_pal.swamp_oak_trees.is_empty());

            // Find a chunk in Desert and verify zero trees + sand/sandstone
            let mut checked_desert = false;
            'desert_search: for gz in (-1200..=1200).step_by(32) {
                for gx in (-1200..=1200).step_by(32) {
                    if sample_biome(gx, gz, 1337) == GenBiome::Desert
                        && terrain_height_for(gx, gz, 1337, WorldGenType::Default) > SEA_LEVEL + 2
                    {
                        let cx = gx.div_euclid(16);
                        let cz = gz.div_euclid(16);
                        let lx = gx.rem_euclid(16) as usize;
                        let lz = gz.rem_euclid(16) as usize;
                        let h = terrain_height_for(gx, gz, 1337, WorldGenType::Default);
                        let cy = h.div_euclid(16);
                        let ly = h.rem_euclid(16) as usize;
                        let ch = generate_chunk_for_world(
                            IVec3::new(cx, cy, cz),
                            1337,
                            WorldGenType::Default,
                            &real_pal,
                            &reg,
                        );
                        assert_eq!(ch.get(lx, ly, lz).0, reg.get_content_id("mcl_core:sand"));
                        checked_desert = true;
                        break 'desert_search;
                    }
                }
            }
            assert!(checked_desert, "Expected to inspect a Desert chunk");
        }
    }
}
