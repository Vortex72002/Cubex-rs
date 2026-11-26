use glam::IVec3;
use rayon::prelude::*;
use std::collections::HashSet;

use super::biome::{fbm_2d, hash2d};
use super::light::propagate_world_light;
use super::structures::{
    populate_structure_chests_for_chunk, stamp_nether_structures_into_chunk, StructurePalette,
};
use super::{local_index, Chunk, WorldGenType, WorldStore, CHUNK_SIZE};
use crate::registry::{EngineRegistries, CONTENT_AIR, CONTENT_UNKNOWN};

pub const NETHER_MIN_Y: i32 = 0;
pub const NETHER_MAX_Y: i32 = 127;
pub const NETHER_LAVA_LEVEL: i32 = 31;
pub const NETHER_CHUNK_Y_RANGE: std::ops::RangeInclusive<i32> = 0..=7;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NetherBiome {
    NetherWastes,
    CrimsonForest,
    WarpedForest,
    BasaltDeltas,
    SoulSandValley,
}

impl NetherBiome {
    pub const ALL: [NetherBiome; 5] = [
        Self::NetherWastes,
        Self::CrimsonForest,
        Self::WarpedForest,
        Self::BasaltDeltas,
        Self::SoulSandValley,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::NetherWastes => "Nether Wastes",
            Self::CrimsonForest => "Crimson Forest",
            Self::WarpedForest => "Warped Forest",
            Self::BasaltDeltas => "Basalt Deltas",
            Self::SoulSandValley => "Soul Sand Valley",
        }
    }
}

#[derive(Debug, Clone)]
pub struct NetherPalette {
    pub netherrack: u16,
    pub bedrock: u16,
    pub lava_source: u16,
    pub glowstone: u16,
    pub magma: u16,
    pub quartz_ore: u16,
    pub gold_ore: u16,
    pub ancient_debris: u16,
    pub soul_sand: u16,
    pub soul_soil: u16,
    pub gravel: u16,
    pub crimson_nylium: u16,
    pub warped_nylium: u16,
    pub nether_wart_block: u16,
    pub warped_wart_block: u16,
    pub shroomlight: u16,
    pub crimson_stem: u16,
    pub warped_stem: u16,
    pub crimson_roots: u16,
    pub crimson_fungus: u16,
    pub warped_roots: u16,
    pub nether_sprouts: u16,
    pub warped_fungus: u16,
    pub blackstone: u16,
    pub basalt: u16,
    pub basalt_smooth: u16,
    pub fire: u16,
    pub soul_fire: u16,
    pub structures: StructurePalette,
}

impl NetherPalette {
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

        let stone_fb = resolve(&["mcl_core:stone"], CONTENT_AIR);
        let netherrack = resolve(&["mcl_nether:netherrack"], stone_fb);
        let bedrock = resolve(&["mcl_core:bedrock"], netherrack);
        let lava_source = resolve(
            &[
                "mcl_nether:nether_lava_source",
                "mcl_core:lava_source",
            ],
            CONTENT_AIR,
        );
        let glowstone = resolve(&["mcl_nether:glowstone"], netherrack);
        let magma = resolve(&["mcl_nether:magma"], netherrack);
        let quartz_ore = resolve(&["mcl_nether:quartz_ore"], netherrack);
        let gold_ore = resolve(&["mcl_blackstone:nether_gold"], quartz_ore);
        let ancient_debris = resolve(&["mcl_nether:ancient_debris"], bedrock);
        let soul_sand = resolve(&["mcl_nether:soul_sand"], netherrack);
        let soul_soil = resolve(&["mcl_blackstone:soul_soil"], soul_sand);
        let gravel = resolve(&["mcl_core:gravel"], netherrack);
        let crimson_nylium = resolve(&["mcl_crimson:crimson_nylium"], netherrack);
        let warped_nylium = resolve(&["mcl_crimson:warped_nylium"], netherrack);
        let nether_wart_block = resolve(&["mcl_nether:nether_wart_block"], crimson_nylium);
        let warped_wart_block = resolve(&["mcl_crimson:warped_wart_block"], warped_nylium);
        let shroomlight = resolve(&["mcl_crimson:shroomlight"], glowstone);
        let crimson_stem = resolve(
            &["mcl_crimson:crimson_hyphae", "mcl_crimson:crimson_hyphae_wood"],
            nether_wart_block,
        );
        let warped_stem = resolve(
            &["mcl_crimson:warped_hyphae", "mcl_crimson:warped_hyphae_wood"],
            warped_wart_block,
        );
        let blackstone = resolve(&["mcl_blackstone:blackstone"], netherrack);
        let basalt = resolve(&["mcl_blackstone:basalt"], blackstone);
        let basalt_smooth = resolve(&["mcl_blackstone:basalt_smooth"], basalt);

        Self {
            netherrack,
            bedrock,
            lava_source,
            glowstone,
            magma,
            quartz_ore,
            gold_ore,
            ancient_debris,
            soul_sand,
            soul_soil,
            gravel,
            crimson_nylium,
            warped_nylium,
            nether_wart_block,
            warped_wart_block,
            shroomlight,
            crimson_stem,
            warped_stem,
            crimson_roots: resolve(&["mcl_crimson:crimson_roots"], CONTENT_AIR),
            crimson_fungus: resolve(&["mcl_crimson:crimson_fungus"], CONTENT_AIR),
            warped_roots: resolve(&["mcl_crimson:warped_roots"], CONTENT_AIR),
            nether_sprouts: resolve(&["mcl_crimson:nether_sprouts"], CONTENT_AIR),
            warped_fungus: resolve(&["mcl_crimson:warped_fungus"], CONTENT_AIR),
            blackstone,
            basalt,
            basalt_smooth,
            fire: resolve(&["mcl_fire:fire"], CONTENT_AIR),
            soul_fire: resolve(&["mcl_fire:soul_fire", "mcl_fire:fire"], CONTENT_AIR),
            structures: StructurePalette::from_registries(reg),
        }
    }
}

#[inline]
fn hash3(x: i32, y: i32, z: i32, seed: u64) -> u32 {
    let mut n = (x as i64).wrapping_mul(374761393)
        ^ (y as i64).wrapping_mul(1664525)
        ^ (z as i64).wrapping_mul(668265263)
        ^ (seed as i64).wrapping_mul(1274126177);
    n = (n ^ (n >> 13)).wrapping_mul(1274126177);
    (n ^ (n >> 16)) as u32
}

#[inline]
fn hash3f(x: i32, y: i32, z: i32, seed: u64) -> f32 {
    (hash3(x, y, z, seed) & 0xFFFF) as f32 / 65535.0
}

#[inline]
fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn value_noise_3d(x: f32, y: f32, z: f32, seed: u64) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let zi = z.floor() as i32;
    let xf = smoothstep(x - xi as f32);
    let yf = smoothstep(y - yi as f32);
    let zf = smoothstep(z - zi as f32);

    let c000 = hash3f(xi, yi, zi, seed);
    let c100 = hash3f(xi + 1, yi, zi, seed);
    let c010 = hash3f(xi, yi + 1, zi, seed);
    let c110 = hash3f(xi + 1, yi + 1, zi, seed);
    let c001 = hash3f(xi, yi, zi + 1, seed);
    let c101 = hash3f(xi + 1, yi, zi + 1, seed);
    let c011 = hash3f(xi, yi + 1, zi + 1, seed);
    let c111 = hash3f(xi + 1, yi + 1, zi + 1, seed);

    let x00 = c000 + (c100 - c000) * xf;
    let x10 = c010 + (c110 - c010) * xf;
    let x01 = c001 + (c101 - c001) * xf;
    let x11 = c011 + (c111 - c011) * xf;

    let y0 = x00 + (x10 - x00) * yf;
    let y1 = x01 + (x11 - x01) * yf;
    y0 + (y1 - y0) * zf
}

/// Deterministic Nether biome selection from warped climate noise fields.
pub fn sample_nether_biome(wx: i32, wz: i32, seed: u64) -> NetherBiome {
    let ns = seed ^ 0x4E45_5448_4552_u64;
    let heat = fbm_2d(wx as f32 * 0.0065, wz as f32 * 0.0065, ns ^ 0x11A5, 3);
    let weir = fbm_2d(wx as f32 * 0.0065, wz as f32 * 0.0065, ns ^ 0x77C9, 3);
    let h = ((heat - 0.5) * 2.5 + 0.5).clamp(0.0, 1.0);
    let w = ((weir - 0.5) * 2.5 + 0.5).clamp(0.0, 1.0);

    if h > 0.66 {
        if w > 0.52 {
            NetherBiome::BasaltDeltas
        } else {
            NetherBiome::CrimsonForest
        }
    } else if h < 0.34 {
        if w > 0.48 {
            NetherBiome::WarpedForest
        } else {
            NetherBiome::SoulSandValley
        }
    } else {
        NetherBiome::NetherWastes
    }
}

pub fn nether_biome_name_at(wx: i32, wz: i32, seed: u64) -> &'static str {
    sample_nether_biome(wx, wz, seed).name()
}

#[inline]
fn eval_nether_density(wx: i32, wy: i32, wz: i32, ns: u64) -> f32 {
    if wy <= 0 || wy >= 127 {
        return 1.0;
    }
    let y = wy as f32;
    let fbm_cave = value_noise_3d(wx as f32 * 0.036, y * 0.052, wz as f32 * 0.036, ns) * 0.68
        + value_noise_3d(wx as f32 * 0.085, y * 0.11, wz as f32 * 0.085, ns ^ 0x99F1) * 0.32;

    // 2D continent shelf height carves deep y=20..31 lava sea basins alongside y=34..48 plateaus
    let shelf = fbm_2d(wx as f32 * 0.015, wz as f32 * 0.015, ns ^ 0x5EA1, 3);
    let floor_y = 19.0 + ((shelf - 0.34) * 44.0).clamp(0.0, 29.0);
    let floor_bias = if y < floor_y {
        (floor_y - y) * 0.062
    } else if y < 60.0 {
        ((floor_y - y) * 0.032).max(-0.26)
    } else {
        -0.26
    };
    let ceil_bias = if y > 92.0 { (y - 92.0) * 0.048 } else { 0.0 };

    // Occasional magma/netherrack arches spanning floor to ceiling
    let ridge = (value_noise_3d(wx as f32 * 0.018, y * 0.022, wz as f32 * 0.018, ns ^ 0x4B21) - 0.5)
        .abs();
    let pillar = (0.075 - ridge).max(0.0) * 4.5;

    (fbm_cave - 0.50) + floor_bias + ceil_bias + pillar
}

#[inline]
fn trilinear_density(grid: &[f32; 125], lx: usize, ly: usize, lz: usize) -> f32 {
    let gx = lx >> 2;
    let gy = ly >> 2;
    let gz = lz >> 2;
    let tx = (lx & 3) as f32 * 0.25;
    let ty = (ly & 3) as f32 * 0.25;
    let tz = (lz & 3) as f32 * 0.25;

    let idx = |x: usize, y: usize, z: usize| -> usize { (y * 5 + z) * 5 + x };
    let c000 = grid[idx(gx, gy, gz)];
    let c100 = grid[idx(gx + 1, gy, gz)];
    let c010 = grid[idx(gx, gy + 1, gz)];
    let c110 = grid[idx(gx + 1, gy + 1, gz)];
    let c001 = grid[idx(gx, gy, gz + 1)];
    let c101 = grid[idx(gx + 1, gy, gz + 1)];
    let c011 = grid[idx(gx, gy + 1, gz + 1)];
    let c111 = grid[idx(gx + 1, gy + 1, gz + 1)];

    let x00 = c000 + (c100 - c000) * tx;
    let x10 = c010 + (c110 - c010) * tx;
    let x01 = c001 + (c101 - c001) * tx;
    let x11 = c011 + (c111 - c011) * tx;
    let y0 = x00 + (x10 - x00) * ty;
    let y1 = x01 + (x11 - x01) * ty;
    y0 + (y1 - y0) * tz
}

/// Generate one 16x16x16 Nether chunk (`y = 0..127`, bedrock floor/roof, lava ocean `<= 31`, 5 biomes).
pub fn generate_nether_chunk(
    cpos: IVec3,
    seed: u64,
    pal: &NetherPalette,
    _reg: &EngineRegistries,
) -> Chunk {
    let mut chunk = Chunk::new(cpos);
    chunk.param1.fill(0);

    let ns = seed ^ 0x4E45_5448_4552_u64;
    if cpos.y < *NETHER_CHUNK_Y_RANGE.start() {
        chunk.nodes.fill(pal.bedrock);
        return chunk;
    }
    if cpos.y > *NETHER_CHUNK_Y_RANGE.end() {
        chunk.nodes.fill(CONTENT_AIR);
        return chunk;
    }

    let ox = cpos.x * 16;
    let oy = cpos.y * 16;
    let oz = cpos.z * 16;

    // 5x5x5 coarse 3D cavern field (step = 4 voxels)
    let mut grid = [0.0f32; 125];
    for gy in 0..5usize {
        let wy = oy + (gy as i32) * 4;
        for gz in 0..5usize {
            let wz = oz + (gz as i32) * 4;
            for gx in 0..5usize {
                let wx = ox + (gx as i32) * 4;
                grid[(gy * 5 + gz) * 5 + gx] = eval_nether_density(wx, wy, wz, ns);
            }
        }
    }

    let mut biomes = [NetherBiome::NetherWastes; 256];
    for lz in 0..CHUNK_SIZE {
        for lx in 0..CHUNK_SIZE {
            biomes[lz * 16 + lx] = sample_nether_biome(ox + lx as i32, oz + lz as i32, seed);
        }
    }

    // 1. Base stone / air / lava ocean / bedrock bounds pass
    for ly in 0..CHUNK_SIZE {
        let wy = oy + ly as i32;
        for lz in 0..CHUNK_SIZE {
            let wz = oz + lz as i32;
            for lx in 0..CHUNK_SIZE {
                let wx = ox + lx as i32;
                let idx = local_index(lx, ly, lz);

                if wy == 0 || wy == 127 {
                    chunk.nodes[idx] = pal.bedrock;
                    continue;
                }
                if (1..=4).contains(&wy) {
                    let h = hash3(wx, wy, wz, ns ^ 0xBD01) % 5;
                    if (h as i32) >= wy {
                        chunk.nodes[idx] = pal.bedrock;
                        continue;
                    }
                }
                if (123..=126).contains(&wy) {
                    let dist_top = 127 - wy;
                    let h = hash3(wx, wy, wz, ns ^ 0xBD02) % 5;
                    if (h as i32) >= dist_top {
                        chunk.nodes[idx] = pal.bedrock;
                        continue;
                    }
                }

                let dens = trilinear_density(&grid, lx, ly, lz);
                if dens > 0.0 {
                    let biome = biomes[lz * 16 + lx];
                    let base = match biome {
                        NetherBiome::BasaltDeltas => {
                            let h = hash3(wx, wy, wz, ns ^ 0xBA5A);
                            if h % 11 < 4 {
                                pal.basalt
                            } else {
                                pal.blackstone
                            }
                        }
                        _ => {
                            let h = hash3(wx, wy, wz, ns ^ 0x0AE5);
                            if h % 47 == 0 {
                                pal.quartz_ore
                            } else if h % 79 == 0 {
                                pal.gold_ore
                            } else if wy <= 24 && h % 260 == 0 {
                                pal.ancient_debris
                            } else {
                                pal.netherrack
                            }
                        }
                    };
                    chunk.nodes[idx] = base;
                } else if wy <= NETHER_LAVA_LEVEL {
                    chunk.nodes[idx] = pal.lava_source;
                } else {
                    chunk.nodes[idx] = CONTENT_AIR;
                }
            }
        }
    }

    // 2. Biome topsoil, ceiling glowstone clusters, basalt columns, and flora pass
    for lz in 0..CHUNK_SIZE {
        let wz = oz + lz as i32;
        for lx in 0..CHUNK_SIZE {
            let wx = ox + lx as i32;
            let biome = biomes[lz * 16 + lx];
            for ly in 0..CHUNK_SIZE {
                let wy = oy + ly as i32;
                if wy <= 4 || wy >= 123 {
                    continue;
                }
                let idx = local_index(lx, ly, lz);
                let cur = chunk.nodes[idx];
                if cur == CONTENT_AIR || cur == pal.lava_source || cur == pal.bedrock {
                    continue;
                }

                let above_id = if ly + 1 < CHUNK_SIZE {
                    chunk.nodes[local_index(lx, ly + 1, lz)]
                } else {
                    if eval_nether_density(wx, wy + 1, wz, ns) > 0.0 {
                        pal.netherrack
                    } else if wy + 1 <= NETHER_LAVA_LEVEL {
                        pal.lava_source
                    } else {
                        CONTENT_AIR
                    }
                };
                let below_id = if ly > 0 {
                    chunk.nodes[local_index(lx, ly - 1, lz)]
                } else {
                    if eval_nether_density(wx, wy - 1, wz, ns) > 0.0 {
                        pal.netherrack
                    } else if wy - 1 <= NETHER_LAVA_LEVEL {
                        pal.lava_source
                    } else {
                        CONTENT_AIR
                    }
                };

                // Exposed floor surface above lava sea
                if above_id == CONTENT_AIR && wy > NETHER_LAVA_LEVEL {
                    let h = hash3(wx, wy, wz, ns ^ 0x50F1);
                    match biome {
                        NetherBiome::CrimsonForest => {
                            chunk.nodes[idx] = pal.crimson_nylium;
                            if ly + 1 < CHUNK_SIZE && h % 29 == 0 && pal.crimson_roots != CONTENT_AIR
                            {
                                chunk.nodes[local_index(lx, ly + 1, lz)] = if h % 58 == 0
                                    && pal.crimson_fungus != CONTENT_AIR
                                {
                                    pal.crimson_fungus
                                } else {
                                    pal.crimson_roots
                                };
                            }
                        }
                        NetherBiome::WarpedForest => {
                            chunk.nodes[idx] = pal.warped_nylium;
                            if ly + 1 < CHUNK_SIZE && h % 23 == 0 && pal.warped_roots != CONTENT_AIR
                            {
                                chunk.nodes[local_index(lx, ly + 1, lz)] = if h % 69 == 0
                                    && pal.warped_fungus != CONTENT_AIR
                                {
                                    pal.warped_fungus
                                } else if h % 2 == 0 && pal.nether_sprouts != CONTENT_AIR {
                                    pal.nether_sprouts
                                } else {
                                    pal.warped_roots
                                };
                            }
                        }
                        NetherBiome::SoulSandValley => {
                            chunk.nodes[idx] = if h % 3 == 0 {
                                pal.soul_soil
                            } else {
                                pal.soul_sand
                            };
                            if ly > 0
                                && chunk.nodes[local_index(lx, ly - 1, lz)] == pal.netherrack
                            {
                                chunk.nodes[local_index(lx, ly - 1, lz)] = pal.soul_sand;
                            }
                            if ly + 1 < CHUNK_SIZE
                                && wy < 90
                                && h % 97 == 0
                                && pal.soul_fire != CONTENT_AIR
                            {
                                chunk.nodes[local_index(lx, ly + 1, lz)] = pal.soul_fire;
                            }
                        }
                        NetherBiome::BasaltDeltas => {
                            if h % 19 == 0 && wy <= 42 {
                                chunk.nodes[idx] = pal.magma;
                            } else if h % 13 < 4 {
                                chunk.nodes[idx] = pal.basalt_smooth;
                            }
                            // Basalt spire columns inside chunk bounds
                            if h % 29 == 0 && ly + 3 < CHUNK_SIZE {
                                let spire_h = (1 + (h % 4) as usize).min(CHUNK_SIZE - 1 - ly);
                                for dy in 1..=spire_h {
                                    let si = local_index(lx, ly + dy, lz);
                                    if chunk.nodes[si] == CONTENT_AIR {
                                        chunk.nodes[si] = pal.basalt;
                                    }
                                }
                            }
                        }
                        NetherBiome::NetherWastes => {
                            if wy <= 35 && (h % 7) == 0 {
                                chunk.nodes[idx] = pal.magma;
                            } else if wy <= 35 && (h % 11) == 0 {
                                chunk.nodes[idx] = pal.gravel;
                            } else if (h % 53) == 0 && ly + 1 < CHUNK_SIZE && pal.fire != CONTENT_AIR
                            {
                                chunk.nodes[local_index(lx, ly + 1, lz)] = pal.fire;
                            }
                        }
                    }
                }

                // Glowing ceiling glowstone clusters
                if below_id == CONTENT_AIR && wy >= 72 && wy <= 122 {
                    let gnoise = fbm_2d(wx as f32 * 0.08, wz as f32 * 0.08, ns ^ 0x610E, 2);
                    if gnoise > 0.72 && ly >= 2 {
                        chunk.nodes[idx] = pal.glowstone;
                        chunk.nodes[local_index(lx, ly - 1, lz)] = pal.glowstone;
                        if hash3(wx, wy, wz, ns ^ 0x610F) % 3 == 0 {
                            chunk.nodes[local_index(lx, ly - 2, lz)] = pal.glowstone;
                        }
                    }
                }
            }
        }
    }

    // 3. Procedural Huge Crimson & Warped Fungi Trees (safe inside 16x16x16 margin)
    for lz in 2..(CHUNK_SIZE - 2) {
        let wz = oz + lz as i32;
        for lx in 2..(CHUNK_SIZE - 2) {
            let wx = ox + lx as i32;
            let biome = biomes[lz * 16 + lx];
            if !matches!(biome, NetherBiome::CrimsonForest | NetherBiome::WarpedForest) {
                continue;
            }
            if hash2d(wx, wz, ns ^ 0xF091) < 0.968 {
                continue;
            }
            let ground_nylium = if biome == NetherBiome::CrimsonForest {
                pal.crimson_nylium
            } else {
                pal.warped_nylium
            };
            let stem = if biome == NetherBiome::CrimsonForest {
                pal.crimson_stem
            } else {
                pal.warped_stem
            };
            let cap = if biome == NetherBiome::CrimsonForest {
                pal.nether_wart_block
            } else {
                pal.warped_wart_block
            };
            for ly in 1..(CHUNK_SIZE - 6) {
                let idx = local_index(lx, ly, lz);
                if chunk.nodes[idx] == ground_nylium
                    && chunk.nodes[local_index(lx, ly + 1, lz)] == CONTENT_AIR
                    && chunk.nodes[local_index(lx, ly + 5, lz)] == CONTENT_AIR
                {
                    let h = hash3(wx, oy + ly as i32, wz, ns ^ 0x7711);
                    let trunk_h = (3 + (h % 3) as usize).min(CHUNK_SIZE - 2 - ly);
                    for dy in 1..=trunk_h {
                        chunk.nodes[local_index(lx, ly + dy, lz)] = stem;
                    }
                    let cap_base = (ly + trunk_h - 1).min(CHUNK_SIZE - 2);
                    for dcy in 0..=1usize {
                        let py = cap_base + dcy;
                        if py >= CHUNK_SIZE {
                            break;
                        }
                        let rad = if dcy == 0 { 2isize } else { 1isize };
                        for dz in -rad..=rad {
                            for dx in -rad..=rad {
                                if dx.abs() == 2 && dz.abs() == 2 {
                                    continue;
                                }
                                let px = (lx as isize + dx) as usize;
                                let pz = (lz as isize + dz) as usize;
                                let ci = local_index(px, py, pz);
                                if chunk.nodes[ci] == CONTENT_AIR {
                                    let is_light =
                                        dcy == 0 && (dx.abs() + dz.abs() == 1) && (h % 4 == 0);
                                    chunk.nodes[ci] = if is_light { pal.shroomlight } else { cap };
                                }
                            }
                        }
                    }
                    break;
                }
            }
        }
    }

    stamp_nether_structures_into_chunk(&mut chunk, seed, &pal.structures);

    chunk
}

/// Synchronously ensures a square radius of Nether chunk columns (`y = 0..=7`) exists and is lit.
pub fn ensure_nether_columns(
    nether_world: &WorldStore,
    center_cx: i32,
    center_cz: i32,
    radius: i32,
    seed: u64,
    pal: &NetherPalette,
    reg: &EngineRegistries,
) -> Vec<IVec3> {
    let existing: HashSet<IVec3> = nether_world.chunks.read().keys().copied().collect();
    let mut missing = Vec::new();
    for cz in (center_cz - radius)..=(center_cz + radius) {
        for cx in (center_cx - radius)..=(center_cx + radius) {
            for cy in NETHER_CHUNK_Y_RANGE {
                let cpos = IVec3::new(cx, cy, cz);
                if !existing.contains(&cpos) {
                    missing.push(cpos);
                }
            }
        }
    }
    if missing.is_empty() {
        return Vec::new();
    }
    let chunks: Vec<Chunk> = missing
        .into_par_iter()
        .map(|cpos| generate_nether_chunk(cpos, seed, pal, reg))
        .collect();
    let coords: Vec<IVec3> = chunks.iter().map(|c| c.pos).collect();
    for ch in chunks {
        nether_world.insert_chunk(ch);
    }
    for &cpos in &coords {
        populate_structure_chests_for_chunk(nether_world, cpos, seed, WorldGenType::Default, reg);
    }
    propagate_world_light(nether_world, &coords, reg);
    coords
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{DrawType, NodeDef};
    use crate::world::dimension::DimensionId;

    #[test]
    fn nether_worldgen_bedrock_bounds_lava_sea_and_biomes() {
        let mut reg = EngineRegistries::new();
        for (name, dt, light) in [
            ("mcl_nether:netherrack", DrawType::Normal, 0),
            ("mcl_nether:glowstone", DrawType::Normal, 14),
            ("mcl_nether:quartz_ore", DrawType::Normal, 0),
            ("mcl_nether:magma", DrawType::Normal, 3),
            ("mcl_nether:soul_sand", DrawType::Normal, 0),
            ("mcl_nether:nether_lava_source", DrawType::Liquid, 15),
            ("mcl_nether:nether_wart_block", DrawType::Normal, 0),
            ("mcl_nether:ancient_debris", DrawType::Normal, 0),
            ("mcl_crimson:crimson_nylium", DrawType::Normal, 0),
            ("mcl_crimson:warped_nylium", DrawType::Normal, 0),
            ("mcl_crimson:warped_wart_block", DrawType::Normal, 0),
            ("mcl_crimson:shroomlight", DrawType::Normal, 15),
            ("mcl_crimson:crimson_hyphae", DrawType::Normal, 0),
            ("mcl_crimson:warped_hyphae", DrawType::Normal, 0),
            ("mcl_blackstone:blackstone", DrawType::Normal, 0),
            ("mcl_blackstone:basalt", DrawType::Normal, 0),
            ("mcl_blackstone:basalt_smooth", DrawType::Normal, 0),
            ("mcl_blackstone:soul_soil", DrawType::Normal, 0),
            ("mcl_blackstone:nether_gold", DrawType::Normal, 0),
            ("mcl_core:bedrock", DrawType::Normal, 0),
            ("mcl_core:gravel", DrawType::Normal, 0),
        ] {
            reg.register_node(NodeDef {
                name: name.into(),
                drawtype: dt,
                walkable: dt == DrawType::Normal,
                sunlight_propagates: dt != DrawType::Normal,
                light_source: light,
                paramtype: "light".into(),
                ..Default::default()
            });
        }
        let pal = NetherPalette::from_registries(&reg);
        let dir = std::env::temp_dir().join("cubex_rs_nether_worldgen_test");
        let _ = std::fs::remove_dir_all(&dir);
        let nether_world = WorldStore::new_for_dimension(
            &dir,
            777_421,
            DimensionId::Nether,
            crate::world::WorldGenType::Default,
            crate::world::Difficulty::Normal,
            crate::world::Gamerules::default(),
            "Survival",
        );
        let coords = ensure_nether_columns(&nether_world, 0, 0, 1, 777_421, &pal, &reg);
        assert!(!coords.is_empty());

        let mut saw_lava_ocean = false;
        let mut saw_mid_air = false;
        let mut max_block_light = 0u8;
        for z in -16..32 {
            for x in -16..32 {
                assert_eq!(nether_world.get_node(IVec3::new(x, 0, z)).0, pal.bedrock);
                assert_eq!(nether_world.get_node(IVec3::new(x, 127, z)).0, pal.bedrock);
                for y in 0..=127 {
                    let (id, p1, _) = nether_world.get_node(IVec3::new(x, y, z));
                    let sky = p1 & 0x0F;
                    let blk = p1 >> 4;
                    assert_eq!(sky, 0, "Nether must have zero skylight at ({x},{y},{z})");
                    max_block_light = max_block_light.max(blk);
                    if y <= NETHER_LAVA_LEVEL && id == pal.lava_source {
                        saw_lava_ocean = true;
                    }
                    if (40..=90).contains(&y) && id == CONTENT_AIR {
                        saw_mid_air = true;
                    }
                }
            }
        }
        assert!(saw_lava_ocean, "Expected Nether lava ocean at y <= 31");
        assert!(saw_mid_air, "Expected open 3D Nether caverns at y in 40..=90");
        assert!(max_block_light >= 14, "Expected emissive lava/glowstone blocklight");

        // All 5 Nether biomes must appear over a wide sample grid
        let mut found_biomes = HashSet::new();
        for z in (-2500..=2500).step_by(80) {
            for x in (-2500..=2500).step_by(80) {
                found_biomes.insert(sample_nether_biome(x, z, 777_421));
            }
        }
        assert_eq!(
            found_biomes.len(),
            NetherBiome::ALL.len(),
            "All 5 Nether biomes should be reachable: found {:?}",
            found_biomes
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
