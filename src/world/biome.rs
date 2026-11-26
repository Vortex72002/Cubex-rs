use serde::{Deserialize, Serialize};

/// Authentic Mineclonia / Minecraft overworld sea level (`Y = 62`).
pub const SEA_LEVEL: i32 = 62;

/// Core overworld Mineclonia biomes supported by Cubex-RS mapgen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GenBiome {
    Plains,
    Forest,
    BirchForest,
    Desert,
    Taiga,
    Swamp,
    SnowyTundra,
}

impl GenBiome {
    pub const ALL: [GenBiome; 7] = [
        Self::Plains,
        Self::Forest,
        Self::BirchForest,
        Self::Desert,
        Self::Taiga,
        Self::Swamp,
        Self::SnowyTundra,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Plains => "Plains",
            Self::Forest => "Forest",
            Self::BirchForest => "BirchForest",
            Self::Desert => "Desert",
            Self::Taiga => "Taiga",
            Self::Swamp => "Swamp",
            Self::SnowyTundra => "SnowyTundra",
        }
    }

    /// Mineclonia `mcl_core_palette_grass.png` / `mcl_core_palette_leaves.png` low 5-bit palette index.
    pub fn palette_index(self) -> u8 {
        match self {
            Self::Plains => 11,
            Self::Forest => 13,
            Self::BirchForest => 15,
            Self::Desert => 17,
            Self::Taiga => 12,
            Self::Swamp => 28,
            Self::SnowyTundra => 10,
        }
    }
}

/// Terrain elevation & relief parameters for each biome, blended smoothly across borders.
#[derive(Debug, Clone, Copy)]
pub struct BiomeTerrainProfile {
    pub base_height: f32,
    pub continental_amp: f32,
    pub hill_amp: f32,
    pub ridge_amp: f32,
}

#[inline]
pub fn biome_profile(b: GenBiome) -> BiomeTerrainProfile {
    match b {
        GenBiome::Plains => BiomeTerrainProfile {
            base_height: 65.5,
            continental_amp: 8.5,
            hill_amp: 3.6,
            ridge_amp: 2.5,
        },
        GenBiome::Forest => BiomeTerrainProfile {
            base_height: 66.5,
            continental_amp: 11.0,
            hill_amp: 5.5,
            ridge_amp: 4.5,
        },
        GenBiome::BirchForest => BiomeTerrainProfile {
            base_height: 67.0,
            continental_amp: 11.5,
            hill_amp: 6.0,
            ridge_amp: 4.8,
        },
        GenBiome::Desert => BiomeTerrainProfile {
            base_height: 66.0,
            continental_amp: 9.0,
            hill_amp: 6.8,
            ridge_amp: 3.5,
        },
        GenBiome::Taiga => BiomeTerrainProfile {
            base_height: 68.5,
            continental_amp: 13.5,
            hill_amp: 8.5,
            ridge_amp: 10.0,
        },
        GenBiome::Swamp => BiomeTerrainProfile {
            base_height: 62.6,
            continental_amp: 2.2,
            hill_amp: 1.8,
            ridge_amp: 0.8,
        },
        GenBiome::SnowyTundra => BiomeTerrainProfile {
            base_height: 66.8,
            continental_amp: 9.5,
            hill_amp: 4.2,
            ridge_amp: 3.4,
        },
    }
}

#[inline]
pub fn hash2d(x: i32, z: i32, seed: u64) -> f32 {
    let mut n = (x as i64).wrapping_mul(374761393)
        ^ (z as i64).wrapping_mul(668265263)
        ^ (seed as i64).wrapping_mul(1274126177);
    n = (n ^ (n >> 13)).wrapping_mul(1274126177);
    ((n ^ (n >> 16)) & 0xFFFF) as f32 / 65535.0
}

#[inline]
fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn value_noise_2d(x: f32, z: f32, seed: u64) -> f32 {
    let xi = x.floor() as i32;
    let zi = z.floor() as i32;
    let xf = smoothstep(x - xi as f32);
    let zf = smoothstep(z - zi as f32);

    let v00 = hash2d(xi, zi, seed);
    let v10 = hash2d(xi + 1, zi, seed);
    let v01 = hash2d(xi, zi + 1, seed);
    let v11 = hash2d(xi + 1, zi + 1, seed);

    let a = v00 + (v10 - v00) * xf;
    let b = v01 + (v11 - v01) * xf;
    a + (b - a) * zf
}

pub fn fbm_2d(x: f32, z: f32, seed: u64, octaves: u32) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut freq = 1.0;
    let mut norm = 0.0;
    for o in 0..octaves {
        sum += value_noise_2d(x * freq, z * freq, seed.wrapping_add(o as u64 * 997)) * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.02;
    }
    sum / norm.max(0.001)
}

#[inline]
pub fn climate_at(wx: i32, wz: i32, seed: u64) -> (f32, f32) {
    let raw_t = fbm_2d(wx as f32 * 0.0052, wz as f32 * 0.0052, seed ^ 0xB10E, 3);
    let raw_h = fbm_2d(wx as f32 * 0.0052, wz as f32 * 0.0052, seed ^ 0xC4F3, 3);
    let temp = ((raw_t - 0.5) * 2.45 + 0.5).clamp(0.0, 1.0);
    let hum = ((raw_h - 0.5) * 2.45 + 0.5).clamp(0.0, 1.0);
    (temp, hum)
}

/// Deterministic biome lookup at world `(wx, wz)` for a given seed.
pub fn sample_biome(wx: i32, wz: i32, seed: u64) -> GenBiome {
    let (temp, hum) = climate_at(wx, wz, seed);
    if temp < 0.28 {
        if hum < 0.48 {
            GenBiome::SnowyTundra
        } else {
            GenBiome::Taiga
        }
    } else if temp >= 0.70 {
        if hum < 0.62 {
            GenBiome::Desert
        } else {
            GenBiome::Swamp
        }
    } else if hum < 0.35 {
        GenBiome::Plains
    } else if hum < 0.58 {
        GenBiome::Forest
    } else if hum < 0.78 {
        GenBiome::BirchForest
    } else {
        GenBiome::Swamp
    }
}

/// Human-readable biome name at world `(wx, wz)` (for F3 HUD & diagnostics).
pub fn biome_name_at(wx: i32, wz: i32, seed: u64) -> &'static str {
    sample_biome(wx, wz, seed).name()
}

/// Smoothly interpolate terrain elevation parameters across neighboring biomes using
/// a C1-continuous, compactly-supported 4x4 kernel (`(1 - (u/2)^2)^2`) evaluated on a fixed
/// 8-block world lattice (`16`-block blend radius). Because weights vanish smoothly at `|u| = 2`,
/// adjacent columns never suffer discrete tap-aliasing jumps across biome borders.
pub fn blended_biome_profile(wx: i32, wz: i32, seed: u64) -> BiomeTerrainProfile {
    const STEP: i32 = 8;
    let gx = wx.div_euclid(STEP);
    let gz = wz.div_euclid(STEP);
    let fx = wx.rem_euclid(STEP) as f32 / STEP as f32;
    let fz = wz.rem_euclid(STEP) as f32 / STEP as f32;

    #[inline]
    fn kernel(u: f32) -> f32 {
        let t = (u * 0.5).abs();
        if t >= 1.0 {
            0.0
        } else {
            let v = 1.0 - t * t;
            v * v
        }
    }

    let wx_k = [
        kernel(-1.0 - fx),
        kernel(0.0 - fx),
        kernel(1.0 - fx),
        kernel(2.0 - fx),
    ];
    let wz_k = [
        kernel(-1.0 - fz),
        kernel(0.0 - fz),
        kernel(1.0 - fz),
        kernel(2.0 - fz),
    ];

    let mut sum_w = 0.0f32;
    let mut base_h = 0.0f32;
    let mut cont_a = 0.0f32;
    let mut hill_a = 0.0f32;
    let mut ridge_a = 0.0f32;

    for (iz, &kz) in wz_k.iter().enumerate() {
        if kz <= 0.0 {
            continue;
        }
        let sz = (gz + iz as i32 - 1) * STEP;
        for (ix, &kx) in wx_k.iter().enumerate() {
            if kx <= 0.0 {
                continue;
            }
            let w = kx * kz;
            let sx = (gx + ix as i32 - 1) * STEP;
            let p = biome_profile(sample_biome(sx, sz, seed));
            base_h += p.base_height * w;
            cont_a += p.continental_amp * w;
            hill_a += p.hill_amp * w;
            ridge_a += p.ridge_amp * w;
            sum_w += w;
        }
    }

    let inv_w = if sum_w > 0.0 { 1.0 / sum_w } else { 1.0 };
    BiomeTerrainProfile {
        base_height: base_h * inv_w,
        continental_amp: cont_a * inv_w,
        hill_amp: hill_a * inv_w,
        ridge_amp: ridge_a * inv_w,
    }
}
