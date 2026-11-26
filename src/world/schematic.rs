use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use glam::IVec3;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

use crate::registry::{EngineRegistries, CONTENT_AIR, CONTENT_IGNORE, CONTENT_UNKNOWN};
use crate::world::WorldStore;

pub const MTSCHEM_FILE_SIGNATURE: u32 = 0x4d54_534d; // "MTSM"
pub const MTSCHEM_PROB_MASK: u8 = 0x7F;
pub const MTSCHEM_PROB_NEVER: u8 = 0x00;
pub const MTSCHEM_PROB_ALWAYS: u8 = 0x7F;
pub const MTSCHEM_PROB_ALWAYS_OLD: u8 = 0xFF;
pub const MTSCHEM_FORCE_PLACE: u8 = 0x80;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchematicRotation {
    Rot0,
    Rot90,
    Rot180,
    Rot270,
    Random,
}

impl SchematicRotation {
    pub fn from_str(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "90" => Self::Rot90,
            "180" => Self::Rot180,
            "270" => Self::Rot270,
            "random" => Self::Random,
            _ => Self::Rot0,
        }
    }

    pub fn resolve(self, rng_seed: u64) -> u8 {
        match self {
            Self::Rot0 => 0,
            Self::Rot90 => 1,
            Self::Rot180 => 2,
            Self::Rot270 => 3,
            Self::Random => ((rng_seed ^ (rng_seed >> 17)) % 4) as u8,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlaceFlags {
    pub place_center_x: bool,
    pub place_center_y: bool,
    pub place_center_z: bool,
    pub force_placement: bool,
}

impl PlaceFlags {
    pub fn from_str(s: &str) -> Self {
        let mut flags = Self::default();
        for tok in s.split(',').map(str::trim) {
            match tok {
                "place_center_x" => flags.place_center_x = true,
                "place_center_y" => flags.place_center_y = true,
                "place_center_z" => flags.place_center_z = true,
                "force_placement" => flags.force_placement = true,
                _ => {}
            }
        }
        flags
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchematicNode {
    pub name: String,
    /// Placement probability in 0..=127 (`MTSCHEM_PROB_NEVER..=MTSCHEM_PROB_ALWAYS`).
    pub prob: u8,
    pub force_place: bool,
    pub param2: u8,
}

impl Default for SchematicNode {
    fn default() -> Self {
        Self {
            name: "air".into(),
            prob: MTSCHEM_PROB_ALWAYS,
            force_place: false,
            param2: 0,
        }
    }
}

/// Port of Luanti's `MapgenSchematic` (`mg_schematic.cpp`).
/// Data is stored in `(z, y, x)` order: `index = z * (sx * sy) + y * sx + x`.
#[derive(Debug, Clone)]
pub struct Schematic {
    pub size: IVec3,
    pub yslice_prob: Vec<u8>,
    pub node_names: Vec<String>,
    pub data: Vec<SchematicNode>,
}

/// Deterministic splitmix64 helper for schematic probability & rotation rolls.
#[inline]
fn schem_hash(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Port of `MapNode::rotateAlongYAxis` from `src/mapnode.cpp`.
pub fn rotate_param2_along_y(paramtype2: &str, mut param2: u8, rot: u8) -> u8 {
    let rot = rot & 3;
    if rot == 0 {
        return param2;
    }
    #[rustfmt::skip]
    const ROTATE_FACEDIR: [u8; 24 * 4] = [
        0, 1, 2, 3,
        1, 2, 3, 0,
        2, 3, 0, 1,
        3, 0, 1, 2,

        4, 13, 10, 19,
        5, 14, 11, 16,
        6, 15, 8, 17,
        7, 12, 9, 18,

        8, 17, 6, 15,
        9, 18, 7, 12,
        10, 19, 4, 13,
        11, 16, 5, 14,

        12, 9, 18, 7,
        13, 10, 19, 4,
        14, 11, 16, 5,
        15, 8, 17, 6,

        16, 5, 14, 11,
        17, 6, 15, 8,
        18, 7, 12, 9,
        19, 4, 13, 10,

        20, 23, 22, 21,
        21, 20, 23, 22,
        22, 21, 20, 23,
        23, 22, 21, 20,
    ];
    const WALLMOUNTED_TO_ROT: [u8; 4] = [3, 1, 0, 2]; // wallmounted 2,3,4,5 -> rotation
    const ROT_TO_WALLMOUNTED: [u8; 4] = [4, 3, 5, 2];

    match paramtype2 {
        "facedir" | "colorfacedir" => {
            let facedir = (param2 & 31) % 24;
            let idx = (facedir as usize) * 4 + (rot as usize);
            param2 &= !31;
            param2 |= ROTATE_FACEDIR[idx];
        }
        "4dir" | "color4dir" => {
            let fourdir = param2 & 3;
            let idx = (fourdir as usize) * 4 + (rot as usize);
            param2 &= !3;
            param2 |= ROTATE_FACEDIR[idx];
        }
        "wallmounted" | "colorwallmounted" => {
            let wmount = (param2 & 0x07).min(5);
            if wmount >= 2 {
                let oldrot = WALLMOUNTED_TO_ROT[(wmount - 2) as usize];
                let newrot = oldrot.wrapping_sub(rot) & 3;
                param2 &= !7;
                param2 |= ROT_TO_WALLMOUNTED[newrot as usize];
            }
        }
        "degrotate" => {
            let angle = ((param2 as u16) + 60 * (rot as u16)) % 240;
            param2 = angle as u8;
        }
        "colordegrotate" => {
            let angle = ((param2 & 0x1F) as u16 + 6 * (rot as u16)) % 24;
            let color = param2 & 0xE0;
            param2 = color | (angle as u8);
        }
        _ => {}
    }
    param2
}

impl Schematic {
    pub fn load_from_file(path: &Path) -> anyhow::Result<Self> {
        let raw = std::fs::read(path)?;
        Self::from_mts_bytes(&raw)
    }

    /// Byte-for-byte port of `Schematic::deserializeFromMts` from `mg_schematic.cpp`.
    pub fn from_mts_bytes(raw: &[u8]) -> anyhow::Result<Self> {
        if raw.len() < 12 {
            anyhow::bail!("MTS file too short ({} bytes)", raw.len());
        }
        let sig = u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]);
        if sig != MTSCHEM_FILE_SIGNATURE {
            anyhow::bail!("Invalid MTS signature: 0x{sig:08X}");
        }
        let version = u16::from_be_bytes([raw[4], raw[5]]);
        if version > 4 {
            anyhow::bail!("Unsupported MTS version: {version}");
        }
        let sx = u16::from_be_bytes([raw[6], raw[7]]) as usize;
        let sy = u16::from_be_bytes([raw[8], raw[9]]) as usize;
        let sz = u16::from_be_bytes([raw[10], raw[11]]) as usize;
        let mut cursor = 12usize;

        let mut yslice_prob = Vec::with_capacity(sy);
        if version >= 3 {
            if cursor + sy > raw.len() {
                anyhow::bail!("Truncated MTS yslice_prob");
            }
            for i in 0..sy {
                let mut p = raw[cursor + i];
                if version < 4 {
                    p >>= 1;
                }
                yslice_prob.push(p & MTSCHEM_PROB_MASK);
            }
            cursor += sy;
        } else {
            yslice_prob.resize(sy, MTSCHEM_PROB_ALWAYS);
        }

        if cursor + 2 > raw.len() {
            anyhow::bail!("Truncated MTS node name count");
        }
        let nidmapcount = u16::from_be_bytes([raw[cursor], raw[cursor + 1]]) as usize;
        cursor += 2;

        let mut node_names = Vec::with_capacity(nidmapcount);
        let mut cignore_idx: Option<usize> = None;
        for i in 0..nidmapcount {
            if cursor + 2 > raw.len() {
                anyhow::bail!("Truncated MTS node name length");
            }
            let slen = u16::from_be_bytes([raw[cursor], raw[cursor + 1]]) as usize;
            cursor += 2;
            if cursor + slen > raw.len() {
                anyhow::bail!("Truncated MTS node name string");
            }
            let mut nm = String::from_utf8_lossy(&raw[cursor..cursor + slen]).to_string();
            cursor += slen;
            if nm == "ignore" {
                nm = "air".into();
                cignore_idx = Some(i);
            }
            node_names.push(nm);
        }

        let nodecount = sx
            .checked_mul(sy)
            .and_then(|v| v.checked_mul(sz))
            .ok_or_else(|| anyhow::anyhow!("Schematic size overflow"))?;

        let mut decomp = Vec::new();
        if cursor < raw.len() {
            let mut dec = ZlibDecoder::new(&raw[cursor..]);
            dec.read_to_end(&mut decomp)?;
        }
        if decomp.len() < nodecount * 4 {
            anyhow::bail!(
                "Decompressed MTS node data too short: {} < {}",
                decomp.len(),
                nodecount * 4
            );
        }

        let mut data = Vec::with_capacity(nodecount);
        for i in 0..nodecount {
            let cid = u16::from_be_bytes([decomp[i * 2], decomp[i * 2 + 1]]) as usize;
            let mut p1 = decomp[nodecount * 2 + i];
            let p2 = decomp[nodecount * 3 + i];

            if version < 2 {
                if p1 == 0 {
                    p1 = MTSCHEM_PROB_ALWAYS_OLD;
                }
                if cignore_idx == Some(cid) {
                    p1 = MTSCHEM_PROB_NEVER;
                }
            }
            if version < 4 {
                p1 >>= 1;
            }

            let name = node_names
                .get(cid)
                .cloned()
                .unwrap_or_else(|| "air".into());
            data.push(SchematicNode {
                name,
                prob: p1 & MTSCHEM_PROB_MASK,
                force_place: (p1 & MTSCHEM_FORCE_PLACE) != 0,
                param2: p2,
            });
        }

        Ok(Self {
            size: IVec3::new(sx as i32, sy as i32, sz as i32),
            yslice_prob,
            node_names,
            data,
        })
    }

    pub fn to_mts_bytes(&self) -> anyhow::Result<Vec<u8>> {
        let sx = self.size.x.max(0) as u16;
        let sy = self.size.y.max(0) as u16;
        let sz = self.size.z.max(0) as u16;
        let nodecount = (sx as usize) * (sy as usize) * (sz as usize);

        let mut name_to_cid: HashMap<String, u16> = HashMap::new();
        let mut names: Vec<String> = Vec::new();
        for node in &self.data {
            if !name_to_cid.contains_key(&node.name) {
                let id = names.len() as u16;
                names.push(node.name.clone());
                name_to_cid.insert(node.name.clone(), id);
            }
        }

        let mut out = Vec::new();
        out.extend_from_slice(&MTSCHEM_FILE_SIGNATURE.to_be_bytes());
        out.extend_from_slice(&4u16.to_be_bytes());
        out.extend_from_slice(&sx.to_be_bytes());
        out.extend_from_slice(&sy.to_be_bytes());
        out.extend_from_slice(&sz.to_be_bytes());
        for y in 0..(sy as usize) {
            out.push(self.yslice_prob.get(y).copied().unwrap_or(MTSCHEM_PROB_ALWAYS) & MTSCHEM_PROB_MASK);
        }
        out.extend_from_slice(&(names.len() as u16).to_be_bytes());
        for nm in &names {
            let bytes = nm.as_bytes();
            out.extend_from_slice(&(bytes.len() as u16).to_be_bytes());
            out.extend_from_slice(bytes);
        }

        let mut bulk = vec![0u8; nodecount * 4];
        for (i, node) in self.data.iter().take(nodecount).enumerate() {
            let cid = name_to_cid[&node.name];
            let be = cid.to_be_bytes();
            bulk[i * 2] = be[0];
            bulk[i * 2 + 1] = be[1];
            bulk[nodecount * 2 + i] = (node.prob & MTSCHEM_PROB_MASK)
                | if node.force_place { MTSCHEM_FORCE_PLACE } else { 0 };
            bulk[nodecount * 3 + i] = node.param2;
        }
        let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
        enc.write_all(&bulk)?;
        out.extend_from_slice(&enc.finish()?);
        Ok(out)
    }

    pub fn apply_replacements(&mut self, replacements: &HashMap<String, String>) {
        if replacements.is_empty() {
            return;
        }
        for n in &mut self.node_names {
            if let Some(rep) = replacements.get(n) {
                *n = rep.clone();
            }
        }
        for node in &mut self.data {
            if let Some(rep) = replacements.get(&node.name) {
                node.name = rep.clone();
            }
        }
    }

    /// Compute effective rotated dimensions and start origin after `PlaceFlags`.
    pub fn adjusted_origin(
        &self,
        mut pos: IVec3,
        rot_steps: u8,
        flags: PlaceFlags,
    ) -> (IVec3, IVec3) {
        let eff_size = if rot_steps == 1 || rot_steps == 3 {
            IVec3::new(self.size.z, self.size.y, self.size.x)
        } else {
            self.size
        };
        if flags.place_center_x {
            pos.x -= (eff_size.x - 1) / 2;
        }
        if flags.place_center_y {
            pos.y -= (eff_size.y - 1) / 2;
        }
        if flags.place_center_z {
            pos.z -= (eff_size.z - 1) / 2;
        }
        (pos, eff_size)
    }

    /// Iterate placed world nodes following Luanti's exact `(z, y, x)` stride and rotation formulas.
    #[allow(clippy::too_many_arguments)]
    pub fn visit_rotated_nodes<F>(
        &self,
        pos: IVec3,
        rotation: SchematicRotation,
        flags: PlaceFlags,
        force_place: bool,
        replacements: Option<&HashMap<String, String>>,
        rng_seed: u64,
        mut f: F,
    ) where
        F: FnMut(IVec3, &str, u8, bool),
    {
        let mut sx = self.size.x.max(0) as isize;
        let sy = self.size.y.max(0) as isize;
        let mut sz = self.size.z.max(0) as isize;
        if sx == 0 || sy == 0 || sz == 0 || self.data.len() < (sx * sy * sz) as usize {
            return;
        }

        let rot = rotation.resolve(schem_hash(rng_seed));
        let (origin, _) = self.adjusted_origin(pos, rot, flags);

        let xstride = 1isize;
        let ystride = sx;
        let zstride = sx * sy;

        let (i_start, i_step_x, i_step_z) = match rot {
            1 => {
                let st = sx - 1;
                std::mem::swap(&mut sx, &mut sz);
                (st, zstride, -xstride)
            }
            2 => (zstride * (sz - 1) + sx - 1, -xstride, -zstride),
            3 => {
                let st = zstride * (sz - 1);
                std::mem::swap(&mut sx, &mut sz);
                (st, -zstride, xstride)
            }
            _ => (0, xstride, zstride),
        };

        let effective_force = force_place || flags.force_placement;
        let mut roll_counter = 1u64;
        let mut y_map = origin.y;

        for y in 0..sy {
            let slice_prob = self
                .yslice_prob
                .get(y as usize)
                .copied()
                .unwrap_or(MTSCHEM_PROB_ALWAYS);
            if slice_prob != MTSCHEM_PROB_ALWAYS {
                let roll = ((schem_hash(rng_seed ^ roll_counter) % (MTSCHEM_PROB_ALWAYS as u64)) + 1) as u8;
                roll_counter += 1;
                if slice_prob <= roll {
                    continue;
                }
            }

            for z in 0..sz {
                let mut i = z * i_step_z + y * ystride + i_start;
                for x in 0..sx {
                    let snode = &self.data[i as usize];
                    i += i_step_x;

                    let node_name = replacements
                        .and_then(|r| r.get(&snode.name))
                        .map(String::as_str)
                        .unwrap_or(snode.name.as_str());

                    if node_name == "ignore" {
                        continue;
                    }
                    if snode.prob == MTSCHEM_PROB_NEVER {
                        continue;
                    }
                    if snode.prob != MTSCHEM_PROB_ALWAYS {
                        let roll = ((schem_hash(rng_seed ^ roll_counter)
                            % (MTSCHEM_PROB_ALWAYS as u64))
                            + 1) as u8;
                        roll_counter += 1;
                        if snode.prob <= roll {
                            continue;
                        }
                    }

                    let world_pos = IVec3::new(origin.x + x as i32, y_map, origin.z + z as i32);
                    f(
                        world_pos,
                        node_name,
                        snode.param2,
                        effective_force || snode.force_place,
                    );
                }
            }
            y_map += 1;
        }
    }

    pub fn place_on_world(
        &self,
        world: &WorldStore,
        reg: &EngineRegistries,
        pos: IVec3,
        rotation: SchematicRotation,
        flags: PlaceFlags,
        replacements: Option<&HashMap<String, String>>,
        rng_seed: u64,
    ) {
        let rot = rotation.resolve(schem_hash(rng_seed));
        let fixed_rot = match rot {
            1 => SchematicRotation::Rot90,
            2 => SchematicRotation::Rot180,
            3 => SchematicRotation::Rot270,
            _ => SchematicRotation::Rot0,
        };
        self.visit_rotated_nodes(
            pos,
            fixed_rot,
            flags,
            flags.force_placement,
            replacements,
            rng_seed,
            |wpos, name, raw_p2, force| {
                let id = reg.get_content_id(name);
                if id == CONTENT_IGNORE || id == CONTENT_UNKNOWN {
                    return;
                }
                if !force {
                    let (existing, _, _) = world.get_node(wpos);
                    if existing != CONTENT_AIR && existing != CONTENT_IGNORE {
                        return;
                    }
                }
                let p2 = if rot == 0 {
                    raw_p2
                } else {
                    let pt2 = reg
                        .get_node_def(id)
                        .map(|d| d.paramtype2.as_str())
                        .unwrap_or("none");
                    rotate_param2_along_y(pt2, raw_p2, rot)
                };
                let p1 = reg
                    .get_node_def(id)
                    .map(|d| {
                        if d.light_source > 0 {
                            (d.light_source.min(15) << 4) | 15
                        } else {
                            15
                        }
                    })
                    .unwrap_or(15);
                world.set_node(wpos, id, p1, p2);
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn place_on_vmanip(
        &self,
        emin: IVec3,
        emax: IVec3,
        data: &mut [u16],
        light_data: &mut [u8],
        param2_data: &mut [u8],
        reg: &EngineRegistries,
        pos: IVec3,
        rotation: SchematicRotation,
        flags: PlaceFlags,
        force_place: bool,
        replacements: Option<&HashMap<String, String>>,
        rng_seed: u64,
    ) -> bool {
        let sx = (emax.x - emin.x + 1).max(1) as usize;
        let sy = (emax.y - emin.y + 1).max(1) as usize;
        let rot = rotation.resolve(schem_hash(rng_seed));
        let fixed_rot = match rot {
            1 => SchematicRotation::Rot90,
            2 => SchematicRotation::Rot180,
            3 => SchematicRotation::Rot270,
            _ => SchematicRotation::Rot0,
        };
        let (adj_origin, eff_size) = self.adjusted_origin(pos, rot, flags);
        let fits = adj_origin.x >= emin.x
            && adj_origin.y >= emin.y
            && adj_origin.z >= emin.z
            && adj_origin.x + eff_size.x - 1 <= emax.x
            && adj_origin.y + eff_size.y - 1 <= emax.y
            && adj_origin.z + eff_size.z - 1 <= emax.z;

        self.visit_rotated_nodes(
            pos,
            fixed_rot,
            flags,
            force_place,
            replacements,
            rng_seed,
            |wpos, name, raw_p2, force| {
                if wpos.x < emin.x
                    || wpos.x > emax.x
                    || wpos.y < emin.y
                    || wpos.y > emax.y
                    || wpos.z < emin.z
                    || wpos.z > emax.z
                {
                    return;
                }
                let idx = (wpos.z - emin.z) as usize * sy * sx
                    + (wpos.y - emin.y) as usize * sx
                    + (wpos.x - emin.x) as usize;
                if idx >= data.len() {
                    return;
                }
                let id = reg.get_content_id(name);
                if id == CONTENT_IGNORE || id == CONTENT_UNKNOWN {
                    return;
                }
                if !force {
                    let cur = data[idx];
                    if cur != CONTENT_AIR && cur != CONTENT_IGNORE {
                        return;
                    }
                }
                let p2 = if rot == 0 {
                    raw_p2
                } else {
                    let pt2 = reg
                        .get_node_def(id)
                        .map(|d| d.paramtype2.as_str())
                        .unwrap_or("none");
                    rotate_param2_along_y(pt2, raw_p2, rot)
                };
                data[idx] = id;
                light_data[idx] = 0;
                param2_data[idx] = p2;
            },
        );

        fits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paths::EnginePaths;
    use crate::registry::{DrawType, NodeDef};

    #[test]
    fn mts_synthetic_roundtrip_and_rotations() {
        let mut reg = EngineRegistries::new();
        let trunk_id = reg.register_node(NodeDef {
            name: "mcl_trees:tree_oak".into(),
            paramtype2: "facedir".into(),
            ..Default::default()
        });
        let leaves_id = reg.register_node(NodeDef {
            name: "mcl_trees:leaves_oak".into(),
            drawtype: DrawType::AllfacesOptional,
            ..Default::default()
        });

        // 3x2x2 schematic in (z, y, x) order
        let sx = 3;
        let sy = 2;
        let sz = 2;
        let mut data = vec![SchematicNode::default(); sx * sy * sz];
        let idx = |x: usize, y: usize, z: usize| z * (sx * sy) + y * sx + x;
        // (0, 0, 0) trunk with facedir = 0; (2, 1, 1) leaves
        data[idx(0, 0, 0)] = SchematicNode {
            name: "mcl_trees:tree_oak".into(),
            prob: MTSCHEM_PROB_ALWAYS,
            force_place: true,
            param2: 0,
        };
        data[idx(2, 1, 1)] = SchematicNode {
            name: "mcl_trees:leaves_oak".into(),
            prob: MTSCHEM_PROB_ALWAYS,
            force_place: false,
            param2: 64,
        };

        let schem = Schematic {
            size: IVec3::new(sx as i32, sy as i32, sz as i32),
            yslice_prob: vec![MTSCHEM_PROB_ALWAYS; sy],
            node_names: vec![
                "air".into(),
                "mcl_trees:tree_oak".into(),
                "mcl_trees:leaves_oak".into(),
            ],
            data,
        };

        let bytes = schem.to_mts_bytes().expect("encode mts");
        assert_eq!(&bytes[0..4], b"MTSM");
        let decoded = Schematic::from_mts_bytes(&bytes).expect("decode mts");
        assert_eq!(decoded.size, IVec3::new(3, 2, 2));
        assert_eq!(decoded.data[idx(0, 0, 0)].name, "mcl_trees:tree_oak");
        assert!(decoded.data[idx(0, 0, 0)].force_place);
        assert_eq!(decoded.data[idx(2, 1, 1)].param2, 64);

        let dir = std::env::temp_dir().join("cubex_rs_schem_rot_test");
        let _ = std::fs::remove_dir_all(&dir);
        let world = WorldStore::new(&dir, 42);
        decoded.place_on_world(
            &world,
            &reg,
            IVec3::new(10, 5, 10),
            SchematicRotation::Rot0,
            PlaceFlags::default(),
            None,
            1,
        );
        assert_eq!(world.get_node(IVec3::new(10, 5, 10)).0, trunk_id);
        assert_eq!(world.get_node(IVec3::new(10, 5, 10)).2, 0);
        assert_eq!(world.get_node(IVec3::new(12, 6, 11)).0, leaves_id);

        // Rotate 90: effective size becomes (sz=2, sy=2, sx=3) and (x=0, z=0) moves to x=1, z=2
        let world90 = WorldStore::new(dir.join("rot90"), 42);
        decoded.place_on_world(
            &world90,
            &reg,
            IVec3::new(0, 0, 0),
            SchematicRotation::Rot90,
            PlaceFlags::default(),
            None,
            1,
        );
        assert_eq!(world90.get_node(IVec3::new(0, 0, 2)).0, trunk_id);
        assert_eq!(world90.get_node(IVec3::new(0, 0, 2)).2, 1);

        // VManip placement test
        let emin = IVec3::new(0, 0, 0);
        let emax = IVec3::new(15, 15, 15);
        let mut vm_data = vec![CONTENT_AIR; 4096];
        let mut vm_light = vec![0u8; 4096];
        let mut vm_p2 = vec![0u8; 4096];
        let fits = decoded.place_on_vmanip(
            emin,
            emax,
            &mut vm_data,
            &mut vm_light,
            &mut vm_p2,
            &reg,
            IVec3::new(2, 2, 2),
            SchematicRotation::Rot0,
            PlaceFlags::default(),
            false,
            None,
            7,
        );
        assert!(fits);
        let vidx = (2usize) * 16 * 16 + (2usize) * 16 + 2usize;
        assert_eq!(vm_data[vidx], trunk_id);
    }

    #[test]
    fn real_mineclonia_trees_decode_and_place() {
        let paths = EnginePaths::resolve();
        let core_schems = paths.game_dir.join("mods/ITEMS/mcl_core/schematics");
        let crimson_schems = paths.game_dir.join("mods/ITEMS/mcl_crimson/schematics");
        if !core_schems.is_dir() {
            return;
        }
        let samples = [
            core_schems.join("mcl_core_oak_classic.mts"),
            core_schems.join("mcl_core_birch.mts"),
            core_schems.join("mcl_core_spruce_1.mts"),
            core_schems.join("mcl_core_jungle_tree.mts"),
            core_schems.join("mcl_core_acacia_1.mts"),
            core_schems.join("mcl_core_dark_oak.mts"),
            crimson_schems.join("crimson_fungus_1.mts"),
            crimson_schems.join("warped_fungus_1.mts"),
        ];
        for p in &samples {
            let s = Schematic::load_from_file(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            assert!(s.size.x > 0 && s.size.y > 0 && s.size.z > 0);
            assert_eq!(
                s.data.len(),
                (s.size.x * s.size.y * s.size.z) as usize
            );
        }

        let oak = Schematic::load_from_file(&core_schems.join("mcl_core_oak_classic.mts")).unwrap();
        let mut reg = EngineRegistries::new();
        let log_id = reg.register_node(NodeDef {
            name: "mcl_trees:tree_oak".into(),
            paramtype2: "facedir".into(),
            ..Default::default()
        });
        let leaf_id = reg.register_node(NodeDef {
            name: "mcl_trees:leaves_oak".into(),
            drawtype: DrawType::AllfacesOptional,
            ..Default::default()
        });

        let dir = std::env::temp_dir().join("cubex_rs_real_oak_test");
        let _ = std::fs::remove_dir_all(&dir);
        let world = WorldStore::new(&dir, 100);
        oak.place_on_world(
            &world,
            &reg,
            IVec3::new(0, 1, 0),
            SchematicRotation::Rot0,
            PlaceFlags {
                place_center_x: true,
                place_center_y: false,
                place_center_z: true,
                force_placement: false,
            },
            None,
            100,
        );
        // Trunk at (0, 1..=5, 0) must have vertical param2 = 0
        for y in 1..=5 {
            let (id, _, p2) = world.get_node(IVec3::new(0, y, 0));
            assert_eq!(id, log_id, "missing trunk at y={y}");
            assert_eq!(p2, 0, "expected vertical param2=0 at y={y}");
        }
        // Canopy must contain leaves around y=5..=8
        let mut leaf_count = 0;
        for dy in 4..=8 {
            for dz in -2..=2 {
                for dx in -2..=2 {
                    if world.get_node(IVec3::new(dx, dy, dz)).0 == leaf_id {
                        leaf_count += 1;
                    }
                }
            }
        }
        assert!(leaf_count > 10, "expected full oak leaf canopy, got {leaf_count}");
    }
}
