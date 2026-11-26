use bytemuck::{Pod, Zeroable};
use glam::IVec3;
use rayon::prelude::*;
use std::collections::HashMap;

use crate::registry::{DrawType, EngineRegistries, NodeBoxSpec, NodeDef, CONTENT_AIR, CONTENT_IGNORE};
use crate::textures::{ObjMesh, TextureAtlas, TILE_SIZE};
use crate::world::light::{get_block_light, get_sky_light};
use crate::world::{ChunkSnapshot, CHUNK_SIZE};

pub const ATLAS_HALF_TEXEL_INSET: f32 = 0.5 / TILE_SIZE as f32;
pub const GREEDY_UV_EDGE_EPS: f32 = 0.0005;

#[inline]
pub fn inset_unit_uv(v: f32) -> f32 {
    v.clamp(0.0, 1.0) * (1.0 - 2.0 * ATLAS_HALF_TEXEL_INSET) + ATLAS_HALF_TEXEL_INSET
}

#[inline]
pub fn inset_unit_uvs(uvs: [[f32; 2]; 4]) -> [[f32; 2]; 4] {
    uvs.map(|[u, v]| [inset_unit_uv(u), inset_unit_uv(v)])
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct VoxelVertex {
    pub pos: [f32; 3],
    pub uv: [f32; 2],
    /// Packed attributes:
    /// bits 0..11  (11): texture array layer (0..2047)
    /// bits 11..14 (3) : face direction (0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z)
    /// bits 14..18 (4) : skylight (0..15)
    /// bits 18..22 (4) : blocklight (0..15)
    /// bits 22..24 (2) : AO (0..3)
    /// bit  24     (1) : waving flag
    /// bit  25     (1) : alpha_cutout flag
    pub packed: u32,
    pub tint: [u8; 4],
}

impl VoxelVertex {
    pub const FLAG_SPECULAR: u32 = 1u32 << 28;
    pub const FLAG_EMISSIVE: u32 = 1u32 << 29;

    pub const ATTRS: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x2,
        2 => Uint32,
        3 => Unorm8x4,
    ];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<VoxelVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }

    #[inline(always)]
    pub fn pack(
        layer: u16,
        face_dir: u8,
        skylight: u8,
        blocklight: u8,
        ao: u8,
        waving: bool,
        alpha_cutout: bool,
    ) -> u32 {
        ((layer as u32) & 0x7FF)
            | (((face_dir as u32) & 0x7) << 11)
            | (((skylight as u32) & 0xF) << 14)
            | (((blocklight as u32) & 0xF) << 18)
            | (((ao as u32) & 0x3) << 22)
            | ((waving as u32) << 24)
            | ((alpha_cutout as u32) << 25)
    }
}

#[derive(Debug, Clone)]
pub struct ChunkMeshData {
    pub chunk_pos: IVec3,
    pub opaque_verts: Vec<VoxelVertex>,
    pub opaque_indices: Vec<u32>,
    pub trans_verts: Vec<VoxelVertex>,
    pub trans_indices: Vec<u32>,
}

#[derive(Copy, Clone, PartialEq, Eq)]
struct MaskCell {
    layer: u16,
    skylight: u8,
    blocklight: u8,
    ao: u8,
    waving: bool,
    cutout: bool,
    extra_bits: u32,
    tint: [u8; 4],
}

#[inline]
fn node_material_extra_bits(def: &NodeDef) -> u32 {
    let lower = def.name.to_ascii_lowercase();
    let is_lava = lower.contains("lava");
    let mut bits = 0u32;
    if (!is_lava && matches!(def.drawtype, DrawType::Liquid | DrawType::FlowingLiquid))
        || (!is_lava && (lower.contains("water") || lower.contains("ice")))
        || lower.contains("_ore")
        || lower.contains("stone_with_")
        || lower.contains("deepslate_with_")
        || lower.contains("diamondblock")
        || lower.contains("goldblock")
        || lower.contains("ironblock")
        || lower.contains("emeraldblock")
    {
        bits |= VoxelVertex::FLAG_SPECULAR;
    }
    if def.light_source >= 8
        || is_lava
        || lower.contains("torch")
        || lower.contains("glowstone")
        || lower.contains("lantern")
        || lower.contains("fire")
        || lower.contains("shroomlight")
    {
        bits |= VoxelVertex::FLAG_EMISSIVE;
    }
    bits
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum TileRot {
    R0 = 0,
    R90 = 1,
    R180 = 2,
    R270 = 3,
}

/// Luanti `wallmounted_to_facedir` table (`src/util/directiontables.cpp`).
pub const WALLMOUNTED_TO_FACEDIR: [u8; 8] = [20, 0, 17, 15, 8, 6, 21, 1];

/// Lookup `(luanti_tile_slot, uv_rotation)` for world face `dir_idx` (0=+X,1=-X,2=+Y,3=-Y,4=+Z,5=-Z)
/// given `facedir` in `0..24`. Ported 1:1 from Luanti `src/client/mapblock_mesh.cpp::getNodeTile`.
#[inline]
pub fn facedir_face_tile(facedir: u8, world_face: usize) -> (usize, TileRot) {
    // Map world_face (0=+X,1=-X,2=+Y,3=-Y,4=+Z,5=-Z) to Luanti dir_i:
    // +X=1, +Y=2, +Z=3, -Z=5, -Y=6, -X=7
    const WF_TO_DIR_I: [usize; 6] = [1, 7, 2, 6, 3, 5];
    const R0: TileRot = TileRot::R0;
    const R1: TileRot = TileRot::R90;
    const R2: TileRot = TileRot::R180;
    const R3: TileRot = TileRot::R270;

    // Luanti tile slot indices: 0=+Y(top), 1=-Y(bot), 2=+X(right), 3=-X(left), 4=+Z(back), 5=-Z(front)
    #[rustfmt::skip]
    const DIR_TO_TILE: [[(u8, TileRot); 8]; 24] = [
        //  0        +X       +Y       +Z       _        -Z       -Y       -X
        [(0,R0), (2,R0), (0,R0), (4,R0), (0,R0), (5,R0), (1,R0), (3,R0)], // 0..3: y+
        [(0,R0), (4,R0), (0,R3), (3,R0), (0,R0), (2,R0), (1,R1), (5,R0)],
        [(0,R0), (3,R0), (0,R2), (5,R0), (0,R0), (4,R0), (1,R2), (2,R0)],
        [(0,R0), (5,R0), (0,R1), (2,R0), (0,R0), (3,R0), (1,R3), (4,R0)],

        [(0,R0), (2,R3), (5,R0), (0,R2), (0,R0), (1,R0), (4,R2), (3,R1)], // 4..7: z+
        [(0,R0), (4,R3), (2,R0), (0,R1), (0,R0), (1,R1), (3,R2), (5,R1)],
        [(0,R0), (3,R3), (4,R0), (0,R0), (0,R0), (1,R2), (5,R2), (2,R1)],
        [(0,R0), (5,R3), (3,R0), (0,R3), (0,R0), (1,R3), (2,R2), (4,R1)],

        [(0,R0), (2,R1), (4,R2), (1,R2), (0,R0), (0,R0), (5,R0), (3,R3)], // 8..11: z-
        [(0,R0), (4,R1), (3,R2), (1,R3), (0,R0), (0,R3), (2,R0), (5,R3)],
        [(0,R0), (3,R1), (5,R2), (1,R0), (0,R0), (0,R2), (4,R0), (2,R3)],
        [(0,R0), (5,R1), (2,R2), (1,R1), (0,R0), (0,R1), (3,R0), (4,R3)],

        [(0,R0), (0,R3), (3,R3), (4,R1), (0,R0), (5,R3), (2,R3), (1,R3)], // 12..15: x+
        [(0,R0), (0,R2), (5,R3), (3,R1), (0,R0), (2,R3), (4,R3), (1,R0)],
        [(0,R0), (0,R1), (2,R3), (5,R1), (0,R0), (4,R3), (3,R3), (1,R1)],
        [(0,R0), (0,R0), (4,R3), (2,R1), (0,R0), (3,R3), (5,R3), (1,R2)],

        [(0,R0), (1,R1), (2,R1), (4,R3), (0,R0), (5,R1), (3,R1), (0,R1)], // 16..19: x-
        [(0,R0), (1,R2), (4,R1), (3,R3), (0,R0), (2,R1), (5,R1), (0,R0)],
        [(0,R0), (1,R3), (3,R1), (5,R3), (0,R0), (4,R1), (2,R1), (0,R3)],
        [(0,R0), (1,R0), (5,R1), (2,R3), (0,R0), (3,R1), (4,R1), (0,R2)],

        [(0,R0), (3,R2), (1,R2), (4,R2), (0,R0), (5,R2), (0,R2), (2,R2)], // 20..23: y-
        [(0,R0), (5,R2), (1,R3), (3,R2), (0,R0), (2,R2), (0,R1), (4,R2)],
        [(0,R0), (2,R2), (1,R0), (5,R2), (0,R0), (4,R2), (0,R0), (3,R2)],
        [(0,R0), (4,R2), (1,R1), (2,R2), (0,R0), (3,R2), (0,R3), (5,R2)],
    ];

    // Map Luanti tile slot [0=+Y,1=-Y,2=+X,3=-X,4=+Z,5=-Z] back to our face_layers index [0=+X,1=-X,2=+Y,3=-Y,4=+Z,5=-Z]
    const LUANTI_TO_FACE_IDX: [usize; 6] = [2, 3, 0, 1, 4, 5];

    let fd = (facedir as usize) % 24;
    let di = WF_TO_DIR_I[world_face.min(5)];
    let (lt_slot, rot) = DIR_TO_TILE[fd][di];
    (LUANTI_TO_FACE_IDX[lt_slot as usize], rot)
}

#[inline]
pub fn get_node_facedir(def: &NodeDef, param2: u8) -> u8 {
    match def.paramtype2.as_str() {
        "facedir" | "colorfacedir" => (param2 & 0x1F) % 24,
        "4dir" | "color4dir" => param2 & 0x03,
        "wallmounted" | "colorwallmounted" => WALLMOUNTED_TO_FACEDIR[(param2 & 0x07) as usize],
        _ => 0,
    }
}

#[inline]
pub fn get_node_wallmounted(def: &NodeDef, param2: u8) -> u8 {
    match def.paramtype2.as_str() {
        "wallmounted" | "colorwallmounted" => (param2 & 0x07).min(7),
        _ if matches!(
            def.drawtype,
            DrawType::Signlike | DrawType::Torchlike | DrawType::Plantlike
        ) =>
        {
            1
        }
        _ => 0,
    }
}

/// Irrlicht-compatible axis rotations (`irr/include/vector3d.h`).
#[inline]
fn rotate_xz(p: [f32; 3], deg: f32) -> [f32; 3] {
    let rad = deg.to_radians();
    let (s, c) = rad.sin_cos();
    [p[0] * c - p[2] * s, p[1], p[0] * s + p[2] * c]
}

#[inline]
fn rotate_xy(p: [f32; 3], deg: f32) -> [f32; 3] {
    let rad = deg.to_radians();
    let (s, c) = rad.sin_cos();
    [p[0] * c - p[1] * s, p[0] * s + p[1] * c, p[2]]
}

#[inline]
fn rotate_yz(p: [f32; 3], deg: f32) -> [f32; 3] {
    let rad = deg.to_radians();
    let (s, c) = rad.sin_cos();
    [p[0], p[1] * c - p[2] * s, p[1] * s + p[2] * c]
}

/// Rotate a local point in `[-0.5, 0.5]^3` by Luanti 6D `facedir` (`0..23`), matching `src/mapnode.cpp`.
pub fn rotate_pt_by_facedir(mut p: [f32; 3], facedir: u8) -> [f32; 3] {
    let fd = (facedir % 24) as u8;
    let axisdir = fd >> 2;
    let rot = fd & 0x03;
    match axisdir {
        0 => match rot {
            1 => p = rotate_xz(p, -90.0),
            2 => p = rotate_xz(p, 180.0),
            3 => p = rotate_xz(p, 90.0),
            _ => {}
        },
        1 => {
            p = rotate_yz(p, 90.0);
            match rot {
                1 => p = rotate_xy(p, 90.0),
                2 => p = rotate_xy(p, 180.0),
                3 => p = rotate_xy(p, -90.0),
                _ => {}
            }
        }
        2 => {
            p = rotate_yz(p, -90.0);
            match rot {
                1 => p = rotate_xy(p, -90.0),
                2 => p = rotate_xy(p, 180.0),
                3 => p = rotate_xy(p, 90.0),
                _ => {}
            }
        }
        3 => {
            p = rotate_xy(p, -90.0);
            match rot {
                1 => p = rotate_yz(p, 90.0),
                2 => p = rotate_yz(p, 180.0),
                3 => p = rotate_yz(p, -90.0),
                _ => {}
            }
        }
        4 => {
            p = rotate_xy(p, 90.0);
            match rot {
                1 => p = rotate_yz(p, -90.0),
                2 => p = rotate_yz(p, 180.0),
                3 => p = rotate_yz(p, 90.0),
                _ => {}
            }
        }
        _ => {
            p = rotate_xy(p, -180.0);
            match rot {
                1 => p = rotate_xz(p, 90.0),
                2 => p = rotate_xz(p, 180.0),
                3 => p = rotate_xz(p, -90.0),
                _ => {}
            }
        }
    }
    p
}

#[inline]
fn rotate_aabb_by_facedir(b: [f32; 6], facedir: u8) -> [f32; 6] {
    if facedir == 0 {
        return b;
    }
    let p0 = rotate_pt_by_facedir([b[0], b[1], b[2]], facedir);
    let p1 = rotate_pt_by_facedir([b[3], b[4], b[5]], facedir);
    [
        p0[0].min(p1[0]),
        p0[1].min(p1[1]),
        p0[2].min(p1[2]),
        p0[0].max(p1[0]),
        p0[1].max(p1[1]),
        p0[2].max(p1[2]),
    ]
}

#[inline]
fn rotate_uv(uv: [f32; 2], rot: TileRot) -> [f32; 2] {
    match rot {
        TileRot::R0 => uv,
        TileRot::R90 => [-uv[1], uv[0]],
        TileRot::R180 => [-uv[0], -uv[1]],
        TileRot::R270 => [uv[1], -uv[0]],
    }
}

#[inline]
fn is_valid_palette_sample(c: [u8; 4]) -> bool {
    c[3] >= 128 && (c[0] as u16 + c[1] as u16 + c[2] as u16) >= 48
}

#[inline]
fn is_precolored_green_tile(atlas: &TextureAtlas, def: &NodeDef, face_idx: usize) -> bool {
    let Some(layers) = atlas.node_face_layers.get(&def.id) else {
        return false;
    };
    let layer = layers[face_idx.min(5)] as usize;
    if layer == 0 {
        return false;
    }
    let Some(img) = atlas.layers.get(layer) else {
        return false;
    };
    let mut sum_r = 0u64;
    let mut sum_g = 0u64;
    let mut sum_b = 0u64;
    let mut cnt = 0u64;
    for px in img.pixels() {
        if px.0[3] >= 128 {
            sum_r += px.0[0] as u64;
            sum_g += px.0[1] as u64;
            sum_b += px.0[2] as u64;
            cnt += 1;
        }
    }
    if cnt == 0 {
        return false;
    }
    let avg_r = sum_r as f32 / cnt as f32;
    let avg_g = sum_g as f32 / cnt as f32;
    let avg_b = sum_b as f32 / cnt as f32;
    avg_g > avg_r + 5.0 && avg_g > avg_b + 5.0 && avg_g < 140.0
}

#[inline]
fn resolve_face_tint(
    def: &NodeDef,
    face_idx: usize,
    param2: u8,
    atlas: Option<&TextureAtlas>,
) -> [u8; 4] {
    // Convert our face_idx [0=+X,1=-X,2=+Y,3=-Y,4=+Z,5=-Z] to Luanti tile index [0=+Y,1=-Y,2=+X,3=-X,4=+Z,5=-Z]
    const FACE_TO_LUANTI: [usize; 6] = [2, 3, 0, 1, 4, 5];
    let lt_idx = FACE_TO_LUANTI[face_idx.min(5)];
    if !def.tiles.is_empty() {
        let tile = &def.tiles[lt_idx.min(def.tiles.len() - 1)];
        if let Some(tc) = tile.color {
            return tc;
        }
    }

    let is_leaves = matches!(def.drawtype, DrawType::Allfaces | DrawType::AllfacesOptional)
        || def.groups.get("leaves").copied().unwrap_or(0) > 0
        || def.name.contains("leaves");
    let is_foliage_plant = def.name.contains("tallgrass") || def.name.contains("fern");

    // Pre-colored resource-pack leaves/tallgrass (e.g. Craft & Ruin) already bake dark green pigment
    // into their PNGs; multiplying them again by a palette/color tint would crush them to black.
    if (is_leaves || is_foliage_plant) && !def.name.contains("cherry") && !def.name.contains("snow") {
        if let Some(atl) = atlas {
            if is_precolored_green_tile(atl, def, face_idx) {
                return [255, 255, 255, 255];
            }
        }
    }

    if let Some(ref pname) = def.palette {
        if !pname.is_empty() {
            if let Some(atl) = atlas {
                if let Some(c) = atl.sample_palette_color(pname, param2) {
                    if is_valid_palette_sample(c) {
                        return c;
                    }
                }
            }
            if let Some(c) = def.color {
                if is_valid_palette_sample(c) {
                    return c;
                }
            }
            if pname.contains("grass") {
                return [157, 189, 99, 255];
            }
            if pname.contains("leaves") {
                return [119, 171, 47, 255];
            }
        }
    }

    if let Some(c) = def.color {
        return c;
    }

    if is_leaves
        && !def.name.contains("cherry")
        && !def.name.contains("azalea")
        && !def.name.contains("snow")
    {
        return [119, 171, 47, 255];
    }
    if (def.name.contains("dirt_with_grass") || def.name.contains("grass")) && face_idx == 2 {
        return [157, 189, 99, 255];
    }
    if is_foliage_plant {
        return [119, 171, 47, 255];
    }
    if def.name.contains("water") {
        return [62, 132, 232, 188];
    }
    [255, 255, 255, 255]
}

#[inline]
fn resolve_overlay_tint(def: &NodeDef, face_idx: usize, param2: u8, atlas: Option<&TextureAtlas>) -> [u8; 4] {
    const FACE_TO_LUANTI: [usize; 6] = [2, 3, 0, 1, 4, 5];
    let lt_idx = FACE_TO_LUANTI[face_idx.min(5)];
    if !def.overlay_tiles.is_empty() {
        let ov = &def.overlay_tiles[lt_idx.min(def.overlay_tiles.len() - 1)];
        if let Some(c) = ov.color {
            return c;
        }
    }
    if let Some(ref pname) = def.palette {
        if !pname.is_empty() {
            if let Some(atl) = atlas {
                if let Some(c) = atl.sample_palette_color(pname, param2) {
                    if is_valid_palette_sample(c) {
                        return c;
                    }
                }
            }
            if pname.contains("grass") {
                return [157, 189, 99, 255];
            }
        }
    }
    def.color.unwrap_or([157, 189, 99, 255])
}

/// Only opaque full `DrawType::Normal` cubes occlude adjacent faces.
/// Leaves (`Allfaces`/`AllfacesOptional`), glass, nodeboxes, plants, liquids, meshes NEVER cull neighbors!
#[inline]
pub fn is_opaque_occluder(id: u16, registries: &EngineRegistries) -> bool {
    if id == CONTENT_AIR || id == CONTENT_IGNORE {
        return false;
    }
    registries
        .get_node_def(id)
        .map(|d| d.is_opaque())
        .unwrap_or(false)
}

#[inline]
fn vertex_ao(side1: bool, side2: bool, corner: bool) -> u8 {
    if side1 && side2 {
        0
    } else {
        3 - (side1 as u8 + side2 as u8 + corner as u8)
    }
}

/// Compute per-vertex Ambient Occlusion `[p0, p1, p2, p3]` aligned with `emit_single_unit_face`
/// vertex order so concave corners darken smoothly while flat open regions stay uniform `[3; 4]`.
fn compute_face_corner_aos(
    snap: &ChunkSnapshot,
    registries: &EngineRegistries,
    nx: isize,
    ny: isize,
    nz: isize,
    face_dir: usize,
) -> [u8; 4] {
    let (_, u_ax, v_ax, _) = AXES[face_dir];
    let corner = |du: isize, dv: isize| -> u8 {
        let mut su = [0isize; 3];
        let mut sv = [0isize; 3];
        su[u_ax] = du;
        sv[v_ax] = dv;
        let s1 = is_opaque_occluder(snap.get(nx + su[0], ny + su[1], nz + su[2]).0, registries);
        let s2 = is_opaque_occluder(snap.get(nx + sv[0], ny + sv[1], nz + sv[2]).0, registries);
        let cn = is_opaque_occluder(
            snap.get(nx + su[0] + sv[0], ny + su[1] + sv[1], nz + su[2] + sv[2]).0,
            registries,
        );
        vertex_ao(s1, s2, cn)
    };
    let ao00 = corner(-1, -1);
    let ao01 = corner(-1, 1);
    let ao11 = corner(1, 1);
    let ao10 = corner(1, -1);
    match face_dir {
        0 | 5 => [ao00, ao01, ao11, ao10],
        1 | 4 => [ao10, ao11, ao01, ao00],
        2 => [ao01, ao11, ao10, ao00],
        _ => [ao00, ao10, ao11, ao01],
    }
}

#[inline]
fn node_uses_cutout(def: &NodeDef) -> bool {
    if def.use_texture_alpha.eq_ignore_ascii_case("blend") {
        return false;
    }
    if def.use_texture_alpha.eq_ignore_ascii_case("clip") {
        return true;
    }
    !matches!(
        def.drawtype,
        DrawType::Normal | DrawType::Liquid | DrawType::FlowingLiquid
    )
}

#[inline]
fn node_uses_blend(def: &NodeDef) -> bool {
    def.use_texture_alpha.eq_ignore_ascii_case("blend")
        || matches!(def.drawtype, DrawType::Liquid | DrawType::FlowingLiquid)
            && !def.name.contains("lava")
            && !def.name.contains("ice")
}

/// Parallel meshing over a snapshot batch on rayon threads (backwards-compatible signature).
pub fn mesh_chunks_parallel(
    snapshots: Vec<ChunkSnapshot>,
    registries: &EngineRegistries,
    face_layers: &HashMap<u16, [u16; 6]>,
    smooth_lighting: bool,
) -> Vec<ChunkMeshData> {
    snapshots
        .into_par_iter()
        .map(|snap| mesh_chunk_greedy(&snap, registries, face_layers, smooth_lighting))
        .collect()
}

/// Parallel meshing with full `TextureAtlas` context (overlays, palettes, `.obj` meshes).
pub fn mesh_chunks_parallel_with_atlas(
    snapshots: Vec<ChunkSnapshot>,
    registries: &EngineRegistries,
    atlas: &TextureAtlas,
    smooth_lighting: bool,
) -> Vec<ChunkMeshData> {
    snapshots
        .into_par_iter()
        .map(|snap| mesh_chunk_with_atlas(&snap, registries, atlas, smooth_lighting))
        .collect()
}

pub fn mesh_chunk_with_atlas(
    snap: &ChunkSnapshot,
    registries: &EngineRegistries,
    atlas: &TextureAtlas,
    smooth_lighting: bool,
) -> ChunkMeshData {
    mesh_chunk_impl(snap, registries, &atlas.node_face_layers, Some(atlas), smooth_lighting)
}

pub fn mesh_chunk_greedy(
    snap: &ChunkSnapshot,
    registries: &EngineRegistries,
    face_layers: &HashMap<u16, [u16; 6]>,
    smooth_lighting: bool,
) -> ChunkMeshData {
    mesh_chunk_impl(snap, registries, face_layers, None, smooth_lighting)
}

const FACE_NORMALS: [[isize; 3]; 6] = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
];

const AXES: [(usize, usize, usize, bool); 6] = [
    (0, 2, 1, true),  // +X: u=Z, v=Y
    (0, 2, 1, false), // -X
    (1, 0, 2, true),  // +Y: u=X, v=Z
    (1, 0, 2, false), // -Y
    (2, 0, 1, true),  // +Z: u=X, v=Y
    (2, 0, 1, false), // -Z
];

fn mesh_chunk_impl(
    snap: &ChunkSnapshot,
    registries: &EngineRegistries,
    face_layers: &HashMap<u16, [u16; 6]>,
    atlas: Option<&TextureAtlas>,
    smooth_lighting: bool,
) -> ChunkMeshData {
    let mut opaque_verts = Vec::with_capacity(2048);
    let mut opaque_indices = Vec::with_capacity(3072);
    let mut trans_verts = Vec::with_capacity(512);
    let mut trans_indices = Vec::with_capacity(768);

    let ox = (snap.pos.x * 16) as f32;
    let oy = (snap.pos.y * 16) as f32;
    let oz = (snap.pos.z * 16) as f32;

    // ---------------------------------------------------------------------
    // Pass 1: Greedy Meshing for unrotated, non-overlay opaque Normal cubes
    // Face directions: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
    // ---------------------------------------------------------------------
    let mut mask: [Option<MaskCell>; 256] = [None; 256];

    for face_dir in 0..6usize {
        let (d_ax, u_ax, v_ax, pos_norm) = AXES[face_dir];
        let n_dir = FACE_NORMALS[face_dir];

        for d in 0..CHUNK_SIZE {
            mask.fill(None);

            for v in 0..CHUNK_SIZE {
                for u in 0..CHUNK_SIZE {
                    let mut p = [0isize; 3];
                    p[d_ax] = d as isize;
                    p[u_ax] = u as isize;
                    p[v_ax] = v as isize;

                    let (id, _, p2) = snap.get(p[0], p[1], p[2]);
                    if !is_opaque_occluder(id, registries) {
                        continue;
                    }
                    let nx = p[0] + n_dir[0];
                    let ny = p[1] + n_dir[1];
                    let nz = p[2] + n_dir[2];
                    let (nid, np1, _) = snap.get(nx, ny, nz);
                    if is_opaque_occluder(nid, registries) {
                        continue;
                    }

                    let Some(def) = registries.get_node_def(id) else {
                        continue;
                    };
                    let facedir = get_node_facedir(def, p2);
                    let (tile_face_idx, tile_rot) = facedir_face_tile(facedir, face_dir);
                    let arr = face_layers.get(&id).copied().unwrap_or([0; 6]);
                    let layer = arr[tile_face_idx];
                    let overlay_layer = atlas
                        .and_then(|atl| atl.overlay_face_layers.get(&id))
                        .map(|o| o[tile_face_idx])
                        .unwrap_or(0);

                    let sky = get_sky_light(np1);
                    let block = get_block_light(np1);
                    let corner_aos = if smooth_lighting {
                        compute_face_corner_aos(snap, registries, nx, ny, nz, face_dir)
                    } else {
                        [3; 4]
                    };
                    let uniform_ao = corner_aos[0] == corner_aos[1]
                        && corner_aos[1] == corner_aos[2]
                        && corner_aos[2] == corner_aos[3];
                    let ao = corner_aos[0];
                    let tint = resolve_face_tint(def, tile_face_idx, p2, atlas);

                    // Rotated UVs, overlay tiles, or varying per-vertex AO faces are emitted individually
                    let extra_bits = node_material_extra_bits(def);
                    if tile_rot != TileRot::R0 || overlay_layer > 0 || !uniform_ao {
                        let bx = ox + p[0] as f32;
                        let by = oy + p[1] as f32;
                        let bz = oz + p[2] as f32;
                        let vstart = opaque_verts.len();
                        emit_single_unit_face(
                            &mut opaque_verts,
                            &mut opaque_indices,
                            [bx, by, bz],
                            face_dir,
                            layer,
                            tile_rot,
                            sky,
                            block,
                            corner_aos,
                            false,
                            false,
                            tint,
                            0.0,
                        );
                        if overlay_layer > 0 {
                            let ov_tint = resolve_overlay_tint(def, tile_face_idx, p2, atlas);
                            emit_single_unit_face(
                                &mut opaque_verts,
                                &mut opaque_indices,
                                [bx, by, bz],
                                face_dir,
                                overlay_layer,
                                tile_rot,
                                sky,
                                block,
                                corner_aos,
                                false,
                                true,
                                ov_tint,
                                0.0015,
                            );
                        }
                        if extra_bits != 0 {
                            for v in &mut opaque_verts[vstart..] {
                                v.packed |= extra_bits;
                            }
                        }
                        continue;
                    }

                    mask[v * 16 + u] = Some(MaskCell {
                        layer,
                        skylight: sky,
                        blocklight: block,
                        ao,
                        waving: false,
                        cutout: false,
                        extra_bits,
                        tint,
                    });
                }
            }

            // Merge greedy rectangles across mask
            for v in 0..CHUNK_SIZE {
                let mut u = 0usize;
                while u < CHUNK_SIZE {
                    let Some(cell) = mask[v * 16 + u] else {
                        u += 1;
                        continue;
                    };

                    let mut w = 1usize;
                    while u + w < CHUNK_SIZE && mask[v * 16 + (u + w)] == Some(cell) {
                        w += 1;
                    }

                    let mut h = 1usize;
                    'height_loop: while v + h < CHUNK_SIZE {
                        for k in 0..w {
                            if mask[(v + h) * 16 + (u + k)] != Some(cell) {
                                break 'height_loop;
                            }
                        }
                        h += 1;
                    }

                    for dv in 0..h {
                        for du in 0..w {
                            mask[(v + dv) * 16 + (u + du)] = None;
                        }
                    }

                    let d_pos = d as f32 + if pos_norm { 1.0 } else { 0.0 };
                    let u0 = u as f32;
                    let u1 = (u + w) as f32;
                    let v0 = v as f32;
                    let v1 = (v + h) as f32;

                    let make_pt = |uu: f32, vv: f32| -> [f32; 3] {
                        let mut c = [0.0f32; 3];
                        c[d_ax] = d_pos;
                        c[u_ax] = uu;
                        c[v_ax] = vv;
                        [ox + c[0], oy + c[1], oz + c[2]]
                    };

                    let u_lo = if w == 1 { ATLAS_HALF_TEXEL_INSET } else { GREEDY_UV_EDGE_EPS };
                    let u_hi = if w == 1 {
                        1.0 - ATLAS_HALF_TEXEL_INSET
                    } else {
                        w as f32 - GREEDY_UV_EDGE_EPS
                    };
                    let v_lo = if h == 1 { ATLAS_HALF_TEXEL_INSET } else { GREEDY_UV_EDGE_EPS };
                    let v_hi = if h == 1 {
                        1.0 - ATLAS_HALF_TEXEL_INSET
                    } else {
                        h as f32 - GREEDY_UV_EDGE_EPS
                    };

                    let (p0, p1, p2, p3, uvs) = match face_dir {
                        0 => (
                            make_pt(u0, v0),
                            make_pt(u0, v1),
                            make_pt(u1, v1),
                            make_pt(u1, v0),
                            [[u_lo, v_hi], [u_lo, v_lo], [u_hi, v_lo], [u_hi, v_hi]],
                        ),
                        1 => (
                            make_pt(u1, v0),
                            make_pt(u1, v1),
                            make_pt(u0, v1),
                            make_pt(u0, v0),
                            [[u_lo, v_hi], [u_lo, v_lo], [u_hi, v_lo], [u_hi, v_hi]],
                        ),
                        2 => (
                            make_pt(u0, v1),
                            make_pt(u1, v1),
                            make_pt(u1, v0),
                            make_pt(u0, v0),
                            [[u_lo, v_lo], [u_hi, v_lo], [u_hi, v_hi], [u_lo, v_hi]],
                        ),
                        3 => (
                            make_pt(u0, v0),
                            make_pt(u1, v0),
                            make_pt(u1, v1),
                            make_pt(u0, v1),
                            [[u_lo, v_lo], [u_hi, v_lo], [u_hi, v_hi], [u_lo, v_hi]],
                        ),
                        4 => (
                            make_pt(u1, v0),
                            make_pt(u1, v1),
                            make_pt(u0, v1),
                            make_pt(u0, v0),
                            [[u_lo, v_hi], [u_lo, v_lo], [u_hi, v_lo], [u_hi, v_hi]],
                        ),
                        _ => (
                            make_pt(u0, v0),
                            make_pt(u0, v1),
                            make_pt(u1, v1),
                            make_pt(u1, v0),
                            [[u_lo, v_hi], [u_lo, v_lo], [u_hi, v_lo], [u_hi, v_hi]],
                        ),
                    };

                    let packed = VoxelVertex::pack(
                        cell.layer,
                        face_dir as u8,
                        cell.skylight,
                        cell.blocklight,
                        cell.ao,
                        cell.waving,
                        cell.cutout,
                    ) | cell.extra_bits;
                    let base = opaque_verts.len() as u32;
                    for (pos, uv) in [p0, p1, p2, p3].into_iter().zip(uvs) {
                        opaque_verts.push(VoxelVertex {
                            pos,
                            uv,
                            packed,
                            tint: cell.tint,
                        });
                    }
                    opaque_indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);

                    u += w;
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // Pass 2: Non-Opaque / Specialized DrawTypes (Normal-alpha, Allfaces,
    // Glasslike, Plantlike, Torchlike, Signlike, Raillike, Fencelike,
    // Firelike, NodeBox, Liquid/FlowingLiquid, Mesh)
    // ---------------------------------------------------------------------
    for ly in 0..CHUNK_SIZE {
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let (id, p1, p2) = snap.get(lx as isize, ly as isize, lz as isize);
                if id == CONTENT_AIR || id == CONTENT_IGNORE {
                    continue;
                }
                let Some(def) = registries.get_node_def(id) else {
                    continue;
                };
                if def.drawtype == DrawType::Airlike || is_opaque_occluder(id, registries) {
                    continue;
                }

                let mut self_sky = get_sky_light(p1);
                let mut self_block = get_block_light(p1);
                let emit = def.light_source.min(15);
                if emit > self_block {
                    self_block = emit;
                }
                // If node doesn't store internal light (e.g. non-paramtype=light nodebox), sample brightest neighbor
                let mut best_sky = self_sky;
                let mut best_block = self_block;
                for n in FACE_NORMALS {
                    let (_, np1, _) = snap.get(lx as isize + n[0], ly as isize + n[1], lz as isize + n[2]);
                    best_sky = best_sky.max(get_sky_light(np1));
                    best_block = best_block.max(get_block_light(np1));
                }
                if self_sky == 0 && def.paramtype != "light" {
                    self_sky = best_sky;
                }
                if self_block == 0 && def.paramtype != "light" {
                    self_block = best_block;
                }

                let layers = face_layers.get(&id).copied().unwrap_or([0; 6]);
                let bx = ox + lx as f32;
                let by = oy + ly as f32;
                let bz = oz + lz as f32;
                let extra_bits = node_material_extra_bits(def);
                let o_start = opaque_verts.len();
                let t_start = trans_verts.len();

                match def.drawtype {
                    DrawType::Normal => {
                        // Non-opaque Normal block (clip or blend)
                        let blend = node_uses_blend(def);
                        let cutout = !blend && node_uses_cutout(def);
                        let (verts, idxs) = if blend {
                            (&mut trans_verts, &mut trans_indices)
                        } else {
                            (&mut opaque_verts, &mut opaque_indices)
                        };
                        let facedir = get_node_facedir(def, p2);
                        for f in 0..6usize {
                            let n = FACE_NORMALS[f];
                            let (nb, np1, _) = snap.get(lx as isize + n[0], ly as isize + n[1], lz as isize + n[2]);
                            if is_opaque_occluder(nb, registries) || (blend && nb == id) {
                                continue;
                            }
                            let (tf, rot) = facedir_face_tile(facedir, f);
                            let tint = resolve_face_tint(def, tf, p2, atlas);
                            emit_single_unit_face(
                                verts,
                                idxs,
                                [bx, by, bz],
                                f,
                                layers[tf],
                                rot,
                                self_sky.max(get_sky_light(np1)),
                                self_block.max(get_block_light(np1)),
                                [3; 4],
                                false,
                                cutout,
                                tint,
                                0.0,
                            );
                        }
                    }

                    DrawType::Liquid | DrawType::FlowingLiquid => {
                        emit_liquid_node(
                            snap,
                            registries,
                            lx as isize,
                            ly as isize,
                            lz as isize,
                            [bx, by, bz],
                            id,
                            p2,
                            def,
                            layers,
                            self_sky.max(best_sky),
                            self_block.max(best_block),
                            atlas,
                            &mut opaque_verts,
                            &mut opaque_indices,
                            &mut trans_verts,
                            &mut trans_indices,
                        );
                    }

                    DrawType::Allfaces | DrawType::AllfacesOptional => {
                        let blend = def.use_texture_alpha.eq_ignore_ascii_case("blend");
                        let cutout = !blend;
                        let (verts, idxs) = if blend {
                            (&mut trans_verts, &mut trans_indices)
                        } else {
                            (&mut opaque_verts, &mut opaque_indices)
                        };
                        let waving = def.waving > 0 || def.name.contains("leaves") || def.groups.contains_key("leaves");
                        for f in 0..6usize {
                            let n = FACE_NORMALS[f];
                            let (nb, np1, _) = snap.get(lx as isize + n[0], ly as isize + n[1], lz as isize + n[2]);
                            // Allfaces NEVER culls against adjacent leaves! Only solid opaque occluders hide it.
                            if is_opaque_occluder(nb, registries) {
                                continue;
                            }
                            let tint = resolve_face_tint(def, f, p2, atlas);
                            let f_sky_raw = self_sky.max(best_sky).max(get_sky_light(np1));
                            let f_sky = if f_sky_raw > 0 { f_sky_raw.max(13) } else { 0 };
                            let f_blk = self_block.max(best_block).max(get_block_light(np1));
                            emit_single_unit_face(
                                verts,
                                idxs,
                                [bx, by, bz],
                                f,
                                layers[f],
                                TileRot::R0,
                                f_sky,
                                f_blk,
                                [3; 4],
                                waving,
                                cutout,
                                tint,
                                0.0,
                            );
                        }
                    }

                    DrawType::Glasslike | DrawType::GlasslikeFramed | DrawType::GlasslikeFramedOptional => {
                        let blend = def.use_texture_alpha.eq_ignore_ascii_case("blend");
                        let cutout = !blend;
                        let (verts, idxs) = if blend {
                            (&mut trans_verts, &mut trans_indices)
                        } else {
                            (&mut opaque_verts, &mut opaque_indices)
                        };
                        let is_framed = matches!(
                            def.drawtype,
                            DrawType::GlasslikeFramed | DrawType::GlasslikeFramedOptional
                        );
                        for f in 0..6usize {
                            let n = FACE_NORMALS[f];
                            let (nb, np1, _) = snap.get(lx as isize + n[0], ly as isize + n[1], lz as isize + n[2]);
                            if nb == id || is_opaque_occluder(nb, registries) {
                                continue;
                            }
                            let tint = resolve_face_tint(def, f, p2, atlas);
                            let f_sky = self_sky.max(get_sky_light(np1));
                            let f_blk = self_block.max(get_block_light(np1));
                            let glass_layer = if is_framed && def.tiles.len() >= 2 {
                                // In GlasslikeFramed, tiles[0] can be frame or tiles[last] is glass interior
                                layers[4]
                            } else {
                                layers[0]
                            };
                            emit_single_unit_face(
                                verts,
                                idxs,
                                [bx, by, bz],
                                f,
                                glass_layer,
                                TileRot::R0,
                                f_sky,
                                f_blk,
                                [3; 4],
                                false,
                                cutout,
                                tint,
                                0.0,
                            );
                        }
                        if is_framed {
                            // Emit thin outer frame edges where adjacent blocks are not identical glass
                            emit_glass_frame_edges(
                                snap,
                                lx as isize,
                                ly as isize,
                                lz as isize,
                                [bx, by, bz],
                                id,
                                layers[2],
                                self_sky,
                                self_block,
                                resolve_face_tint(def, 2, p2, atlas),
                                cutout,
                                verts,
                                idxs,
                            );
                        }
                    }

                    DrawType::Plantlike => {
                        let tint = resolve_face_tint(def, 2, p2, atlas);
                        let packed_top = VoxelVertex::pack(layers[2], 2, self_sky, self_block, 3, true, true);
                        let packed_bot = VoxelVertex::pack(layers[2], 2, self_sky, self_block, 3, false, true);
                        let scale = def.visual_scale.clamp(0.4, 2.0);
                        emit_plantlike(
                            &mut opaque_verts,
                            &mut opaque_indices,
                            [bx, by, bz],
                            packed_bot,
                            packed_top,
                            tint,
                            scale,
                            def,
                            p2,
                        );
                    }

                    DrawType::Firelike => {
                        let tint = resolve_face_tint(def, 2, p2, atlas);
                        let packed = VoxelVertex::pack(layers[2], 2, 15, 15, 3, true, true);
                        emit_firelike(
                            snap,
                            lx as isize,
                            ly as isize,
                            lz as isize,
                            [bx, by, bz],
                            id,
                            packed,
                            tint,
                            &mut opaque_verts,
                            &mut opaque_indices,
                        );
                    }

                    DrawType::Torchlike => {
                        let wall = get_node_wallmounted(def, p2);
                        let tile_layer = match wall {
                            0 | 6 => layers[3], // ceiling -> tiles[1]
                            1 | 7 => layers[2], // floor -> tiles[0]
                            _ => layers[0],     // wall -> tiles[2]
                        };
                        emit_torchlike_3d(
                            &mut opaque_verts,
                            &mut opaque_indices,
                            [bx, by, bz],
                            wall,
                            [tile_layer; 6],
                            self_sky.max(12),
                            self_block.max(14),
                        );
                    }

                    DrawType::Signlike => {
                        let wall = get_node_wallmounted(def, p2);
                        let tint = resolve_face_tint(def, 2, p2, atlas);
                        let packed = VoxelVertex::pack(layers[2], 2, self_sky, self_block, 3, false, true);
                        emit_signlike(
                            &mut opaque_verts,
                            &mut opaque_indices,
                            [bx, by, bz],
                            wall,
                            def.visual_scale.clamp(0.4, 2.0),
                            packed,
                            tint,
                        );
                    }

                    DrawType::Raillike => {
                        emit_raillike(
                            snap,
                            registries,
                            lx as isize,
                            ly as isize,
                            lz as isize,
                            [bx, by, bz],
                            id,
                            def,
                            layers,
                            self_sky,
                            self_block,
                            resolve_face_tint(def, 2, p2, atlas),
                            &mut opaque_verts,
                            &mut opaque_indices,
                        );
                    }

                    DrawType::Fencelike => {
                        let tint = resolve_face_tint(def, 2, p2, atlas);
                        emit_fencelike(
                            snap,
                            registries,
                            lx as isize,
                            ly as isize,
                            lz as isize,
                            [bx, by, bz],
                            layers,
                            self_sky,
                            self_block,
                            tint,
                            &mut opaque_verts,
                            &mut opaque_indices,
                        );
                    }

                    DrawType::NodeBox => {
                        let blend = def.use_texture_alpha.eq_ignore_ascii_case("blend");
                        let cutout = !blend;
                        let (verts, idxs) = if blend {
                            (&mut trans_verts, &mut trans_indices)
                        } else {
                            (&mut opaque_verts, &mut opaque_indices)
                        };
                        emit_nodebox_node(
                            snap,
                            registries,
                            lx as isize,
                            ly as isize,
                            lz as isize,
                            [bx, by, bz],
                            id,
                            p2,
                            def,
                            layers,
                            self_sky,
                            self_block,
                            cutout,
                            atlas,
                            verts,
                            idxs,
                        );
                    }

                    DrawType::Mesh => {
                        let blend = def.use_texture_alpha.eq_ignore_ascii_case("blend");
                        let cutout = !blend;
                        let (verts, idxs) = if blend {
                            (&mut trans_verts, &mut trans_indices)
                        } else {
                            (&mut opaque_verts, &mut opaque_indices)
                        };
                        let obj_opt = atlas.and_then(|atl| {
                            def.mesh.as_deref().and_then(|m| atl.get_obj_mesh(m))
                        });
                        if let Some(obj) = obj_opt {
                            emit_obj_mesh(
                                obj,
                                [bx, by, bz],
                                def,
                                p2,
                                layers,
                                self_sky,
                                self_block,
                                cutout,
                                atlas,
                                verts,
                                idxs,
                            );
                        } else if !def.node_boxes.is_empty() || !def.selection_boxes.is_empty() {
                            let boxes = if !def.node_boxes.is_empty() {
                                &def.node_boxes
                            } else {
                                &def.selection_boxes
                            };
                            let facedir = get_node_facedir(def, p2);
                            for b in boxes {
                                let rb = rotate_aabb_by_facedir(*b, facedir);
                                emit_cuboid_with_facedir(
                                    verts,
                                    idxs,
                                    [bx, by, bz],
                                    rb,
                                    layers,
                                    facedir,
                                    self_sky,
                                    self_block,
                                    false,
                                    cutout,
                                    |tf| resolve_face_tint(def, tf, p2, atlas),
                                    0b0000_0000,
                                );
                            }
                        } else {
                            let tint = resolve_face_tint(def, 2, p2, atlas);
                            emit_box(
                                verts,
                                idxs,
                                [bx + 0.125, by, bz + 0.125],
                                [bx + 0.875, by + 0.875, bz + 0.875],
                                layers,
                                self_sky,
                                self_block,
                                false,
                                cutout,
                                tint,
                                None::<fn(usize) -> bool>,
                            );
                        }
                    }

                    DrawType::Airlike => {}
                }
                if extra_bits != 0 {
                    for v in &mut opaque_verts[o_start..] {
                        v.packed |= extra_bits;
                    }
                    for v in &mut trans_verts[t_start..] {
                        v.packed |= extra_bits;
                    }
                }
            }
        }
    }

    ChunkMeshData {
        chunk_pos: snap.pos,
        opaque_verts,
        opaque_indices,
        trans_verts,
        trans_indices,
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_single_unit_face(
    verts: &mut Vec<VoxelVertex>,
    indices: &mut Vec<u32>,
    base_pos: [f32; 3],
    face_dir: usize,
    layer: u16,
    rot: TileRot,
    sky: u8,
    block: u8,
    corner_aos: [u8; 4],
    waving: bool,
    cutout: bool,
    tint: [u8; 4],
    normal_offset: f32,
) {
    let [x0, y0, z0] = base_pos;
    let [x1, y1, z1] = [x0 + 1.0, y0 + 1.0, z0 + 1.0];
    let n = FACE_NORMALS[face_dir];
    let dx = n[0] as f32 * normal_offset;
    let dy = n[1] as f32 * normal_offset;
    let dz = n[2] as f32 * normal_offset;

    let (pts, raw_uvs) = match face_dir {
        0 => (
            [[x1 + dx, y0, z0], [x1 + dx, y1, z0], [x1 + dx, y1, z1], [x1 + dx, y0, z1]],
            [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        ),
        1 => (
            [[x0 + dx, y0, z1], [x0 + dx, y1, z1], [x0 + dx, y1, z0], [x0 + dx, y0, z0]],
            [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        ),
        2 => (
            [[x0, y1 + dy, z1], [x1, y1 + dy, z1], [x1, y1 + dy, z0], [x0, y1 + dy, z0]],
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        ),
        3 => (
            [[x0, y0 + dy, z0], [x1, y0 + dy, z0], [x1, y0 + dy, z1], [x0, y0 + dy, z1]],
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        ),
        4 => (
            [[x1, y0, z1 + dz], [x1, y1, z1 + dz], [x0, y1, z1 + dz], [x0, y0, z1 + dz]],
            [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        ),
        _ => (
            [[x0, y0, z0 + dz], [x0, y1, z0 + dz], [x1, y1, z0 + dz], [x1, y0, z0 + dz]],
            [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        ),
    };

    let base = verts.len() as u32;
    for (idx, (pos, uv)) in pts.into_iter().zip(raw_uvs).enumerate() {
        let packed = VoxelVertex::pack(
            layer,
            face_dir as u8,
            sky,
            block,
            corner_aos[idx],
            waving,
            cutout,
        );
        verts.push(VoxelVertex {
            pos,
            uv: rotate_uv([inset_unit_uv(uv[0]), inset_unit_uv(uv[1])], rot),
            packed,
            tint,
        });
    }
    if corner_aos[0] + corner_aos[2] >= corner_aos[1] + corner_aos[3] {
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    } else {
        indices.extend_from_slice(&[base + 1, base + 2, base + 3, base + 1, base + 3, base]);
    }
}

/// Emit a sub-voxel cuboid `b = [min_x, min_y, min_z, max_x, max_y, max_z]` (in `[-0.5, 0.5]^3`)
/// with Luanti `generateCuboidTextureCoords` UVs, `facedir` tile/UV rotation, and face-cull `skip_mask`.
#[allow(clippy::too_many_arguments)]
fn emit_cuboid_with_facedir<F: Fn(usize) -> [u8; 4]>(
    verts: &mut Vec<VoxelVertex>,
    indices: &mut Vec<u32>,
    origin: [f32; 3],
    b: [f32; 6],
    layers: [u16; 6],
    facedir: u8,
    sky: u8,
    block: u8,
    waving: bool,
    cutout: bool,
    tint_fn: F,
    skip_mask: u8,
) {
    let x0 = origin[0] + b[0] + 0.5;
    let y0 = origin[1] + b[1] + 0.5;
    let z0 = origin[2] + b[2] + 0.5;
    let x1 = origin[0] + b[3] + 0.5;
    let y1 = origin[1] + b[4] + 0.5;
    let z1 = origin[2] + b[5] + 0.5;

    let tx0 = (b[0] + 0.5).clamp(0.0, 1.0);
    let ty0 = (b[1] + 0.5).clamp(0.0, 1.0);
    let tz0 = (b[2] + 0.5).clamp(0.0, 1.0);
    let tx1 = (b[3] + 0.5).clamp(0.0, 1.0);
    let ty1 = (b[4] + 0.5).clamp(0.0, 1.0);
    let tz1 = (b[5] + 0.5).clamp(0.0, 1.0);

    let faces: [([[f32; 3]; 4], [[f32; 2]; 4]); 6] = [
        // 0: +X
        (
            [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]],
            [[tz0, 1.0 - ty0], [tz0, 1.0 - ty1], [tz1, 1.0 - ty1], [tz1, 1.0 - ty0]],
        ),
        // 1: -X
        (
            [[x0, y0, z1], [x0, y1, z1], [x0, y1, z0], [x0, y0, z0]],
            [[1.0 - tz1, 1.0 - ty0], [1.0 - tz1, 1.0 - ty1], [1.0 - tz0, 1.0 - ty1], [1.0 - tz0, 1.0 - ty0]],
        ),
        // 2: +Y
        (
            [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]],
            [[tx0, 1.0 - tz1], [tx1, 1.0 - tz1], [tx1, 1.0 - tz0], [tx0, 1.0 - tz0]],
        ),
        // 3: -Y
        (
            [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],
            [[tx0, tz0], [tx1, tz0], [tx1, tz1], [tx0, tz1]],
        ),
        // 4: +Z
        (
            [[x1, y0, z1], [x1, y1, z1], [x0, y1, z1], [x0, y0, z1]],
            [[1.0 - tx1, 1.0 - ty0], [1.0 - tx1, 1.0 - ty1], [1.0 - tx0, 1.0 - ty1], [1.0 - tx0, 1.0 - ty0]],
        ),
        // 5: -Z
        (
            [[x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [x1, y0, z0]],
            [[tx0, 1.0 - ty0], [tx0, 1.0 - ty1], [tx1, 1.0 - ty1], [tx1, 1.0 - ty0]],
        ),
    ];

    for (f, (pts, raw_uvs)) in faces.into_iter().enumerate() {
        if (skip_mask & (1 << f)) != 0 {
            continue;
        }
        let (tf, rot) = facedir_face_tile(facedir, f);
        let tint = tint_fn(tf);
        let packed = VoxelVertex::pack(layers[tf], f as u8, sky, block, 3, waving, cutout);
        let base = verts.len() as u32;
        for (pos, uv) in pts.into_iter().zip(raw_uvs) {
            verts.push(VoxelVertex {
                pos,
                uv: rotate_uv(uv, rot),
                packed,
                tint,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}


/// Checks whether a `connected` NodeBox at `(lx, ly, lz)` connects to neighbor in direction `dir`.
/// Ported from Luanti `NodeDefManager::nodeboxConnects` (`src/nodedef.cpp`).
fn nodebox_connects_to(
    def: &NodeDef,
    id: u16,
    neighbor_id: u16,
    neighbor_p2: u8,
    connect_bit: u8,
    registries: &EngineRegistries,
) -> bool {
    if neighbor_id == CONTENT_AIR || neighbor_id == CONTENT_IGNORE || def.connects_to.is_empty() {
        return false;
    }
    let Some(nb_def) = registries.get_node_def(neighbor_id) else {
        return false;
    };

    let matches_target = |source_def: &NodeDef, target_def: &NodeDef| -> bool {
        source_def.connects_to.iter().any(|entry| {
            if let Some(grp) = entry.strip_prefix("group:") {
                target_def.groups.get(grp).copied().unwrap_or(0) > 0
            } else {
                target_def.name == *entry
            }
        })
    };

    if !matches_target(def, nb_def) {
        return false;
    }

    if nb_def.drawtype == DrawType::NodeBox && nb_def.node_box_type == "connected" {
        return matches_target(nb_def, def);
    }

    if !nb_def.connect_sides.is_empty() {
        let mut mask = 0u8;
        for side in &nb_def.connect_sides {
            match side.as_str() {
                "top" => mask |= 2,
                "bottom" => mask |= 1,
                "front" => mask |= 16,
                "left" => mask |= 32,
                "back" => mask |= 4,
                "right" => mask |= 8,
                _ => {}
            }
        }
        if matches!(
            nb_def.paramtype2.as_str(),
            "facedir" | "colorfacedir" | "4dir" | "color4dir"
        ) && connect_bit >= 4
        {
            let rot = (neighbor_p2 & 0x03) as usize;
            let rotated_bit = match connect_bit {
                4 => [4, 32, 16, 8][rot],
                8 => [8, 4, 32, 16][rot],
                16 => [16, 8, 4, 32][rot],
                32 => [32, 16, 8, 4][rot],
                _ => connect_bit,
            };
            return (mask & rotated_bit) != 0;
        }
        return (mask & connect_bit) != 0;
    }

    let _ = id;
    true
}

/// Collect transformed sub-boxes for a `NodeBox` node (`fixed`, `wallmounted`, `connected`, `leveled`, `regular`).
pub fn resolve_nodebox_aabbs(
    snap: &ChunkSnapshot,
    registries: &EngineRegistries,
    lx: isize,
    ly: isize,
    lz: isize,
    id: u16,
    p2: u8,
    def: &NodeDef,
) -> Vec<[f32; 6]> {
    let mut boxes = Vec::new();
    let nb_spec: NodeBoxSpec = def.node_box_spec();
    let box_type = nb_spec.box_type.as_str();

    match box_type {
        "wallmounted" => {
            let wall = get_node_wallmounted(def, p2);
            match wall {
                0 | 6 => {
                    let mut b = nb_spec.wall_top.unwrap_or([-0.5, 0.4375, -0.5, 0.5, 0.5, 0.5]);
                    if wall == 6 {
                        let a = rotate_xz([b[0], b[1], b[2]], 90.0);
                        let c = rotate_xz([b[3], b[4], b[5]], 90.0);
                        b = [a[0].min(c[0]), a[1].min(c[1]), a[2].min(c[2]), a[0].max(c[0]), a[1].max(c[1]), a[2].max(c[2])];
                    }
                    boxes.push(b);
                }
                1 | 7 => {
                    let mut b = nb_spec.wall_bottom.unwrap_or([-0.5, -0.5, -0.5, 0.5, -0.4375, 0.5]);
                    if wall == 7 {
                        let a = rotate_xz([b[0], b[1], b[2]], -90.0);
                        let c = rotate_xz([b[3], b[4], b[5]], -90.0);
                        b = [a[0].min(c[0]), a[1].min(c[1]), a[2].min(c[2]), a[0].max(c[0]), a[1].max(c[1]), a[2].max(c[2])];
                    }
                    boxes.push(b);
                }
                _ => {
                    let side = nb_spec.wall_side.unwrap_or([-0.5, -0.5, -0.5, -0.4375, 0.5, 0.5]);
                    let deg = match wall {
                        2 => 180.0, // +X
                        3 => 0.0,   // -X
                        4 => -90.0, // +Z
                        5 => 90.0,  // -Z
                        _ => 0.0,
                    };
                    let a = rotate_xz([side[0], side[1], side[2]], deg);
                    let c = rotate_xz([side[3], side[4], side[5]], deg);
                    boxes.push([
                        a[0].min(c[0]),
                        a[1].min(c[1]),
                        a[2].min(c[2]),
                        a[0].max(c[0]),
                        a[1].max(c[1]),
                        a[2].max(c[2]),
                    ]);
                }
            }
        }
        "connected" => {
            // Connection check order from Luanti `content_mapblock.cpp:1516`
            const CONN_DIRS: [(isize, isize, isize, u8); 6] = [
                (0, 1, 0, 1),   // top
                (0, -1, 0, 2),  // bottom
                (0, 0, -1, 4),  // front (-Z)
                (-1, 0, 0, 8),  // left (-X)
                (0, 0, 1, 16),  // back (+Z)
                (1, 0, 0, 32),  // right (+X)
            ];
            let mut neighbors_set = 0u8;
            for (dx, dy, dz, bit) in CONN_DIRS {
                let (nid, _, np2) = snap.get(lx + dx, ly + dy, lz + dz);
                if nodebox_connects_to(def, id, nid, np2, bit, registries) {
                    neighbors_set |= bit;
                }
            }
            boxes.extend_from_slice(&nb_spec.fixed);
            if (neighbors_set & 1) != 0 {
                boxes.extend_from_slice(&nb_spec.connect_top);
            } else {
                boxes.extend_from_slice(&nb_spec.disconnected_top);
            }
            if (neighbors_set & 2) != 0 {
                boxes.extend_from_slice(&nb_spec.connect_bottom);
            } else {
                boxes.extend_from_slice(&nb_spec.disconnected_bottom);
            }
            if (neighbors_set & 4) != 0 {
                boxes.extend_from_slice(&nb_spec.connect_front);
            } else {
                boxes.extend_from_slice(&nb_spec.disconnected_front);
            }
            if (neighbors_set & 8) != 0 {
                boxes.extend_from_slice(&nb_spec.connect_left);
            } else {
                boxes.extend_from_slice(&nb_spec.disconnected_left);
            }
            if (neighbors_set & 16) != 0 {
                boxes.extend_from_slice(&nb_spec.connect_back);
            } else {
                boxes.extend_from_slice(&nb_spec.disconnected_back);
            }
            if (neighbors_set & 32) != 0 {
                boxes.extend_from_slice(&nb_spec.connect_right);
            } else {
                boxes.extend_from_slice(&nb_spec.disconnected_right);
            }
            if neighbors_set == 0 {
                boxes.extend_from_slice(&nb_spec.disconnected_all);
            }
            if neighbors_set < 4 {
                boxes.extend_from_slice(&nb_spec.disconnected_sides);
            }
        }
        "fixed" | "leveled" => {
            let facedir = get_node_facedir(def, p2);
            let level = if box_type == "leveled" {
                let l = if def.paramtype2 == "leveled" {
                    p2 & 0x7F
                } else {
                    0
                };
                let eff = if l > 0 { l } else { def.leveled.max(8) };
                (eff.min(64) as f32) / 64.0
            } else {
                0.0
            };
            let src_boxes = if nb_spec.fixed.is_empty() {
                vec![[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]]
            } else {
                nb_spec.fixed
            };
            for mut b in src_boxes {
                if box_type == "leveled" {
                    b[4] = -0.5 + level;
                }
                boxes.push(rotate_aabb_by_facedir(b, facedir));
            }
        }
        _ => {
            if nb_spec.fixed.is_empty() {
                boxes.push([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]);
            } else {
                let facedir = get_node_facedir(def, p2);
                for b in nb_spec.fixed {
                    boxes.push(rotate_aabb_by_facedir(b, facedir));
                }
            }
        }
    }
    boxes
}

#[allow(clippy::too_many_arguments)]
fn emit_nodebox_node(
    snap: &ChunkSnapshot,
    registries: &EngineRegistries,
    lx: isize,
    ly: isize,
    lz: isize,
    origin: [f32; 3],
    id: u16,
    p2: u8,
    def: &NodeDef,
    layers: [u16; 6],
    sky: u8,
    block: u8,
    cutout: bool,
    atlas: Option<&TextureAtlas>,
    verts: &mut Vec<VoxelVertex>,
    indices: &mut Vec<u32>,
) {
    let boxes = resolve_nodebox_aabbs(snap, registries, lx, ly, lz, id, p2, def);
    let facedir = if matches!(def.node_box_type.as_str(), "fixed" | "leveled" | "regular") {
        get_node_facedir(def, p2)
    } else {
        0
    };

    // Determine solid/same-type neighbors for face culling on boundary-touching sub-boxes
    let mut solid_mask = 0u8;
    let mut sametype_mask = 0u8;
    for f in 0..6usize {
        let n = FACE_NORMALS[f];
        let (nid, _, np2) = snap.get(lx + n[0], ly + n[1], lz + n[2]);
        if is_opaque_occluder(nid, registries) {
            solid_mask |= 1 << f;
        }
        if nid == id && p2 == np2 {
            sametype_mask |= 1 << f;
        }
    }

    for b in boxes {
        let eps = 0.0005;
        let mut boundary_mask = 0u8;
        if b[3] <= 0.5 + eps && b[0] >= -0.5 - eps && b[4] <= 0.5 + eps && b[1] >= -0.5 - eps && b[5] <= 0.5 + eps && b[2] >= -0.5 - eps {
            if (b[3] - 0.5).abs() < eps { boundary_mask |= 1 << 0; }
            if (b[0] + 0.5).abs() < eps { boundary_mask |= 1 << 1; }
            if (b[4] - 0.5).abs() < eps { boundary_mask |= 1 << 2; }
            if (b[1] + 0.5).abs() < eps { boundary_mask |= 1 << 3; }
            if (b[5] - 0.5).abs() < eps { boundary_mask |= 1 << 4; }
            if (b[2] + 0.5).abs() < eps { boundary_mask |= 1 << 5; }
        }
        let skip = (boundary_mask & solid_mask)
            | (if def.use_texture_alpha == "opaque" {
                let opp = ((boundary_mask & 0b00_0011) == 0b00_0011) as u8 * 0b00_0011
                    | ((boundary_mask & 0b00_1100) == 0b00_1100) as u8 * 0b00_1100
                    | ((boundary_mask & 0b11_0000) == 0b11_0000) as u8 * 0b11_0000;
                opp & sametype_mask
            } else {
                0
            });

        emit_cuboid_with_facedir(
            verts,
            indices,
            origin,
            b,
            layers,
            facedir,
            sky,
            block,
            def.waving > 0,
            cutout,
            |tf| resolve_face_tint(def, tf, p2, atlas),
            skip,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_liquid_node(
    snap: &ChunkSnapshot,
    registries: &EngineRegistries,
    lx: isize,
    ly: isize,
    lz: isize,
    origin: [f32; 3],
    id: u16,
    p2: u8,
    def: &NodeDef,
    layers: [u16; 6],
    sky: u8,
    block: u8,
    atlas: Option<&TextureAtlas>,
    opaque_verts: &mut Vec<VoxelVertex>,
    opaque_indices: &mut Vec<u32>,
    trans_verts: &mut Vec<VoxelVertex>,
    trans_indices: &mut Vec<u32>,
) {
    let blend = node_uses_blend(def);
    let (verts, indices) = if blend {
        (trans_verts, trans_indices)
    } else {
        (opaque_verts, opaque_indices)
    };

    let src_id = if !def.liquid_alternative_source.is_empty() {
        registries.get_content_id(&def.liquid_alternative_source)
    } else if def.drawtype == DrawType::Liquid {
        id
    } else {
        crate::registry::CONTENT_UNKNOWN
    };
    let flow_id = if !def.liquid_alternative_flowing.is_empty() {
        registries.get_content_id(&def.liquid_alternative_flowing)
    } else if def.drawtype == DrawType::FlowingLiquid {
        id
    } else {
        crate::registry::CONTENT_UNKNOWN
    };

    let is_same_liq = |nid: u16| -> bool {
        nid == id || (src_id != crate::registry::CONTENT_UNKNOWN && nid == src_id)
            || (flow_id != crate::registry::CONTENT_UNKNOWN && nid == flow_id)
    };

    let liq_level_of = |nid: u16, np2: u8| -> f32 {
        if nid == src_id || registries.get_node_def(nid).map(|d| d.drawtype == DrawType::Liquid).unwrap_or(false) {
            0.88
        } else if is_same_liq(nid) {
            let lvl = (np2 & 0x07).max(1) as f32;
            (0.12 + (lvl / 8.0) * 0.76).clamp(0.12, 0.88)
        } else {
            0.0
        }
    };

    let top_id = snap.get(lx, ly + 1, lz).0;
    let top_same = is_same_liq(top_id);
    let bot_id = snap.get(lx, ly - 1, lz).0;
    let draw_bot = !is_same_liq(bot_id) && !is_opaque_occluder(bot_id, registries);

    // Compute 4 corner heights: [z in 0..=1][x in 0..=1] matching Luanti `getCornerLevel`
    let mut corner = [[0.88f32; 2]; 2];
    if top_same {
        corner = [[1.0; 2]; 2];
    } else {
        for cz in 0..=1usize {
            for cx in 0..=1usize {
                let mut sum = 0.0f32;
                let mut cnt = 0u32;
                let mut air_cnt = 0u32;
                let mut has_top_liq = false;
                let mut has_src = false;
                for dz in 0..=1isize {
                    for dx in 0..=1isize {
                        let ox = lx + cx as isize + dx - 1;
                        let oz = lz + cz as isize + dz - 1;
                        let (nid, _, np2) = snap.get(ox, ly, oz);
                        let (aid, _, _) = snap.get(ox, ly + 1, oz);
                        if is_same_liq(aid) {
                            has_top_liq = true;
                        }
                        if nid == src_id
                            || registries
                                .get_node_def(nid)
                                .map(|d| d.drawtype == DrawType::Liquid)
                                .unwrap_or(false)
                        {
                            has_src = true;
                        } else if is_same_liq(nid) {
                            sum += liq_level_of(nid, np2);
                            cnt += 1;
                        } else if nid == CONTENT_AIR {
                            air_cnt += 1;
                        }
                    }
                }
                corner[cz][cx] = if has_top_liq {
                    1.0
                } else if has_src {
                    0.88
                } else if air_cnt >= 2 && def.drawtype == DrawType::FlowingLiquid {
                    0.08
                } else if cnt > 0 {
                    (sum / cnt as f32).clamp(0.08, 0.88)
                } else {
                    liq_level_of(id, p2)
                };
            }
        }
    }

    let mut tint = resolve_face_tint(def, 2, p2, atlas);
    if blend && tint[3] > 210 {
        tint[3] = 192; // Translucent water alpha
    }
    let waving = def.waving > 0 || def.name.contains("water");
    let [bx, by, bz] = origin;

    // Top surface (plus underside when translucent water so diving underwater sees the surface above)
    if !top_same {
        let packed = VoxelVertex::pack(layers[2], 2, sky, block, 3, waving, false);
        let pts = [
            [bx, by + corner[1][0], bz + 1.0],
            [bx + 1.0, by + corner[1][1], bz + 1.0],
            [bx + 1.0, by + corner[0][1], bz],
            [bx, by + corner[0][0], bz],
        ];
        let uvs = inset_unit_uvs([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let base = verts.len() as u32;
        for (pos, uv) in pts.into_iter().zip(uvs) {
            verts.push(VoxelVertex { pos, uv, packed, tint });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        if blend {
            let under_packed = VoxelVertex::pack(layers[3], 3, sky, block, 3, waving, false);
            let ubase = verts.len() as u32;
            for (pos, uv) in [pts[3], pts[2], pts[1], pts[0]].into_iter().zip(uvs) {
                verts.push(VoxelVertex { pos, uv, packed: under_packed, tint });
            }
            indices.extend_from_slice(&[ubase, ubase + 1, ubase + 2, ubase, ubase + 2, ubase + 3]);
        }
    }

    // Bottom surface
    if draw_bot {
        let packed = VoxelVertex::pack(layers[3], 3, sky, block, 3, waving, false);
        let pts = [
            [bx, by, bz],
            [bx + 1.0, by, bz],
            [bx + 1.0, by, bz + 1.0],
            [bx, by, bz + 1.0],
        ];
        let uvs = inset_unit_uvs([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let base = verts.len() as u32;
        for (pos, uv) in pts.into_iter().zip(uvs) {
            verts.push(VoxelVertex { pos, uv, packed, tint });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    // 4 vertical liquid sides with sloped top corners
    let side_defs: [(isize, isize, usize, [f32; 2], [f32; 2], f32, f32); 4] = [
        (1, 0, 0, [bx + 1.0, bz], [bx + 1.0, bz + 1.0], corner[0][1], corner[1][1]),   // +X
        (-1, 0, 1, [bx, bz + 1.0], [bx, bz], corner[1][0], corner[0][0]),              // -X
        (0, 1, 4, [bx + 1.0, bz + 1.0], [bx, bz + 1.0], corner[1][1], corner[1][0]),   // +Z
        (0, -1, 5, [bx, bz], [bx + 1.0, bz], corner[0][0], corner[0][1]),              // -Z
    ];
    for (dx, dz, fdir, p_a, p_b, ha, hb) in side_defs {
        let (nid, _, _) = snap.get(lx + dx, ly, lz + dz);
        let ntop = snap.get(lx + dx, ly + 1, lz + dz).0;
        if is_opaque_occluder(nid, registries) {
            continue;
        }
        if is_same_liq(nid) && (!top_same || is_same_liq(ntop)) {
            continue;
        }
        let packed = VoxelVertex::pack(layers[fdir], fdir as u8, sky, block, 3, waving, false);
        let pts = [
            [p_a[0], by, p_a[1]],
            [p_a[0], by + ha, p_a[1]],
            [p_b[0], by + hb, p_b[1]],
            [p_b[0], by, p_b[1]],
        ];
        let uvs = inset_unit_uvs([[0.0, 1.0], [0.0, 1.0 - ha], [1.0, 1.0 - hb], [1.0, 1.0]]);
        let base = verts.len() as u32;
        for (pos, uv) in pts.into_iter().zip(uvs) {
            verts.push(VoxelVertex { pos, uv, packed, tint });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}


#[allow(clippy::too_many_arguments)]
fn emit_glass_frame_edges(
    snap: &ChunkSnapshot,
    lx: isize,
    ly: isize,
    lz: isize,
    origin: [f32; 3],
    id: u16,
    layer: u16,
    sky: u8,
    block: u8,
    tint: [u8; 4],
    cutout: bool,
    verts: &mut Vec<VoxelVertex>,
    indices: &mut Vec<u32>,
) {
    // Check 6 axis neighbors; draw 4 thin edge bars on exposed edges of non-adjacent glass
    let a = 0.5f32;
    let b = 0.438f32;
    let is_same = |dx: isize, dy: isize, dz: isize| snap.get(lx + dx, ly + dy, lz + dz).0 == id;
    let yp = is_same(0, 1, 0);
    let yn = is_same(0, -1, 0);
    let xp = is_same(1, 0, 0);
    let xn = is_same(-1, 0, 0);
    let zp = is_same(0, 0, 1);
    let zn = is_same(0, 0, -1);

    let edges: [(bool, [f32; 6]); 12] = [
        (!yp && !xp, [b, b, -a, a, a, a]),
        (!yp && !xn, [-a, b, -a, -b, a, a]),
        (!yn && !xp, [b, -a, -a, a, -b, a]),
        (!yn && !xn, [-a, -a, -a, -b, -b, a]),
        (!xp && !zp, [b, -a, b, a, a, a]),
        (!xp && !zn, [b, -a, -a, a, a, -b]),
        (!xn && !zp, [-a, -a, b, -b, a, a]),
        (!xn && !zn, [-a, -a, -a, -b, a, -b]),
        (!yp && !zp, [-a, b, b, a, a, a]),
        (!yn && !zp, [-a, -a, b, a, -b, a]),
        (!yn && !zn, [-a, -a, -a, a, -b, -b]),
        (!yp && !zn, [-a, b, -a, a, a, -b]),
    ];

    for (vis, e) in edges {
        if vis {
            emit_box(
                verts,
                indices,
                [origin[0] + e[0] + 0.5, origin[1] + e[1] + 0.5, origin[2] + e[2] + 0.5],
                [origin[0] + e[3] + 0.5, origin[1] + e[4] + 0.5, origin[2] + e[5] + 0.5],
                [layer; 6],
                sky,
                block,
                false,
                cutout,
                tint,
                None::<fn(usize) -> bool>,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_plantlike(
    verts: &mut Vec<VoxelVertex>,
    indices: &mut Vec<u32>,
    base_pos: [f32; 3],
    packed_bot: u32,
    packed_top: u32,
    tint: [u8; 4],
    visual_scale: f32,
    def: &NodeDef,
    param2: u8,
) {
    let half = 0.44 * visual_scale;
    let height = (0.94 * visual_scale).clamp(0.2, 2.0);
    let mut deg_extra = 0.0f32;
    let mut style = 0u8;
    if def.paramtype2 == "meshoptions" {
        style = param2 & 0x07;
    } else if def.paramtype2 == "degrotate" || def.paramtype2 == "colordegrotate" {
        deg_extra = (param2 as f32) * 1.5;
    }

    let cx = base_pos[0] + 0.5;
    let cy = base_pos[1];
    let cz = base_pos[2] + 0.5;

    let mut emit_rot_quad = |ang_deg: f32, z_off: f32| {
        let rad = (ang_deg + deg_extra).to_radians();
        let (s, c) = rad.sin_cos();
        let rot = |lx: f32, lz: f32| -> [f32; 2] { [cx + lx * c - lz * s, cz + lx * s + lz * c] };
        let l = rot(-half, z_off);
        let r = rot(half, z_off);
        let q1 = ([l[0], cy, l[1]], [l[0], cy + height, l[1]], [r[0], cy + height, r[1]], [r[0], cy, r[1]]);
        let q2 = (q1.3, q1.2, q1.1, q1.0);
        let uvs = inset_unit_uvs([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]);
        let packs = [packed_bot, packed_top, packed_top, packed_bot];
        for (p0, p1, p2, p3) in [q1, q2] {
            let base = verts.len() as u32;
            for ((pos, uv), pk) in [p0, p1, p2, p3].into_iter().zip(uvs).zip(packs) {
                verts.push(VoxelVertex { pos, uv, packed: pk, tint });
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    };

    match style {
        2 => {
            emit_rot_quad(1.0, 0.0);
            emit_rot_quad(121.0, 0.0);
            emit_rot_quad(241.0, 0.0);
        }
        3 | 4 => {
            emit_rot_quad(1.0, 0.25);
            emit_rot_quad(91.0, 0.25);
            emit_rot_quad(181.0, 0.25);
            emit_rot_quad(271.0, 0.25);
        }
        _ => {
            emit_rot_quad(45.0, 0.0);
            emit_rot_quad(-45.0, 0.0);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_firelike(
    snap: &ChunkSnapshot,
    lx: isize,
    ly: isize,
    lz: isize,
    base_pos: [f32; 3],
    id: u16,
    packed: u32,
    tint: [u8; 4],
    verts: &mut Vec<VoxelVertex>,
    indices: &mut Vec<u32>,
) {
    let mut nb = [false; 6];
    let mut any = false;
    for (i, n) in FACE_NORMALS.iter().enumerate() {
        let nid = snap.get(lx + n[0], ly + n[1], lz + n[2]).0;
        if nid != CONTENT_AIR && nid != CONTENT_IGNORE && nid != id {
            nb[i] = true;
            any = true;
        }
    }
    let basic = nb[3] || !any; // floor or floating
    let cx = base_pos[0] + 0.5;
    let cy = base_pos[1];
    let cz = base_pos[2] + 0.5;
    let uvs = [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]];

    let mut push_double = |p0: [f32; 3], p1: [f32; 3], p2: [f32; 3], p3: [f32; 3]| {
        for (a, b, c, d) in [(p0, p1, p2, p3), (p3, p2, p1, p0)] {
            let base = verts.len() as u32;
            for (pos, uv) in [a, b, c, d].into_iter().zip(uvs) {
                verts.push(VoxelVertex { pos, uv, packed, tint });
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    };

    if basic || nb[4] {
        push_double([cx - 0.5, cy, cz + 0.42], [cx - 0.5, cy + 1.0, cz + 0.32], [cx + 0.5, cy + 1.0, cz + 0.32], [cx + 0.5, cy, cz + 0.42]);
    }
    if basic || nb[5] {
        push_double([cx + 0.5, cy, cz - 0.42], [cx + 0.5, cy + 1.0, cz - 0.32], [cx - 0.5, cy + 1.0, cz - 0.32], [cx - 0.5, cy, cz - 0.42]);
    }
    if basic || nb[0] {
        push_double([cx + 0.42, cy, cz - 0.5], [cx + 0.32, cy + 1.0, cz - 0.5], [cx + 0.32, cy + 1.0, cz + 0.5], [cx + 0.42, cy, cz + 0.5]);
    }
    if basic || nb[1] {
        push_double([cx - 0.42, cy, cz + 0.5], [cx - 0.32, cy + 1.0, cz + 0.5], [cx - 0.32, cy + 1.0, cz - 0.5], [cx - 0.42, cy, cz - 0.5]);
    }
    if basic {
        push_double([cx - 0.4, cy, cz - 0.4], [cx - 0.4, cy + 1.0, cz - 0.4], [cx + 0.4, cy + 1.0, cz + 0.4], [cx + 0.4, cy, cz + 0.4]);
        push_double([cx + 0.4, cy, cz - 0.4], [cx + 0.4, cy + 1.0, cz - 0.4], [cx - 0.4, cy + 1.0, cz + 0.4], [cx - 0.4, cy, cz + 0.4]);
    }
}

/// 3D Torchlike geometry (floor-standing and angled wall-mounted by `wallmounted` param2).
fn emit_torchlike_3d(
    verts: &mut Vec<VoxelVertex>,
    indices: &mut Vec<u32>,
    origin: [f32; 3],
    wall: u8,
    layers: [u16; 6],
    sky: u8,
    block: u8,
) {
    let cx = origin[0] + 0.5;
    let cy = origin[1] + 0.5;
    let cz = origin[2] + 0.5;
    let r = 0.0625f32;

    if wall == 0 || wall == 1 || wall == 6 || wall == 7 {
        let y0 = if wall == 0 || wall == 6 { 0.5 - 0.625 } else { -0.5 };
        let y1 = y0 + 0.625;
        emit_box(
            verts,
            indices,
            [cx - r, cy + y0, cz - r],
            [cx + r, cy + y1, cz + r],
            layers,
            sky,
            block,
            false,
            true,
            [255, 244, 210, 255],
            None::<fn(usize) -> bool>,
        );
        return;
    }

    // Wall-mounted torch: angled ~23 degrees away from the mounted wall face
    // wallmounted: 2=+X, 3=-X, 4=+Z, 5=-Z (direction pointing INTO the wall)
    let yaw_deg = match wall {
        2 => 0.0,    // mounted on +X wall -> leans toward -X
        3 => 180.0,  // mounted on -X wall -> leans toward +X
        4 => 90.0,   // mounted on +Z wall -> leans toward -Z
        5 => -90.0,  // mounted on -Z wall -> leans toward +Z
        _ => 0.0,
    };
    let tilt = 0.42f32; // rad (~24 deg)
    let (ts, tc) = tilt.sin_cos();
    let transform = |lx: f32, ly: f32, lz: f32| -> [f32; 3] {
        // Tilt bottom (+0.5 at wall) upward/inward toward center
        let px = lx * tc - ly * ts + 0.44;
        let py = lx * ts + ly * tc - 0.12;
        let pz = lz;
        let rpt = rotate_xz([px, py, pz], yaw_deg);
        [cx + rpt[0], cy + rpt[1], cz + rpt[2]]
    };

    let x0 = -r;
    let x1 = r;
    let y0 = 0.0;
    let y1 = 0.625;
    let z0 = -r;
    let z1 = r;
    let box_faces = [
        (0usize, [transform(x1, y0, z0), transform(x1, y1, z0), transform(x1, y1, z1), transform(x1, y0, z1)]),
        (1usize, [transform(x0, y0, z1), transform(x0, y1, z1), transform(x0, y1, z0), transform(x0, y0, z0)]),
        (2usize, [transform(x0, y1, z1), transform(x1, y1, z1), transform(x1, y1, z0), transform(x0, y1, z0)]),
        (3usize, [transform(x0, y0, z0), transform(x1, y0, z0), transform(x1, y0, z1), transform(x0, y0, z1)]),
        (4usize, [transform(x1, y0, z1), transform(x1, y1, z1), transform(x0, y1, z1), transform(x0, y0, z1)]),
        (5usize, [transform(x0, y0, z0), transform(x0, y1, z0), transform(x1, y1, z0), transform(x1, y0, z0)]),
    ];
    let uvs = [[0.4375, 1.0], [0.4375, 0.375], [0.5625, 0.375], [0.5625, 1.0]];
    for (f, pts) in box_faces {
        let packed = VoxelVertex::pack(layers[f], f as u8, sky, block, 3, false, true);
        let base = verts.len() as u32;
        for (pos, uv) in pts.into_iter().zip(uvs) {
            verts.push(VoxelVertex {
                pos,
                uv,
                packed,
                tint: [255, 244, 210, 255],
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}


fn emit_signlike(
    verts: &mut Vec<VoxelVertex>,
    indices: &mut Vec<u32>,
    origin: [f32; 3],
    wall: u8,
    visual_scale: f32,
    packed: u32,
    tint: [u8; 4],
) {
    let cx = origin[0] + 0.5;
    let cy = origin[1] + 0.5;
    let cz = origin[2] + 0.5;
    let s = 0.5 * visual_scale;
    let x = 0.5 - 1.0 / 16.0;
    let base_pts = [[x, s, s], [x, s, -s], [x, -s, -s], [x, -s, s]];
    let mut pts = [[0.0f32; 3]; 4];
    for (i, mut p) in base_pts.into_iter().enumerate() {
        p = match wall {
            0 => rotate_xy(p, 90.0),
            1 => rotate_xy(p, -90.0),
            2 => p,
            3 => rotate_xz(p, 180.0),
            4 => rotate_xz(p, 90.0),
            5 => rotate_xz(p, -90.0),
            6 => rotate_xz(rotate_xy(p, 90.0), 90.0),
            _ => rotate_xz(rotate_xy(p, -90.0), -90.0),
        };
        pts[i] = [cx + p[0], cy + p[1], cz + p[2]];
    }
    let uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    for quad in [pts, [pts[3], pts[2], pts[1], pts[0]]] {
        let base = verts.len() as u32;
        for (pos, uv) in quad.into_iter().zip(uvs) {
            verts.push(VoxelVertex { pos, uv, packed, tint });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}


#[allow(clippy::too_many_arguments)]
fn emit_raillike(
    snap: &ChunkSnapshot,
    registries: &EngineRegistries,
    lx: isize,
    ly: isize,
    lz: isize,
    origin: [f32; 3],
    id: u16,
    def: &NodeDef,
    layers: [u16; 6],
    sky: u8,
    block: u8,
    tint: [u8; 4],
    verts: &mut Vec<VoxelVertex>,
    indices: &mut Vec<u32>,
) {
    let my_grp = def.groups.get("connect_to_raillike").copied().unwrap_or(0);
    let is_same_rail = |dx: isize, dy: isize, dz: isize| -> bool {
        let nid = snap.get(lx + dx, ly + dy, lz + dz).0;
        if nid == id {
            return true;
        }
        registries
            .get_node_def(nid)
            .map(|d| {
                d.drawtype == DrawType::Raillike
                    && d.groups.get("connect_to_raillike").copied().unwrap_or(0) == my_grp
            })
            .unwrap_or(false)
    };

    const RAIL_DIRS: [(isize, isize); 4] = [(0, 1), (0, -1), (-1, 0), (1, 0)];
    const SLOPE_ANG: [f32; 4] = [0.0, 180.0, 90.0, -90.0];
    // (tile_slot 0=straight,1=curved,2=junction,3=cross; rotation_deg)
    const RAIL_KINDS: [(usize, f32); 16] = [
        (0, 0.0), (0, 0.0), (0, 0.0), (0, 0.0),
        (0, 90.0), (1, 180.0), (1, 270.0), (2, 180.0),
        (0, 90.0), (1, 90.0), (1, 0.0), (2, 0.0),
        (0, 90.0), (2, 90.0), (2, 270.0), (3, 0.0),
    ];

    let mut code = 0usize;
    let mut sloped = false;
    let mut angle = 0.0f32;
    for (dir, &(dx, dz)) in RAIL_DIRS.iter().enumerate() {
        let rail_above = is_same_rail(dx, 1, dz);
        if rail_above {
            sloped = true;
            angle = SLOPE_ANG[dir];
        }
        if rail_above || is_same_rail(dx, 0, dz) || is_same_rail(dx, -1, dz) {
            code |= 1 << dir;
        }
    }

    let tile_slot = if sloped {
        0
    } else {
        angle = RAIL_KINDS[code].1;
        RAIL_KINDS[code].0
    };
    const RAIL_TO_FACE_IDX: [usize; 4] = [2, 3, 0, 1];
    let layer = layers[RAIL_TO_FACE_IDX[tile_slot.min(3)]];
    let packed = VoxelVertex::pack(layer, 2, sky, block, 3, false, true);

    let cx = origin[0] + 0.5;
    let cy = origin[1] + 0.5;
    let cz = origin[2] + 0.5;
    let off = 1.0 / 64.0;
    let y2 = if sloped { 0.5 } else { -0.5 };
    let mut pts = [
        [-0.5, y2 + off, 0.5],
        [0.5, y2 + off, 0.5],
        [0.5, -0.5 + off, -0.5],
        [-0.5, -0.5 + off, -0.5],
    ];
    for p in &mut pts {
        if angle != 0.0 {
            *p = rotate_xz(*p, angle);
        }
        *p = [cx + p[0], cy + p[1], cz + p[2]];
    }
    let uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    for quad in [pts, [pts[3], pts[2], pts[1], pts[0]]] {
        let base = verts.len() as u32;
        for (pos, uv) in quad.into_iter().zip(uvs) {
            verts.push(VoxelVertex { pos, uv, packed, tint });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}


#[allow(clippy::too_many_arguments)]
fn emit_fencelike(
    snap: &ChunkSnapshot,
    registries: &EngineRegistries,
    lx: isize,
    ly: isize,
    lz: isize,
    origin: [f32; 3],
    layers: [u16; 6],
    sky: u8,
    block: u8,
    tint: [u8; 4],
    verts: &mut Vec<VoxelVertex>,
    indices: &mut Vec<u32>,
) {
    let [bx, by, bz] = origin;
    // Central post [-0.125, -0.5, -0.125] .. [0.125, 0.5, 0.125]
    emit_cuboid_with_facedir(
        verts,
        indices,
        origin,
        [-0.125, -0.5, -0.125, 0.125, 0.5, 0.125],
        layers,
        0,
        sky,
        block,
        false,
        true,
        |_| tint,
        0,
    );

    let connects_fence = |dx: isize, dz: isize| -> bool {
        let nid = snap.get(lx + dx, ly, lz + dz).0;
        registries
            .get_node_def(nid)
            .map(|d| d.drawtype == DrawType::Fencelike || (d.drawtype == DrawType::Normal && d.walkable))
            .unwrap_or(false)
    };

    // +X / -X / +Z / -Z rails
    if connects_fence(1, 0) {
        emit_cuboid_with_facedir(verts, indices, [bx, by, bz], [0.125, 0.1875, -0.0625, 0.5, 0.3125, 0.0625], layers, 0, sky, block, false, true, |_| tint, 0);
        emit_cuboid_with_facedir(verts, indices, [bx, by, bz], [0.125, -0.3125, -0.0625, 0.5, -0.1875, 0.0625], layers, 0, sky, block, false, true, |_| tint, 0);
    }
    if connects_fence(-1, 0) {
        emit_cuboid_with_facedir(verts, indices, [bx, by, bz], [-0.5, 0.1875, -0.0625, -0.125, 0.3125, 0.0625], layers, 0, sky, block, false, true, |_| tint, 0);
        emit_cuboid_with_facedir(verts, indices, [bx, by, bz], [-0.5, -0.3125, -0.0625, -0.125, -0.1875, 0.0625], layers, 0, sky, block, false, true, |_| tint, 0);
    }
    if connects_fence(0, 1) {
        emit_cuboid_with_facedir(verts, indices, [bx, by, bz], [-0.0625, 0.1875, 0.125, 0.0625, 0.3125, 0.5], layers, 0, sky, block, false, true, |_| tint, 0);
        emit_cuboid_with_facedir(verts, indices, [bx, by, bz], [-0.0625, -0.3125, 0.125, 0.0625, -0.1875, 0.5], layers, 0, sky, block, false, true, |_| tint, 0);
    }
    if connects_fence(0, -1) {
        emit_cuboid_with_facedir(verts, indices, [bx, by, bz], [-0.0625, 0.1875, -0.5, 0.0625, 0.3125, -0.125], layers, 0, sky, block, false, true, |_| tint, 0);
        emit_cuboid_with_facedir(verts, indices, [bx, by, bz], [-0.0625, -0.3125, -0.5, 0.0625, -0.1875, -0.125], layers, 0, sky, block, false, true, |_| tint, 0);
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_obj_mesh(
    obj: &ObjMesh,
    origin: [f32; 3],
    def: &NodeDef,
    p2: u8,
    layers: [u16; 6],
    sky: u8,
    block: u8,
    cutout: bool,
    atlas: Option<&TextureAtlas>,
    verts: &mut Vec<VoxelVertex>,
    indices: &mut Vec<u32>,
) {
    const SLOT_TO_FACE: [usize; 6] = [2, 3, 0, 1, 4, 5];
    let facedir = get_node_facedir(def, p2);
    let scale = def.visual_scale;
    let cx = origin[0] + 0.5;
    let cy = origin[1] + 0.5;
    let cz = origin[2] + 0.5;

    for tri in &obj.triangles {
        let slot = tri.material_slot.min(5);
        let face_idx = SLOT_TO_FACE[slot];
        let layer = layers[face_idx];
        let tint = resolve_face_tint(def, face_idx, p2, atlas);

        let n = tri.verts[0].normal;
        let rn = rotate_pt_by_facedir(n, facedir);
        let face_dir = dominant_face_dir(rn);
        let packed = VoxelVertex::pack(layer, face_dir, sky, block, 3, false, cutout);

        let base = verts.len() as u32;
        for v in &tri.verts {
            let sp = [v.pos[0] * scale, v.pos[1] * scale, v.pos[2] * scale];
            let rp = rotate_pt_by_facedir(sp, facedir);
            verts.push(VoxelVertex {
                pos: [cx + rp[0], cy + rp[1], cz + rp[2]],
                uv: v.uv,
                packed,
                tint,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2]);
        // Emit back-side triangle as well since Luanti NDT_MESH disables backface culling (`needsBackfaceCulling() == false`)
        indices.extend_from_slice(&[base, base + 2, base + 1]);
    }
}

#[inline]
fn dominant_face_dir(n: [f32; 3]) -> u8 {
    let ax = n[0].abs();
    let ay = n[1].abs();
    let az = n[2].abs();
    if ay >= ax && ay >= az {
        if n[1] >= 0.0 { 2 } else { 3 }
    } else if ax >= az {
        if n[0] >= 0.0 { 0 } else { 1 }
    } else if n[2] >= 0.0 {
        4
    } else {
        5
    }
}

#[allow(clippy::too_many_arguments)]
pub fn emit_box<F: Fn(usize) -> bool>(
    verts: &mut Vec<VoxelVertex>,
    indices: &mut Vec<u32>,
    min: [f32; 3],
    max: [f32; 3],
    layers: [u16; 6],
    sky: u8,
    block: u8,
    waving: bool,
    cutout: bool,
    tint: [u8; 4],
    face_filter: Option<F>,
) {
    let [x0, y0, z0] = min;
    let [x1, y1, z1] = max;
    let faces = [
        ([x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]),
        ([x0, y0, z1], [x0, y1, z1], [x0, y1, z0], [x0, y0, z0]),
        ([x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]),
        ([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]),
        ([x1, y0, z1], [x1, y1, z1], [x0, y1, z1], [x0, y0, z1]),
        ([x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [x1, y0, z0]),
    ];
    let uvs = [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]];

    for (f, (p0, p1, p2, p3)) in faces.into_iter().enumerate() {
        if let Some(ref pred) = face_filter {
            if !pred(f) {
                continue;
            }
        }
        let packed = VoxelVertex::pack(layers[f], f as u8, sky, block, 3, waving, cutout);
        let base = verts.len() as u32;
        for (pos, uv) in [p0, p1, p2, p3].into_iter().zip(uvs) {
            verts.push(VoxelVertex {
                pos,
                uv,
                packed,
                tint,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{NodeDef, TileSpec};
    use crate::world::{Chunk, WorldStore};

    #[test]
    fn leaves_never_cull_adjacent_solid_faces_and_tint_green() {
        let mut reg = EngineRegistries::new();
        let stone = reg.register_node(NodeDef {
            name: "mcl_core:stone".into(),
            drawtype: DrawType::Normal,
            ..Default::default()
        });
        let leaves = reg.register_node(NodeDef {
            name: "mcl_trees:leaves_oak".into(),
            drawtype: DrawType::AllfacesOptional,
            paramtype: "light".into(),
            ..Default::default()
        });
        assert!(!is_opaque_occluder(leaves, &reg));

        let tmp = std::env::temp_dir().join("cubex_mesher_leaves_test");
        let world = WorldStore::new(&tmp, 1);
        let mut ch = Chunk::new(IVec3::ZERO);
        ch.set(4, 4, 4, stone, 15, 0);
        ch.set(4, 5, 4, leaves, 15, 0);
        world.insert_chunk(ch);

        let snap = world.snapshot_chunk(IVec3::ZERO).unwrap();
        let mut layers = HashMap::new();
        layers.insert(stone, [1; 6]);
        layers.insert(leaves, [2; 6]);
        let mesh = mesh_chunk_greedy(&snap, &reg, &layers, false);

        // Stone has all 6 faces (6*4 = 24 verts) because leaves above +Y do NOT cull it,
        // and leaves has 5 faces (not culled by air, 1 bottom face culled by stone) = 20 verts -> total 44 verts!
        assert_eq!(mesh.opaque_verts.len(), 44);
        // Leaves vertices carry green tint (G > R, G > B), never greyscale or black
        let leaf_vert = mesh.opaque_verts.iter().find(|v| (v.packed & 0x7FF) == 2).unwrap();
        assert!(leaf_vert.tint[1] > leaf_vert.tint[0] && leaf_vert.tint[1] > leaf_vert.tint[2]);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn facedir_and_nodebox_variants_generate_valid_geometry() {
        let mut reg = EngineRegistries::new();
        let log = reg.register_node(NodeDef {
            name: "mcl_trees:tree_oak".into(),
            drawtype: DrawType::Normal,
            paramtype2: "facedir".into(),
            tiles: vec![
                TileSpec { name: "top.png".into(), color: None },
                TileSpec { name: "top.png".into(), color: None },
                TileSpec { name: "side.png".into(), color: None },
            ],
            ..Default::default()
        });
        let slab = reg.register_node(NodeDef {
            name: "mcl_stairs:slab_stone".into(),
            drawtype: DrawType::NodeBox,
            node_box_type: "fixed".into(),
            paramtype2: "facedir".into(),
            node_boxes: vec![[-0.5, -0.5, -0.5, 0.5, 0.0, 0.5]],
            ..Default::default()
        });
        let fence = reg.register_node(NodeDef {
            name: "mcl_fences:oak_fence".into(),
            drawtype: DrawType::NodeBox,
            node_box_type: "connected".into(),
            connects_to: vec!["mcl_fences:oak_fence".into()],
            node_boxes: vec![[-0.125, -0.5, -0.125, 0.125, 0.5, 0.125]],
            connect_right: vec![[0.125, 0.2, -0.06, 0.5, 0.4, 0.06]],
            ..Default::default()
        });

        let tmp = std::env::temp_dir().join("cubex_mesher_nb_test");
        let world = WorldStore::new(&tmp, 2);
        let mut ch = Chunk::new(IVec3::ZERO);
        // Horizontal log (facedir=12 rotates +Y top tile onto +X/-X)
        ch.set(2, 2, 2, log, 15, 12);
        ch.set(5, 2, 2, slab, 15, 0);
        ch.set(8, 2, 2, fence, 15, 0);
        ch.set(9, 2, 2, fence, 15, 0);
        world.insert_chunk(ch);

        let snap = world.snapshot_chunk(IVec3::ZERO).unwrap();
        let mut layers = HashMap::new();
        // [0=+X, 1=-X, 2=+Y(top), 3=-Y(bot), 4=+Z, 5=-Z]
        layers.insert(log, [20, 20, 10, 10, 20, 20]);
        layers.insert(slab, [30; 6]);
        layers.insert(fence, [40; 6]);
        let mesh = mesh_chunk_greedy(&snap, &reg, &layers, false);

        // +X face of horizontal log (facedir=12) must receive layer 10 (top ring)
        let log_plus_x = mesh
            .opaque_verts
            .iter()
            .any(|v| (v.packed & 0x7FF) == 10 && ((v.packed >> 11) & 7) == 0);
        assert!(log_plus_x, "Horizontal log +X face should use top ring tile");

        // Slab (24 verts) + 2 fences (post=24 + connected bar=24, plus post=24) -> > 80 verts
        assert!(mesh.opaque_verts.len() >= 96);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn precolored_leaves_avoid_double_darkening_and_greyscale_uses_palette() {
        use image::{Rgba, RgbaImage};
        use std::path::Path;

        let mut reg = EngineRegistries::new();
        let precolored = reg.register_node(NodeDef {
            name: "mcl_trees:leaves_oak".into(),
            drawtype: DrawType::AllfacesOptional,
            paramtype: "light".into(),
            palette: Some("mcl_core_palette_leaves.png".into()),
            tiles: vec![TileSpec {
                name: "cr_leaves.png".into(),
                color: None,
            }],
            ..Default::default()
        });
        let greyscale = reg.register_node(NodeDef {
            name: "mcl_trees:leaves_dark_oak".into(),
            drawtype: DrawType::AllfacesOptional,
            paramtype: "light".into(),
            palette: Some("mcl_core_palette_leaves.png".into()),
            tiles: vec![TileSpec {
                name: "mcl_leaves.png".into(),
                color: None,
            }],
            ..Default::default()
        });

        let tmp = std::env::temp_dir().join("cubex_mesher_leaf_tint_test");
        let tex_dir = tmp.join("textures");
        let _ = std::fs::create_dir_all(&tex_dir);
        RgbaImage::from_pixel(16, 16, Rgba([35, 53, 22, 255]))
            .save(tex_dir.join("cr_leaves.png"))
            .unwrap();
        RgbaImage::from_pixel(16, 16, Rgba([130, 128, 124, 255]))
            .save(tex_dir.join("mcl_leaves.png"))
            .unwrap();
        let mut pal = RgbaImage::from_pixel(16, 16, Rgba([119, 171, 47, 255]));
        // Mirror Mineclonia's 32-entry band repetition in mcl_core_palette_leaves.png (idx 13 & idx 64|13=77)
        pal.put_pixel(13, 0, Rgba([89, 174, 48, 255]));
        pal.put_pixel(13, 4, Rgba([89, 174, 48, 255]));
        pal.save(tex_dir.join("mcl_core_palette_leaves.png")).unwrap();

        let atlas = TextureAtlas::build(&HashMap::new(), &tmp, None, &reg);
        let world = WorldStore::new(tmp.join("world"), 7);
        let mut ch = Chunk::new(IVec3::ZERO);
        // Set dim canopy sky = 3 to verify ambient daylight floor lifts sunlit canopy leaves
        ch.set(4, 4, 4, precolored, 3, 64);
        ch.set(8, 4, 8, greyscale, 3, 64 | 13);
        world.insert_chunk(ch);

        let snap = world.snapshot_chunk(IVec3::ZERO).unwrap();
        let mesh = mesh_chunk_with_atlas(&snap, &reg, &atlas, false);

        let pre_layer = atlas.node_face_layers[&precolored][0] as u32;
        let pre_vert = mesh
            .opaque_verts
            .iter()
            .find(|v| (v.packed & 0x7FF) == pre_layer)
            .unwrap();
        // Pre-colored Craft&Ruin-style leaves must NOT be double-darkened by palette tint
        assert_eq!(pre_vert.tint, [255, 255, 255, 255]);
        assert!(((pre_vert.packed >> 14) & 0xF) >= 13);

        let grey_layer = atlas.node_face_layers[&greyscale][0] as u32;
        let grey_vert = mesh
            .opaque_verts
            .iter()
            .find(|v| (v.packed & 0x7FF) == grey_layer)
            .unwrap();
        // Greyscale Mineclonia leaves get vibrant green palette tint (idx 13 -> [89, 174, 48, 255])
        assert_eq!(grey_vert.tint, [89, 174, 48, 255]);
        assert!(((grey_vert.packed >> 14) & 0xF) >= 13);

        let _ = Path::new(&tmp);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn half_texel_uv_inset_prevents_snow_seams() {
        let mut reg = EngineRegistries::new();
        let snow = reg.register_node(NodeDef {
            name: "mcl_core:snowblock".into(),
            drawtype: DrawType::Normal,
            ..Default::default()
        });
        let water = reg.register_node(NodeDef {
            name: "mcl_core:water_source".into(),
            drawtype: DrawType::Liquid,
            waving: 3,
            ..Default::default()
        });

        let tmp = std::env::temp_dir().join("cubex_mesher_snow_uv_inset_test");
        let _ = std::fs::remove_dir_all(&tmp);
        let world = WorldStore::new(tmp.join("world"), 11);
        let mut ch = Chunk::new(IVec3::ZERO);
        ch.set(2, 4, 2, snow, 15, 0);
        // 3x2 snow plateau merged via greedy meshing
        for z in 8..10 {
            for x in 8..11 {
                ch.set(x, 4, z, snow, 15, 0);
            }
        }
        ch.set(5, 4, 5, water, 15, 0);
        world.insert_chunk(ch);

        let snap = world.snapshot_chunk(IVec3::ZERO).unwrap();
        let mut layers = HashMap::new();
        layers.insert(snow, [12; 6]);
        layers.insert(water, [25; 6]);
        let mesh = mesh_chunk_greedy(&snap, &reg, &layers, false);

        assert!(!mesh.opaque_verts.is_empty());
        for v in &mesh.opaque_verts {
            // Every quad UV must be strictly inset away from exact integers at tile borders
            assert!(v.uv[0] >= GREEDY_UV_EDGE_EPS);
            assert!(v.uv[1] >= GREEDY_UV_EDGE_EPS);
            let frac_u = v.uv[0].fract();
            let frac_v = v.uv[1].fract();
            assert!(
                (frac_u - ATLAS_HALF_TEXEL_INSET).abs() < 1e-5
                    || (frac_u - (1.0 - ATLAS_HALF_TEXEL_INSET)).abs() < 1e-5
                    || (frac_u - GREEDY_UV_EDGE_EPS).abs() < 1e-5
                    || (frac_u - (1.0 - GREEDY_UV_EDGE_EPS)).abs() < 1e-5,
                "Unexpected un-inset opaque U: {}",
                v.uv[0]
            );
            assert!(
                (frac_v - ATLAS_HALF_TEXEL_INSET).abs() < 1e-5
                    || (frac_v - (1.0 - ATLAS_HALF_TEXEL_INSET)).abs() < 1e-5
                    || (frac_v - GREEDY_UV_EDGE_EPS).abs() < 1e-5
                    || (frac_v - (1.0 - GREEDY_UV_EDGE_EPS)).abs() < 1e-5,
                "Unexpected un-inset opaque V: {}",
                v.uv[1]
            );
        }

        assert!(!mesh.trans_verts.is_empty());
        for v in &mesh.trans_verts {
            assert!(
                v.uv[0] >= ATLAS_HALF_TEXEL_INSET - 1e-6
                    && v.uv[0] <= 1.0 - ATLAS_HALF_TEXEL_INSET + 1e-6,
                "Water U {} out of half-texel bounds",
                v.uv[0]
            );
            assert!(
                v.uv[1] >= ATLAS_HALF_TEXEL_INSET - 1e-6
                    && v.uv[1] <= 1.0 - ATLAS_HALF_TEXEL_INSET + 1e-6,
                "Water V {} out of half-texel bounds",
                v.uv[1]
            );
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn per_vertex_voxel_ao_darkens_concave_corners() {
        let mut reg = EngineRegistries::new();
        let stone = reg.register_node(NodeDef {
            name: "mcl_core:stone".into(),
            drawtype: DrawType::Normal,
            ..Default::default()
        });
        let tmp = std::env::temp_dir().join("cubex_mesher_per_vertex_ao_test");
        let _ = std::fs::remove_dir_all(&tmp);
        let world = WorldStore::new(tmp.join("world"), 99);
        let mut ch = Chunk::new(IVec3::ZERO);
        for z in 2..14 {
            for x in 2..14 {
                ch.set(x, 1, z, stone, 15, 0);
            }
        }
        // Raised step/wall at x=5, y=2, z=4..9 creates concave floor corners along x=6 and x=4
        for z in 4..=9 {
            ch.set(5, 2, z, stone, 15, 0);
        }
        world.insert_chunk(ch);

        let snap = world.snapshot_chunk(IVec3::ZERO).unwrap();
        let mut layers = HashMap::new();
        layers.insert(stone, [7; 6]);
        let mesh = mesh_chunk_greedy(&snap, &reg, &layers, true);

        // Floor block (6, 1, 6) +Y top face sits at y=2.0 in x=[6..7], z=[6..7],
        // with step wall blocking -X side -> corners at x=6 must have AO <= 2 while corners at x=7 stay AO == 3.
        let floor_quad = mesh
            .opaque_verts
            .chunks_exact(4)
            .find(|q| {
                let cx = 0.25 * (q[0].pos[0] + q[1].pos[0] + q[2].pos[0] + q[3].pos[0]);
                let cy = 0.25 * (q[0].pos[1] + q[1].pos[1] + q[2].pos[1] + q[3].pos[1]);
                let cz = 0.25 * (q[0].pos[2] + q[1].pos[2] + q[2].pos[2] + q[3].pos[2]);
                ((q[0].packed >> 11) & 7) == 2
                    && (cx - 6.5).abs() < 0.05
                    && (cy - 2.0).abs() < 0.01
                    && (cz - 6.5).abs() < 0.05
            })
            .expect("step-adjacent floor tile must emit per-vertex AO quad");
        let aos: Vec<u32> = floor_quad
            .iter()
            .map(|v| (v.packed >> 22) & 0x3)
            .collect();
        let min_ao = *aos.iter().min().unwrap();
        let max_ao = *aos.iter().max().unwrap();
        assert!(
            min_ao < 3 && max_ao == 3,
            "expected darker wall-adjacent corners and unoccluded outer corners, got {aos:?}"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
