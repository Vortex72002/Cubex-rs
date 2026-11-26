use glam::{Mat4, Vec3};
use image::{Rgba, RgbaImage};
use std::f32::consts::FRAC_PI_2;

use crate::lua_api::userdata::ObjectState;
use crate::registry::{EngineRegistries, CONTENT_AIR};
use crate::render::mesher::{inset_unit_uvs, VoxelVertex};
use crate::textures::TextureAtlas;
use crate::world::WorldStore;

#[derive(Debug, Clone)]
pub struct EntityRenderSnapshot {
    pub id: u64,
    pub entity_name: String,
    pub pos: Vec3,
    pub yaw: f32,
    pub collisionbox: [f32; 6],
    pub visual: String,
    pub visual_size: Vec3,
    pub mesh: String,
    pub texture_name: String,
    pub wield_item: String,
    pub walk_phase: f32,
    pub walk_blend: f32,
    pub anim_time: f32,
    pub vel_y: f32,
    pub hurt_flash: f32,
    pub sky_light: u8,
    pub block_light: u8,
}

impl Default for EntityRenderSnapshot {
    fn default() -> Self {
        Self {
            id: 0,
            entity_name: String::new(),
            pos: Vec3::ZERO,
            yaw: 0.0,
            collisionbox: [-0.4, 0.0, -0.4, 0.4, 1.8, 0.4],
            visual: "mesh".into(),
            visual_size: Vec3::ONE,
            mesh: String::new(),
            texture_name: String::new(),
            wield_item: String::new(),
            walk_phase: 0.0,
            walk_blend: 0.0,
            anim_time: 0.0,
            vel_y: 0.0,
            hurt_flash: 0.0,
            sky_light: 15,
            block_light: 0,
        }
    }
}

impl EntityRenderSnapshot {
    pub fn from_object_state(g: &ObjectState, world: &WorldStore) -> Self {
        let mid_y = g.pos.y + (g.collisionbox[1] + g.collisionbox[4]) * 0.5 + 0.15;
        let sample_pos = Vec3::new(g.pos.x, mid_y, g.pos.z).floor().as_ivec3();
        let (_, p1, _) = world.get_node(sample_pos);
        let is_blank_tex = |t: &str| {
            let clean = t.trim();
            clean.is_empty()
                || clean == "blank.png"
                || clean == "empty.png"
                || clean == "invisible.png"
                || clean == "mobs_mc_empty.png"
                || clean.starts_with("blank.png^")
        };
        let texture_name = if g.textures.len() > 1
            && g.textures[0].starts_with("mobs_mc_sheep_fur.png")
            && !is_blank_tex(&g.textures[1])
        {
            g.textures[1].clone()
        } else {
            g.textures
                .iter()
                .find(|t| !is_blank_tex(t))
                .cloned()
                .unwrap_or_default()
        };
        Self {
            id: g.id,
            entity_name: g.entity_name.clone(),
            pos: g.pos,
            yaw: g.yaw,
            collisionbox: g.collisionbox,
            visual: g.visual.clone(),
            visual_size: g.visual_size,
            mesh: g.mesh.clone(),
            texture_name,
            wield_item: g.wield_item.clone(),
            walk_phase: g.walk_phase,
            walk_blend: g.walk_blend,
            anim_time: g.walk_phase,
            vel_y: g.vel.y,
            hurt_flash: g.hurt_flash,
            sky_light: (p1 & 0x0F).max(6),
            block_light: (p1 >> 4) & 0x0F,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityRigKind {
    Biped,
    ZombiePiglin,
    WitherSkeleton,
    Blaze,
    Ghast,
    MagmaCube,
    Guardian,
    Fireball,
    Cow,
    Pig,
    Sheep,
    Chicken,
    Wolf,
    Rabbit,
    Llama,
    Creeper,
    Spider,
    SmallCube,
    FallingNode,
    DroppedItem,
}

pub fn classify_entity_rig(ent: &EntityRenderSnapshot) -> EntityRigKind {
    if ent.entity_name == "__builtin:falling_node"
        || (ent.entity_name != "__builtin:item"
            && matches!(ent.visual.as_str(), "wielditem" | "item" | "cube")
            && (ent.texture_name.contains(':') || ent.wield_item.contains(':')))
    {
        return EntityRigKind::FallingNode;
    }
    if ent.entity_name == "__builtin:item" || ent.visual == "wielditem" || ent.visual == "item" {
        return EntityRigKind::DroppedItem;
    }
    let lower_name = ent.entity_name.to_ascii_lowercase();
    let lower_mesh = ent.mesh.to_ascii_lowercase();
    if lower_name.contains("fireball") {
        return EntityRigKind::Fireball;
    }
    if lower_name.contains("pigman")
        || lower_name.contains("zombiepig")
        || lower_name.contains("zombified_piglin")
        || lower_name.contains("piglin")
    {
        return EntityRigKind::ZombiePiglin;
    }
    if lower_name.contains("witherskeleton") || lower_name.contains("wither_skeleton") {
        return EntityRigKind::WitherSkeleton;
    }
    if lower_name.contains("blaze") || lower_mesh.contains("blaze") {
        return EntityRigKind::Blaze;
    }
    if lower_name.contains("ghast") || lower_mesh.contains("ghast") {
        return EntityRigKind::Ghast;
    }
    if lower_name.contains("magma") || lower_mesh.contains("magma") {
        return EntityRigKind::MagmaCube;
    }
    if lower_name.contains("guardian") || lower_mesh.contains("guardian") {
        return EntityRigKind::Guardian;
    }
    if lower_name.contains("creeper") || lower_mesh.contains("creeper") {
        return EntityRigKind::Creeper;
    }
    if lower_name.contains("spider") || lower_mesh.contains("spider") {
        return EntityRigKind::Spider;
    }
    if lower_name.contains("cow") || lower_name.contains("mooshroom") || lower_mesh.contains("cow") {
        return EntityRigKind::Cow;
    }
    if lower_name.contains("pig") || lower_mesh.contains("pig") {
        return EntityRigKind::Pig;
    }
    if lower_name.contains("sheep") || lower_mesh.contains("sheep") {
        return EntityRigKind::Sheep;
    }
    if lower_name.contains("chicken") || lower_mesh.contains("chicken") {
        return EntityRigKind::Chicken;
    }
    if lower_name.contains("wolf") || lower_name.contains("fox") || lower_name.contains("dog") || lower_mesh.contains("wolf") {
        return EntityRigKind::Wolf;
    }
    if lower_name.contains("rabbit") || lower_mesh.contains("rabbit") {
        return EntityRigKind::Rabbit;
    }
    if lower_name.contains("llama")
        || lower_name.contains("horse")
        || lower_name.contains("donkey")
        || lower_name.contains("mule")
        || lower_mesh.contains("llama")
    {
        return EntityRigKind::Llama;
    }
    if lower_name.contains("slime")
        || lower_name.contains("bee")
        || lower_name.contains("bat")
        || lower_name.contains("silverfish")
        || lower_name.contains("endermite")
    {
        return EntityRigKind::SmallCube;
    }
    let h = (ent.collisionbox[4] - ent.collisionbox[1]).abs();
    let w = (ent.collisionbox[3] - ent.collisionbox[0])
        .abs()
        .max((ent.collisionbox[5] - ent.collisionbox[2]).abs());
    if h >= 1.25
        || lower_name.contains("zombie")
        || lower_name.contains("husk")
        || lower_name.contains("drowned")
        || lower_name.contains("skeleton")
        || lower_name.contains("stray")
        || lower_name.contains("villager")
        || lower_name.contains("pillager")
        || lower_name.contains("witch")
        || lower_name.contains("enderman")
    {
        EntityRigKind::Biped
    } else if w > h * 1.15 && h >= 0.45 {
        EntityRigKind::Cow
    } else if ent.visual == "cube" || h < 0.9 {
        EntityRigKind::SmallCube
    } else {
        EntityRigKind::Biped
    }
}

/// Canonical base PNG texture filename for Mineclonia `mobs_mc` species if Lua properties omitted one.
pub fn canonical_mob_texture_for_entity(entity_name: &str) -> Option<&'static str> {
    let l = entity_name.to_ascii_lowercase();
    if l.contains("pigman")
        || l.contains("zombiepig")
        || l.contains("zombified_piglin")
        || l.contains("piglin")
    {
        Some("extra_mobs_zombified_piglin.png")
    } else if l.contains("witherskeleton") || l.contains("wither_skeleton") {
        Some("mobs_mc_wither_skeleton.png")
    } else if l.contains("blaze") {
        Some("mobs_mc_blaze.png")
    } else if l.contains("ghast") {
        Some("mobs_mc_ghast.png")
    } else if l.contains("magma") {
        Some("mobs_mc_magmacube.png")
    } else if l.contains("guardian") {
        Some("mobs_mc_guardian.png")
    } else if l.contains("mooshroom") {
        Some("mobs_mc_mooshroom.png")
    } else if l.contains("cow") {
        Some("mobs_mc_cow.png")
    } else if l.contains("pig") {
        Some("mobs_mc_pig.png")
    } else if l.contains("sheep") {
        Some("mobs_mc_sheep.png")
    } else if l.contains("chicken") {
        Some("mobs_mc_chicken.png")
    } else if l.contains("wolf") {
        Some("mobs_mc_wolf.png")
    } else if l.contains("rabbit") {
        Some("mobs_mc_rabbit_brown.png")
    } else if l.contains("llama") {
        Some("mobs_mc_llama_brown.png")
    } else if l.contains("horse") {
        Some("mobs_mc_horse_brown.png")
    } else if l.contains("husk") {
        Some("mobs_mc_husk.png")
    } else if l.contains("zombie") {
        Some("mobs_mc_zombie.png")
    } else if l.contains("skeleton") {
        Some("mobs_mc_skeleton.png")
    } else if l.contains("creeper") {
        Some("mobs_mc_creeper.png")
    } else if l.contains("cave_spider") {
        Some("mobs_mc_cave_spider.png")
    } else if l.contains("spider") {
        Some("mobs_mc_spider.png")
    } else if l.contains("slime") {
        Some("mobs_mc_slime.png")
    } else if l.contains("enderman") {
        Some("mobs_mc_enderman.png")
    } else {
        None
    }
}

/// Replace near-transparent background pixels on mob skins with solid average RGB so
/// zero transparent holes can ever appear on solid mob boxes.
pub fn fill_transparent_mob_pixels(img: &mut RgbaImage) {
    let mut sum_r: u64 = 0;
    let mut sum_g: u64 = 0;
    let mut sum_b: u64 = 0;
    let mut cnt: u64 = 0;
    for px in img.pixels() {
        if px.0[3] >= 128 {
            sum_r += px.0[0] as u64;
            sum_g += px.0[1] as u64;
            sum_b += px.0[2] as u64;
            cnt += 1;
        }
    }
    let fill = if cnt > 0 {
        Rgba([
            (sum_r / cnt).min(255) as u8,
            (sum_g / cnt).min(255) as u8,
            (sum_b / cnt).min(255) as u8,
            255,
        ])
    } else {
        Rgba([142, 134, 122, 255])
    };
    for px in img.pixels_mut() {
        if px.0[3] < 96 {
            *px = fill;
        }
    }
}

#[inline]
pub fn hurt_tint_for_flash(hurt_flash: f32) -> [u8; 4] {
    if hurt_flash <= 0.001 {
        return [255, 255, 255, 255];
    }
    let f = (hurt_flash / 0.25).clamp(0.0, 1.0);
    let gb = (255.0 * (1.0 - 0.78 * f)).round().clamp(0.0, 255.0) as u8;
    [255, gb, gb, 255]
}

#[inline]
fn dominant_face_dir(n: Vec3) -> u8 {
    let ax = n.x.abs();
    let ay = n.y.abs();
    let az = n.z.abs();
    if ax >= ay && ax >= az {
        if n.x >= 0.0 { 0 } else { 1 }
    } else if ay >= ax && ay >= az {
        if n.y >= 0.0 { 2 } else { 3 }
    } else if n.z >= 0.0 {
        4
    } else {
        5
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_box(
    verts: &mut Vec<VoxelVertex>,
    indices: &mut Vec<u32>,
    transform: Mat4,
    min: Vec3,
    max: Vec3,
    uvs_per_face: [[f32; 4]; 6],
    sky: u8,
    block: u8,
    tint: [u8; 4],
    no_alpha_discard: bool,
) {
    let [x0, y0, z0] = min.to_array();
    let [x1, y1, z1] = max.to_array();
    let faces: [(Vec3, [[f32; 3]; 4]); 6] = [
        (Vec3::X, [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]]),
        (Vec3::NEG_X, [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]]),
        (Vec3::Y, [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]]),
        (Vec3::NEG_Y, [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]),
        (Vec3::Z, [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]),
        (Vec3::NEG_Z, [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]]),
    ];

    for (fidx, (local_n, pts)) in faces.into_iter().enumerate() {
        let [u0, v0, u1, v1] = uvs_per_face[fidx];
        let uvs = [[u0, v1], [u1, v1], [u1, v0], [u0, v0]];
        let world_n = transform.transform_vector3(local_n).normalize_or_zero();
        let fdir = dominant_face_dir(world_n);
        let mut packed = VoxelVertex::pack(0, fdir, sky, block, 3, false, false);
        if no_alpha_discard {
            packed |= 1u32 << 27;
        }
        let base = verts.len() as u32;
        for (p, uv) in pts.into_iter().zip(uvs) {
            let wp = transform.transform_point3(Vec3::from_array(p));
            verts.push(VoxelVertex {
                pos: wp.to_array(),
                uv,
                packed,
                tint,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

fn net_box_uvs(ox: f32, oy: f32, w: f32, h: f32, d: f32, tw: f32, th: f32) -> [[f32; 4]; 6] {
    let rect = |x0: f32, y0: f32, x1: f32, y1: f32| -> [f32; 4] {
        [
            (x0 / tw).clamp(0.0, 1.0),
            (y0 / th).clamp(0.0, 1.0),
            (x1 / tw).clamp(0.0, 1.0),
            (y1 / th).clamp(0.0, 1.0),
        ]
    };
    let uv_right = rect(ox, oy + d, ox + d, oy + d + h);
    let uv_front = rect(ox + d, oy + d, ox + d + w, oy + d + h);
    let uv_left = rect(ox + d + w, oy + d, ox + 2.0 * d + w, oy + d + h);
    let uv_back = rect(ox + 2.0 * d + w, oy + d, ox + 2.0 * d + 2.0 * w, oy + d + h);
    let uv_top = rect(ox + d, oy, ox + d + w, oy + d);
    let uv_bot = rect(ox + d + w, oy, ox + d + 2.0 * w, oy + d);
    [uv_left, uv_right, uv_top, uv_bot, uv_front, uv_back]
}

fn full_tile_uvs() -> [[f32; 4]; 6] {
    [[0.0, 0.0, 1.0, 1.0]; 6]
}

/// Resolve `[+X, -X, +Y, -Y, +Z, -Z]` terrain atlas layer indices and tints when an entity
/// (`__builtin:falling_node` or block `wielditem`/`cube`) references a registered node ID.
pub fn resolve_entity_block_layers(
    ent: &EntityRenderSnapshot,
    registries: &EngineRegistries,
    atlas: &TextureAtlas,
) -> Option<([u16; 6], [[u8; 4]; 6])> {
    let raw = if !ent.texture_name.is_empty() {
        ent.texture_name.as_str()
    } else {
        ent.wield_item.as_str()
    };
    let token = raw.split_whitespace().next().unwrap_or("").trim();
    if token.is_empty() || token.ends_with(".png") {
        return None;
    }
    let resolved = registries.resolve_alias(token);
    let cid = registries.get_content_id(resolved);
    if cid == CONTENT_AIR {
        return None;
    }
    let layers = *atlas.node_face_layers.get(&cid)?;
    let hurt = hurt_tint_for_flash(ent.hurt_flash);
    const FACE_TO_LUANTI: [usize; 6] = [2, 3, 0, 1, 4, 5];
    let mut tints = [hurt; 6];
    if let Some(def) = registries.get_node_def(cid) {
        if !def.tiles.is_empty() {
            for (f, t_out) in tints.iter_mut().enumerate() {
                let lt = FACE_TO_LUANTI[f].min(def.tiles.len() - 1);
                if let Some(tc) = def.tiles[lt].color {
                    *t_out = [
                        ((tc[0] as u16 * hurt[0] as u16) / 255) as u8,
                        ((tc[1] as u16 * hurt[1] as u16) / 255) as u8,
                        ((tc[2] as u16 * hurt[2] as u16) / 255) as u8,
                        tc[3],
                    ];
                }
            }
        }
    }
    Some((layers, tints))
}

/// Build a full 1x1x1 world-scale falling block cube textured with the terrain atlas layers
/// (`[+X, -X, +Y, -Y, +Z, -Z]`) so falling sand, gravel, and concrete powder match real blocks.
pub fn build_falling_node_mesh(
    ent: &EntityRenderSnapshot,
    face_layers: [u16; 6],
    face_tints: [[u8; 4]; 6],
) -> (Vec<VoxelVertex>, Vec<u32>) {
    let mut verts = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    let transform = Mat4::from_translation(ent.pos) * Mat4::from_rotation_y(-ent.yaw);
    let [x0, y0, z0] = [-0.5f32; 3];
    let [x1, y1, z1] = [0.5f32; 3];
    let faces: [(Vec3, [[f32; 3]; 4]); 6] = [
        (Vec3::X, [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]]),
        (Vec3::NEG_X, [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]]),
        (Vec3::Y, [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]]),
        (Vec3::NEG_Y, [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]),
        (Vec3::Z, [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]),
        (Vec3::NEG_Z, [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]]),
    ];
    let unit_uvs = inset_unit_uvs([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]);
    for (fidx, (local_n, pts)) in faces.into_iter().enumerate() {
        let world_n = transform.transform_vector3(local_n).normalize_or_zero();
        let fdir = dominant_face_dir(world_n);
        let mut packed = VoxelVertex::pack(
            face_layers[fidx],
            fdir,
            ent.sky_light,
            ent.block_light,
            3,
            false,
            false,
        );
        if ent.hurt_flash > 0.01 {
            packed |= 1u32 << 26;
        }
        let base = verts.len() as u32;
        for (p, uv) in pts.into_iter().zip(unit_uvs) {
            let wp = transform.transform_point3(Vec3::from_array(p));
            verts.push(VoxelVertex {
                pos: wp.to_array(),
                uv,
                packed,
                tint: face_tints[fidx],
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (verts, indices)
}

/// Build a 3D spinning, bobbing compound dropped-item mesh for `cubex:ram_mic`.
pub fn build_dropped_ram_mic_mesh(ent: &EntityRenderSnapshot) -> (Vec<VoxelVertex>, Vec<u32>) {
    let vm = crate::render::viewmodel::create_ram_mic_mesh();
    let phase = (ent.id as f32 * 0.7) + ent.walk_phase;
    let bob = 0.28 + phase.sin() * 0.055;
    let world_mat = Mat4::from_translation(ent.pos + Vec3::new(0.0, bob, 0.0))
        * Mat4::from_rotation_y(-ent.yaw + ent.walk_phase * 0.55)
        * glam::Mat4::from_rotation_z(0.18)
        * Mat4::from_scale(Vec3::splat(0.85));

    let hurt = hurt_tint_for_flash(ent.hurt_flash);
    let mut verts = Vec::with_capacity(vm.verts.len());
    for v in &vm.verts {
        let wp = world_mat.transform_point3(Vec3::from_array(v.pos));
        let old_fdir = ((v.packed >> 11) & 0x7) as u8;
        let local_n = match old_fdir {
            0 => Vec3::X,
            1 => Vec3::NEG_X,
            2 => Vec3::Y,
            3 => Vec3::NEG_Y,
            4 => Vec3::Z,
            _ => Vec3::NEG_Z,
        };
        let world_n = world_mat.transform_vector3(local_n).normalize_or_zero();
        let new_fdir = dominant_face_dir(world_n);
        let emissive = (v.packed & VoxelVertex::FLAG_EMISSIVE) != 0;
        let mut packed = VoxelVertex::pack(
            0,
            new_fdir,
            ent.sky_light,
            if emissive { 15 } else { ent.block_light },
            3,
            false,
            false,
        );
        if emissive {
            packed |= VoxelVertex::FLAG_EMISSIVE;
        }
        if ent.hurt_flash > 0.01 {
            packed |= 1u32 << 26;
        }
        verts.push(VoxelVertex {
            pos: wp.to_array(),
            uv: v.uv,
            packed,
            tint: [
                ((v.tint[0] as u16 * hurt[0] as u16) / 255) as u8,
                ((v.tint[1] as u16 * hurt[1] as u16) / 255) as u8,
                ((v.tint[2] as u16 * hurt[2] as u16) / 255) as u8,
                v.tint[3],
            ],
        });
    }
    (verts, vm.indices)
}

/// Build an articulated, yaw-oriented, walk-animated voxel rig mesh in world coordinates for an entity.
pub fn build_entity_rig_mesh(
    ent: &EntityRenderSnapshot,
    tex_w: u32,
    tex_h: u32,
) -> (Vec<VoxelVertex>, Vec<u32>) {
    let mut verts = Vec::new();
    let mut indices = Vec::new();

    let kind = classify_entity_rig(ent);
    let tint = hurt_tint_for_flash(ent.hurt_flash);
    let sky = ent.sky_light;
    let block = ent.block_light;

    let min_y = ent.collisionbox[1];
    let box_h = (ent.collisionbox[4] - ent.collisionbox[1]).abs().clamp(0.35, 3.6);
    let rel_vs_y = if (ent.visual_size.y - 1.0).abs() < 2.5 {
        ent.visual_size.y.clamp(0.35, 2.5)
    } else {
        1.0
    };
    // Luanti yaw = 0 faces +Z and positive yaw turns CCW (-sin(yaw), 0, cos(yaw)),
    // which matches Mat4::from_rotation_y(-yaw) * Vec3::Z.
    let root = Mat4::from_translation(ent.pos) * Mat4::from_rotation_y(-ent.yaw);
    let base = root * Mat4::from_translation(Vec3::new(0.0, min_y, 0.0));
    let swing = ent.walk_phase.sin() * 0.65 * ent.walk_blend.clamp(0.0, 1.15);

    let has_sheet = tex_w >= 64 && tex_h >= 32;
    let tw = tex_w.max(1) as f32;
    let th = tex_h.max(1) as f32;
    let tile_uv = full_tile_uvs();
    let solid = kind != EntityRigKind::DroppedItem;
    let sc64 = if has_sheet { tw / 64.0 } else { 1.0 };
    let sc64_h = if has_sheet {
        if tex_h >= 64 && tex_w >= 128 { th / 64.0 } else { sc64 }
    } else {
        1.0
    };
    let uv64 = |ox: f32, oy: f32, w: f32, h: f32, d: f32| -> [[f32; 4]; 6] {
        if has_sheet {
            net_box_uvs(ox * sc64, oy * sc64_h, w * sc64, h * sc64_h, d * sc64, tw, th)
        } else {
            tile_uv
        }
    };
    let uv128 = |ox: f32, oy: f32, w: f32, h: f32, d: f32| -> [[f32; 4]; 6] {
        if tex_w >= 128 && tex_h >= 64 {
            let sx = tw / 128.0;
            let sy = th / 64.0;
            net_box_uvs(ox * sx, oy * sy, w * sx, h * sy, d * sx, tw, th)
        } else if has_sheet {
            net_box_uvs(
                ox * 0.5 * sc64,
                oy * 0.5 * sc64_h,
                w * 0.5 * sc64,
                h * 0.5 * sc64_h,
                d * 0.5 * sc64,
                tw,
                th,
            )
        } else {
            tile_uv
        }
    };

    let emit_four_legs = |verts: &mut Vec<VoxelVertex>,
                          indices: &mut Vec<u32>,
                          leg_uv: [[f32; 4]; 6],
                          lw: f32,
                          lh: f32,
                          ld: f32,
                          x_off: f32,
                          z_f: f32,
                          z_b: f32| {
        for &(lx, lz, pm) in &[
            (-x_off, z_f, 1.0f32),
            (x_off, z_f, -1.0f32),
            (-x_off, z_b, -1.0f32),
            (x_off, z_b, 1.0f32),
        ] {
            let leg = base
                * Mat4::from_translation(Vec3::new(lx, lh, lz))
                * Mat4::from_rotation_x(swing * pm);
            emit_box(
                verts,
                indices,
                leg,
                Vec3::new(-lw * 0.5, -lh, -ld * 0.5),
                Vec3::new(lw * 0.5, 0.0, ld * 0.5),
                leg_uv,
                sky,
                block,
                tint,
                true,
            );
        }
    };

    match kind {
        EntityRigKind::FallingNode => {
            let transform = Mat4::from_translation(ent.pos) * Mat4::from_rotation_y(-ent.yaw);
            emit_box(
                &mut verts,
                &mut indices,
                transform,
                Vec3::splat(-0.5),
                Vec3::splat(0.5),
                tile_uv,
                sky,
                block,
                tint,
                false,
            );
        }
        EntityRigKind::DroppedItem => {
            let phase = (ent.id as f32 * 0.7) + ent.walk_phase;
            let bob = 0.16 + phase.sin() * 0.05;
            let spin = Mat4::from_translation(ent.pos + Vec3::new(0.0, bob, 0.0))
                * Mat4::from_rotation_y(-ent.yaw + ent.walk_phase * 0.5);
            let half = 0.15;
            emit_box(
                &mut verts,
                &mut indices,
                spin,
                Vec3::splat(-half),
                Vec3::splat(half),
                tile_uv,
                sky,
                block,
                tint,
                false,
            );
        }
        EntityRigKind::Biped
        | EntityRigKind::ZombiePiglin
        | EntityRigKind::WitherSkeleton => {
            let is_wither = kind == EntityRigKind::WitherSkeleton;
            let is_piglin = kind == EntityRigKind::ZombiePiglin;
            let is_skel = is_wither
                || ent.entity_name.to_ascii_lowercase().contains("skeleton")
                || ent.entity_name.to_ascii_lowercase().contains("stray");
            let is_zombie = is_piglin
                || ent.entity_name.to_ascii_lowercase().contains("zombie")
                || ent.entity_name.to_ascii_lowercase().contains("husk")
                || ent.entity_name.to_ascii_lowercase().contains("drowned");
            let s = if is_wither {
                (box_h / 1.9).max(1.22) * rel_vs_y
            } else {
                (box_h / 1.9) * rel_vs_y
            };
            let p = 0.059 * s;
            let leg_top = 12.0 * p;
            let neck_y = 24.0 * p;

            let limb_w = if is_skel { 2.0 } else { 4.0 };
            let head_uv = uv64(0.0, 0.0, 8.0, 8.0, 8.0);
            let body_uv = uv64(16.0, 16.0, 8.0, 12.0, 4.0);
            let arm_uv = uv64(40.0, 16.0, limb_w, 12.0, limb_w);
            let leg_uv = uv64(0.0, 16.0, limb_w, 12.0, limb_w);

            let head_mat = base * Mat4::from_translation(Vec3::new(0.0, neck_y, 0.0));
            emit_box(
                &mut verts,
                &mut indices,
                head_mat,
                Vec3::new(-4.0 * p, 0.0, -4.0 * p),
                Vec3::new(4.0 * p, 8.0 * p, 4.0 * p),
                head_uv,
                sky,
                block,
                tint,
                solid,
            );
            if is_piglin {
                // Signature piglin snout + floppy ears
                emit_box(
                    &mut verts,
                    &mut indices,
                    head_mat,
                    Vec3::new(-2.0 * p, 1.0 * p, 4.0 * p),
                    Vec3::new(2.0 * p, 3.5 * p, 5.2 * p),
                    uv64(32.0, 0.0, 4.0, 3.0, 1.0),
                    sky,
                    block,
                    tint,
                    solid,
                );
                for sx in [-4.5 * p, 3.8 * p] {
                    let ear = head_mat
                        * Mat4::from_translation(Vec3::new(sx, 4.0 * p, 0.0))
                        * Mat4::from_rotation_z(if sx < 0.0 { 0.35 } else { -0.35 });
                    emit_box(
                        &mut verts,
                        &mut indices,
                        ear,
                        Vec3::new(-0.4 * p, -3.0 * p, -1.5 * p),
                        Vec3::new(0.4 * p, 0.5 * p, 1.5 * p),
                        head_uv,
                        sky,
                        block,
                        tint,
                        solid,
                    );
                }
            }

            let torso_mat = base * Mat4::from_translation(Vec3::new(0.0, leg_top, 0.0));
            emit_box(
                &mut verts,
                &mut indices,
                torso_mat,
                Vec3::new(-4.0 * p, 0.0, -2.0 * p),
                Vec3::new(4.0 * p, 12.0 * p, 2.0 * p),
                body_uv,
                sky,
                block,
                tint,
                solid,
            );
            let r_arm_pitch = if is_zombie { -1.35 + swing * 0.25 } else { -swing * 0.85 };
            let l_arm_pitch = if is_zombie { -1.35 - swing * 0.25 } else { swing * 0.85 };
            let hw = (limb_w * 0.5) * p;
            let arm_x = (4.0 + limb_w * 0.5) * p;
            let r_arm = base
                * Mat4::from_translation(Vec3::new(-arm_x, neck_y - 2.0 * p, 0.0))
                * Mat4::from_rotation_x(r_arm_pitch);
            emit_box(
                &mut verts,
                &mut indices,
                r_arm,
                Vec3::new(-hw, -10.0 * p, -hw),
                Vec3::new(hw, 2.0 * p, hw),
                arm_uv,
                sky,
                block,
                tint,
                solid,
            );
            if is_piglin || is_wither {
                let sword_tint = if is_piglin {
                    [248, 212, 72, 255]
                } else {
                    [102, 100, 108, 255]
                };
                let sword = r_arm
                    * Mat4::from_translation(Vec3::new(0.0, -9.2 * p, 1.5 * p))
                    * Mat4::from_rotation_x(0.85);
                emit_box(
                    &mut verts,
                    &mut indices,
                    sword,
                    Vec3::new(-0.55 * p, -1.0 * p, -0.55 * p),
                    Vec3::new(0.55 * p, 10.5 * p, 0.55 * p),
                    tile_uv,
                    sky,
                    block,
                    sword_tint,
                    true,
                );
                emit_box(
                    &mut verts,
                    &mut indices,
                    sword,
                    Vec3::new(-2.0 * p, 1.2 * p, -0.8 * p),
                    Vec3::new(2.0 * p, 2.3 * p, 0.8 * p),
                    tile_uv,
                    sky,
                    block,
                    sword_tint,
                    true,
                );
            }
            let l_arm = base
                * Mat4::from_translation(Vec3::new(arm_x, neck_y - 2.0 * p, 0.0))
                * Mat4::from_rotation_x(l_arm_pitch);
            emit_box(
                &mut verts,
                &mut indices,
                l_arm,
                Vec3::new(-hw, -10.0 * p, -hw),
                Vec3::new(hw, 2.0 * p, hw),
                arm_uv,
                sky,
                block,
                tint,
                solid,
            );
            let leg_x = 2.0 * p;
            let r_leg = base
                * Mat4::from_translation(Vec3::new(-leg_x, leg_top, 0.0))
                * Mat4::from_rotation_x(swing);
            emit_box(
                &mut verts,
                &mut indices,
                r_leg,
                Vec3::new(-hw, -12.0 * p, -hw),
                Vec3::new(hw, 0.0, hw),
                leg_uv,
                sky,
                block,
                tint,
                solid,
            );
            let l_leg = base
                * Mat4::from_translation(Vec3::new(leg_x, leg_top, 0.0))
                * Mat4::from_rotation_x(-swing);
            emit_box(
                &mut verts,
                &mut indices,
                l_leg,
                Vec3::new(-hw, -12.0 * p, -hw),
                Vec3::new(hw, 0.0, hw),
                leg_uv,
                sky,
                block,
                tint,
                solid,
            );
        }
        EntityRigKind::Blaze => {
            let p = (box_h / 1.8).clamp(0.6, 1.5) * 0.0625 * rel_vs_y;
            let bob = (ent.anim_time * 3.1 + ent.id as f32).sin() * 1.4 * p;
            let head_uv = uv64(0.0, 0.0, 8.0, 8.0, 8.0);
            let rod_uv = uv64(0.0, 16.0, 2.0, 8.0, 2.0);
            let emissive_start = verts.len();

            let head_mat = base * Mat4::from_translation(Vec3::new(0.0, 20.0 * p + bob, 0.0));
            emit_box(
                &mut verts,
                &mut indices,
                head_mat,
                Vec3::new(-4.0 * p, -4.0 * p, -4.0 * p),
                Vec3::new(4.0 * p, 4.0 * p, 4.0 * p),
                head_uv,
                sky,
                15,
                tint,
                true,
            );
            let tiers = [
                (21.0 * p, 9.0 * p, 1.8f32),
                (14.0 * p, 7.0 * p, -2.5),
                (7.5 * p, 5.0 * p, 3.2),
            ];
            for (tier_idx, &(ty, radius, speed)) in tiers.iter().enumerate() {
                for r in 0..4usize {
                    let ang = ent.anim_time * speed
                        + (r as f32) * FRAC_PI_2
                        + (tier_idx as f32) * 0.55;
                    let rx = ang.cos() * radius;
                    let rz = ang.sin() * radius;
                    let ry = ty + (ent.anim_time * 4.0 + r as f32 + tier_idx as f32).sin() * 1.2 * p;
                    let rod_mat = base
                        * Mat4::from_translation(Vec3::new(rx, ry, rz))
                        * Mat4::from_rotation_y(-ang);
                    emit_box(
                        &mut verts,
                        &mut indices,
                        rod_mat,
                        Vec3::new(-1.0 * p, -4.0 * p, -1.0 * p),
                        Vec3::new(1.0 * p, 4.0 * p, 1.0 * p),
                        rod_uv,
                        sky,
                        15,
                        tint,
                        true,
                    );
                }
            }
            for v in &mut verts[emissive_start..] {
                v.packed |= VoxelVertex::FLAG_EMISSIVE;
            }
        }
        EntityRigKind::Ghast => {
            let s = (box_h / 4.0).clamp(0.45, 1.4) * rel_vs_y;
            let body_w = 2.0 * s;
            let tent_w = 0.22 * s;
            let body_uv = uv64(0.0, 0.0, 16.0, 16.0, 16.0);
            let tent_uv = uv64(0.0, 0.0, 2.0, 8.0, 2.0);
            let bob = (ent.anim_time * 1.8 + ent.id as f32).sin() * 0.12;
            let g_base = base * Mat4::from_translation(Vec3::new(0.0, 0.9 * s + bob, 0.0));

            emit_box(
                &mut verts,
                &mut indices,
                g_base,
                Vec3::new(-body_w, 0.0, -body_w),
                Vec3::new(body_w, 4.0 * s, body_w),
                body_uv,
                sky,
                block.max(10),
                tint,
                true,
            );
            let mut idx = 0usize;
            for tz in [-1.25f32, 0.0, 1.25] {
                for tx in [-1.25f32, 0.0, 1.25] {
                    idx += 1;
                    let tent_len = (1.1 + (idx % 3) as f32 * 0.35) * s;
                    let sway_x = (ent.anim_time * 3.1 + idx as f32 * 0.85).sin() * 0.28;
                    let sway_z = (ent.anim_time * 2.5 + idx as f32 * 1.1).cos() * 0.18;
                    let tmat = g_base
                        * Mat4::from_translation(Vec3::new(tx * s, 0.0, tz * s))
                        * Mat4::from_rotation_x(sway_x)
                        * Mat4::from_rotation_z(sway_z);
                    emit_box(
                        &mut verts,
                        &mut indices,
                        tmat,
                        Vec3::new(-tent_w, -tent_len, -tent_w),
                        Vec3::new(tent_w, 0.05, tent_w),
                        tent_uv,
                        sky,
                        block.max(10),
                        tint,
                        true,
                    );
                }
            }
        }
        EntityRigKind::MagmaCube => {
            let half_w = ((ent.collisionbox[3] - ent.collisionbox[0]).abs() * 0.5).clamp(0.24, 1.4);
            let h = box_h * rel_vs_y;
            let jump_lift = (ent.vel_y.max(0.0) * 0.055).clamp(0.0, 0.28);
            let shell_uv = uv128(0.0, 0.0, 16.0, 16.0, 16.0);
            let core_uv = uv128(0.0, 32.0, 8.0, 8.0, 8.0);

            let core_start = verts.len();
            emit_box(
                &mut verts,
                &mut indices,
                base,
                Vec3::new(-half_w * 0.52, h * 0.22, -half_w * 0.52),
                Vec3::new(half_w * 0.52, h * 0.78 + jump_lift * 4.0, half_w * 0.52),
                core_uv,
                sky,
                15,
                tint,
                true,
            );
            for v in &mut verts[core_start..] {
                v.packed |= VoxelVertex::FLAG_EMISSIVE;
            }
            let slice_h = h / 8.0;
            for slice in 0..8usize {
                let y0 = slice as f32 * (slice_h + jump_lift);
                let y1 = y0 + slice_h * 0.96;
                emit_box(
                    &mut verts,
                    &mut indices,
                    base,
                    Vec3::new(-half_w, y0, -half_w),
                    Vec3::new(half_w, y1, half_w),
                    shell_uv,
                    sky,
                    12,
                    tint,
                    true,
                );
            }
        }
        EntityRigKind::Guardian => {
            let r = (box_h * 0.5).clamp(0.32, 0.85) * rel_vs_y;
            let body_uv = uv64(0.0, 0.0, 12.0, 12.0, 12.0);
            let eye_uv = uv64(8.0, 0.0, 4.0, 4.0, 1.0);
            let tail_uv = uv64(40.0, 0.0, 4.0, 4.0, 8.0);
            let spike_uv = uv64(0.0, 0.0, 2.0, 4.0, 2.0);
            let swim_bob = (ent.anim_time * 2.6 + ent.id as f32).sin() * 0.06;
            let g_base = base * Mat4::from_translation(Vec3::new(0.0, r + swim_bob, 0.0));

            emit_box(
                &mut verts,
                &mut indices,
                g_base,
                Vec3::new(-r, -r, -r),
                Vec3::new(r, r, r),
                body_uv,
                sky,
                block.max(6),
                tint,
                true,
            );
            let eye_start = verts.len();
            emit_box(
                &mut verts,
                &mut indices,
                g_base,
                Vec3::new(-r * 0.34, -r * 0.34, r * 0.92),
                Vec3::new(r * 0.34, r * 0.34, r * 1.05),
                eye_uv,
                sky,
                15,
                tint,
                true,
            );
            for v in &mut verts[eye_start..] {
                v.packed |= VoxelVertex::FLAG_EMISSIVE;
            }
            let ext = 0.28 * r + 0.08 * r * (ent.anim_time * 4.5).sin().abs();
            let spike_dirs = [
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(-1.0, 1.0, 0.0),
                Vec3::new(1.0, -1.0, 0.0),
                Vec3::new(-1.0, -1.0, 0.0),
                Vec3::new(0.0, 1.0, 1.0),
                Vec3::new(0.0, 1.0, -1.0),
                Vec3::new(0.0, -1.0, 1.0),
                Vec3::new(0.0, -1.0, -1.0),
                Vec3::new(1.0, 0.0, 1.0),
                Vec3::new(-1.0, 0.0, 1.0),
                Vec3::new(1.0, 0.0, -1.0),
                Vec3::new(-1.0, 0.0, -1.0),
            ];
            for d in spike_dirs {
                let c = d.normalize() * (r + ext * 0.5);
                let sp_half = Vec3::splat(r * 0.12) + d.abs() * (ext * 0.55);
                emit_box(
                    &mut verts,
                    &mut indices,
                    g_base,
                    c - sp_half,
                    c + sp_half,
                    spike_uv,
                    sky,
                    block.max(6),
                    tint,
                    true,
                );
            }
            let tail_wag = (ent.anim_time * 5.2 + ent.id as f32).sin() * 0.32;
            let mut t_mat = g_base * Mat4::from_translation(Vec3::new(0.0, 0.0, -r));
            for seg in 0..3usize {
                let len = r * (0.62 - seg as f32 * 0.12);
                let w = r * (0.34 - seg as f32 * 0.07);
                t_mat = t_mat * Mat4::from_rotation_y(tail_wag * (0.5 + seg as f32 * 0.35));
                emit_box(
                    &mut verts,
                    &mut indices,
                    t_mat,
                    Vec3::new(-w, -w, -len),
                    Vec3::new(w, w, 0.0),
                    tail_uv,
                    sky,
                    block.max(6),
                    tint,
                    true,
                );
                t_mat = t_mat * Mat4::from_translation(Vec3::new(0.0, 0.0, -len));
            }
        }
        EntityRigKind::Fireball => {
            let r = ((ent.collisionbox[3] - ent.collisionbox[0]).abs() * 0.5).clamp(0.16, 0.55);
            let spin = Mat4::from_translation(ent.pos)
                * Mat4::from_rotation_y(ent.anim_time * 4.2)
                * Mat4::from_rotation_x(ent.anim_time * 3.1);
            let start = verts.len();
            emit_box(
                &mut verts,
                &mut indices,
                spin,
                Vec3::splat(-r),
                Vec3::splat(r),
                tile_uv,
                15,
                15,
                [255, 195, 70, 255],
                true,
            );
            emit_box(
                &mut verts,
                &mut indices,
                spin * Mat4::from_rotation_y(0.78),
                Vec3::splat(-r * 0.72),
                Vec3::splat(r * 0.72),
                tile_uv,
                15,
                15,
                [255, 245, 165, 255],
                true,
            );
            for v in &mut verts[start..] {
                v.packed |= VoxelVertex::FLAG_EMISSIVE;
            }
        }
        EntityRigKind::Cow => {
            let p = (box_h / 1.4).clamp(0.55, 1.8) * 0.0625 * rel_vs_y;
            let leg_h = 12.0 * p;
            let leg_uv = uv64(0.0, 16.0, 4.0, 12.0, 4.0);
            let body_uv = uv64(18.0, 4.0, 12.0, 18.0, 10.0);
            let head_uv = uv64(0.0, 0.0, 8.0, 8.0, 6.0);
            let horn_uv = uv64(22.0, 0.0, 1.0, 3.0, 1.0);

            let body_mat = base
                * Mat4::from_translation(Vec3::new(0.0, leg_h + 5.0 * p, 0.0))
                * Mat4::from_rotation_x(FRAC_PI_2);
            emit_box(
                &mut verts,
                &mut indices,
                body_mat,
                Vec3::new(-6.0 * p, -9.0 * p, -5.0 * p),
                Vec3::new(6.0 * p, 9.0 * p, 5.0 * p),
                body_uv,
                sky,
                block,
                tint,
                true,
            );
            let head_mat = base * Mat4::from_translation(Vec3::new(0.0, leg_h + 6.0 * p, 11.0 * p));
            emit_box(
                &mut verts,
                &mut indices,
                head_mat,
                Vec3::new(-4.0 * p, -4.0 * p, -3.0 * p),
                Vec3::new(4.0 * p, 4.0 * p, 3.0 * p),
                head_uv,
                sky,
                block,
                tint,
                true,
            );
            for sx in [-4.5 * p, 3.5 * p] {
                emit_box(
                    &mut verts,
                    &mut indices,
                    head_mat,
                    Vec3::new(sx, 3.0 * p, -1.0 * p),
                    Vec3::new(sx + p, 6.0 * p, 0.0),
                    horn_uv,
                    sky,
                    block,
                    tint,
                    true,
                );
            }
            emit_four_legs(&mut verts, &mut indices, leg_uv, 4.0 * p, leg_h, 4.0 * p, 3.8 * p, 6.5 * p, -6.5 * p);
        }
        EntityRigKind::Pig => {
            let p = (box_h / 0.88).clamp(0.55, 1.8) * 0.0625 * rel_vs_y;
            let leg_h = 6.0 * p;
            let leg_uv = uv64(0.0, 16.0, 4.0, 6.0, 4.0);
            let body_uv = uv64(28.0, 8.0, 10.0, 16.0, 8.0);
            let head_uv = uv64(0.0, 0.0, 8.0, 8.0, 8.0);
            let snout_uv = uv64(16.0, 16.0, 4.0, 3.0, 1.0);

            let body_mat = base
                * Mat4::from_translation(Vec3::new(0.0, leg_h + 4.0 * p, 0.0))
                * Mat4::from_rotation_x(FRAC_PI_2);
            emit_box(
                &mut verts,
                &mut indices,
                body_mat,
                Vec3::new(-5.0 * p, -8.0 * p, -4.0 * p),
                Vec3::new(5.0 * p, 8.0 * p, 4.0 * p),
                body_uv,
                sky,
                block,
                tint,
                true,
            );
            let head_mat = base * Mat4::from_translation(Vec3::new(0.0, leg_h + 5.5 * p, 9.5 * p));
            emit_box(
                &mut verts,
                &mut indices,
                head_mat,
                Vec3::new(-4.0 * p, -4.0 * p, -4.0 * p),
                Vec3::new(4.0 * p, 4.0 * p, 4.0 * p),
                head_uv,
                sky,
                block,
                tint,
                true,
            );
            emit_box(
                &mut verts,
                &mut indices,
                head_mat,
                Vec3::new(-2.0 * p, -3.0 * p, 4.0 * p),
                Vec3::new(2.0 * p, 0.0, 5.0 * p),
                snout_uv,
                sky,
                block,
                tint,
                true,
            );
            emit_four_legs(&mut verts, &mut indices, leg_uv, 4.0 * p, leg_h, 4.0 * p, 3.0 * p, 5.5 * p, -5.5 * p);
        }
        EntityRigKind::Sheep => {
            let p = (box_h / 1.3).clamp(0.55, 1.8) * 0.0625 * rel_vs_y;
            let leg_h = 12.0 * p;
            let leg_uv = uv64(0.0, 16.0, 4.0, 12.0, 4.0);
            let body_uv = uv64(28.0, 8.0, 8.0, 16.0, 6.0);
            let head_uv = uv64(0.0, 0.0, 6.0, 6.0, 8.0);

            let body_mat = base
                * Mat4::from_translation(Vec3::new(0.0, leg_h + 3.5 * p, 0.0))
                * Mat4::from_rotation_x(FRAC_PI_2);
            emit_box(
                &mut verts,
                &mut indices,
                body_mat,
                Vec3::new(-4.5 * p, -8.0 * p, -3.5 * p),
                Vec3::new(4.5 * p, 8.0 * p, 3.5 * p),
                body_uv,
                sky,
                block,
                tint,
                true,
            );
            let head_mat = base * Mat4::from_translation(Vec3::new(0.0, leg_h + 6.0 * p, 9.5 * p));
            emit_box(
                &mut verts,
                &mut indices,
                head_mat,
                Vec3::new(-3.0 * p, -3.0 * p, -4.0 * p),
                Vec3::new(3.0 * p, 3.0 * p, 4.0 * p),
                head_uv,
                sky,
                block,
                tint,
                true,
            );
            emit_four_legs(&mut verts, &mut indices, leg_uv, 3.8 * p, leg_h, 3.8 * p, 2.8 * p, 5.5 * p, -5.5 * p);
        }
        EntityRigKind::Chicken => {
            let p = (box_h / 0.7).clamp(0.55, 1.8) * 0.058 * rel_vs_y;
            let leg_h = 5.0 * p;
            let head_uv = uv64(0.0, 0.0, 4.0, 6.0, 3.0);
            let beak_uv = uv64(14.0, 0.0, 4.0, 2.0, 2.0);
            let wattle_uv = uv64(14.0, 4.0, 2.0, 2.0, 2.0);
            let body_uv = uv64(0.0, 9.0, 6.0, 8.0, 6.0);
            let wing_uv = uv64(24.0, 13.0, 1.0, 4.0, 6.0);
            let foot_uv = uv64(29.0, 0.0, 3.0, 3.0, 1.0);

            let body_mat = base
                * Mat4::from_translation(Vec3::new(0.0, leg_h + 3.0 * p, 0.0))
                * Mat4::from_rotation_x(FRAC_PI_2);
            emit_box(
                &mut verts,
                &mut indices,
                body_mat,
                Vec3::new(-3.0 * p, -4.0 * p, -3.0 * p),
                Vec3::new(3.0 * p, 4.0 * p, 3.0 * p),
                body_uv,
                sky,
                block,
                tint,
                true,
            );
            let head_mat = base * Mat4::from_translation(Vec3::new(0.0, leg_h + 6.5 * p, 4.0 * p));
            emit_box(
                &mut verts,
                &mut indices,
                head_mat,
                Vec3::new(-2.0 * p, -3.0 * p, -1.5 * p),
                Vec3::new(2.0 * p, 3.0 * p, 1.5 * p),
                head_uv,
                sky,
                block,
                tint,
                true,
            );
            emit_box(
                &mut verts,
                &mut indices,
                head_mat,
                Vec3::new(-2.0 * p, -1.0 * p, 1.5 * p),
                Vec3::new(2.0 * p, 1.0 * p, 3.5 * p),
                beak_uv,
                sky,
                block,
                tint,
                true,
            );
            emit_box(
                &mut verts,
                &mut indices,
                head_mat,
                Vec3::new(-1.0 * p, -3.0 * p, 1.2 * p),
                Vec3::new(1.0 * p, -1.0 * p, 2.8 * p),
                wattle_uv,
                sky,
                block,
                tint,
                true,
            );
            for sx in [-3.5 * p, 3.5 * p] {
                emit_box(
                    &mut verts,
                    &mut indices,
                    base * Mat4::from_translation(Vec3::new(sx, leg_h + 3.0 * p, 0.0)),
                    Vec3::new(-0.5 * p, -2.0 * p, -3.0 * p),
                    Vec3::new(0.5 * p, 2.0 * p, 3.0 * p),
                    wing_uv,
                    sky,
                    block,
                    tint,
                    true,
                );
            }
            for &(lx, pm) in &[(-1.6 * p, 1.0f32), (1.6 * p, -1.0f32)] {
                let leg = base
                    * Mat4::from_translation(Vec3::new(lx, leg_h, 0.0))
                    * Mat4::from_rotation_x(swing * pm);
                emit_box(
                    &mut verts,
                    &mut indices,
                    leg,
                    Vec3::new(-0.7 * p, -leg_h, -1.2 * p),
                    Vec3::new(0.7 * p, 0.0, 1.2 * p),
                    foot_uv,
                    sky,
                    block,
                    tint,
                    true,
                );
            }
        }
        EntityRigKind::Wolf => {
            let p = (box_h / 0.85).clamp(0.55, 1.8) * 0.06 * rel_vs_y;
            let leg_h = 8.0 * p;
            let head_uv = uv64(0.0, 0.0, 6.0, 6.0, 4.0);
            let nose_uv = uv64(0.0, 10.0, 3.0, 3.0, 4.0);
            let mane_uv = uv64(21.0, 0.0, 8.0, 6.0, 7.0);
            let body_uv = uv64(18.0, 14.0, 6.0, 9.0, 6.0);
            let leg_uv = uv64(0.0, 18.0, 2.0, 8.0, 2.0);
            let tail_uv = uv64(9.0, 18.0, 2.0, 8.0, 2.0);
            let ear_uv = uv64(16.0, 14.0, 2.0, 2.0, 1.0);

            let body_mat = base
                * Mat4::from_translation(Vec3::new(0.0, leg_h + 3.0 * p, -2.0 * p))
                * Mat4::from_rotation_x(FRAC_PI_2);
            emit_box(
                &mut verts,
                &mut indices,
                body_mat,
                Vec3::new(-3.0 * p, -4.5 * p, -3.0 * p),
                Vec3::new(3.0 * p, 4.5 * p, 3.0 * p),
                body_uv,
                sky,
                block,
                tint,
                true,
            );
            let mane_mat = base
                * Mat4::from_translation(Vec3::new(0.0, leg_h + 3.2 * p, 5.0 * p))
                * Mat4::from_rotation_x(FRAC_PI_2);
            emit_box(
                &mut verts,
                &mut indices,
                mane_mat,
                Vec3::new(-4.0 * p, -3.0 * p, -3.5 * p),
                Vec3::new(4.0 * p, 3.0 * p, 3.5 * p),
                mane_uv,
                sky,
                block,
                tint,
                true,
            );
            let head_mat = base * Mat4::from_translation(Vec3::new(0.0, leg_h + 3.5 * p, 9.5 * p));
            emit_box(
                &mut verts,
                &mut indices,
                head_mat,
                Vec3::new(-3.0 * p, -3.0 * p, -2.0 * p),
                Vec3::new(3.0 * p, 3.0 * p, 2.0 * p),
                head_uv,
                sky,
                block,
                tint,
                true,
            );
            emit_box(
                &mut verts,
                &mut indices,
                head_mat,
                Vec3::new(-1.5 * p, -3.0 * p, 2.0 * p),
                Vec3::new(1.5 * p, 0.0, 5.5 * p),
                nose_uv,
                sky,
                block,
                tint,
                true,
            );
            for sx in [-2.8 * p, 0.8 * p] {
                emit_box(
                    &mut verts,
                    &mut indices,
                    head_mat,
                    Vec3::new(sx, 3.0 * p, -0.5 * p),
                    Vec3::new(sx + 2.0 * p, 5.0 * p, 0.5 * p),
                    ear_uv,
                    sky,
                    block,
                    tint,
                    true,
                );
            }
            let tail_mat = base
                * Mat4::from_translation(Vec3::new(0.0, leg_h + 4.5 * p, -6.5 * p))
                * Mat4::from_rotation_x(-1.15);
            emit_box(
                &mut verts,
                &mut indices,
                tail_mat,
                Vec3::new(-1.0 * p, -7.0 * p, -1.0 * p),
                Vec3::new(1.0 * p, 0.0, 1.0 * p),
                tail_uv,
                sky,
                block,
                tint,
                true,
            );
            emit_four_legs(&mut verts, &mut indices, leg_uv, 2.0 * p, leg_h, 2.0 * p, 1.8 * p, 5.0 * p, -5.0 * p);
        }
        EntityRigKind::Rabbit => {
            let p = (box_h / 0.5).clamp(0.55, 1.8) * 0.045 * rel_vs_y;
            let body_uv = uv64(0.0, 0.0, 6.0, 5.0, 10.0);
            let head_uv = uv64(32.0, 0.0, 5.0, 4.0, 5.0);
            let ear_l_uv = uv64(52.0, 0.0, 2.0, 5.0, 1.0);
            let ear_r_uv = uv64(58.0, 0.0, 2.0, 5.0, 1.0);
            let foot_uv = uv64(8.0, 24.0, 2.0, 1.0, 7.0);
            let leg_f_uv = uv64(8.0, 15.0, 2.0, 4.0, 2.0);
            let tail_uv = uv64(52.0, 6.0, 3.0, 3.0, 2.0);

            emit_box(
                &mut verts,
                &mut indices,
                base * Mat4::from_translation(Vec3::new(0.0, 3.5 * p, -0.5 * p)),
                Vec3::new(-3.0 * p, -2.5 * p, -4.5 * p),
                Vec3::new(3.0 * p, 2.5 * p, 4.5 * p),
                body_uv,
                sky,
                block,
                tint,
                true,
            );
            let head_mat = base * Mat4::from_translation(Vec3::new(0.0, 6.5 * p, 4.5 * p));
            emit_box(
                &mut verts,
                &mut indices,
                head_mat,
                Vec3::new(-2.5 * p, -2.0 * p, -2.5 * p),
                Vec3::new(2.5 * p, 2.0 * p, 2.5 * p),
                head_uv,
                sky,
                block,
                tint,
                true,
            );
            emit_box(
                &mut verts,
                &mut indices,
                head_mat,
                Vec3::new(-2.4 * p, 2.0 * p, -1.0 * p),
                Vec3::new(-0.4 * p, 7.0 * p, 0.2 * p),
                ear_l_uv,
                sky,
                block,
                tint,
                true,
            );
            emit_box(
                &mut verts,
                &mut indices,
                head_mat,
                Vec3::new(0.4 * p, 2.0 * p, -1.0 * p),
                Vec3::new(2.4 * p, 7.0 * p, 0.2 * p),
                ear_r_uv,
                sky,
                block,
                tint,
                true,
            );
            emit_box(
                &mut verts,
                &mut indices,
                base,
                Vec3::new(-1.5 * p, 3.0 * p, -6.5 * p),
                Vec3::new(1.5 * p, 5.5 * p, -4.5 * p),
                tail_uv,
                sky,
                block,
                tint,
                true,
            );
            for &(lx, lz, pm, uv) in &[
                (-2.0 * p, 3.2 * p, 1.0f32, leg_f_uv),
                (2.0 * p, 3.2 * p, -1.0f32, leg_f_uv),
                (-2.0 * p, -2.8 * p, -1.0f32, foot_uv),
                (2.0 * p, -2.8 * p, 1.0f32, foot_uv),
            ] {
                let leg = base
                    * Mat4::from_translation(Vec3::new(lx, 2.0 * p, lz))
                    * Mat4::from_rotation_x(swing * pm);
                emit_box(
                    &mut verts,
                    &mut indices,
                    leg,
                    Vec3::new(-1.0 * p, -2.0 * p, -1.5 * p),
                    Vec3::new(1.0 * p, 0.0, 2.5 * p),
                    uv,
                    sky,
                    block,
                    tint,
                    true,
                );
            }
        }
        EntityRigKind::Llama => {
            let p = (box_h / 1.87).clamp(0.55, 1.8) * 0.056 * rel_vs_y;
            let leg_h = 14.0 * p;
            let head_uv = uv128(0.0, 14.0, 8.0, 18.0, 6.0);
            let body_uv = uv128(29.0, 0.0, 12.0, 18.0, 10.0);
            let leg_uv = uv128(29.0, 29.0, 4.0, 14.0, 4.0);
            let ear_uv = uv128(17.0, 0.0, 3.0, 3.0, 2.0);

            let body_mat = base
                * Mat4::from_translation(Vec3::new(0.0, leg_h + 5.0 * p, -1.0 * p))
                * Mat4::from_rotation_x(FRAC_PI_2);
            emit_box(
                &mut verts,
                &mut indices,
                body_mat,
                Vec3::new(-6.0 * p, -9.0 * p, -5.0 * p),
                Vec3::new(6.0 * p, 9.0 * p, 5.0 * p),
                body_uv,
                sky,
                block,
                tint,
                true,
            );
            let head_mat = base * Mat4::from_translation(Vec3::new(0.0, leg_h + 10.0 * p, 9.0 * p));
            emit_box(
                &mut verts,
                &mut indices,
                head_mat,
                Vec3::new(-4.0 * p, -9.0 * p, -3.0 * p),
                Vec3::new(4.0 * p, 9.0 * p, 3.0 * p),
                head_uv,
                sky,
                block,
                tint,
                true,
            );
            for sx in [-3.8 * p, 1.2 * p] {
                emit_box(
                    &mut verts,
                    &mut indices,
                    head_mat,
                    Vec3::new(sx, 9.0 * p, -1.0 * p),
                    Vec3::new(sx + 2.6 * p, 12.0 * p, 1.0 * p),
                    ear_uv,
                    sky,
                    block,
                    tint,
                    true,
                );
            }
            emit_four_legs(&mut verts, &mut indices, leg_uv, 4.0 * p, leg_h, 4.0 * p, 3.8 * p, 6.0 * p, -7.0 * p);
        }
        EntityRigKind::Creeper => {
            let p = (box_h / 1.7).clamp(0.55, 1.8) * 0.065 * rel_vs_y;
            let leg_h = 6.0 * p;
            let head_uv = uv64(0.0, 0.0, 8.0, 8.0, 8.0);
            let body_uv = uv64(16.0, 16.0, 8.0, 12.0, 4.0);
            let leg_uv = uv64(0.0, 16.0, 4.0, 6.0, 4.0);

            emit_box(
                &mut verts,
                &mut indices,
                base * Mat4::from_translation(Vec3::new(0.0, leg_h, 0.0)),
                Vec3::new(-4.0 * p, 0.0, -2.0 * p),
                Vec3::new(4.0 * p, 12.0 * p, 2.0 * p),
                body_uv,
                sky,
                block,
                tint,
                true,
            );
            emit_box(
                &mut verts,
                &mut indices,
                base * Mat4::from_translation(Vec3::new(0.0, leg_h + 12.0 * p, 0.0)),
                Vec3::new(-4.0 * p, 0.0, -4.0 * p),
                Vec3::new(4.0 * p, 8.0 * p, 4.0 * p),
                head_uv,
                sky,
                block,
                tint,
                true,
            );
            emit_four_legs(&mut verts, &mut indices, leg_uv, 4.0 * p, leg_h, 4.0 * p, 2.2 * p, 3.8 * p, -3.8 * p);
        }
        EntityRigKind::Spider => {
            let s = (box_h / 0.9).clamp(0.5, 1.6) * rel_vs_y;
            let p = 0.0625 * s;
            let head_uv = uv64(32.0, 4.0, 8.0, 8.0, 8.0);
            let thorax_uv = uv64(0.0, 0.0, 6.0, 6.0, 6.0);
            let abd_uv = uv64(0.0, 12.0, 10.0, 8.0, 12.0);
            let leg_uv = uv64(18.0, 0.0, 16.0, 2.0, 2.0);

            let cy = 6.5 * p;
            // Head (+Z front)
            emit_box(
                &mut verts,
                &mut indices,
                base * Mat4::from_translation(Vec3::new(0.0, cy, 8.0 * p)),
                Vec3::new(-4.0 * p, -4.0 * p, -4.0 * p),
                Vec3::new(4.0 * p, 4.0 * p, 4.0 * p),
                head_uv,
                sky,
                block,
                tint,
                true,
            );
            // Thorax (center)
            emit_box(
                &mut verts,
                &mut indices,
                base * Mat4::from_translation(Vec3::new(0.0, cy, 1.0 * p)),
                Vec3::new(-3.0 * p, -3.0 * p, -3.0 * p),
                Vec3::new(3.0 * p, 3.0 * p, 3.0 * p),
                thorax_uv,
                sky,
                block,
                tint,
                true,
            );
            // Abdomen (-Z rear)
            emit_box(
                &mut verts,
                &mut indices,
                base * Mat4::from_translation(Vec3::new(0.0, cy + 0.5 * p, -8.0 * p)),
                Vec3::new(-5.0 * p, -4.0 * p, -6.0 * p),
                Vec3::new(5.0 * p, 4.0 * p, 6.0 * p),
                abd_uv,
                sky,
                block,
                tint,
                true,
            );
            // 8 articulated angled spider legs
            for side in [-1.0f32, 1.0f32] {
                for seg in 0..4 {
                    let z_off = (-2.0 + seg as f32 * 2.2) * p;
                    let phase_mul = if (seg % 2 == 0) ^ (side > 0.0) { 1.0 } else { -1.0 };
                    let sweep = (seg as f32 - 1.5) * 0.24 + swing * 0.42 * phase_mul;
                    let leg = base
                        * Mat4::from_translation(Vec3::new(side * 3.0 * p, cy - 0.5 * p, z_off))
                        * Mat4::from_rotation_y(side * sweep)
                        * Mat4::from_rotation_z(-side * 0.42);
                    let lx0 = if side < 0.0 { -13.0 * p } else { 0.0 };
                    let lx1 = if side < 0.0 { 0.0 } else { 13.0 * p };
                    emit_box(
                        &mut verts,
                        &mut indices,
                        leg,
                        Vec3::new(lx0, -1.0 * p, -1.0 * p),
                        Vec3::new(lx1, 1.0 * p, 1.0 * p),
                        leg_uv,
                        sky,
                        block,
                        tint,
                        true,
                    );
                }
            }
        }
        EntityRigKind::SmallCube => {
            let half_w = ((ent.collisionbox[3] - ent.collisionbox[0]).abs() * 0.5).clamp(0.22, 1.2);
            let half_d = ((ent.collisionbox[5] - ent.collisionbox[2]).abs() * 0.5).clamp(0.22, 1.2);
            let h = box_h * rel_vs_y;
            let cube_uv = if ent.entity_name.to_ascii_lowercase().contains("slime")
                || ent.entity_name.to_ascii_lowercase().contains("magma")
            {
                uv128(0.0, 0.0, 16.0, 16.0, 16.0)
            } else {
                uv64(0.0, 0.0, 8.0, 8.0, 8.0)
            };
            emit_box(
                &mut verts,
                &mut indices,
                base,
                Vec3::new(-half_w, 0.0, -half_d),
                Vec3::new(half_w, h, half_d),
                cube_uv,
                sky,
                block,
                tint,
                true,
            );
        }
    }

    if ent.hurt_flash > 0.01 {
        for v in &mut verts {
            v.packed |= 1u32 << 26;
        }
    }

    (verts, indices)
}

/// Synthesize a crisp 16x16 textured fallback mob skin when an entity's texture file is absent.
pub fn make_fallback_mob_texture(entity_name: &str) -> RgbaImage {
    let lower = entity_name.to_ascii_lowercase();
    let (base, accent): ([u8; 4], [u8; 4]) = if lower.contains("fireball") {
        ([255, 165, 36, 255], [255, 238, 128, 255])
    } else if lower.contains("blaze") {
        ([244, 176, 36, 255], [255, 240, 120, 255])
    } else if lower.contains("ghast") {
        ([242, 242, 244, 255], [190, 56, 56, 255])
    } else if lower.contains("witherskeleton") || lower.contains("wither_skeleton") {
        ([44, 42, 46, 255], [160, 158, 164, 255])
    } else if lower.contains("magma") {
        ([54, 26, 20, 255], [255, 154, 24, 255])
    } else if lower.contains("guardian") {
        ([82, 162, 146, 255], [238, 142, 74, 255])
    } else if lower.contains("pigman")
        || lower.contains("zombiepig")
        || lower.contains("zombified_piglin")
        || lower.contains("piglin")
    {
        ([188, 124, 124, 255], [92, 142, 66, 255])
    } else if lower.contains("zombie") || lower.contains("creeper") || lower.contains("slime") {
        ([76, 138, 66, 255], [46, 92, 42, 255])
    } else if lower.contains("skeleton") {
        ([214, 214, 206, 255], [142, 142, 136, 255])
    } else if lower.contains("pig") {
        ([232, 152, 160, 255], [198, 108, 120, 255])
    } else if lower.contains("cow") || lower.contains("mooshroom") {
        ([112, 74, 48, 255], [228, 224, 216, 255])
    } else if lower.contains("chicken") {
        ([244, 244, 240, 255], [208, 42, 42, 255])
    } else if lower.contains("wolf") || lower.contains("rabbit") || lower.contains("llama") {
        ([184, 174, 158, 255], [116, 104, 90, 255])
    } else if lower.contains("sheep") {
        ([236, 234, 228, 255], [196, 190, 180, 255])
    } else if lower.contains("spider") || lower.contains("enderman") {
        ([38, 36, 44, 255], [176, 36, 48, 255])
    } else if lower.contains("bee") {
        ([236, 190, 42, 255], [38, 30, 22, 255])
    } else {
        ([156, 128, 98, 255], [96, 74, 54, 255])
    };
    let mut img = RgbaImage::new(16, 16);
    for y in 0..16 {
        for x in 0..16 {
            let eye = (y == 5 || y == 6) && ((4..=5).contains(&x) || (10..=11).contains(&x));
            let col = if eye {
                accent
            } else if (x + y) % 3 == 0 {
                [
                    base[0].saturating_sub(14),
                    base[1].saturating_sub(14),
                    base[2].saturating_sub(14),
                    255,
                ]
            } else {
                base
            };
            img.put_pixel(x, y, Rgba(col));
        }
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_mob(name: &str, cbox: [f32; 6]) -> EntityRenderSnapshot {
        EntityRenderSnapshot {
            id: 1,
            entity_name: name.into(),
            pos: Vec3::ZERO,
            yaw: 0.0,
            collisionbox: cbox,
            visual: "mesh".into(),
            visual_size: Vec3::ONE,
            mesh: String::new(),
            texture_name: String::new(),
            wield_item: String::new(),
            walk_phase: 0.5,
            walk_blend: 1.0,
            anim_time: 0.5,
            vel_y: 0.0,
            hurt_flash: 0.0,
            sky_light: 15,
            block_light: 0,
        }
    }

    #[test]
    fn entity_rig_mesh_animates_and_flashes_red_on_damage() {
        let mut ent = mk_mob("mobs_mc:zombie", [-0.3, 0.0, -0.3, 0.3, 1.95, 0.3]);
        ent.yaw = 0.4;
        ent.mesh = "mobs_mc_zombie.b3d".into();
        ent.texture_name = "mobs_mc_zombie.png".into();

        let (v_idle, i_idle) = build_entity_rig_mesh(&ent, 64, 64);
        assert!(!v_idle.is_empty());
        assert_eq!(i_idle.len() % 6, 0);
        assert!(v_idle.iter().all(|v| v.tint == [255, 255, 255, 255]));
        // Every solid mob vertex carries bit 27 (`no_alpha_discard`) and valid [0..1] UVs
        assert!(v_idle.iter().all(|v| ((v.packed >> 27) & 1) == 1));
        assert!(v_idle.iter().all(|v| v.uv[0] >= 0.0 && v.uv[0] <= 1.0 && v.uv[1] >= 0.0 && v.uv[1] <= 1.0));

        ent.hurt_flash = 0.25;
        let (v_hurt, _) = build_entity_rig_mesh(&ent, 64, 64);
        assert!(
            v_hurt.iter().all(|v| v.tint[0] == 255 && v.tint[1] < 90 && v.tint[2] < 90),
            "damaged entity vertices must carry vivid red hurt tint"
        );
    }

    #[test]
    fn species_classification_and_uv_integrity_across_mineclonia_mobs() {
        let cases = [
            ("mobs_mc:cow", [-0.45, 0.0, -0.45, 0.45, 1.4, 0.45], EntityRigKind::Cow, "mobs_mc_cow.png"),
            ("mobs_mc:pig", [-0.45, 0.0, -0.45, 0.45, 0.865, 0.45], EntityRigKind::Pig, "mobs_mc_pig.png"),
            ("mobs_mc:sheep", [-0.45, 0.0, -0.45, 0.45, 1.3, 0.45], EntityRigKind::Sheep, "mobs_mc_sheep.png"),
            ("mobs_mc:chicken", [-0.2, 0.0, -0.2, 0.2, 0.7, 0.2], EntityRigKind::Chicken, "mobs_mc_chicken.png"),
            ("mobs_mc:wolf", [-0.3, 0.0, -0.3, 0.3, 0.85, 0.3], EntityRigKind::Wolf, "mobs_mc_wolf.png"),
            ("mobs_mc:rabbit", [-0.2, 0.0, -0.2, 0.2, 0.5, 0.2], EntityRigKind::Rabbit, "mobs_mc_rabbit_brown.png"),
            ("mobs_mc:llama", [-0.45, 0.0, -0.45, 0.45, 1.87, 0.45], EntityRigKind::Llama, "mobs_mc_llama_brown.png"),
            ("mobs_mc:creeper", [-0.3, 0.0, -0.3, 0.3, 1.7, 0.3], EntityRigKind::Creeper, "mobs_mc_creeper.png"),
            ("mobs_mc:spider", [-0.7, 0.0, -0.7, 0.7, 0.9, 0.7], EntityRigKind::Spider, "mobs_mc_spider.png"),
            ("mobs_mc:zombie", [-0.3, 0.0, -0.3, 0.3, 1.95, 0.3], EntityRigKind::Biped, "mobs_mc_zombie.png"),
            ("mobs_mc:skeleton", [-0.3, 0.0, -0.3, 0.3, 1.95, 0.3], EntityRigKind::Biped, "mobs_mc_skeleton.png"),
        ];
        let tex_dir = crate::paths::EnginePaths::resolve()
            .game_dir
            .join("mods/ENTITIES/mobs_mc/textures");

        for &(name, cbox, expected_kind, tex_file) in &cases {
            let mut mob = mk_mob(name, cbox);
            assert_eq!(classify_entity_rig(&mob), expected_kind, "species={name}");

            let p = tex_dir.join(tex_file);
            let (tw, th, img_opt) = if p.is_file() {
                let mut im = image::open(&p).expect("open mob png").to_rgba8();
                fill_transparent_mob_pixels(&mut im);
                (im.width(), im.height(), Some(im))
            } else {
                (64, 32, None)
            };
            let (verts, idx) = build_entity_rig_mesh(&mob, tw, th);
            assert!(!verts.is_empty() && !idx.is_empty());
            assert!(verts.iter().all(|v| ((v.packed >> 27) & 1) == 1));
            assert!(verts.iter().all(|v| v.uv[0] >= 0.0 && v.uv[0] <= 1.0 && v.uv[1] >= 0.0 && v.uv[1] <= 1.0));

            if let Some(im) = img_opt {
                for quad in verts.chunks_exact(4) {
                    let cu = 0.25 * (quad[0].uv[0] + quad[1].uv[0] + quad[2].uv[0] + quad[3].uv[0]);
                    let cv = 0.25 * (quad[0].uv[1] + quad[1].uv[1] + quad[2].uv[1] + quad[3].uv[1]);
                    let px = ((cu * tw as f32).floor() as u32).min(tw - 1);
                    let py = ((cv * th as f32).floor() as u32).min(th - 1);
                    assert!(
                        im.get_pixel(px, py).0[3] >= 128,
                        "opaque UV pixel expected at ({px},{py}) on {tex_file} for {name}"
                    );
                }
            }

            // Verify Luanti yaw convention: yaw=0 places head/snout at +Z; yaw=pi/2 turns to -X
            mob.yaw = std::f32::consts::FRAC_PI_2;
            let (v_rot, _) = build_entity_rig_mesh(&mob, tw, th);
            let min_x = v_rot.iter().map(|v| v.pos[0]).fold(f32::INFINITY, f32::min);
            let max_x = v_rot.iter().map(|v| v.pos[0]).fold(f32::NEG_INFINITY, f32::max);
            assert!(min_x < max_x);
        }
    }

    #[test]
    fn falling_node_resolves_block_atlas_uvs_not_fallback() {
        use crate::registry::{DrawType, NodeDef, TileSpec};
        use image::{Rgba, RgbaImage};

        let mut reg = EngineRegistries::new();
        let sand = reg.register_node(NodeDef {
            name: "mcl_core:sand".into(),
            drawtype: DrawType::Normal,
            tiles: vec![TileSpec {
                name: "sand_top.png".into(),
                color: None,
            }],
            ..Default::default()
        });
        let tmp = std::env::temp_dir().join("cubex_falling_node_atlas_test");
        let _ = std::fs::remove_dir_all(&tmp);
        let tex_dir = tmp.join("textures");
        let _ = std::fs::create_dir_all(&tex_dir);
        RgbaImage::from_pixel(16, 16, Rgba([219, 207, 163, 255]))
            .save(tex_dir.join("sand_top.png"))
            .unwrap();
        let atlas = TextureAtlas::build(&std::collections::HashMap::new(), &tmp, None, &reg);

        let ent = EntityRenderSnapshot {
            id: 42,
            entity_name: "__builtin:falling_node".into(),
            pos: Vec3::new(10.5, 65.5, 10.5),
            yaw: 0.0,
            collisionbox: [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5],
            visual: "wielditem".into(),
            visual_size: Vec3::splat(0.667),
            mesh: String::new(),
            texture_name: "mcl_core:sand".into(),
            wield_item: String::new(),
            walk_phase: 0.0,
            walk_blend: 0.0,
            anim_time: 0.0,
            vel_y: 0.0,
            hurt_flash: 0.0,
            sky_light: 15,
            block_light: 0,
        };
        assert_eq!(classify_entity_rig(&ent), EntityRigKind::FallingNode);

        let (layers, tints) =
            resolve_entity_block_layers(&ent, &reg, &atlas).expect("mcl_core:sand must resolve block layers");
        let expected_layers = atlas.node_face_layers[&sand];
        assert_eq!(layers, expected_layers);
        assert!(layers.iter().all(|&l| l > 0), "resolved sand layer must not be fallback 0");

        let (verts, indices) = build_falling_node_mesh(&ent, layers, tints);
        assert_eq!(verts.len(), 24);
        assert_eq!(indices.len(), 36);
        let min_y = verts.iter().map(|v| v.pos[1]).fold(f32::INFINITY, f32::min);
        let max_y = verts.iter().map(|v| v.pos[1]).fold(f32::NEG_INFINITY, f32::max);
        assert!((min_y - 65.0).abs() < 1e-4 && (max_y - 66.0).abs() < 1e-4);
        assert!(verts.iter().all(|v| (v.packed & 0x7FF) == expected_layers[0] as u32));
        // Terrain atlas 32x32 half-texel UVs must stay strictly inset and clear bit 27
        assert!(verts.iter().all(|v| ((v.packed >> 27) & 1) == 0));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn dropped_ram_mic_uses_compound_3d_mesh_with_emissive_led() {
        let ent = EntityRenderSnapshot {
            id: 77,
            entity_name: "__builtin:item".into(),
            pos: Vec3::new(4.0, 10.0, 4.0),
            yaw: 0.25,
            collisionbox: [-0.18, -0.18, -0.18, 0.18, 0.18, 0.18],
            visual: "wielditem".into(),
            visual_size: Vec3::splat(0.67),
            mesh: String::new(),
            texture_name: String::new(),
            wield_item: crate::registry::RAM_MIC_ITEM_ID.into(),
            walk_phase: 0.4,
            walk_blend: 0.0,
            anim_time: 0.4,
            vel_y: 0.0,
            hurt_flash: 0.0,
            sky_light: 14,
            block_light: 0,
        };
        assert_eq!(classify_entity_rig(&ent), EntityRigKind::DroppedItem);

        let (verts, indices) = build_dropped_ram_mic_mesh(&ent);
        assert!(
            verts.len() > 240,
            "Dropped cubex:ram_mic must use full 3D compound mesh, got {} verts",
            verts.len()
        );
        assert!(indices.len() > 360);
        assert!(verts
            .iter()
            .any(|v| (v.packed & VoxelVertex::FLAG_EMISSIVE) != 0));
    }

    #[test]
    fn nether_and_monument_mobs_build_valid_3d_voxel_meshes() {
        let species = [
            ("mobs_mc:pigman", [-0.3, 0.0, -0.3, 0.3, 1.95, 0.3], EntityRigKind::ZombiePiglin, 64, 64),
            ("mobs_mc:witherskeleton", [-0.35, 0.0, -0.35, 0.35, 2.4, 0.35], EntityRigKind::WitherSkeleton, 64, 32),
            ("mobs_mc:blaze", [-0.35, 0.0, -0.35, 0.35, 1.8, 0.35], EntityRigKind::Blaze, 64, 32),
            ("mobs_mc:ghast", [-1.9, 0.0, -1.9, 1.9, 4.0, 1.9], EntityRigKind::Ghast, 64, 32),
            ("mobs_mc:magma_cube", [-0.8, 0.0, -0.8, 0.8, 1.6, 0.8], EntityRigKind::MagmaCube, 128, 64),
            ("mobs_mc:guardian", [-0.45, 0.0, -0.45, 0.45, 0.9, 0.45], EntityRigKind::Guardian, 64, 64),
            ("cubex:fireball", [-0.25, -0.25, -0.25, 0.25, 0.25, 0.25], EntityRigKind::Fireball, 16, 16),
        ];
        for (name, cbox, expected_kind, tw, th) in species {
            let mut mob = mk_mob(name, cbox);
            mob.anim_time = 1.25;
            mob.vel_y = 2.0;
            assert_eq!(classify_entity_rig(&mob), expected_kind, "species={name}");
            let (verts, idx) = build_entity_rig_mesh(&mob, tw, th);
            assert!(
                !verts.is_empty() && idx.len() % 6 == 0,
                "Expected valid indexed mesh for {name}"
            );
            assert!(verts.iter().all(|v| ((v.packed >> 27) & 1) == 1));
            assert!(verts
                .iter()
                .all(|v| v.uv[0] >= 0.0 && v.uv[0] <= 1.0 && v.uv[1] >= 0.0 && v.uv[1] <= 1.0));
        }
    }
}
