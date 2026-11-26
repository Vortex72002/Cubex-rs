use glam::{IVec3, Mat4, Vec3};
use std::time::Instant;

use crate::config::EngineSettings;
use crate::registry::{DrawType, EngineRegistries, NodeBoxSpec, NodeDef, CONTENT_AIR, CONTENT_IGNORE};
use crate::world::{Difficulty, WorldStore};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameMode {
    Survival,
    Creative,
    Spectator,
}

impl GameMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Survival => "survival",
            Self::Creative => "creative",
            Self::Spectator => "spectator",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Survival => "Survival",
            Self::Creative => "Creative",
            Self::Spectator => "Spectator",
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct MoveInput {
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub jump: bool,
    pub sneak: bool,
    pub sprint: bool,
    pub zoom: bool,
    /// Optional analog strafe deflection in `-1.0..=1.0` (`+1.0` = right).
    pub analog_x: f32,
    /// Optional analog forward deflection in `-1.0..=1.0` (`+1.0` = forward).
    pub analog_y: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FallEvent {
    pub damage: i32,
    pub node_id: u16,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct StepEvents {
    pub fall: Option<FallEvent>,
    pub footstep_node: Option<u16>,
    pub regains_ground: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollisionAxis {
    None = -1,
    X = 0,
    Y = 1,
    Z = 2,
}

#[derive(Debug, Clone)]
pub struct CollisionInfo {
    pub axis: CollisionAxis,
    pub node_pos: IVec3,
    pub old_speed: Vec3,
    pub new_speed: Vec3,
    pub new_pos: Vec3,
}

#[derive(Debug, Clone, Default)]
pub struct CollisionMoveResult {
    pub touching_ground: bool,
    pub collides: bool,
    pub collisions: Vec<CollisionInfo>,
}

#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    #[inline]
    pub const fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    #[inline]
    pub fn from_array(b: [f32; 6]) -> Self {
        let min = Vec3::new(b[0].min(b[3]), b[1].min(b[4]), b[2].min(b[5]));
        let max = Vec3::new(b[0].max(b[3]), b[1].max(b[4]), b[2].max(b[5]));
        Self { min, max }
    }

    #[inline]
    pub fn translate(self, offset: Vec3) -> Self {
        Self {
            min: self.min + offset,
            max: self.max + offset,
        }
    }

    #[inline]
    pub fn center(self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    #[inline]
    pub fn intersects(self, other: Self) -> bool {
        self.min.x < other.max.x
            && self.max.x > other.min.x
            && self.min.y < other.max.y
            && self.max.y > other.min.y
            && self.min.z < other.max.z
            && self.max.z > other.min.z
    }
}

#[derive(Debug, Clone)]
struct NearbyBox {
    box_world: Aabb,
    node_pos: IVec3,
    bouncy: i32,
    is_step_up: bool,
}

// =========================================================================
// Luanti Voxel NodeBox / Facedir / Wallmounted / Connected Box Transforms
// =========================================================================

/// Luanti `wallmounted_to_facedir` table (`util/directiontables.cpp`).
const WALLMOUNTED_TO_FACEDIR: [u8; 8] = [20, 0, 17, 15, 8, 6, 21, 1];

#[inline]
fn rotate_xz(v: Vec3, deg: i32) -> Vec3 {
    match deg.rem_euclid(360) {
        90 => Vec3::new(-v.z, v.y, v.x),
        180 => Vec3::new(-v.x, v.y, -v.z),
        270 => Vec3::new(v.z, v.y, -v.x),
        _ => v,
    }
}

#[inline]
fn rotate_xy(v: Vec3, deg: i32) -> Vec3 {
    match deg.rem_euclid(360) {
        90 => Vec3::new(-v.y, v.x, v.z),
        180 => Vec3::new(-v.x, -v.y, v.z),
        270 => Vec3::new(v.y, -v.x, v.z),
        _ => v,
    }
}

#[inline]
fn rotate_yz(v: Vec3, deg: i32) -> Vec3 {
    match deg.rem_euclid(360) {
        90 => Vec3::new(v.x, -v.z, v.y),
        180 => Vec3::new(v.x, -v.y, -v.z),
        270 => Vec3::new(v.x, v.z, -v.y),
        _ => v,
    }
}

#[inline]
fn repair_box(a: Vec3, b: Vec3) -> Aabb {
    Aabb::new(a.min(b), a.max(b))
}

#[inline]
pub(crate) fn is_liquid_node(def: &NodeDef) -> bool {
    def.liquidtype != "none"
        || matches!(def.drawtype, DrawType::Liquid | DrawType::FlowingLiquid)
        || def.liquid_viscosity > 0
        || def.groups.get("liquid").copied().unwrap_or(0) > 0
}

fn node_facedir(def: &NodeDef, param2: u8) -> u8 {
    match def.paramtype2.as_str() {
        "facedir" | "colorfacedir" => (param2 & 0x1F) % 24,
        "4dir" | "color4dir" => param2 & 0x03,
        "wallmounted" | "colorwallmounted" => WALLMOUNTED_TO_FACEDIR[(param2 & 0x07) as usize],
        _ => 0,
    }
}

fn node_wallmounted(def: &NodeDef, param2: u8) -> u8 {
    match def.paramtype2.as_str() {
        "wallmounted" | "colorwallmounted" => param2 & 0x07,
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

/// Port of Luanti `transformNodeBox` (`src/mapnode.cpp`) for `NODEBOX_FIXED` / `NODEBOX_LEVELED`.
fn rotate_facedir_aabb(b: [f32; 6], facedir: u8) -> Aabb {
    let mut p0 = Vec3::new(b[0], b[1], b[2]);
    let mut p1 = Vec3::new(b[3], b[4], b[5]);
    let axisdir = facedir >> 2;
    let rot = facedir & 0x03;

    match axisdir {
        0 => {
            let deg = match rot {
                1 => -90,
                2 => 180,
                3 => 90,
                _ => 0,
            };
            p0 = rotate_xz(p0, deg);
            p1 = rotate_xz(p1, deg);
        }
        1 => {
            p0 = rotate_yz(p0, 90);
            p1 = rotate_yz(p1, 90);
            let deg = match rot {
                1 => 90,
                2 => 180,
                3 => -90,
                _ => 0,
            };
            p0 = rotate_xy(p0, deg);
            p1 = rotate_xy(p1, deg);
        }
        2 => {
            p0 = rotate_yz(p0, -90);
            p1 = rotate_yz(p1, -90);
            let deg = match rot {
                1 => -90,
                2 => 180,
                3 => 90,
                _ => 0,
            };
            p0 = rotate_xy(p0, deg);
            p1 = rotate_xy(p1, deg);
        }
        3 => {
            p0 = rotate_xy(p0, -90);
            p1 = rotate_xy(p1, -90);
            let deg = match rot {
                1 => 90,
                2 => 180,
                3 => -90,
                _ => 0,
            };
            p0 = rotate_yz(p0, deg);
            p1 = rotate_yz(p1, deg);
        }
        4 => {
            p0 = rotate_xy(p0, 90);
            p1 = rotate_xy(p1, 90);
            let deg = match rot {
                1 => -90,
                2 => 180,
                3 => 90,
                _ => 0,
            };
            p0 = rotate_yz(p0, deg);
            p1 = rotate_yz(p1, deg);
        }
        5 => {
            p0 = rotate_xy(p0, -180);
            p1 = rotate_xy(p1, -180);
            let deg = match rot {
                1 => 90,
                2 => 180,
                3 => -90,
                _ => 0,
            };
            p0 = rotate_xz(p0, deg);
            p1 = rotate_xz(p1, deg);
        }
        _ => {}
    }

    repair_box(p0, p1)
}

fn matches_connect_list(target_def: &NodeDef, list: &[String], registries: &EngineRegistries) -> bool {
    if list.is_empty() {
        return false;
    }
    let resolved_name = registries.resolve_alias(&target_def.name);
    for entry in list {
        if let Some(grp) = entry.strip_prefix("group:") {
            if target_def.groups.get(grp).copied().unwrap_or(0) != 0 {
                return true;
            }
        } else {
            let entry_resolved = registries.resolve_alias(entry);
            if entry_resolved == resolved_name || entry == &target_def.name {
                return true;
            }
        }
    }
    false
}

fn connect_sides_mask(sides: &[String]) -> u8 {
    let mut mask = 0u8;
    for s in sides {
        match s.as_str() {
            "top" => mask |= 1,
            "bottom" => mask |= 2,
            "front" => mask |= 4,
            "left" => mask |= 8,
            "back" => mask |= 16,
            "right" => mask |= 32,
            _ => {}
        }
    }
    mask
}

fn nodebox_connects(
    from_def: &NodeDef,
    to_def: &NodeDef,
    to_param2: u8,
    connect_face: u8,
    registries: &EngineRegistries,
) -> bool {
    if !matches_connect_list(to_def, &from_def.connects_to, registries) {
        return false;
    }
    if to_def.drawtype == DrawType::NodeBox && to_def.node_box_type == "connected" {
        return matches_connect_list(from_def, &to_def.connects_to, registries);
    }
    let cs = connect_sides_mask(&to_def.connect_sides);
    if cs > 0 {
        if matches!(
            to_def.paramtype2.as_str(),
            "facedir" | "colorfacedir" | "4dir" | "color4dir"
        ) && connect_face >= 4
        {
            let rot_idx = (to_param2 & 0x03) as usize;
            let rotated = match connect_face {
                4 => [4u8, 32, 16, 8][rot_idx],
                8 => [8u8, 4, 32, 16][rot_idx],
                16 => [16u8, 8, 4, 32][rot_idx],
                32 => [32u8, 16, 8, 4][rot_idx],
                _ => connect_face,
            };
            return (cs & rotated) != 0;
        }
        return (cs & connect_face) != 0;
    }
    true
}

/// Compute Luanti's 6-bit neighbor mask for `NODEBOX_CONNECTED`.
pub(crate) fn compute_connected_neighbors(
    pos: IVec3,
    def: &NodeDef,
    world: &WorldStore,
    registries: &EngineRegistries,
) -> u8 {
    if def.drawtype != DrawType::NodeBox || def.node_box_type != "connected" {
        return 0;
    }
    const OFFSETS: [(IVec3, u8); 6] = [
        (IVec3::Y, 1),     // top
        (IVec3::NEG_Y, 2), // bottom
        (IVec3::NEG_Z, 4), // front
        (IVec3::NEG_X, 8), // left
        (IVec3::Z, 16),    // back
        (IVec3::X, 32),    // right
    ];
    let mut mask = 0u8;
    for (delta, bit) in OFFSETS {
        let (nid, _, np2) = world.get_node(pos + delta);
        if nid == CONTENT_AIR || nid == CONTENT_IGNORE {
            continue;
        }
        if let Some(ndef) = registries.get_node_def(nid) {
            if nodebox_connects(def, ndef, np2, bit, registries) {
                mask |= bit;
            }
        }
    }
    mask
}

pub(crate) fn effective_collision_spec(def: &NodeDef) -> NodeBoxSpec {
    if def.collision_box.box_type != "regular" || !def.collision_box.fixed.is_empty() {
        return def.collision_box.clone();
    }
    if !def.collision_boxes.is_empty() {
        let mut spec = def.node_box_spec();
        spec.fixed = def.collision_boxes.clone();
        if spec.box_type == "regular" {
            spec.box_type = "fixed".into();
        }
        return spec;
    }
    if def.node_box_type != "regular" || !def.node_boxes.is_empty() {
        let mut spec = def.node_box_spec();
        if spec.box_type == "regular" && !spec.fixed.is_empty() {
            spec.box_type = "fixed".into();
        }
        return spec;
    }
    NodeBoxSpec::default()
}

pub(crate) fn effective_selection_spec(def: &NodeDef) -> NodeBoxSpec {
    if def.selection_box.box_type != "regular" || !def.selection_box.fixed.is_empty() {
        return def.selection_box.clone();
    }
    if !def.selection_boxes.is_empty() {
        let mut spec = def.node_box_spec();
        spec.fixed = def.selection_boxes.clone();
        if spec.box_type == "regular" {
            spec.box_type = "fixed".into();
        }
        return spec;
    }
    if def.node_box_type != "regular" || !def.node_boxes.is_empty() {
        let mut spec = def.node_box_spec();
        if spec.box_type == "regular" && !spec.fixed.is_empty() {
            spec.box_type = "fixed".into();
        }
        return spec;
    }
    NodeBoxSpec::default()
}

/// Transform a `NodeBoxSpec` from local `[-0.5, 0.5]` node coordinates to world-space `Aabb`s
/// anchored at voxel `(pos.x, pos.y, pos.z)` spanning `[pos, pos + 1]`.
pub(crate) fn transform_node_box_world(
    pos: IVec3,
    def: &NodeDef,
    spec: &NodeBoxSpec,
    param2: u8,
    neighbors: u8,
) -> Vec<Aabb> {
    let shift = pos.as_vec3() + Vec3::splat(0.5);
    let mut local_boxes: Vec<Aabb> = Vec::new();
    let box_type = if spec.box_type == "regular" && !spec.fixed.is_empty() {
        "fixed"
    } else {
        spec.box_type.as_str()
    };

    match box_type {
        "fixed" | "leveled" => {
            let facedir = node_facedir(def, param2);
            let source = if spec.fixed.is_empty() {
                if box_type == "leveled" {
                    vec![[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]]
                } else {
                    Vec::new()
                }
            } else {
                spec.fixed.clone()
            };
            for mut b in source {
                if box_type == "leveled" {
                    let lvl = if (param2 & 0x7F) > 0 {
                        param2 & 0x7F
                    } else {
                        def.leveled
                    };
                    if lvl > 0 {
                        b[4] = (-0.5 + (lvl as f32) / 64.0).clamp(-0.5, 0.5);
                    }
                }
                local_boxes.push(rotate_facedir_aabb(b, facedir));
            }
        }
        "wallmounted" => {
            let wall = node_wallmounted(def, param2);
            let top_box = spec
                .wall_top
                .or(spec.wall_bottom)
                .or_else(|| spec.fixed.first().copied())
                .unwrap_or([-0.5, 0.0, -0.5, 0.5, 0.5, 0.5]);
            let bot_box = spec
                .wall_bottom
                .or(spec.wall_top)
                .or_else(|| spec.fixed.first().copied())
                .unwrap_or([-0.5, -0.5, -0.5, 0.5, 0.0, 0.5]);
            let side_box = spec
                .wall_side
                .or(spec.wall_bottom)
                .or_else(|| spec.fixed.first().copied())
                .unwrap_or([-0.5, -0.5, -0.5, 0.0, 0.5, 0.5]);

            let (raw, deg) = match wall {
                0 => (top_box, 0),
                6 => (top_box, 90),
                1 => (bot_box, 0),
                7 => (bot_box, -90),
                2 => (side_box, 180),
                3 => (side_box, 0),
                4 => (side_box, -90),
                5 => (side_box, 90),
                _ => (top_box, 0),
            };
            let p0 = rotate_xz(Vec3::new(raw[0], raw[1], raw[2]), deg);
            let p1 = rotate_xz(Vec3::new(raw[3], raw[4], raw[5]), deg);
            local_boxes.push(repair_box(p0, p1));
        }
        "connected" => {
            for b in &spec.fixed {
                local_boxes.push(Aabb::from_array(*b));
            }
            let branches: [(u8, &[ [f32; 6] ], &[ [f32; 6] ]); 6] = [
                (1, &spec.connect_top, &spec.disconnected_top),
                (2, &spec.connect_bottom, &spec.disconnected_bottom),
                (4, &spec.connect_front, &spec.disconnected_front),
                (8, &spec.connect_left, &spec.disconnected_left),
                (16, &spec.connect_back, &spec.disconnected_back),
                (32, &spec.connect_right, &spec.disconnected_right),
            ];
            for (bit, conn, disc) in branches {
                let slice = if (neighbors & bit) != 0 { conn } else { disc };
                for b in slice {
                    local_boxes.push(Aabb::from_array(*b));
                }
            }
            if neighbors == 0 {
                for b in &spec.disconnected_all {
                    local_boxes.push(Aabb::from_array(*b));
                }
            }
            if neighbors < 4 {
                for b in &spec.disconnected_sides {
                    local_boxes.push(Aabb::from_array(*b));
                }
            }
            if local_boxes.is_empty() {
                local_boxes.push(Aabb::from_array([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]));
            }
        }
        _ => {
            local_boxes.push(Aabb::from_array([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]));
        }
    }

    local_boxes
        .into_iter()
        .map(|b| b.translate(shift))
        .collect()
}

pub(crate) fn world_collision_boxes(
    pos: IVec3,
    id: u16,
    param2: u8,
    world: &WorldStore,
    registries: &EngineRegistries,
) -> Vec<Aabb> {
    if id == CONTENT_AIR || id == CONTENT_IGNORE {
        return Vec::new();
    }
    let Some(def) = registries.get_node_def(id) else {
        return Vec::new();
    };
    if !def.walkable {
        return Vec::new();
    }
    let spec = effective_collision_spec(def);
    let neighbors = if spec.box_type == "connected" {
        compute_connected_neighbors(pos, def, world, registries)
    } else {
        0
    };
    transform_node_box_world(pos, def, &spec, param2, neighbors)
}

pub(crate) fn world_selection_boxes_with_neighbors(
    pos: IVec3,
    def: &NodeDef,
    param2: u8,
    neighbors: u8,
) -> Vec<Aabb> {
    let spec = effective_selection_spec(def);
    transform_node_box_world(pos, def, &spec, param2, neighbors)
}

// =========================================================================
// Luanti `axisAlignedCollision` & `collisionMoveSimple`
// =========================================================================

fn axis_aligned_collision(
    static_box: Aabb,
    moving_box: Aabb,
    speed: Vec3,
    dtime: &mut f32,
) -> CollisionAxis {
    let sum_w = (moving_box.max - moving_box.min) + (static_box.max - static_box.min);
    let outer_w = moving_box.max.max(static_box.max) - moving_box.min.min(static_box.min);
    let dtime_max = *dtime;

    if speed.y != 0.0 {
        let distance = outer_w.y - sum_w.y;
        *dtime = distance / speed.y.abs();
        let time = (*dtime).max(0.0);
        if *dtime <= dtime_max {
            let inner_margin = (-0.5 * (static_box.max.y - static_box.min.y)).max(-0.2);
            if (speed.y > 0.0 && static_box.min.y - moving_box.max.y > inner_margin)
                || (speed.y < 0.0 && moving_box.min.y - static_box.max.y > inner_margin)
            {
                let span_x = (moving_box.max.x + speed.x * time).max(static_box.max.x)
                    - (moving_box.min.x + speed.x * time).min(static_box.min.x)
                    - sum_w.x;
                let span_z = (moving_box.max.z + speed.z * time).max(static_box.max.z)
                    - (moving_box.min.z + speed.z * time).min(static_box.min.z)
                    - sum_w.z;
                if span_x < 0.0 && span_z < 0.0 {
                    return CollisionAxis::Y;
                }
            }
        } else {
            return CollisionAxis::None;
        }
    }

    if speed.x != 0.0 {
        let distance = outer_w.x - sum_w.x;
        *dtime = distance / speed.x.abs();
        let time = (*dtime).max(0.0);
        if *dtime <= dtime_max {
            let inner_margin = (-0.5 * (static_box.max.x - static_box.min.x)).max(-0.2);
            // Guard against floor-seam snagging when feet rest on static_box.max.y
            let flush_floor_seam = moving_box.min.y >= static_box.max.y - 1e-4;
            if !flush_floor_seam
                && ((speed.x > 0.0 && static_box.min.x - moving_box.max.x > inner_margin)
                    || (speed.x < 0.0 && moving_box.min.x - static_box.max.x > inner_margin))
            {
                let span_y = (moving_box.max.y + speed.y * time).max(static_box.max.y)
                    - (moving_box.min.y + speed.y * time).min(static_box.min.y)
                    - sum_w.y;
                let span_z = (moving_box.max.z + speed.z * time).max(static_box.max.z)
                    - (moving_box.min.z + speed.z * time).min(static_box.min.z)
                    - sum_w.z;
                if span_y < 0.0 && span_z < 0.0 {
                    return CollisionAxis::X;
                }
            }
        } else {
            return CollisionAxis::None;
        }
    }

    if speed.z != 0.0 {
        let distance = outer_w.z - sum_w.z;
        *dtime = distance / speed.z.abs();
        let time = (*dtime).max(0.0);
        if *dtime <= dtime_max {
            let inner_margin = (-0.5 * (static_box.max.z - static_box.min.z)).max(-0.2);
            let flush_floor_seam = moving_box.min.y >= static_box.max.y - 1e-4;
            if !flush_floor_seam
                && ((speed.z > 0.0 && static_box.min.z - moving_box.max.z > inner_margin)
                    || (speed.z < 0.0 && moving_box.min.z - static_box.max.z > inner_margin))
            {
                let span_x = (moving_box.max.x + speed.x * time).max(static_box.max.x)
                    - (moving_box.min.x + speed.x * time).min(static_box.min.x)
                    - sum_w.x;
                let span_y = (moving_box.max.y + speed.y * time).max(static_box.max.y)
                    - (moving_box.min.y + speed.y * time).min(static_box.min.y)
                    - sum_w.y;
                if span_x < 0.0 && span_y < 0.0 {
                    return CollisionAxis::Z;
                }
            }
        }
    }

    CollisionAxis::None
}

fn would_collide_with_ceiling(
    cinfo: &[NearbyBox],
    step_idx: usize,
    step_box: Aabb,
    y_increase: f32,
    tolerance: f32,
) -> bool {
    if y_increase <= 0.0 {
        return false;
    }
    let shifted_min_y = step_box.min.y + y_increase + tolerance;
    let shifted_max_y = step_box.max.y + y_increase;
    for (i, it) in cinfo.iter().enumerate() {
        if i == step_idx {
            continue;
        }
        let sb = it.box_world;
        let overlap_y = ((step_box.max.y - tolerance <= sb.min.y) && (shifted_max_y > sb.min.y))
            || (shifted_min_y < sb.max.y && shifted_max_y > sb.min.y + tolerance);
        if overlap_y
            && (step_box.min.x + 1e-4 < sb.max.x)
            && (step_box.max.x - 1e-4 > sb.min.x)
            && (step_box.min.z + 1e-4 < sb.max.z)
            && (step_box.max.z - 1e-4 > sb.min.z)
        {
            return true;
        }
    }
    false
}

fn collect_nearby_boxes(
    min_pos: IVec3,
    max_pos: IVec3,
    world: &WorldStore,
    registries: &EngineRegistries,
) -> Vec<NearbyBox> {
    let mut cinfo = Vec::new();
    for z in min_pos.z..=max_pos.z {
        for y in min_pos.y..=max_pos.y {
            for x in min_pos.x..=max_pos.x {
                let p = IVec3::new(x, y, z);
                let (id, _, p2) = world.get_node(p);
                if id == CONTENT_AIR || id == CONTENT_IGNORE {
                    continue;
                }
                let Some(def) = registries.get_node_def(id) else {
                    continue;
                };
                if !def.walkable {
                    continue;
                }
                let bouncy = def.groups.get("bouncy").copied().unwrap_or(0).abs();
                let boxes = world_collision_boxes(p, id, p2, world, registries);
                for box_world in boxes {
                    cinfo.push(NearbyBox {
                        box_world,
                        node_pos: p,
                        bouncy,
                        is_step_up: false,
                    });
                }
            }
        }
    }
    cinfo
}

pub fn collision_move_simple(
    world: &WorldStore,
    registries: &EngineRegistries,
    box_0: Aabb,
    stepheight: f32,
    dtime: f32,
    pos: &mut Vec3,
    speed: &mut Vec3,
    accel: Vec3,
) -> CollisionMoveResult {
    collision_move_simple_ex(
        world,
        registries,
        box_0,
        stepheight,
        dtime,
        pos,
        speed,
        accel,
        1.0,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn collision_move_simple_ex(
    world: &WorldStore,
    registries: &EngineRegistries,
    box_0: Aabb,
    stepheight: f32,
    dtime: f32,
    pos: &mut Vec3,
    speed: &mut Vec3,
    accel: Vec3,
    restitution_scale: f32,
) -> CollisionMoveResult {
    let mut result = CollisionMoveResult::default();
    let mut rem_dt = dtime.clamp(0.0, 0.5);
    if rem_dt <= 0.0 {
        return result;
    }

    let dpos = (*speed + accel * (0.5 * rem_dt)) * rem_dt;
    let new_pos = *pos + dpos;
    *speed += accel * rem_dt;
    if dpos == Vec3::ZERO && *speed == Vec3::ZERO {
        // Still perform ground-touch check below
        let min_p = (*pos + box_0.min - Vec3::ONE).floor().as_ivec3();
        let max_p = (*pos + box_0.max + Vec3::ONE).floor().as_ivec3();
        let cinfo = collect_nearby_boxes(min_p, max_p, world, registries);
        let cur_box = box_0.translate(*pos);
        for info in &cinfo {
            let cb = info.box_world;
            if cb.max.x > cur_box.min.x
                && cb.min.x < cur_box.max.x
                && cb.max.z > cur_box.min.z
                && cb.min.z < cur_box.max.z
                && (cb.max.y - cur_box.min.y).abs() < 0.015
            {
                result.touching_ground = true;
                break;
            }
        }
        return result;
    }

    speed.x = speed.x.clamp(-500.0, 500.0);
    speed.y = speed.y.clamp(-500.0, 500.0);
    speed.z = speed.z.clamp(-500.0, 500.0);

    let min_pos_f = Vec3::new(
        pos.x.min(new_pos.x),
        pos.y.min(new_pos.y) + 0.01,
        pos.z.min(new_pos.z),
    );
    let max_pos_f = pos.max(new_pos);
    let min_i = (min_pos_f + box_0.min).floor().as_ivec3() - IVec3::ONE;
    let max_i = (max_pos_f + box_0.max).floor().as_ivec3() + IVec3::ONE;

    let mut cinfo = collect_nearby_boxes(min_i, max_i, world, registries);
    let ceiling_tol = 0.015;

    let mut loop_count = 0;
    while rem_dt > 1e-8 {
        loop_count += 1;
        if loop_count >= 100 {
            break;
        }

        let moving_box = box_0.translate(*pos);
        let mut nearest_collided = CollisionAxis::None;
        let mut nearest_dtime = rem_dt;
        let mut nearest_idx: Option<usize> = None;

        for (idx, box_info) in cinfo.iter().enumerate() {
            if box_info.is_step_up {
                continue;
            }
            let mut dt_tmp = nearest_dtime;
            let collided = axis_aligned_collision(box_info.box_world, moving_box, *speed, &mut dt_tmp);
            if collided == CollisionAxis::None || dt_tmp >= nearest_dtime {
                continue;
            }
            nearest_dtime = dt_tmp;
            nearest_collided = collided;
            nearest_idx = Some(idx);
        }

        if nearest_collided == CollisionAxis::None {
            *pos += *speed * rem_dt;
            rem_dt = 0.0;
        } else {
            let idx = nearest_idx.expect("nearest box index set");
            let cbox = cinfo[idx].box_world;
            let bouncy = cinfo[idx].bouncy;
            let node_pos = cinfo[idx].node_pos;

            let mut step_box = moving_box;
            step_box.min.x += speed.x * rem_dt;
            step_box.min.z += speed.z * rem_dt;
            step_box.max.x += speed.x * rem_dt;
            step_box.max.z += speed.z * rem_dt;

            let step_up = nearest_collided != CollisionAxis::Y
                && moving_box.min.y < cbox.max.y
                && moving_box.min.y + stepheight > cbox.max.y
                && !would_collide_with_ceiling(
                    &cinfo,
                    idx,
                    step_box,
                    cbox.max.y - moving_box.min.y,
                    ceiling_tol,
                );

            if nearest_dtime < 0.0 {
                if !step_up {
                    match nearest_collided {
                        CollisionAxis::X => pos.x += speed.x * nearest_dtime,
                        CollisionAxis::Y => pos.y += speed.y * nearest_dtime,
                        CollisionAxis::Z => pos.z += speed.z * nearest_dtime,
                        CollisionAxis::None => {}
                    }
                }
            } else {
                *pos += *speed * nearest_dtime;
                rem_dt -= nearest_dtime;
            }

            let old_speed = *speed;
            let eff_bouncy = if restitution_scale <= 0.0 { 0 } else { bouncy };
            let bounce = -(eff_bouncy as f32) / 100.0 * restitution_scale.max(0.0);
            let mut record_collision = true;

            if step_up {
                cinfo[idx].is_step_up = true;
                record_collision = false;
            } else {
                match nearest_collided {
                    CollisionAxis::X => {
                        if speed.x.abs() > 3.0 && eff_bouncy > 0 {
                            speed.x *= bounce;
                        } else {
                            speed.x = 0.0;
                        }
                        result.collides = true;
                    }
                    CollisionAxis::Y => {
                        if speed.y.abs() > 3.0 && eff_bouncy > 0 {
                            speed.y *= bounce;
                        } else {
                            speed.y = 0.0;
                        }
                        result.collides = true;
                    }
                    CollisionAxis::Z => {
                        if speed.z.abs() > 3.0 && eff_bouncy > 0 {
                            speed.z *= bounce;
                        } else {
                            speed.z = 0.0;
                        }
                        result.collides = true;
                    }
                    CollisionAxis::None => {}
                }
            }

            if (*speed - old_speed).length() < 0.1 {
                record_collision = false;
            }
            if record_collision {
                result.collisions.push(CollisionInfo {
                    axis: nearest_collided,
                    node_pos,
                    old_speed,
                    new_speed: *speed,
                    new_pos: *pos,
                });
            }
        }
    }

    // Apply any stepped-up boxes and check ground contact
    let mut final_box = box_0.translate(*pos);
    for info in &cinfo {
        let cb = info.box_world;
        if cb.max.x > final_box.min.x
            && cb.min.x < final_box.max.x
            && cb.max.z > final_box.min.z
            && cb.min.z < final_box.max.z
        {
            if info.is_step_up && cb.max.y > final_box.min.y {
                pos.y += cb.max.y - final_box.min.y;
                final_box = box_0.translate(*pos);
            }
            if (cb.max.y - final_box.min.y).abs() < 0.015 {
                result.touching_ground = true;
            }
        }
    }

    result
}

// =========================================================================
// PlayerController
// =========================================================================

pub struct PlayerController {
    pub pos: Vec3, // feet center
    pub vel: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub on_ground: bool,
    pub flying: bool,
    pub gamemode: GameMode,
    pub hp: i32,
    pub hunger: i32,
    pub selected_slot: usize,
    pub current_fov: f32,
    pub last_space_down: Option<Instant>,
    pub eye_height: f32,

    // Movement telemetry & physics state
    pub horizontal_speed: f32,
    pub step_distance_accum: f32,
    pub footstep_node: Option<u16>,
    pub is_sprinting: bool,
    pub is_sneaking: bool,
    pub in_liquid: bool,
    pub in_liquid_stable: bool,
    pub is_climbing: bool,
    pub camera_impact: f32,
    pub last_fall_event: Option<FallEvent>,
    pub is_dead: bool,
    pub has_dropped_death_items: bool,
    pub death_cause: String,
    pub spawn_invuln_timer: f32,
    pub suffocation_timer: f32,
    pub saturation: f32,
    pub exhaustion: f32,
    pub regen_timer: f32,
    pub starve_timer: f32,
    pub eat_cooldown: f32,

    prev_jump: bool,
    prev_forward: bool,
    last_w_down: Option<Instant>,
    double_tap_sprint: bool,
    max_fall_speed: f32,
    disable_jump: bool,
}

impl Default for PlayerController {
    fn default() -> Self {
        Self {
            pos: Vec3::new(0.5, 24.0, 0.5),
            vel: Vec3::ZERO,
            yaw: 0.0,
            pitch: -0.15,
            on_ground: false,
            flying: false,
            gamemode: GameMode::Survival,
            hp: 20,
            hunger: 20,
            selected_slot: 0,
            current_fov: 80.0,
            last_space_down: None,
            eye_height: Self::EYE_HEIGHT_STANDING,
            horizontal_speed: 0.0,
            step_distance_accum: 0.0,
            footstep_node: None,
            is_sprinting: false,
            is_sneaking: false,
            in_liquid: false,
            in_liquid_stable: false,
            is_climbing: false,
            camera_impact: 0.0,
            last_fall_event: None,
            is_dead: false,
            has_dropped_death_items: false,
            death_cause: String::new(),
            spawn_invuln_timer: 0.0,
            suffocation_timer: 0.0,
            saturation: 5.0,
            exhaustion: 0.0,
            regen_timer: 0.0,
            starve_timer: 0.0,
            eat_cooldown: 0.0,
            prev_jump: false,
            prev_forward: false,
            last_w_down: None,
            double_tap_sprint: false,
            max_fall_speed: 0.0,
            disable_jump: false,
        }
    }
}

impl PlayerController {
    pub const HALF_WIDTH: f32 = 0.30;
    pub const HEIGHT: f32 = 1.75;
    pub const EYE_HEIGHT: f32 = 1.55;
    pub const EYE_HEIGHT_STANDING: f32 = 1.55;
    pub const EYE_HEIGHT_SNEAKING: f32 = 1.38;
    pub const SUFFOCATION_GRACE_SEC: f32 = 0.50;
    pub const STEP_HEIGHT: f32 = 0.60;
    pub const GRAVITY: f32 = 19.62;
    pub const JUMP_SPEED: f32 = 7.45;
    pub const SAFE_FALL_SPEED: f32 = 12.5;

    #[inline]
    pub fn collision_box() -> Aabb {
        Aabb::new(
            Vec3::new(-Self::HALF_WIDTH, 0.0, -Self::HALF_WIDTH),
            Vec3::new(Self::HALF_WIDTH, Self::HEIGHT, Self::HALF_WIDTH),
        )
    }

    pub fn eye_pos(&self) -> Vec3 {
        self.pos + Vec3::new(0.0, self.eye_height, 0.0)
    }

    pub fn add_exhaustion(&mut self, amount: f32) {
        if self.gamemode != GameMode::Survival || self.is_dead || amount <= 0.0 {
            return;
        }
        self.exhaustion += amount;
        while self.exhaustion >= 4.0 {
            self.exhaustion -= 4.0;
            if self.saturation > 0.0 {
                self.saturation = (self.saturation - 1.0).max(0.0);
            } else if self.hunger > 0 {
                self.hunger -= 1;
            }
        }
    }

    pub fn eat_food(&mut self, hunger_gain: i32, saturation_gain: f32, bonus_hp: i32) {
        self.hunger = (self.hunger + hunger_gain).clamp(0, 20);
        self.saturation = (self.saturation + saturation_gain).clamp(0.0, self.hunger as f32);
        if bonus_hp > 0 {
            self.hp = (self.hp + bonus_hp).clamp(0, 20);
        }
        self.eat_cooldown = 0.25;
    }

    /// Returns `true` ONLY when the exact block at the player's eye position
    /// `(pos.x, pos.y + 1.55, pos.z)` is a full opaque solid cube (ignoring air,
    /// slabs, stairs, foliage, water, and half-blocks).
    pub fn is_eye_position_suffocating(
        pos: Vec3,
        world: &WorldStore,
        registries: &EngineRegistries,
    ) -> bool {
        let eye = IVec3::new(
            pos.x.floor() as i32,
            (pos.y + Self::EYE_HEIGHT_STANDING).floor() as i32,
            pos.z.floor() as i32,
        );
        let (nid, _, _) = world.get_node(eye);
        if nid == CONTENT_AIR || nid == CONTENT_IGNORE {
            return false;
        }
        let Some(def) = registries.get_node_def(nid) else {
            return false;
        };
        def.drawtype == DrawType::Normal
            && def.walkable
            && def.is_opaque()
            && (def.node_box_type.is_empty() || def.node_box_type == "regular")
            && def.node_boxes.is_empty()
            && (def.collision_box.box_type.is_empty() || def.collision_box.box_type == "regular")
            && def.collision_boxes.is_empty()
            && def.groups.get("disable_suffocation").copied().unwrap_or(0) == 0
    }

    pub fn look_dir(&self) -> Vec3 {
        let cp = self.pitch.cos();
        Vec3::new(-self.yaw.sin() * cp, self.pitch.sin(), self.yaw.cos() * cp).normalize_or_zero()
    }

    pub fn view_proj(&self, aspect: f32, settings: &EngineSettings) -> Mat4 {
        let fov_rad = self.current_fov.to_radians();
        let proj = Mat4::perspective_rh(
            fov_rad,
            aspect.max(0.1),
            0.1,
            (settings.render_distance_chunks as f32) * 16.0 + 48.0,
        );
        let eye = self.eye_pos();
        let view = Mat4::look_to_rh(eye, self.look_dir(), Vec3::Y);
        proj * view
    }

    pub fn facing_label(&self) -> &'static str {
        let dir = self.look_dir();
        if dir.x.abs() > dir.z.abs() {
            if dir.x >= 0.0 {
                "East (+X)"
            } else {
                "West (-X)"
            }
        } else if dir.z >= 0.0 {
            "South (+Z)"
        } else {
            "North (-Z)"
        }
    }

    pub fn apply_mouse_look(&mut self, dx: f64, dy: f64, settings: &EngineSettings) {
        let sens = settings.mouse_sensitivity * 0.0045;
        // In RH look_to_rh with dir=(-sin(yaw), 0, cos(yaw)), screen-right is +yaw.
        self.yaw += (dx as f32) * sens;
        let sign = if settings.invert_mouse { 1.0 } else { -1.0 };
        self.pitch = (self.pitch + (dy as f32) * sens * sign).clamp(-1.55, 1.55);
    }

    pub fn apply_gamepad_look(&mut self, rx: f32, ry: f32, dt: f32, settings: &EngineSettings) {
        if rx == 0.0 && ry == 0.0 {
            return;
        }
        let rad_per_sec = 2.6 * settings.controller_sensitivity.clamp(0.25, 3.0);
        self.yaw += rx * rad_per_sec * dt;
        // Up on right stick is positive axis (+ry), which looks upward (+pitch) unless inverted.
        let sign = if settings.controller_invert_y { -1.0 } else { 1.0 };
        self.pitch = (self.pitch + ry * rad_per_sec * 0.85 * dt * sign).clamp(-1.55, 1.55);
    }

    fn detect_environment_medium(
        &self,
        world: &WorldStore,
        registries: &EngineRegistries,
    ) -> (bool, bool, u8, bool) {
        let min_x = (self.pos.x - Self::HALF_WIDTH + 0.05).floor() as i32;
        let max_x = (self.pos.x + Self::HALF_WIDTH - 0.05).floor() as i32;
        let min_y = (self.pos.y + 0.05).floor() as i32;
        let max_y = (self.pos.y + Self::HEIGHT - 0.05).floor() as i32;
        let min_z = (self.pos.z - Self::HALF_WIDTH + 0.05).floor() as i32;
        let max_z = (self.pos.z + Self::HALF_WIDTH - 0.05).floor() as i32;

        let mut in_liq = false;
        let mut max_visc = 0u8;
        let mut climbing = false;

        for y in min_y..=max_y {
            for z in min_z..=max_z {
                for x in min_x..=max_x {
                    let (id, _, _) = world.get_node(IVec3::new(x, y, z));
                    if id == CONTENT_AIR || id == CONTENT_IGNORE {
                        continue;
                    }
                    if let Some(def) = registries.get_node_def(id) {
                        if is_liquid_node(def) {
                            in_liq = true;
                            max_visc = max_visc.max(def.liquid_viscosity.max(1));
                        }
                        if def.climbable {
                            climbing = true;
                        }
                    }
                }
            }
        }

        let feet_pos = IVec3::new(
            self.pos.x.floor() as i32,
            self.pos.y.floor() as i32,
            self.pos.z.floor() as i32,
        );
        let (fid, _, _) = world.get_node(feet_pos);
        let in_liq_stable = registries
            .get_node_def(fid)
            .map(is_liquid_node)
            .unwrap_or(false);

        (in_liq, in_liq_stable, max_visc, climbing && !self.flying)
    }

    fn has_ground_support_at(
        &self,
        test_pos: Vec3,
        margin: f32,
        world: &WorldStore,
        registries: &EngineRegistries,
    ) -> bool {
        let pmin_x = test_pos.x - Self::HALF_WIDTH;
        let pmax_x = test_pos.x + Self::HALF_WIDTH;
        let pmin_z = test_pos.z - Self::HALF_WIDTH;
        let pmax_z = test_pos.z + Self::HALF_WIDTH;

        let min_vx = pmin_x.floor() as i32;
        let max_vx = pmax_x.floor() as i32;
        let min_vy = (test_pos.y - 1.0).floor() as i32;
        let max_vy = (test_pos.y + 0.6).floor() as i32;
        let min_vz = pmin_z.floor() as i32;
        let max_vz = pmax_z.floor() as i32;

        for vy in min_vy..=max_vy {
            for vz in min_vz..=max_vz {
                for vx in min_vx..=max_vx {
                    let vp = IVec3::new(vx, vy, vz);
                    let (id, _, p2) = world.get_node(vp);
                    for b in world_collision_boxes(vp, id, p2, world, registries) {
                        if b.max.y >= test_pos.y - 0.06
                            && b.max.y <= test_pos.y + Self::STEP_HEIGHT
                            && b.min.y <= test_pos.y + 0.05
                            && b.max.x > pmin_x + margin
                            && b.min.x < pmax_x - margin
                            && b.max.z > pmin_z + margin
                            && b.min.z < pmax_z - margin
                        {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    fn clamp_sneak_edge_axis(
        &self,
        base_pos: Vec3,
        delta: f32,
        axis: usize,
        world: &WorldStore,
        registries: &EngineRegistries,
    ) -> f32 {
        if delta.abs() < 1e-6 {
            return delta;
        }
        let unit = if axis == 0 { Vec3::X } else { Vec3::Z };
        let margin = 0.003;
        if self.has_ground_support_at(base_pos + unit * delta, margin, world, registries) {
            return delta;
        }
        if !self.has_ground_support_at(base_pos, 0.0, world, registries) {
            return delta;
        }
        let mut lo = 0.0f32;
        let mut hi = delta;
        for _ in 0..16 {
            let mid = (lo + hi) * 0.5;
            if self.has_ground_support_at(base_pos + unit * mid, margin, world, registries) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    fn footstep_node_at(
        &self,
        world: &WorldStore,
        registries: &EngineRegistries,
    ) -> Option<u16> {
        let probe = if self.in_liquid_stable {
            IVec3::new(
                self.pos.x.floor() as i32,
                self.pos.y.floor() as i32,
                self.pos.z.floor() as i32,
            )
        } else {
            IVec3::new(
                self.pos.x.floor() as i32,
                (self.pos.y - 0.05).floor() as i32,
                self.pos.z.floor() as i32,
            )
        };
        let (id, _, _) = world.get_node(probe);
        if id == CONTENT_AIR || id == CONTENT_IGNORE {
            return None;
        }
        if registries.get_node_def(id).is_some() {
            Some(id)
        } else {
            None
        }
    }

    pub fn step(
        &mut self,
        dt: f32,
        input: &MoveInput,
        world: &WorldStore,
        registries: &EngineRegistries,
        settings: &EngineSettings,
    ) -> StepEvents {
        let dt = dt.clamp(0.0001, 0.05);
        let mut events = StepEvents::default();
        self.footstep_node = None;
        self.last_fall_event = None;
        if self.is_dead {
            self.vel = Vec3::ZERO;
            self.horizontal_speed = 0.0;
            self.suffocation_timer = 0.0;
            return events;
        }
        self.spawn_invuln_timer = (self.spawn_invuln_timer - dt).max(0.0);
        self.eat_cooldown = (self.eat_cooldown - dt).max(0.0);

        // 1. Double-tap Space flight toggle in Creative/Spectator
        let just_pressed_jump = input.jump && !self.prev_jump;
        self.prev_jump = input.jump;
        if just_pressed_jump && matches!(self.gamemode, GameMode::Creative | GameMode::Spectator) {
            let now = Instant::now();
            if let Some(prev) = self.last_space_down {
                if now.duration_since(prev).as_millis() <= 280 {
                    self.flying = !self.flying;
                    self.vel.y = 0.0;
                    self.last_space_down = None;
                } else {
                    self.last_space_down = Some(now);
                }
            } else {
                self.last_space_down = Some(now);
            }
        }
        if self.gamemode == GameMode::Spectator {
            self.flying = true;
        } else if self.gamemode == GameMode::Survival {
            self.flying = false;
        }

        // 2. Double-tap W (280ms) or Ctrl (`aux1`) sprint detection
        let fwd_active = input.forward || input.analog_y > 0.25;
        let just_pressed_forward = fwd_active && !self.prev_forward;
        self.prev_forward = fwd_active;
        if just_pressed_forward {
            let now = Instant::now();
            if let Some(prev) = self.last_w_down {
                if now.duration_since(prev).as_millis() <= 280 {
                    self.double_tap_sprint = true;
                    self.last_w_down = None;
                } else {
                    self.last_w_down = Some(now);
                }
            } else {
                self.last_w_down = Some(now);
            }
        }
        if !fwd_active || input.sneak {
            self.double_tap_sprint = false;
        }

        self.is_sneaking = input.sneak && !self.flying;
        let can_sprint = self.gamemode != GameMode::Survival || self.hunger > 6;
        self.is_sprinting = (input.sprint || self.double_tap_sprint)
            && fwd_active
            && !self.is_sneaking
            && can_sprint;

        // 3. Smooth eye height interpolation (1.625 standing -> 1.45 sneaking)
        let target_eye = if self.is_sneaking {
            Self::EYE_HEIGHT_SNEAKING
        } else {
            Self::EYE_HEIGHT_STANDING
        };
        let eye_lerp = (dt * 18.0).clamp(0.0, 1.0);
        self.eye_height += (target_eye - self.eye_height) * eye_lerp;

        // 4. Smooth C-Key Optical Zoom & Sprint FOV interpolation (+8 deg)
        let target_fov = if input.zoom {
            24.0
        } else if self.is_sprinting {
            (settings.fov + 8.0).min(115.0)
        } else {
            settings.fov
        };
        let fov_lerp = (dt * 14.0).clamp(0.0, 1.0);
        self.current_fov += (target_fov - self.current_fov) * fov_lerp;

        // 5. Medium detection (liquids & climbables)
        let (in_liq, in_liq_stable, liquid_visc, climbing) =
            self.detect_environment_medium(world, registries);
        self.in_liquid = in_liq;
        self.in_liquid_stable = in_liq_stable;
        self.is_climbing = climbing;

        // Standing node properties (`disable_jump`, `slippery`)
        let stand_pos = IVec3::new(
            self.pos.x.floor() as i32,
            (self.pos.y - 0.05).floor() as i32,
            self.pos.z.floor() as i32,
        );
        let at_feet_pos = IVec3::new(
            self.pos.x.floor() as i32,
            (self.pos.y + 0.05).floor() as i32,
            self.pos.z.floor() as i32,
        );
        let (stand_id, _, _) = world.get_node(stand_pos);
        let (feet_id, _, _) = world.get_node(at_feet_pos);
        let stand_def = registries.get_node_def(stand_id);
        let feet_def = registries.get_node_def(feet_id);
        self.disable_jump = stand_def
            .and_then(|d| d.groups.get("disable_jump").copied())
            .unwrap_or(0)
            != 0
            || feet_def
                .and_then(|d| d.groups.get("disable_jump").copied())
                .unwrap_or(0)
                != 0;
        let slippery = stand_def
            .filter(|d| d.walkable)
            .and_then(|d| d.groups.get("slippery").copied())
            .unwrap_or(0);

        // 6. Horizontal wish direction (screen-right = flat_forward x Y), combining digital + analog sticks
        let flat_forward = Vec3::new(-self.yaw.sin(), 0.0, self.yaw.cos()).normalize_or_zero();
        let flat_right = flat_forward.cross(Vec3::Y).normalize_or_zero();
        let mut dz_in = (if input.forward { 1.0 } else { 0.0 })
            - (if input.backward { 1.0 } else { 0.0 })
            + input.analog_y;
        let mut dx_in = (if input.right { 1.0 } else { 0.0 })
            - (if input.left { 1.0 } else { 0.0 })
            + input.analog_x;
        let mag = (dx_in * dx_in + dz_in * dz_in).sqrt();
        if mag > 1.0 {
            dx_in /= mag;
            dz_in /= mag;
        }
        let wish = flat_forward * dz_in + flat_right * dx_in;

        let visc_scale = if self.in_liquid {
            1.0 / (1.0 + 0.35 * (liquid_visc.saturating_sub(1) as f32))
        } else {
            1.0
        };

        let base_speed = if self.flying {
            if input.sprint {
                18.0
            } else {
                10.5
            }
        } else if self.in_liquid {
            let s = if self.is_sprinting { 3.6 } else { 2.6 };
            s * visc_scale
        } else if self.is_climbing {
            3.0
        } else if self.is_sneaking {
            1.35
        } else if self.is_sprinting {
            6.2
        } else {
            4.3
        };

        let target_hx = wish.x * base_speed;
        let target_hz = wish.z * base_speed;
        if slippery >= 1 && self.on_ground && !self.flying && !self.in_liquid {
            let accel_t = (dt * (4.0 / (slippery as f32 + 1.0))).clamp(0.0, 1.0);
            self.vel.x += (target_hx - self.vel.x) * accel_t;
            self.vel.z += (target_hz - self.vel.z) * accel_t;
        } else if self.in_liquid {
            let drag_t = (dt * 7.5 * visc_scale).clamp(0.0, 1.0);
            self.vel.x += (target_hx - self.vel.x) * drag_t;
            self.vel.z += (target_hz - self.vel.z) * drag_t;
        } else {
            self.vel.x = target_hx;
            self.vel.z = target_hz;
        }

        // 7. Vertical physics (Flying / Spectator / Liquid / Climbing / Gravity + Jump)
        let was_on_ground = self.on_ground;
        let pre_step_pos = self.pos;

        let mut jumped_this_tick = false;
        if self.flying {
            let mut vy = 0.0;
            if input.jump {
                vy += base_speed * 0.85;
            }
            if input.sneak {
                vy -= base_speed * 0.85;
            }
            self.vel.y = vy;
            self.max_fall_speed = 0.0;
            if self.gamemode == GameMode::Spectator {
                self.pos += self.vel * dt;
                self.on_ground = false;
                self.horizontal_speed =
                    Vec3::new(self.pos.x - pre_step_pos.x, 0.0, self.pos.z - pre_step_pos.z).length() / dt;
                return events;
            }
        } else if self.is_climbing {
            self.max_fall_speed = 0.0;
            if input.jump && !self.disable_jump {
                self.vel.y = 3.2;
            } else if input.sneak {
                self.vel.y = -3.2;
            } else {
                self.vel.y = 0.0;
            }
        } else if self.in_liquid {
            self.max_fall_speed = 0.0;
            let target_vy = if input.jump && !self.disable_jump {
                4.2 * visc_scale
            } else if input.sneak {
                -4.2 * visc_scale
            } else {
                // Gentle buoyancy / liquid sink instead of full gravity
                -1.6 * visc_scale
            };
            let drag_y = (dt * 9.0 * visc_scale).clamp(0.0, 1.0);
            self.vel.y += (target_vy - self.vel.y) * drag_y;
        } else {
            if input.jump && self.on_ground && !self.disable_jump {
                self.vel.y = Self::JUMP_SPEED;
                self.on_ground = false;
                self.max_fall_speed = 0.0;
                jumped_this_tick = true;
            }
        }

        // 8. Sneak edge-catch prevention (independent X and Z clamping so sliding along ledges works!)
        let jumping_this_step = !self.on_ground && self.vel.y > 0.0;
        if was_on_ground
            && self.is_sneaking
            && !self.flying
            && !self.in_liquid
            && !self.is_climbing
            && !jumping_this_step
        {
            let dx = self.vel.x * dt;
            let safe_dx = self.clamp_sneak_edge_axis(self.pos, dx, 0, world, registries);
            if (safe_dx - dx).abs() > 1e-5 {
                self.vel.x = safe_dx / dt;
            }
            let pos_after_x = self.pos + Vec3::new(safe_dx, 0.0, 0.0);
            let dz = self.vel.z * dt;
            let safe_dz = self.clamp_sneak_edge_axis(pos_after_x, dz, 1, world, registries);
            if (safe_dz - dz).abs() > 1e-5 {
                self.vel.z = safe_dz / dt;
            }
        }

        // 9. Swept AABB Collision Resolution (`collisionMoveSimple`) with 0.6-block `stepheight`
        let accel = if self.flying || self.is_climbing || self.in_liquid {
            Vec3::ZERO
        } else {
            Vec3::new(0.0, -Self::GRAVITY, 0.0)
        };
        let stepheight = if self.flying || self.in_liquid {
            0.0
        } else if was_on_ground {
            Self::STEP_HEIGHT
        } else {
            0.20
        };

        if !self.on_ground && !self.flying && !self.in_liquid && !self.is_climbing && self.vel.y < 0.0 {
            self.max_fall_speed = self.max_fall_speed.max(-self.vel.y);
        }

        let initial_pos = self.pos;
        let initial_vel = self.vel;
        let move_res = collision_move_simple(
            world,
            registries,
            Self::collision_box(),
            stepheight,
            dt,
            &mut self.pos,
            &mut self.vel,
            accel,
        );

        // Also track peak downward velocity across the frame if accelerated by gravity
        if !move_res.touching_ground && !self.flying && !self.in_liquid && !self.is_climbing && self.vel.y < 0.0 {
            self.max_fall_speed = self.max_fall_speed.max(-self.vel.y);
        }

        // Re-verify sneak edge guard after swept collision in case stepheight shifted position
        if was_on_ground
            && self.is_sneaking
            && !self.flying
            && !self.in_liquid
            && !self.is_climbing
            && !jumping_this_step
            && !move_res.touching_ground
            && self.has_ground_support_at(initial_pos, 0.0, world, registries)
        {
            self.pos = initial_pos;
            self.vel.x = 0.0;
            self.vel.y = 0.0;
            self.vel.z = 0.0;
            self.on_ground = true;
        } else {
            self.on_ground = move_res.touching_ground;
        }

        if self.flying && self.on_ground && initial_vel.y < 0.0 {
            self.flying = false;
        }

        // 10. Optional full-block `auto_jump` (separate from 0.6-block `stepheight`)
        if settings.auto_jump
            && was_on_ground
            && !self.flying
            && !input.sneak
            && !input.jump
            && !self.in_liquid
            && !self.is_climbing
            && !self.disable_jump
            && wish.length_squared() > 0.01
        {
            let hit_horiz = move_res
                .collisions
                .iter()
                .any(|c| matches!(c.axis, CollisionAxis::X | CollisionAxis::Z));
            if hit_horiz {
                let headroom_clear = !self.collides_world(
                    self.pos + Vec3::new(0.0, 1.05, 0.0),
                    world,
                    registries,
                );
                if headroom_clear {
                    let test_pos = self.pos + Vec3::new(wish.x * 0.35, 1.05, wish.z * 0.35);
                    if !self.collides_world(test_pos, world, registries) {
                        self.vel.y = Self::JUMP_SPEED;
                        self.on_ground = false;
                    }
                }
            }
        }

        // 11. Fall Damage on landing transition (`!was_on_ground -> on_ground`)
        if !was_on_ground && self.on_ground {
            events.regains_ground = true;
            let impact_speed = self.max_fall_speed.max(
                move_res
                    .collisions
                    .iter()
                    .filter(|c| c.axis == CollisionAxis::Y && c.old_speed.y < 0.0)
                    .map(|c| -c.old_speed.y)
                    .fold(0.0f32, f32::max),
            );
            self.camera_impact = impact_speed;

            if !self.in_liquid
                && !self.is_climbing
                && !self.flying
                && self.gamemode == GameMode::Survival
                && self.spawn_invuln_timer <= 0.0
                && world.gamerules.read().fall_damage
            {
                let landing_node_pos = move_res
                    .collisions
                    .iter()
                    .find(|c| c.axis == CollisionAxis::Y && c.old_speed.y < 0.0)
                    .map(|c| c.node_pos)
                    .unwrap_or_else(|| {
                        IVec3::new(
                            self.pos.x.floor() as i32,
                            (self.pos.y - 0.05).floor() as i32,
                            self.pos.z.floor() as i32,
                        )
                    });
                let (landing_id, _, _) = world.get_node(landing_node_pos);
                let add_percent = registries
                    .get_node_def(landing_id)
                    .and_then(|d| d.groups.get("fall_damage_add_percent").copied())
                    .unwrap_or(0);
                let factor = (1.0 + add_percent as f32 / 100.0).max(0.0);
                let effective_speed = impact_speed * factor;

                if effective_speed > Self::SAFE_FALL_SPEED && factor > 0.0 {
                    let dmg = ((effective_speed - Self::SAFE_FALL_SPEED) + 0.5).floor() as i32;
                    if dmg > 0 {
                        self.hp = (self.hp - dmg).max(0);
                        if self.hp == 0 {
                            self.death_cause = "Fell from a high place".into();
                        }
                        let fe = FallEvent {
                            damage: dmg,
                            node_id: landing_id,
                        };
                        self.last_fall_event = Some(fe);
                        events.fall = Some(fe);
                    }
                }
            }
            self.max_fall_speed = 0.0;
            if let Some(nid) = self.footstep_node_at(world, registries) {
                events.footstep_node = Some(nid);
                self.footstep_node = Some(nid);
            }
        } else if self.on_ground || self.flying || self.in_liquid || self.is_climbing {
            self.max_fall_speed = 0.0;
        }

        // Exact eye-position suffocation check with 0.5s grace period
        if self.gamemode == GameMode::Survival && !self.is_dead {
            if Self::is_eye_position_suffocating(self.pos, world, registries) {
                if self.spawn_invuln_timer <= 0.0 {
                    self.suffocation_timer += dt;
                    if self.suffocation_timer >= Self::SUFFOCATION_GRACE_SEC {
                        self.suffocation_timer -= Self::SUFFOCATION_GRACE_SEC;
                        self.hp = (self.hp - 1).max(0);
                        if self.hp == 0 {
                            self.death_cause = "Suffocated in a wall".into();
                        }
                    }
                }
            } else {
                self.suffocation_timer = 0.0;
            }
        } else {
            self.suffocation_timer = 0.0;
        }

        // 12. Horizontal movement telemetry, exhaustion, regen/starvation & footsteps
        let disp = self.pos - pre_step_pos;
        let horiz_dist = Vec3::new(disp.x, 0.0, disp.z).length();
        self.horizontal_speed = horiz_dist / dt;

        if self.gamemode == GameMode::Survival && !self.is_dead {
            if jumped_this_tick {
                self.add_exhaustion(if self.is_sprinting { 0.20 } else { 0.05 });
            }
            if self.is_sprinting && horiz_dist > 0.0 {
                self.add_exhaustion(0.10 * horiz_dist);
            } else if self.in_liquid && horiz_dist > 0.0 {
                self.add_exhaustion(0.015 * horiz_dist);
            }

            if settings.difficulty == Difficulty::Peaceful {
                self.starve_timer = 0.0;
                if self.hunger < 20 {
                    self.regen_timer += dt;
                    if self.regen_timer >= 1.5 {
                        self.regen_timer -= 1.5;
                        self.hunger = (self.hunger + 1).min(20);
                        self.saturation = self.saturation.max(3.0).min(self.hunger as f32);
                    }
                } else if self.hp > 0 && self.hp < 20 {
                    self.regen_timer += dt;
                    if self.regen_timer >= 2.0 {
                        self.regen_timer -= 2.0;
                        self.hp = (self.hp + 1).min(20);
                    }
                } else {
                    self.regen_timer = 0.0;
                }
            } else {
                if self.hunger >= 18 && self.hp > 0 && self.hp < 20 {
                    self.regen_timer += dt;
                    if self.regen_timer >= 2.5 {
                        self.regen_timer -= 2.5;
                        self.hp = (self.hp + 1).min(20);
                        self.add_exhaustion(3.0);
                    }
                } else {
                    self.regen_timer = 0.0;
                }

                if self.hunger == 0 && self.hp > 0 && self.spawn_invuln_timer <= 0.0 {
                    let min_hp = if settings.difficulty == Difficulty::Hard { 0 } else { 1 };
                    if self.hp > min_hp {
                        self.starve_timer += dt;
                        if self.starve_timer >= 4.0 {
                            self.starve_timer -= 4.0;
                            self.hp = (self.hp - 1).max(min_hp);
                            if self.hp == 0 {
                                self.death_cause = "Starved to death".into();
                            }
                        }
                    } else {
                        self.starve_timer = 0.0;
                    }
                } else {
                    self.starve_timer = 0.0;
                }
            }
        } else {
            self.regen_timer = 0.0;
            self.starve_timer = 0.0;
        }

        if (self.on_ground || self.in_liquid_stable) && !self.flying && !self.is_climbing {
            self.step_distance_accum += horiz_dist;
            let stride = if self.is_sneaking { 1.5 } else { 1.85 };
            if self.step_distance_accum >= stride {
                self.step_distance_accum %= stride;
                if let Some(nid) = self.footstep_node_at(world, registries) {
                    self.footstep_node = Some(nid);
                    events.footstep_node = Some(nid);
                }
            }
        }

        events
    }

    pub fn collides_world(
        &self,
        pos: Vec3,
        world: &WorldStore,
        registries: &EngineRegistries,
    ) -> bool {
        let pbox = Self::collision_box().translate(pos);
        let min_x = pbox.min.x.floor() as i32;
        let max_x = pbox.max.x.floor() as i32;
        let min_y = pbox.min.y.floor() as i32;
        let max_y = (pbox.max.y - 1e-4).floor() as i32;
        let min_z = pbox.min.z.floor() as i32;
        let max_z = pbox.max.z.floor() as i32;

        for y in min_y..=max_y {
            for z in min_z..=max_z {
                for x in min_x..=max_x {
                    let vp = IVec3::new(x, y, z);
                    let (id, _, p2) = world.get_node(vp);
                    for cb in world_collision_boxes(vp, id, p2, world, registries) {
                        if pbox.intersects(cb) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
}
