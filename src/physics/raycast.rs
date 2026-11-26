use glam::{IVec3, Vec3};
use mlua::Table;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

use crate::lua_api::userdata::{ItemStackData, ObjectState};
use crate::lua_api::LuaRuntime;
use crate::registry::{
    compute_dig_params_with_wear, DrawType, EngineRegistries, NodeDef, SoundSpec,
    ToolCapabilities, CONTENT_AIR, CONTENT_IGNORE, CONTENT_UNKNOWN,
};
use crate::world::light::relight_around_pos;
use crate::world::portal::{extinguish_portal_around, is_obsidian, is_portal_igniter, try_ignite_portal};
use crate::world::{DimensionId, WorldStore};

use super::collision::{
    compute_connected_neighbors, is_liquid_node, world_collision_boxes,
    world_selection_boxes_with_neighbors, Aabb, GameMode, PlayerController,
};

#[derive(Debug, Clone)]
pub struct RaycastHit {
    /// Hit voxel coordinate (alias of `under`, kept for backwards compatibility).
    pub pos: IVec3,
    /// Adjacent placement voxel coordinate (alias of `above`, kept for backwards compatibility).
    pub prev_pos: IVec3,
    /// Integer outward normal of hit box face.
    pub normal: IVec3,
    /// Hit voxel content ID.
    pub node_id: u16,
    /// Luanti `pointed_thing.under` (hit voxel).
    pub under: IVec3,
    /// Luanti `pointed_thing.above` (adjacent placement voxel).
    pub above: IVec3,
    /// Exact world-space intersection point on the selection box.
    pub intersection_point: Vec3,
    /// Unit outward normal at the intersection point.
    pub intersection_normal: Vec3,
    /// Index of the hit selection sub-box.
    pub box_id: usize,
    /// Squared distance from ray origin to `intersection_point`.
    pub distance_sq: f32,
    /// `param2` of the hit node (used for rotated selection wireframe rendering).
    pub param2: u8,
    /// Connected-node neighbor bitmask of the hit node.
    pub neighbors: u8,
}

/// Luanti `boxLineCollision` (`src/raycast.cpp`): tests ray segment `[start, start + dir]`
/// against AABB `[box_min, box_max]`. Returns `(intersection_point, intersection_normal)`.
pub fn box_line_collision(
    box_min: Vec3,
    box_max: Vec3,
    start: Vec3,
    dir: Vec3,
) -> Option<(Vec3, Vec3)> {
    if start.x >= box_min.x
        && start.x <= box_max.x
        && start.y >= box_min.y
        && start.y <= box_max.y
        && start.z >= box_min.z
        && start.z <= box_max.z
    {
        return Some((start, Vec3::ZERO));
    }

    let eps = 1e-5f32;
    let mut best_m = f32::INFINITY;
    let mut best_pt = Vec3::ZERO;
    let mut best_n = Vec3::ZERO;

    if dir.x.abs() > 1e-8 {
        let plane = if dir.x > 0.0 { box_min.x } else { box_max.x };
        let m = (plane - start.x) / dir.x;
        if (0.0..=1.0).contains(&m) && m < best_m {
            let pt = start + dir * m;
            if pt.y >= box_min.y - eps
                && pt.y <= box_max.y + eps
                && pt.z >= box_min.z - eps
                && pt.z <= box_max.z + eps
            {
                best_m = m;
                best_pt = Vec3::new(
                    plane,
                    pt.y.clamp(box_min.y, box_max.y),
                    pt.z.clamp(box_min.z, box_max.z),
                );
                best_n = Vec3::new(if dir.x > 0.0 { -1.0 } else { 1.0 }, 0.0, 0.0);
            }
        }
    }

    if dir.y.abs() > 1e-8 {
        let plane = if dir.y > 0.0 { box_min.y } else { box_max.y };
        let m = (plane - start.y) / dir.y;
        if (0.0..=1.0).contains(&m) && m < best_m {
            let pt = start + dir * m;
            if pt.x >= box_min.x - eps
                && pt.x <= box_max.x + eps
                && pt.z >= box_min.z - eps
                && pt.z <= box_max.z + eps
            {
                best_m = m;
                best_pt = Vec3::new(
                    pt.x.clamp(box_min.x, box_max.x),
                    plane,
                    pt.z.clamp(box_min.z, box_max.z),
                );
                best_n = Vec3::new(0.0, if dir.y > 0.0 { -1.0 } else { 1.0 }, 0.0);
            }
        }
    }

    if dir.z.abs() > 1e-8 {
        let plane = if dir.z > 0.0 { box_min.z } else { box_max.z };
        let m = (plane - start.z) / dir.z;
        if (0.0..=1.0).contains(&m) && m < best_m {
            let pt = start + dir * m;
            if pt.x >= box_min.x - eps
                && pt.x <= box_max.x + eps
                && pt.y >= box_min.y - eps
                && pt.y <= box_max.y + eps
            {
                best_m = m;
                best_pt = Vec3::new(
                    pt.x.clamp(box_min.x, box_max.x),
                    pt.y.clamp(box_min.y, box_max.y),
                    plane,
                );
                best_n = Vec3::new(0.0, 0.0, if dir.z > 0.0 { -1.0 } else { 1.0 });
            }
        }
    }

    if best_m <= 1.0 {
        Some((best_pt, best_n))
    } else {
        None
    }
}

#[inline]
fn is_node_pointable(def: &NodeDef, liquids_pointable: bool) -> bool {
    if is_liquid_node(def) {
        return liquids_pointable || def.pointable;
    }
    def.pointable
}

/// Exact 3D DDA traversal state matching Luanti `RaycastState`.
pub struct RaycastState {
    pub origin: Vec3,
    pub dir: Vec3,
    pub max_dist: f32,
    pub liquids_pointable: bool,
    cell: IVec3,
    prev_cell: IVec3,
    step: IVec3,
    t_max: Vec3,
    t_delta: Vec3,
    dist: f32,
    entry_normal: IVec3,
    done: bool,
}

impl RaycastState {
    pub const DEFAULT_REACH: f32 = 5.0;

    pub fn new(origin: Vec3, dir: Vec3, max_dist: f32, liquids_pointable: bool) -> Self {
        let d = dir.normalize_or_zero();
        let valid = d.length_squared() >= 1e-8 && max_dist > 0.0;
        let cell = origin.floor().as_ivec3();
        let step = IVec3::new(
            if d.x >= 0.0 { 1 } else { -1 },
            if d.y >= 0.0 { 1 } else { -1 },
            if d.z >= 0.0 { 1 } else { -1 },
        );
        let t_delta = Vec3::new(
            if d.x.abs() > 1e-7 { (1.0 / d.x).abs() } else { 1e30 },
            if d.y.abs() > 1e-7 { (1.0 / d.y).abs() } else { 1e30 },
            if d.z.abs() > 1e-7 { (1.0 / d.z).abs() } else { 1e30 },
        );
        let next_bound = |o: f32, s: i32, c: i32| -> f32 {
            if s > 0 {
                (c + 1) as f32 - o
            } else {
                o - c as f32
            }
        };
        let t_max = Vec3::new(
            next_bound(origin.x, step.x, cell.x) * t_delta.x,
            next_bound(origin.y, step.y, cell.y) * t_delta.y,
            next_bound(origin.z, step.z, cell.z) * t_delta.z,
        );

        Self {
            origin,
            dir: d,
            max_dist,
            liquids_pointable,
            cell,
            prev_cell: cell,
            step,
            t_max,
            t_delta,
            dist: 0.0,
            entry_normal: IVec3::ZERO,
            done: !valid,
        }
    }

    pub fn next_hit(
        &mut self,
        world: &WorldStore,
        registries: &EngineRegistries,
    ) -> Option<RaycastHit> {
        if self.done {
            return None;
        }
        let ray_vec = self.dir * self.max_dist;

        while self.dist <= self.max_dist {
            let cur = self.cell;
            let prev = self.prev_cell;
            let entry_n = self.entry_normal;

            // Advance DDA to next cell before returning or continuing
            self.prev_cell = self.cell;
            if self.t_max.x < self.t_max.y && self.t_max.x < self.t_max.z {
                self.dist = self.t_max.x;
                self.t_max.x += self.t_delta.x;
                self.cell.x += self.step.x;
                self.entry_normal = IVec3::new(-self.step.x, 0, 0);
            } else if self.t_max.y < self.t_max.z {
                self.dist = self.t_max.y;
                self.t_max.y += self.t_delta.y;
                self.cell.y += self.step.y;
                self.entry_normal = IVec3::new(0, -self.step.y, 0);
            } else {
                self.dist = self.t_max.z;
                self.t_max.z += self.t_delta.z;
                self.cell.z += self.step.z;
                self.entry_normal = IVec3::new(0, 0, -self.step.z);
            }

            let (id, _, p2) = world.get_node(cur);
            if id == CONTENT_AIR || id == CONTENT_IGNORE {
                continue;
            }
            let Some(def) = registries.get_node_def(id) else {
                continue;
            };
            if !is_node_pointable(def, self.liquids_pointable) {
                continue;
            }

            let neighbors = compute_connected_neighbors(cur, def, world, registries);
            let boxes = world_selection_boxes_with_neighbors(cur, def, p2, neighbors);

            let mut best_dist_sq = f32::INFINITY;
            let mut best_hit: Option<(Vec3, Vec3, usize)> = None;

            for (box_id, b) in boxes.iter().enumerate() {
                if let Some((ipt, inorm)) = box_line_collision(b.min, b.max, self.origin, ray_vec) {
                    let dsq = ipt.distance_squared(self.origin);
                    if dsq < best_dist_sq {
                        best_dist_sq = dsq;
                        best_hit = Some((ipt, inorm, box_id));
                    }
                }
            }

            if let Some((ipt, inorm, box_id)) = best_hit {
                let normal = if inorm.length_squared() > 0.5 {
                    inorm.round().as_ivec3()
                } else {
                    entry_n
                };
                let under = cur;
                let above = if normal != IVec3::ZERO {
                    cur + normal
                } else {
                    prev
                };
                return Some(RaycastHit {
                    pos: under,
                    prev_pos: above,
                    normal,
                    node_id: id,
                    under,
                    above,
                    intersection_point: ipt,
                    intersection_normal: inorm,
                    box_id,
                    distance_sq: best_dist_sq,
                    param2: p2,
                    neighbors,
                });
            }
        }

        self.done = true;
        None
    }
}

/// Result of intersecting an attack/interaction ray with an active LuaEntity AABB.
#[derive(Debug, Clone)]
pub struct EntityHit {
    pub entity_id: u64,
    pub entity_name: String,
    pub intersection_point: Vec3,
    pub intersection_normal: Vec3,
    pub distance_sq: f32,
}

/// Fast 3D DDA voxel raycast with sub-voxel selection box intersection (`liquids_pointable = false`).
pub fn raycast_voxel(
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
    world: &WorldStore,
    registries: &EngineRegistries,
) -> Option<RaycastHit> {
    raycast_voxel_ex(origin, dir, max_dist, false, world, registries)
}

/// Raycast up to `max_dist` blocks against active pointable non-item entities.
pub fn raycast_entities(
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
    entities: &[Arc<RwLock<ObjectState>>],
) -> Option<EntityHit> {
    let dir_n = dir.normalize_or_zero();
    if max_dist <= 0.0 || dir_n.length_squared() < 1e-6 {
        return None;
    }
    let ray_vec = dir_n * max_dist;
    let max_dist_sq = max_dist * max_dist;
    let mut best: Option<EntityHit> = None;

    for arc in entities {
        let g = arc.read();
        if g.removed
            || g.is_player
            || !g.pointable
            || !g.is_visible
            || g.entity_name == "__builtin:item"
        {
            continue;
        }
        let s = g.selectionbox;
        let sb_valid =
            (s[3] - s[0]).abs() > 0.01 && (s[4] - s[1]).abs() > 0.01 && (s[5] - s[2]).abs() > 0.01;
        let raw = if sb_valid { s } else { g.collisionbox };
        if (raw[3] - raw[0]).abs() <= 0.005
            || (raw[4] - raw[1]).abs() <= 0.005
            || (raw[5] - raw[2]).abs() <= 0.005
        {
            continue;
        }
        let aabb = Aabb::from_array(raw).translate(g.pos);
        if let Some((ipt, inorm)) = box_line_collision(aabb.min, aabb.max, origin, ray_vec) {
            let dsq = ipt.distance_squared(origin);
            if dsq <= max_dist_sq && best.as_ref().map_or(true, |b| dsq < b.distance_sq) {
                best = Some(EntityHit {
                    entity_id: g.id,
                    entity_name: g.entity_name.clone(),
                    intersection_point: ipt,
                    intersection_normal: inorm,
                    distance_sq: dsq,
                });
            }
        }
    }
    best
}

/// Raycast up to `max_dist` blocks against a single live peer player (co-op PvP).
pub fn raycast_player(
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
    target: &Arc<RwLock<ObjectState>>,
) -> Option<EntityHit> {
    let dir_n = dir.normalize_or_zero();
    if max_dist <= 0.0 || dir_n.length_squared() < 1e-6 {
        return None;
    }
    let g = target.read();
    if g.removed || !g.is_player {
        return None;
    }
    let aabb = Aabb::from_array([-0.32, 0.0, -0.32, 0.32, 1.82, 0.32]).translate(g.pos);
    let ray_vec = dir_n * max_dist;
    let (ipt, inorm) = box_line_collision(aabb.min, aabb.max, origin, ray_vec)?;
    let dsq = ipt.distance_squared(origin);
    if dsq <= max_dist * max_dist {
        Some(EntityHit {
            entity_id: g.id,
            entity_name: g.name.clone(),
            intersection_point: ipt,
            intersection_normal: inorm,
            distance_sq: dsq,
        })
    } else {
        None
    }
}

/// 3D DDA voxel raycast with configurable `liquids_pointable`.
pub fn raycast_voxel_ex(
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
    liquids_pointable: bool,
    world: &WorldStore,
    registries: &EngineRegistries,
) -> Option<RaycastHit> {
    let mut state = RaycastState::new(origin, dir, max_dist, liquids_pointable);
    state.next_hit(world, registries)
}

/// Returns world-space `(min, max)` selection AABBs for wireframe rendering around the targeted node.
pub fn selected_world_boxes(
    hit: &RaycastHit,
    registries: &EngineRegistries,
) -> Vec<([f32; 3], [f32; 3])> {
    let Some(def) = registries.get_node_def(hit.node_id) else {
        let p = hit.under.as_vec3();
        return vec![(p.into(), (p + Vec3::ONE).into())];
    };
    let boxes = world_selection_boxes_with_neighbors(hit.under, def, hit.param2, hit.neighbors);
    if boxes.is_empty() {
        let p = hit.under.as_vec3();
        return vec![(p.into(), (p + Vec3::ONE).into())];
    }
    boxes
        .into_iter()
        .map(|b| (b.min.to_array(), b.max.to_array()))
        .collect()
}

// =========================================================================
// DigController State Machine
// =========================================================================

#[derive(Debug, Clone)]
pub struct CompletedDig {
    pub pos: IVec3,
    pub node_id: u16,
    pub node_name: String,
    pub drops: Vec<String>,
    pub picked_up_any: bool,
    pub sound_dug: Option<SoundSpec>,
    pub burst_particles: bool,
}

#[derive(Debug, Clone, Default)]
pub struct DigTickResult {
    pub swing_triggered: bool,
    pub is_digging: bool,
    pub crack_stage: Option<u8>,
    pub spawn_dig_particles: bool,
    pub sound_dig: Option<SoundSpec>,
    pub completed_dig: Option<CompletedDig>,
}

pub struct DigController {
    pub dig_progress: f32,
    pub target_pos: Option<IVec3>,
    pub active_slot: usize,
    pub elapsed_dig_time: f32,
    pub dig_time_complete: f32,
    nodig_delay_timer: f32,
    feedback_timer: f32,
    just_started: bool,
}

impl Default for DigController {
    fn default() -> Self {
        Self::new()
    }
}

impl DigController {
    pub fn new() -> Self {
        Self {
            dig_progress: 0.0,
            target_pos: None,
            active_slot: 0,
            elapsed_dig_time: 0.0,
            dig_time_complete: 0.0,
            nodig_delay_timer: 0.0,
            feedback_timer: 0.0,
            just_started: false,
        }
    }

    pub fn reset(&mut self) {
        self.dig_progress = 0.0;
        self.target_pos = None;
        self.elapsed_dig_time = 0.0;
        self.dig_time_complete = 0.0;
        self.feedback_timer = 0.0;
        self.just_started = false;
    }

    fn resolve_dig_sound(def: &NodeDef, tool_group: Option<&str>) -> Option<SoundSpec> {
        if let Some(ref spec) = def.sounds.dig {
            if spec.name == "__group" {
                if let Some(grp) = tool_group {
                    return Some(SoundSpec {
                        name: format!("default_dig_{grp}"),
                        gain: 0.5,
                        pitch: spec.pitch,
                    });
                }
                return None;
            }
            return Some(spec.clone());
        }
        None
    }

    fn best_matching_group(
        def: &NodeDef,
        tool: Option<&crate::registry::ToolCapabilities>,
    ) -> Option<String> {
        let tp = tool?;
        let level = def.groups.get("level").copied().unwrap_or(0);
        let mut best: Option<(f32, String)> = None;
        for (gname, cap) in &tp.groupcaps {
            let leveldiff = cap.maxlevel - level;
            if leveldiff < 0 {
                continue;
            }
            let rating = def.groups.get(gname).copied().unwrap_or(0);
            if let Some(&t) = cap.times.get(&rating) {
                let eff_t = if leveldiff > 1 {
                    t / leveldiff as f32
                } else {
                    t
                };
                if best.as_ref().map(|(bt, _)| eff_t < *bt).unwrap_or(true) {
                    best = Some((eff_t, gname.clone()));
                }
            }
        }
        best.map(|(_, g)| g)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        dt: f32,
        lmb_down: bool,
        hit: Option<&RaycastHit>,
        world: &WorldStore,
        registries: &EngineRegistries,
        player: &PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        lua_rt: Option<&LuaRuntime>,
    ) -> DigTickResult {
        self.update_with_physical_drops(
            dt,
            lmb_down,
            hit,
            world,
            registries,
            player,
            player_state,
            lua_rt,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_with_physical_drops(
        &mut self,
        dt: f32,
        lmb_down: bool,
        hit: Option<&RaycastHit>,
        world: &WorldStore,
        registries: &EngineRegistries,
        player: &PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        lua_rt: Option<&LuaRuntime>,
        physical_drops: bool,
    ) -> DigTickResult {
        let mut out = DigTickResult::default();
        let dt = dt.max(0.0);
        self.nodig_delay_timer = (self.nodig_delay_timer - dt).max(0.0);

        if !lmb_down || player.gamemode == GameMode::Spectator {
            self.reset();
            return out;
        }
        let Some(hit) = hit else {
            self.reset();
            return out;
        };

        if self.target_pos != Some(hit.under) || self.active_slot != player.selected_slot {
            self.reset();
            self.target_pos = Some(hit.under);
            self.active_slot = player.selected_slot;
            self.just_started = true;
        }

        if self.nodig_delay_timer > 0.0 {
            return out;
        }

        let Some(def) = registries.get_node_def(hit.node_id) else {
            self.reset();
            return out;
        };
        let creative = player.gamemode == GameMode::Creative;
        if !def.diggable && !creative {
            if self.just_started {
                self.just_started = false;
                out.swing_triggered = true;
            }
            return out;
        }

        let (held_name, held_wear, hand_name) = {
            let ps = player_state.read();
            let mg = ps.meta.read();
            let slot_idx = player.selected_slot;
            let held = mg
                .inventories
                .get("main")
                .and_then(|v| v.get(slot_idx))
                .filter(|s| !s.is_empty());
            let hand = mg
                .inventories
                .get("hand")
                .and_then(|v| v.first())
                .filter(|s| !s.is_empty());
            (
                held.map(|s| s.name.clone()),
                held.map(|s| s.wear).unwrap_or(0),
                hand.map(|s| s.name.clone()),
            )
        };

        let held_item_def = held_name
            .as_deref()
            .and_then(|n| registries.get_item_def(n));
        let held_tc = held_item_def.and_then(|d| d.tool_capabilities.as_ref());
        let hand_tc = effective_hand_tc(registries, hand_name.as_deref(), creative);

        let mut used_tc = held_tc;
        let mut params = compute_dig_params_with_wear(&def.groups, held_tc, creative, held_wear);
        if !params.diggable {
            used_tc = hand_tc;
            params = compute_dig_params_with_wear(&def.groups, hand_tc, creative, 0);
        }

        if !params.diggable {
            self.dig_progress = 0.0;
            self.elapsed_dig_time = 0.0;
            self.dig_time_complete = 0.0;
            if self.just_started {
                self.just_started = false;
                out.swing_triggered = true;
            }
            return out;
        }

        self.dig_time_complete = params.time;
        self.elapsed_dig_time += dt;
        self.dig_progress = if self.dig_time_complete <= 1e-4 {
            1.0
        } else {
            (self.elapsed_dig_time / self.dig_time_complete).clamp(0.0, 1.0)
        };

        self.feedback_timer -= dt;
        let matched_group = Self::best_matching_group(def, used_tc);
        if self.just_started || self.feedback_timer <= 0.0 {
            self.just_started = false;
            self.feedback_timer = 0.24;
            out.swing_triggered = true;
            out.spawn_dig_particles = true;
            out.sound_dig = Self::resolve_dig_sound(def, matched_group.as_deref());
        }

        if self.dig_progress >= 1.0 {
            let held_groups = held_item_def.map(|d| &d.groups);
            let outcome = complete_dig_node(
                hit,
                def,
                world,
                registries,
                player.selected_slot,
                held_name.as_deref(),
                held_groups,
                player_state,
                lua_rt,
                creative,
                params.wear,
                physical_drops,
            );

            self.nodig_delay_timer = if self.dig_time_complete <= 1e-4 {
                0.18
            } else {
                (self.dig_time_complete / 10.0).clamp(0.10, 0.25)
            };
            self.reset();
            out.crack_stage = None;
            out.is_digging = false;

            if let Some((drops, picked_up_any)) = outcome {
                out.completed_dig = Some(CompletedDig {
                    pos: hit.under,
                    node_id: hit.node_id,
                    node_name: def.name.clone(),
                    drops,
                    picked_up_any,
                    sound_dug: def.sounds.dug.clone(),
                    burst_particles: true,
                });
            }
        } else {
            let stage = ((self.dig_progress * 10.0).floor() as u8).min(9);
            out.crack_stage = Some(stage);
            out.is_digging = true;
        }

        out
    }
}

fn effective_hand_tc<'a>(
    registries: &'a EngineRegistries,
    hand_name: Option<&str>,
    creative: bool,
) -> Option<&'a ToolCapabilities> {
    let direct = registries.get_effective_tool_capabilities(None, hand_name);
    if direct
        .map(|tc| !tc.groupcaps.is_empty())
        .unwrap_or(false)
    {
        return direct;
    }
    if hand_name.is_none() {
        let suffix = if creative { "_crea" } else { "_surv" };
        let preferred = if creative {
            "mcl_meshhand:hand_crea"
        } else {
            "mcl_meshhand:hand_surv"
        };
        if let Some(tc) = registries
            .get_item_def(preferred)
            .and_then(|d| d.tool_capabilities.as_ref())
            .filter(|tc| !tc.groupcaps.is_empty())
        {
            return Some(tc);
        }
        if let Some(tc) = registries
            .items
            .values()
            .find(|d| d.name.starts_with("mcl_meshhand:") && d.name.ends_with(suffix))
            .and_then(|d| d.tool_capabilities.as_ref())
            .filter(|tc| !tc.groupcaps.is_empty())
        {
            return Some(tc);
        }
    }
    direct
}

fn snapshot_main_inv_counts(player_state: &Arc<RwLock<ObjectState>>) -> HashMap<String, u32> {
    let mut counts = HashMap::new();
    let ps = player_state.read();
    let mg = ps.meta.read();
    if let Some(inv) = mg.inventories.get("main") {
        for slot in inv {
            if !slot.is_empty() {
                *counts.entry(slot.name.clone()).or_insert(0) += slot.count as u32;
            }
        }
    }
    counts
}

fn diff_inserted_drops(
    before: &HashMap<String, u32>,
    after: &HashMap<String, u32>,
) -> Vec<String> {
    let mut added = Vec::new();
    for (name, &after_cnt) in after {
        let before_cnt = before.get(name).copied().unwrap_or(0);
        if after_cnt > before_cnt {
            let delta = after_cnt - before_cnt;
            if delta == 1 {
                added.push(name.clone());
            } else {
                added.push(format!("{name} {delta}"));
            }
        }
    }
    added.sort();
    added
}

fn apply_survival_tool_wear(
    player_state: &Arc<RwLock<ObjectState>>,
    slot_idx: usize,
    wear: u16,
) {
    if wear == 0 {
        return;
    }
    let ps = player_state.read();
    let mut mg = ps.meta.write();
    if let Some(inv) = mg.inventories.get_mut("main") {
        if let Some(slot) = inv.get_mut(slot_idx) {
            if !slot.is_empty() {
                let next_wear = (slot.wear as u32) + (wear as u32);
                if next_wear >= 65536 {
                    *slot = ItemStackData::default();
                } else {
                    slot.wear = next_wear as u16;
                }
            }
        }
    }
}

pub fn add_drops_to_inventory(
    player_state: &Arc<RwLock<ObjectState>>,
    drops: &[String],
    registries: &EngineRegistries,
) -> bool {
    let ps = player_state.read();
    let mut mg = ps.meta.write();
    let inv = mg
        .inventories
        .entry("main".into())
        .or_insert_with(|| vec![ItemStackData::default(); 36]);
    if inv.len() < 36 {
        inv.resize_with(36, ItemStackData::default);
    }

    let mut picked_up_any = false;
    for spec in drops {
        let mut stack = ItemStackData::from_spec(spec);
        if stack.is_empty() {
            continue;
        }
        let stack_max = registries
            .get_item_def(&stack.name)
            .map(|d| d.stack_max)
            .or_else(|| {
                let id = registries.get_content_id(&stack.name);
                registries.get_node_def(id).map(|d| d.stack_max)
            })
            .unwrap_or(64)
            .max(1);

        // Pass 1: merge into existing matching stacks across 0..36 (hotbar 0..9 first, then 9..36)
        if stack.wear == 0 {
            for slot in inv.iter_mut() {
                if !slot.is_empty()
                    && slot.name == stack.name
                    && slot.wear == 0
                    && slot.count < stack_max
                {
                    let room = stack_max - slot.count;
                    let take = room.min(stack.count);
                    if take > 0 {
                        slot.count += take;
                        stack.count -= take;
                        picked_up_any = true;
                    }
                    if stack.count == 0 {
                        break;
                    }
                }
            }
        }

        // Pass 2: place remainder into empty slots (hotbar 0..9 first, then 9..36)
        if stack.count > 0 {
            let hotbar_end = inv.len().min(9);
            let (hotbar, rest) = inv.split_at_mut(hotbar_end);
            for slot in hotbar.iter_mut().chain(rest.iter_mut()) {
                if slot.is_empty() {
                    let take = stack.count.min(stack_max);
                    if take > 0 {
                        slot.name = stack.name.clone();
                        slot.count = take;
                        slot.wear = stack.wear;
                        stack.count -= take;
                        picked_up_any = true;
                    }
                    if stack.count == 0 {
                        break;
                    }
                }
            }
        }
    }
    picked_up_any
}

#[allow(clippy::too_many_arguments)]
fn complete_dig_node(
    hit: &RaycastHit,
    def: &NodeDef,
    world: &WorldStore,
    registries: &EngineRegistries,
    slot_idx: usize,
    held_name: Option<&str>,
    held_groups: Option<&HashMap<String, i32>>,
    player_state: &Arc<RwLock<ObjectState>>,
    lua_rt: Option<&LuaRuntime>,
    creative: bool,
    wear: u16,
    physical_drops: bool,
) -> Option<(Vec<String>, bool)> {
    if let Some(rt) = lua_rt {
        let in_lua = rt
            .lua
            .globals()
            .get::<Table>("core")
            .ok()
            .and_then(|c| c.get::<Table>("registered_nodes").ok())
            .map(|rn| rn.contains_key(def.name.as_str()).unwrap_or(false))
            .unwrap_or(false);

        if in_lua {
            {
                let mut ps = player_state.write();
                ps.wield_index = slot_idx + 1;
                let mut mg = ps.meta.write();
                let hand = mg
                    .inventories
                    .entry("hand".into())
                    .or_insert_with(|| vec![ItemStackData::default(); 1]);
                if hand.is_empty() {
                    hand.push(ItemStackData::default());
                }
                if hand[0].is_empty() {
                    let suffix = if creative { "_crea" } else { "_surv" };
                    let fallback_hand = registries
                        .items
                        .values()
                        .find(|d| d.name.starts_with("mcl_meshhand:") && d.name.ends_with(suffix))
                        .map(|d| d.name.clone());
                    if let Some(hn) = fallback_hand {
                        hand[0] = ItemStackData::from_spec(&hn);
                    }
                }
            }

            let inv_before = snapshot_main_inv_counts(player_state);
            let slot_before = player_state
                .read()
                .meta
                .read()
                .inventories
                .get("main")
                .and_then(|v| v.get(slot_idx))
                .cloned()
                .unwrap_or_default();
            let eid_before = *rt.shared.next_entity_id.read();

            let dug = rt.dig_node(hit.under, player_state);
            let removed = dug || world.get_node(hit.under).0 != hit.node_id;
            if !removed {
                return None;
            }
            world.mark_priority_dirty_with_neighbors(hit.under);

            if physical_drops && !creative {
                let boosted = rt.boost_spawned_item_drops(eid_before, hit.under);
                if wear > 0 && !slot_before.is_empty() {
                    let slot_after = player_state
                        .read()
                        .meta
                        .read()
                        .inventories
                        .get("main")
                        .and_then(|v| v.get(slot_idx))
                        .cloned()
                        .unwrap_or_default();
                    if slot_after.name == slot_before.name && slot_after.wear == slot_before.wear {
                        apply_survival_tool_wear(player_state, slot_idx, wear);
                    }
                }
                if !boosted.is_empty() {
                    return Some((boosted, false));
                }
                let inv_after = snapshot_main_inv_counts(player_state);
                let lua_inserted = diff_inserted_drops(&inv_before, &inv_after);
                if !lua_inserted.is_empty() {
                    return Some((lua_inserted, true));
                }
                if rt.has_mcl_item_entity() {
                    return Some((Vec::new(), false));
                }
                let drops = rt.get_node_drops(&def.name, held_name, Some(player_state));
                for d in &drops {
                    rt.spawn_physical_item_drop(hit.under, d);
                }
                return Some((drops, false));
            }

            let spawned = rt.take_spawned_item_drops(eid_before);
            let inv_after = snapshot_main_inv_counts(player_state);
            let lua_inserted = diff_inserted_drops(&inv_before, &inv_after);

            if creative {
                return Some((Vec::new(), false));
            }

            // Ensure tool wear was applied in Survival (without double-wearing if Lua already wore it)
            if wear > 0 && !slot_before.is_empty() {
                let slot_after = player_state
                    .read()
                    .meta
                    .read()
                    .inventories
                    .get("main")
                    .and_then(|v| v.get(slot_idx))
                    .cloned()
                    .unwrap_or_default();
                if slot_after.name == slot_before.name && slot_after.wear == slot_before.wear {
                    apply_survival_tool_wear(player_state, slot_idx, wear);
                }
            }

            if !lua_inserted.is_empty() {
                return Some((lua_inserted, true));
            }
            if !spawned.is_empty() {
                let picked = add_drops_to_inventory(player_state, &spawned, registries);
                return Some((spawned, picked));
            }
            if rt.has_mcl_item_entity() {
                // Mineclonia's handle_node_drops already ran and yielded 0 drops (e.g. drop="",
                // leaves RNG miss, or tool tier insufficient via can_harvest).
                return Some((Vec::new(), false));
            }
            let drops = rt.get_node_drops(&def.name, held_name, Some(player_state));
            let picked = add_drops_to_inventory(player_state, &drops, registries);
            return Some((drops, picked));
        }
    }

    // Pure-Rust fallback (no LuaRuntime or node absent from Lua core.registered_nodes)
    if world.get_node(hit.under).0 == CONTENT_AIR {
        return None;
    }
    world.set_node(hit.under, CONTENT_AIR, 15, 0);
    world.remove_node_meta(hit.under);
    world.mark_priority_dirty_with_neighbors(hit.under);

    let drops = def
        .drop
        .resolve_drops(&def.name, held_name, held_groups)
        .into_iter()
        .filter(|s| !s.trim().is_empty())
        .collect::<Vec<_>>();

    if !creative {
        apply_survival_tool_wear(player_state, slot_idx, wear);
        if physical_drops {
            if let Some(rt) = lua_rt {
                for d in &drops {
                    rt.spawn_physical_item_drop(hit.under, d);
                }
                return Some((drops, false));
            }
        }
        let picked = add_drops_to_inventory(player_state, &drops, registries);
        return Some((drops, picked));
    }

    Some((drops, false))
}

/// Legacy single-call dig helper (backwards-compatible).
pub fn dig_targeted(
    hit: &RaycastHit,
    world: &WorldStore,
    registries: &EngineRegistries,
    player_state: &Arc<RwLock<ObjectState>>,
    lua_rt: Option<&LuaRuntime>,
) -> bool {
    let Some(def) = registries.get_node_def(hit.node_id) else {
        return false;
    };
    if !def.diggable {
        return false;
    }
    let slot_idx = player_state.read().wield_index.saturating_sub(1);
    let held_name = player_state
        .read()
        .meta
        .read()
        .inventories
        .get("main")
        .and_then(|v| v.get(slot_idx))
        .filter(|s| !s.is_empty())
        .map(|s| s.name.clone());
    let held_groups = held_name
        .as_deref()
        .and_then(|n| registries.get_item_def(n))
        .map(|d| &d.groups);
    complete_dig_node(
        hit,
        def,
        world,
        registries,
        slot_idx,
        held_name.as_deref(),
        held_groups,
        player_state,
        lua_rt,
        false,
        0,
        false,
    )
    .is_some()
}

// =========================================================================
// Right-Click Interaction & Node Placement (`try_interact_or_place`)
// =========================================================================

#[derive(Debug, Clone, Default)]
pub struct PlaceResult {
    pub interacted: bool,
    pub placed: bool,
    pub placed_pos: Option<IVec3>,
    pub sound_place: Option<SoundSpec>,
    pub portal_ignited: Option<IVec3>,
    pub water_evaporated_pos: Option<IVec3>,
    pub bed_exploded_pos: Option<IVec3>,
    pub open_chest_pos: Option<IVec3>,
}

/// Returns `true` when `item_name` is an empty bucket item (`mcl_buckets:bucket_empty`).
pub fn is_empty_bucket_item(item_name: &str) -> bool {
    let lower = item_name.trim().to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "mcl_buckets:bucket_empty" | "bucket:bucket_empty"
    )
}

/// Returns `true` when `player`'s currently selected hotbar slot holds an empty bucket.
pub fn held_item_is_empty_bucket(
    player: &PlayerController,
    player_state: &Arc<RwLock<ObjectState>>,
) -> bool {
    let slot_idx = player.selected_slot.min(8);
    let ps = player_state.read();
    let mg = ps.meta.read();
    mg.inventories
        .get("main")
        .and_then(|v| v.get(slot_idx))
        .filter(|s| !s.is_empty())
        .is_some_and(|s| is_empty_bucket_item(&s.name))
}

#[inline]
fn is_water_source_name(name: &str) -> bool {
    let lower = name.trim().to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "mcl_core:water_source"
            | "mcl_core:river_water_source"
            | "default:water_source"
            | "default:river_water_source"
    )
}

#[inline]
fn is_any_water_name(name: &str) -> bool {
    let lower = name.trim().to_ascii_lowercase();
    is_water_source_name(&lower)
        || lower.ends_with(":water_flowing")
        || lower.ends_with(":river_water_flowing")
}

#[inline]
fn is_lava_source_name(name: &str) -> bool {
    let lower = name.trim().to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "mcl_core:lava_source" | "mcl_nether:nether_lava_source" | "default:lava_source"
    )
}

/// Cools any `lava_source` at or orthogonally adjacent to `center` that touches water from above or horizontally.
pub fn cool_lava_around(
    world: &WorldStore,
    registries: &EngineRegistries,
    center: IVec3,
) -> Vec<IVec3> {
    let obsidian_id = registries.get_content_id("mcl_core:obsidian");
    if obsidian_id == CONTENT_UNKNOWN {
        return Vec::new();
    }
    let candidates = [
        center,
        center - IVec3::Y,
        center + IVec3::X,
        center - IVec3::X,
        center + IVec3::Z,
        center - IVec3::Z,
        center + IVec3::Y,
    ];
    let mut cooled = Vec::new();
    for lp in candidates {
        if cooled.contains(&lp) {
            continue;
        }
        let (nid, _, _) = world.get_node(lp);
        if nid == CONTENT_AIR || nid == CONTENT_IGNORE {
            continue;
        }
        let nname = registries.get_name_from_content_id(nid);
        if !is_lava_source_name(nname) {
            continue;
        }
        let water_touches = [
            lp + IVec3::Y,
            lp + IVec3::X,
            lp - IVec3::X,
            lp + IVec3::Z,
            lp - IVec3::Z,
        ]
        .into_iter()
        .any(|wp| {
            let (wid, _, _) = world.get_node(wp);
            wid != CONTENT_AIR
                && wid != CONTENT_IGNORE
                && is_any_water_name(registries.get_name_from_content_id(wid))
        });
        if water_touches {
            world.set_node(lp, obsidian_id, 15, 0);
            world.mark_priority_dirty_with_neighbors(lp);
            relight_around_pos(world, lp, registries);
            cooled.push(lp);
        }
    }
    cooled
}

/// Returns `true` when `node_name` is an interactive 27-slot storage/loot chest.
pub fn is_chest_node_name(node_name: &str) -> bool {
    let lower = node_name.trim().to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "mcl_chests:chest"
            | "mcl_chests:chest_small"
            | "mcl_chests:trapped_chest"
            | "mcl_chests:trapped_chest_small"
            | "default:chest"
            | "default:chest_locked"
    )
}

/// Luanti `core.dir_to_facedir` for horizontal placement (`0..3`).
pub fn dir_to_facedir(dir: Vec3) -> u8 {
    if dir.x.abs() > dir.z.abs() {
        if dir.x < 0.0 {
            3
        } else {
            1
        }
    } else if dir.z < 0.0 {
        2
    } else {
        0
    }
}

/// Luanti `core.dir_to_wallmounted` (`0..5`) where `dir` points from placed node towards attachment wall (`under - above`).
pub fn dir_to_wallmounted(dir: IVec3) -> u8 {
    if dir.y.abs() > dir.x.abs().max(dir.z.abs()) {
        if dir.y < 0 {
            1
        } else {
            0
        }
    } else if dir.x.abs() > dir.z.abs() {
        if dir.x < 0 {
            3
        } else {
            2
        }
    } else if dir.z < 0 {
        5
    } else {
        4
    }
}

fn compute_placement_param2(
    def: &NodeDef,
    hit: &RaycastHit,
    player: &PlayerController,
) -> u8 {
    match def.paramtype2.as_str() {
        "wallmounted" | "colorwallmounted" => {
            let dir = if hit.normal != IVec3::ZERO {
                -hit.normal
            } else {
                hit.under - hit.above
            };
            dir_to_wallmounted(dir)
        }
        "facedir" | "colorfacedir" | "4dir" | "color4dir" => dir_to_facedir(player.look_dir()),
        _ => 0,
    }
}

fn placement_overlaps_player(
    place_pos: IVec3,
    def: &NodeDef,
    param2: u8,
    world: &WorldStore,
    registries: &EngineRegistries,
    player: &PlayerController,
) -> bool {
    if !def.walkable {
        return false;
    }
    let player_box = PlayerController::collision_box().translate(player.pos);
    let mut tmp_boxes = world_collision_boxes(place_pos, def.id, param2, world, registries);
    if tmp_boxes.is_empty() {
        let p = place_pos.as_vec3();
        tmp_boxes.push(Aabb::new(p, p + Vec3::ONE));
    }
    tmp_boxes.into_iter().any(|b| player_box.intersects(b))
}

fn node_has_on_rightclick(lua_rt: &LuaRuntime, node_name: &str) -> bool {
    let Ok(core) = lua_rt.lua.globals().get::<Table>("core") else {
        return false;
    };
    core.get::<Table>("registered_nodes")
        .ok()
        .and_then(|rn| rn.get::<Table>(node_name).ok())
        .and_then(|def| def.get::<mlua::Function>("on_rightclick").ok())
        .is_some()
}

fn node_has_meta_formspec(world: &WorldStore, pos: IVec3) -> bool {
    world
        .node_meta
        .read()
        .get(&pos)
        .and_then(|m| {
            m.read()
                .fields
                .get("formspec")
                .map(|s| !s.trim().is_empty())
        })
        .unwrap_or(false)
}

/// Full Right-Click interaction & placement handler:
/// 1. Prioritizes `on_rightclick` (`LuaRuntime::rightclick_or_place_node_with_sneak`) for chests,
///    crafting tables, furnaces, doors, buttons, and levers.
/// 2. Falls back to `item_place_node` checking `buildable_to`, player-AABB overlap protection,
///    `facedir`/`wallmounted` `param2`, survival item consumption, priority remesh, and `place` sound.
pub fn try_interact_or_place(
    hit: &RaycastHit,
    world: &WorldStore,
    registries: &EngineRegistries,
    player: &PlayerController,
    player_state: &Arc<RwLock<ObjectState>>,
    lua_rt: Option<&LuaRuntime>,
) -> PlaceResult {
    if player.gamemode == GameMode::Spectator {
        return PlaceResult::default();
    }

    let creative = player.gamemode == GameMode::Creative;
    let slot_idx = player.selected_slot.min(8);
    let held_name = {
        let ps = player_state.read();
        let mg = ps.meta.read();
        mg.inventories
            .get("main")
            .and_then(|v| v.get(slot_idx))
            .filter(|s| !s.is_empty())
            .map(|s| s.name.clone())
    };

    let (under_id, _, _) = world.get_node(hit.under);
    let under_def = registries.get_node_def(under_id);
    let under_name = under_def.map(|d| d.name.as_str()).unwrap_or("");
    let under_buildable_to = under_def.map(|d| d.buildable_to).unwrap_or(false);
    let target_pos = if under_buildable_to { hit.under } else { hit.above };

    // 1. Pre-Lua Nether Bed Explosion Intercept (Intentional Game Design!)
    if world.dimension == DimensionId::Nether && !player.is_sneaking {
        let lower_under = under_name.to_ascii_lowercase();
        if lower_under.contains("mcl_beds:")
            && (lower_under.contains("bed") || lower_under.ends_with("_bottom") || lower_under.ends_with("_top"))
        {
            let center = hit.under;
            world.set_node(center, CONTENT_AIR, 0, 0);
            world.mark_priority_dirty_with_neighbors(center);
            for dx in -4..=4 {
                for dy in -4..=4 {
                    for dz in -4..=4 {
                        if dx * dx + dy * dy + dz * dz > 16 {
                            continue;
                        }
                        let p = center + IVec3::new(dx, dy, dz);
                        if p.y <= 0 || p.y >= 127 {
                            continue;
                        }
                        let (nid, _, _) = world.get_node(p);
                        if nid == CONTENT_AIR || nid == CONTENT_IGNORE {
                            continue;
                        }
                        let nname = registries.get_name_from_content_id(nid).to_ascii_lowercase();
                        if nname.contains("bedrock") || nname.contains("obsidian") {
                            continue;
                        }
                        world.set_node(p, CONTENT_AIR, 0, 0);
                        world.mark_priority_dirty_with_neighbors(p);
                        let _ = extinguish_portal_around(world, registries, p);
                    }
                }
            }
            relight_around_pos(world, center, registries);
            return PlaceResult {
                interacted: true,
                bed_exploded_pos: Some(center),
                ..Default::default()
            };
        }
    }

    // 2. Pre-Lua Nether Water Bucket Evaporation Intercept
    if world.dimension == DimensionId::Nether {
        if let Some(ref hname) = held_name {
            let hl = hname.to_ascii_lowercase();
            let is_water_item = hl.contains("bucket_water")
                || hl.contains("bucket_river_water")
                || hl.contains("water_bucket")
                || hl.ends_with(":water_source")
                || hl.ends_with(":river_water_source");
            if is_water_item {
                if !creative {
                    let ps = player_state.write();
                    let mut mg = ps.meta.write();
                    if let Some(main) = mg.inventories.get_mut("main") {
                        if let Some(slot) = main.get_mut(slot_idx) {
                            if hl.contains("bucket") && registries.items.contains_key("mcl_buckets:bucket_empty") {
                                *slot = ItemStackData::from_spec("mcl_buckets:bucket_empty 1");
                            } else if slot.count > 1 {
                                slot.count -= 1;
                            } else {
                                *slot = ItemStackData::default();
                            }
                        }
                    }
                }
                return PlaceResult {
                    interacted: true,
                    water_evaporated_pos: Some(target_pos),
                    ..Default::default()
                };
            }
        }
    }

    // 3. Pre-Lua Obsidian Portal Ignition (Flint & Steel / Fire Charge in Overworld or Nether)
    if let Some(ref hname) = held_name {
        if is_portal_igniter(hname) && is_obsidian(under_id, registries) {
            if let Some(frame) = try_ignite_portal(world, registries, hit.above) {
                if !creative {
                    let ps = player_state.write();
                    let mut mg = ps.meta.write();
                    if let Some(main) = mg.inventories.get_mut("main") {
                        if let Some(slot) = main.get_mut(slot_idx) {
                            if hname.to_ascii_lowercase().contains("fire_charge") {
                                if slot.count > 1 {
                                    slot.count -= 1;
                                } else {
                                    *slot = ItemStackData::default();
                                }
                            } else {
                                let nw = (slot.wear as u32).saturating_add(1024);
                                if nw >= 65535 {
                                    *slot = ItemStackData::default();
                                } else {
                                    slot.wear = nw as u16;
                                }
                            }
                        }
                    }
                }
                return PlaceResult {
                    interacted: true,
                    portal_ignited: Some(frame.origin),
                    ..Default::default()
                };
            }
        }
    }

    // 3b. Pre-Lua Empty Bucket Liquid Scoop
    if let Some(ref hname) = held_name {
        if is_empty_bucket_item(hname) {
            let filled_bucket = if is_water_source_name(under_name) {
                Some("mcl_buckets:bucket_water")
            } else if is_lava_source_name(under_name) {
                Some("mcl_buckets:bucket_lava")
            } else {
                None
            };
            if let Some(target_bucket) = filled_bucket {
                if registries.items.contains_key(target_bucket) {
                    world.set_node(hit.under, CONTENT_AIR, 15, 0);
                    world.mark_priority_dirty_with_neighbors(hit.under);
                    relight_around_pos(world, hit.under, registries);
                    if !creative {
                        let mut overflow_item = false;
                        {
                            let ps = player_state.read();
                            let mut mg = ps.meta.write();
                            if let Some(main) = mg.inventories.get_mut("main") {
                                if let Some(slot) = main.get_mut(slot_idx) {
                                    if slot.count <= 1 {
                                        *slot = ItemStackData::from_spec(&format!(
                                            "{target_bucket} 1"
                                        ));
                                    } else {
                                        slot.count -= 1;
                                        overflow_item = true;
                                    }
                                }
                            }
                        }
                        if overflow_item {
                            add_drops_to_inventory(
                                player_state,
                                &[format!("{target_bucket} 1")],
                                registries,
                            );
                        }
                    }
                    return PlaceResult {
                        interacted: true,
                        placed: true,
                        placed_pos: Some(hit.under),
                        ..Default::default()
                    };
                }
            }
        }
    }

    // 3c. Pre-Lua Water/Lava Bucket Placement & Instant Lava-to-Obsidian Cooling
    if let Some(ref hname) = held_name {
        let hl = hname.trim().to_ascii_lowercase();
        let place_liq_name = if hl == "mcl_buckets:bucket_water"
            || hl == "mcl_buckets:bucket_river_water"
            || hl == "bucket:bucket_water"
        {
            Some("mcl_core:water_source")
        } else if hl == "mcl_buckets:bucket_lava" || hl == "bucket:bucket_lava" {
            Some("mcl_core:lava_source")
        } else {
            None
        };
        if let Some(liq_name) = place_liq_name {
            let hit_interactive = !player.is_sneaking
                && (is_chest_node_name(under_name)
                    || node_has_meta_formspec(world, hit.under)
                    || lua_rt.is_some_and(|rt| node_has_on_rightclick(rt, under_name)));
            if !hit_interactive {
                let liq_id = registries.get_content_id(liq_name);
                if liq_id != CONTENT_UNKNOWN {
                    let liq_place_pos = if under_buildable_to
                        && !is_water_source_name(under_name)
                        && !is_lava_source_name(under_name)
                    {
                        hit.under
                    } else {
                        hit.above
                    };
                    let (cur_id, _, _) = world.get_node(liq_place_pos);
                    let can_replace = cur_id == CONTENT_AIR
                        || registries
                            .get_node_def(cur_id)
                            .is_some_and(|d| d.buildable_to);
                    if can_replace {
                        let p1 = registries
                            .get_node_def(liq_id)
                            .map(|d| {
                                if d.light_source > 0 {
                                    (d.light_source << 4) | 15
                                } else {
                                    15
                                }
                            })
                            .unwrap_or(15);
                        world.set_node(liq_place_pos, liq_id, p1, 0);
                        world.mark_priority_dirty_with_neighbors(liq_place_pos);
                        relight_around_pos(world, liq_place_pos, registries);
                        let _ = cool_lava_around(world, registries, liq_place_pos);
                        if !creative {
                            let ps = player_state.read();
                            let mut mg = ps.meta.write();
                            if let Some(main) = mg.inventories.get_mut("main") {
                                if let Some(slot) = main.get_mut(slot_idx) {
                                    if registries.items.contains_key("mcl_buckets:bucket_empty") {
                                        *slot =
                                            ItemStackData::from_spec("mcl_buckets:bucket_empty 1");
                                    } else if slot.count > 1 {
                                        slot.count -= 1;
                                    } else {
                                        *slot = ItemStackData::default();
                                    }
                                }
                            }
                        }
                        return PlaceResult {
                            interacted: true,
                            placed: true,
                            placed_pos: Some(liq_place_pos),
                            ..Default::default()
                        };
                    }
                }
            }
        }
    }

    // 4. Pre-Lua Generated Structure Loot Chest OR Nether Chest Open Intercept
    if !player.is_sneaking && is_chest_node_name(under_name) {
        let is_structure_chest = crate::world::structure_chest_at(
            hit.under,
            world.dimension,
            world.seed,
            *world.world_type.read(),
        )
        .is_some();
        if is_structure_chest || lua_rt.is_none() || world.dimension.is_nether() {
            crate::world::ensure_structure_chest_loot(
                world,
                hit.under,
                world.seed,
                *world.world_type.read(),
                registries,
            );
            return PlaceResult {
                interacted: true,
                open_chest_pos: Some(hit.under),
                ..Default::default()
            };
        }
    }

    // Determine whether held item is a placeable node and whether it overlaps the player AABB
    let held_node_def = held_name.as_deref().and_then(|name| {
        let id = registries.get_content_id(name);
        if id == CONTENT_UNKNOWN {
            return None;
        }
        registries.get_node_def(id)
    });

    if let Some(rt) = lua_rt {
        let can_rightclick = !player.is_sneaking
            && (node_has_on_rightclick(rt, under_name) || node_has_meta_formspec(world, hit.under));
        let would_overlap_player = if !can_rightclick {
            if let Some(def) = held_node_def {
                let p2 = compute_placement_param2(def, hit, player);
                placement_overlaps_player(target_pos, def, p2, world, registries, player)
            } else {
                false
            }
        } else {
            false
        };

        if can_rightclick || !would_overlap_player {
            let old_under = world.get_node(hit.under);
            let old_above = world.get_node(hit.above);
            if rt.rightclick_or_place_node_with_sneak(
                hit.under,
                hit.above,
                player_state,
                player.is_sneaking,
            ) {
                let new_above = world.get_node(hit.above);
                let new_under = world.get_node(hit.under);
                let placed_pos = if new_above != old_above {
                    Some(hit.above)
                } else if new_under != old_under && under_buildable_to {
                    Some(hit.under)
                } else {
                    None
                };
                if let Some(pp) = placed_pos {
                    world.mark_priority_dirty_with_neighbors(pp);
                    let placed_id = world.get_node(pp).0;
                    let snd = registries
                        .get_node_def(placed_id)
                        .and_then(|d| d.sounds.place.clone());
                    return PlaceResult {
                        interacted: can_rightclick,
                        placed: true,
                        placed_pos: Some(pp),
                        sound_place: snd,
                        ..Default::default()
                    };
                }
                world.mark_priority_dirty_with_neighbors(hit.under);
                return PlaceResult {
                    interacted: true,
                    placed: false,
                    placed_pos: None,
                    sound_place: None,
                    ..Default::default()
                };
            }
        } else {
            return PlaceResult::default();
        }
    }

    // Fallback `item_place_node`
    let Some(def) = held_node_def else {
        return PlaceResult::default();
    };
    if def.drawtype == DrawType::Airlike {
        return PlaceResult::default();
    }

    if !under_buildable_to {
        let (above_id, _, _) = world.get_node(hit.above);
        let above_buildable_to = if above_id == CONTENT_AIR {
            true
        } else {
            registries
                .get_node_def(above_id)
                .map(|d| d.buildable_to)
                .unwrap_or(false)
        };
        if !above_buildable_to {
            return PlaceResult::default();
        }
    }

    let param2 = compute_placement_param2(def, hit, player);
    if placement_overlaps_player(target_pos, def, param2, world, registries, player) {
        return PlaceResult::default();
    }

    let p1 = if def.light_source > 0 {
        (def.light_source << 4) | 15
    } else {
        15
    };
    world.set_node(target_pos, def.id, p1, param2);
    if let Some(rt) = lua_rt {
        rt.invoke_on_construct(target_pos);
    }
    world.mark_priority_dirty_with_neighbors(target_pos);

    if !creative {
        let ps = player_state.read();
        let mut mg = ps.meta.write();
        if let Some(inv) = mg.inventories.get_mut("main") {
            if let Some(slot) = inv.get_mut(slot_idx) {
                slot.count = slot.count.saturating_sub(1);
                if slot.count == 0 {
                    slot.name.clear();
                    slot.wear = 0;
                }
            }
        }
    }

    PlaceResult {
        interacted: false,
        placed: true,
        placed_pos: Some(target_pos),
        sound_place: def.sounds.place.clone(),
        ..Default::default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FoodStats {
    pub hunger: i32,
    pub saturation: f32,
    pub bonus_hp: i32,
    pub always_edible: bool,
}

pub fn resolve_food_stats(item_name: &str, registries: &EngineRegistries) -> Option<FoodStats> {
    if item_name.trim().is_empty() {
        return None;
    }
    let resolved = registries.resolve_alias(item_name.trim());
    let short = resolved.rsplit(':').next().unwrap_or(resolved);
    let named = match short {
        "apple" => Some(FoodStats { hunger: 4, saturation: 2.4, bonus_hp: 0, always_edible: false }),
        "bread" => Some(FoodStats { hunger: 5, saturation: 6.0, bonus_hp: 0, always_edible: false }),
        "cooked_beef" | "steak" => Some(FoodStats { hunger: 8, saturation: 12.8, bonus_hp: 0, always_edible: false }),
        "cooked_porkchop" => Some(FoodStats { hunger: 8, saturation: 12.8, bonus_hp: 0, always_edible: false }),
        "beef_raw" | "porkchop_raw" => Some(FoodStats { hunger: 3, saturation: 1.8, bonus_hp: 0, always_edible: false }),
        "potato_item_baked" | "baked_potato" => Some(FoodStats { hunger: 5, saturation: 6.0, bonus_hp: 0, always_edible: false }),
        "potato_item" | "potato" => Some(FoodStats { hunger: 1, saturation: 0.6, bonus_hp: 0, always_edible: false }),
        "carrot_item" | "carrot" => Some(FoodStats { hunger: 3, saturation: 3.6, bonus_hp: 0, always_edible: false }),
        "carrot_item_gold" | "golden_carrot" => Some(FoodStats { hunger: 6, saturation: 14.4, bonus_hp: 0, always_edible: false }),
        "apple_gold" | "golden_apple" => Some(FoodStats { hunger: 4, saturation: 9.6, bonus_hp: 4, always_edible: true }),
        "apple_gold_enchanted" | "enchanted_golden_apple" => Some(FoodStats { hunger: 4, saturation: 9.6, bonus_hp: 8, always_edible: true }),
        "cooked_chicken" | "cooked_mutton" | "cooked_salmon" => Some(FoodStats { hunger: 6, saturation: 7.2, bonus_hp: 0, always_edible: false }),
        "cooked_rabbit" | "cooked_cod" => Some(FoodStats { hunger: 5, saturation: 6.0, bonus_hp: 0, always_edible: false }),
        "mushroom_stew" | "rabbit_stew" | "beetroot_soup" => Some(FoodStats { hunger: 6, saturation: 7.2, bonus_hp: 0, always_edible: false }),
        "pumpkin_pie" => Some(FoodStats { hunger: 8, saturation: 4.8, bonus_hp: 0, always_edible: false }),
        "melon_item" | "cookie" | "sweet_berries" | "glow_berries" => Some(FoodStats { hunger: 2, saturation: 1.2, bonus_hp: 0, always_edible: false }),
        "beetroot_item" | "beetroot" => Some(FoodStats { hunger: 1, saturation: 1.2, bonus_hp: 0, always_edible: false }),
        _ => None,
    };
    if named.is_some() {
        return named;
    }
    let def = registries.get_item_def(resolved)?;
    let eatable = def.groups.get("eatable").copied().unwrap_or(0);
    let food_grp = def.groups.get("food").copied().unwrap_or(0);
    let gain = eatable.max(if food_grp > 1 { food_grp } else { 0 });
    if gain > 0 {
        Some(FoodStats {
            hunger: gain.clamp(1, 20),
            saturation: (gain as f32 * 0.8).max(1.2),
            bonus_hp: 0,
            always_edible: def.groups.get("can_eat_when_full").copied().unwrap_or(0) > 0,
        })
    } else {
        None
    }
}

pub fn targeted_node_has_interaction(
    hit: &RaycastHit,
    world: &WorldStore,
    registries: &EngineRegistries,
    lua_rt: Option<&LuaRuntime>,
) -> bool {
    let under_name = registries
        .get_node_def(world.get_node(hit.under).0)
        .map(|d| d.name.as_str())
        .unwrap_or("");
    let has_rc = lua_rt.is_some_and(|rt| node_has_on_rightclick(rt, under_name));
    has_rc || node_has_meta_formspec(world, hit.under) || is_chest_node_name(under_name)
}

pub fn try_eat_held_food(
    player: &mut PlayerController,
    player_state: &Arc<RwLock<ObjectState>>,
    registries: &EngineRegistries,
    lua_rt: Option<&LuaRuntime>,
) -> bool {
    if player.is_dead || player.eat_cooldown > 0.0 || player.gamemode == GameMode::Spectator {
        return false;
    }
    let slot_idx = player.selected_slot.min(8);
    let held_name = {
        let ps = player_state.read();
        let mg = ps.meta.read();
        mg.inventories
            .get("main")
            .and_then(|v| v.get(slot_idx))
            .filter(|s| !s.is_empty())
            .map(|s| s.name.clone())
    };
    let Some(item_name) = held_name else {
        return false;
    };

    if let Some(stats) = resolve_food_stats(&item_name, registries) {
        let can_eat = player.hunger < 20 || stats.always_edible || player.gamemode == GameMode::Creative;
        if !can_eat {
            return false;
        }
        player.eat_food(stats.hunger, stats.saturation, stats.bonus_hp);
        if player.gamemode != GameMode::Creative {
            let ps = player_state.read();
            let mut mg = ps.meta.write();
            if let Some(inv) = mg.inventories.get_mut("main") {
                if let Some(slot) = inv.get_mut(slot_idx) {
                    slot.count = slot.count.saturating_sub(1);
                    if slot.count == 0 {
                        *slot = ItemStackData::default();
                    }
                }
            }
        }
        {
            let mut ps = player_state.write();
            ps.hunger = player.hunger;
            ps.hp = player.hp;
            ps.meta.write().fields.insert("mcl_health".into(), player.hp.to_string());
        }
        return true;
    }

    if let Some(rt) = lua_rt {
        let is_node = registries.get_item_def(&item_name).is_some_and(|d| d.is_node);
        if !is_node && rt.invoke_item_on_use(player_state) {
            player.eat_cooldown = 0.25;
            return true;
        }
    }
    false
}

/// Legacy single-call placement helper (backwards-compatible).
pub fn place_targeted(
    hit: &RaycastHit,
    world: &WorldStore,
    registries: &EngineRegistries,
    player: &PlayerController,
    player_state: &Arc<RwLock<ObjectState>>,
    creative: bool,
) -> bool {
    let mut tmp_player = PlayerController::default();
    tmp_player.pos = player.pos;
    tmp_player.yaw = player.yaw;
    tmp_player.pitch = player.pitch;
    tmp_player.selected_slot = player.selected_slot;
    tmp_player.is_sneaking = player.is_sneaking;
    tmp_player.gamemode = if creative {
        GameMode::Creative
    } else {
        GameMode::Survival
    };
    tmp_player.eye_height = player.eye_height;
    try_interact_or_place(hit, world, registries, &tmp_player, player_state, None).placed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{ItemDef, NodeDrop};

    #[test]
    fn inventory_drop_stacking_hotbar_first_and_empty_drop() {
        let mut reg = EngineRegistries::new();
        reg.register_item(ItemDef {
            name: "test:stone".into(),
            stack_max: 64,
            ..Default::default()
        });
        reg.register_item(ItemDef {
            name: "test:pick".into(),
            item_type: "tool".into(),
            stack_max: 1,
            ..Default::default()
        });
        let glass = reg.register_node(NodeDef {
            name: "test:glass".into(),
            drop: NodeDrop::Empty,
            groups: HashMap::from([("dig_immediate".into(), 3)]),
            ..Default::default()
        });

        let ps = Arc::new(RwLock::new(ObjectState::new_player("InvTester")));
        // Pre-fill hotbar slots 0..9 except slot 3, and put a partial stack of test:stone 62 in slot 5
        {
            let p = ps.read();
            let mut m = p.meta.write();
            let inv = m.inventories.get_mut("main").unwrap();
            for (i, s) in inv.iter_mut().enumerate().take(9) {
                if i == 5 {
                    *s = ItemStackData::from_spec("test:stone 62");
                } else if i != 3 {
                    *s = ItemStackData::from_spec("test:filler 1");
                }
            }
        }

        // Insert 5 test:stone -> 2 stack into slot 5 (to 64), remaining 3 go into hotbar slot 3 (before 9..36!)
        let picked = add_drops_to_inventory(&ps, &["test:stone 5".into()], &reg);
        assert!(picked);
        {
            let p = ps.read();
            let m = p.meta.read();
            let inv = &m.inventories["main"];
            assert_eq!(inv[5].count, 64);
            assert_eq!(inv[3].name, "test:stone");
            assert_eq!(inv[3].count, 3);
            assert!(inv[9].is_empty());
        }

        // Insert tool with stack_max = 1 -> goes to main inventory slot 9 (since hotbar is full)
        assert!(add_drops_to_inventory(&ps, &["test:pick 2".into()], &reg));
        {
            let p = ps.read();
            let m = p.meta.read();
            let inv = &m.inventories["main"];
            assert_eq!(inv[9].name, "test:pick");
            assert_eq!(inv[9].count, 1);
            assert_eq!(inv[10].name, "test:pick");
            assert_eq!(inv[10].count, 1);
        }

        // Empty drop produces picked_up_any == false
        assert!(!add_drops_to_inventory(&ps, &["".into()], &reg));
        let dir = std::env::temp_dir().join("cubex_rs_raycast_empty_drop");
        let _ = std::fs::remove_dir_all(&dir);
        let world = WorldStore::new(&dir, 1);
        world.set_node(IVec3::new(0, 0, 2), glass, 15, 0);
        let hit = raycast_voxel(Vec3::new(0.5, 0.5, 0.5), Vec3::Z, 5.0, &world, &reg).unwrap();
        let mut pctrl = PlayerController::default();
        pctrl.pos = Vec3::new(0.5, 0.0, 0.5);
        let mut digger = DigController::new();
        let tick = digger.update(0.05, true, Some(&hit), &world, &reg, &pctrl, &ps, None);
        let comp = tick.completed_dig.expect("glass breaks immediately");
        assert!(comp.drops.is_empty());
        assert!(!comp.picked_up_any);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn structure_chest_rightclick_opens_27_slot_loot_formspec() {
        use crate::registry::DrawType;
        use crate::ui::formspec::{build_loot_chest_formspec, parse_formspec, FormElement};
        use crate::world::structures::{locate_nearest_structure, StructureKind};
        use crate::world::WorldGenType;

        let mut reg = EngineRegistries::new();
        let chest_id = reg.register_node(NodeDef {
            name: "mcl_chests:chest".into(),
            drawtype: DrawType::Mesh,
            walkable: true,
            pointable: true,
            ..Default::default()
        });
        let anchor = locate_nearest_structure(
            StructureKind::NetherFortress,
            1337,
            WorldGenType::Default,
            (0, 0),
        )
        .expect("fortress anchor");
        let chest_pos = anchor.chests[0].0;

        let dir = std::env::temp_dir().join("cubex_rs_struct_chest_rc_test");
        let _ = std::fs::remove_dir_all(&dir);
        let world = WorldStore::new_for_dimension(
            &dir,
            1337,
            DimensionId::Nether,
            WorldGenType::Default,
            crate::world::Difficulty::Normal,
            crate::world::Gamerules::default(),
            "Survival",
        );
        world.set_node(chest_pos, chest_id, 15, 0);

        let hit = RaycastHit {
            pos: chest_pos,
            prev_pos: chest_pos - IVec3::Z,
            normal: IVec3::NEG_Z,
            node_id: chest_id,
            under: chest_pos,
            above: chest_pos - IVec3::Z,
            intersection_point: chest_pos.as_vec3(),
            intersection_normal: Vec3::NEG_Z,
            box_id: 0,
            distance_sq: 2.0,
            param2: 0,
            neighbors: 0,
        };
        let ps = Arc::new(RwLock::new(ObjectState::new_player("Looter")));
        let pctrl = PlayerController::default();

        assert!(targeted_node_has_interaction(&hit, &world, &reg, None));
        let pr = try_interact_or_place(&hit, &world, &reg, &pctrl, &ps, None);
        assert!(pr.interacted && !pr.placed);
        assert_eq!(pr.open_chest_pos, Some(chest_pos));

        let meta = world.get_node_meta(chest_pos);
        let g = meta.read();
        let inv = g.inventories.get("main").expect("27-slot chest inventory");
        assert_eq!(inv.len(), 27);
        assert!(inv.iter().any(|s| !s.is_empty()), "loot chest must contain items");

        let title = g
            .fields
            .get("infotext")
            .cloned()
            .unwrap_or_else(|| "Loot Chest".into());
        let raw_fs = build_loot_chest_formspec(chest_pos, &title);
        let parsed = parse_formspec(&raw_fs);
        assert!(parsed.elements.iter().any(|e| matches!(
            e,
            FormElement::InvList { loc, list, w: 9, h: 3, .. }
                if loc.starts_with("nodemeta:") && list == "main"
        )));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
