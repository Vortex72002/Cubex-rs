use glam::IVec3;
use std::collections::{HashSet, VecDeque};

use super::dimension::DimensionId;
use super::light::relight_around_pos;
use super::nether::NETHER_LAVA_LEVEL;
use super::WorldStore;
use crate::registry::{EngineRegistries, CONTENT_AIR, CONTENT_IGNORE, NETHER_PORTAL_NODE_ID};

pub const MIN_PORTAL_INNER_WIDTH: i32 = 2;
pub const MIN_PORTAL_INNER_HEIGHT: i32 = 3;
pub const MAX_PORTAL_INNER_SIZE: i32 = 21;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortalAxis {
    /// Portal aperture spans X horizontally and Y vertically (thin slab facing +-Z, facedir param2 = 0).
    X,
    /// Portal aperture spans Z horizontally and Y vertically (thin slab facing +-X, facedir param2 = 1).
    Z,
}

impl PortalAxis {
    pub fn param2(self) -> u8 {
        match self {
            Self::X => 0,
            Self::Z => 1,
        }
    }

    pub fn from_param2(p2: u8) -> Self {
        if (p2 & 0x03) == 1 {
            Self::Z
        } else {
            Self::X
        }
    }

    pub fn step_horiz(self) -> IVec3 {
        match self {
            Self::X => IVec3::X,
            Self::Z => IVec3::Z,
        }
    }

    pub fn step_depth(self) -> IVec3 {
        match self {
            Self::X => IVec3::Z,
            Self::Z => IVec3::X,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PortalFrame {
    pub axis: PortalAxis,
    /// Bottom-left interior cell (inside the obsidian frame).
    pub origin: IVec3,
    /// Inner air/portal width (`2..=21`).
    pub width: i32,
    /// Inner air/portal height (`3..=21`).
    pub height: i32,
}

impl PortalFrame {
    pub fn interior_positions(&self) -> Vec<IVec3> {
        let h_dir = self.axis.step_horiz();
        let mut out = Vec::with_capacity((self.width * self.height) as usize);
        for dy in 0..self.height {
            for dw in 0..self.width {
                out.push(self.origin + h_dir * dw + IVec3::Y * dy);
            }
        }
        out
    }

    pub fn bottom_center(&self) -> IVec3 {
        self.origin + self.axis.step_horiz() * (self.width / 2)
    }
}

#[inline]
pub fn is_obsidian(id: u16, reg: &EngineRegistries) -> bool {
    if id == CONTENT_AIR || id == CONTENT_IGNORE {
        return false;
    }
    let name = reg.get_name_from_content_id(id);
    name == "mcl_core:obsidian" || name.ends_with(":obsidian")
}

#[inline]
pub fn is_nether_portal_node(id: u16, reg: &EngineRegistries) -> bool {
    if id == CONTENT_AIR || id == CONTENT_IGNORE {
        return false;
    }
    let name = reg.get_name_from_content_id(id);
    name == NETHER_PORTAL_NODE_ID || name == "mcl_portals:portal"
}

#[inline]
pub fn is_portal_igniter(item_name: &str) -> bool {
    let lower = item_name.trim().to_ascii_lowercase();
    lower.contains("flint_and_steel") || lower.contains("fire_charge")
}

#[inline]
fn is_replaceable_interior(id: u16, reg: &EngineRegistries) -> bool {
    if id == CONTENT_AIR {
        return true;
    }
    if id == CONTENT_IGNORE {
        return false;
    }
    if is_nether_portal_node(id, reg) {
        return true;
    }
    let name = reg.get_name_from_content_id(id);
    name.contains(":fire") || name == "mcl_fire:fire" || name == "mcl_fire:soul_fire"
}

fn validate_frame_for_axis(
    world: &WorldStore,
    reg: &EngineRegistries,
    seed_air: IVec3,
    axis: PortalAxis,
    require_all_portal_cells: bool,
) -> Option<PortalFrame> {
    let h_dir = axis.step_horiz();
    let check_int = |p: IVec3| -> bool {
        let id = world.get_node(p).0;
        if require_all_portal_cells {
            is_nether_portal_node(id, reg)
        } else {
            is_replaceable_interior(id, reg)
        }
    };
    if !check_int(seed_air) {
        return None;
    }

    // 1. Walk downward to bottom interior row above obsidian floor
    let mut bottom_pos = seed_air;
    let mut down_steps = 0;
    while down_steps < MAX_PORTAL_INNER_SIZE && check_int(bottom_pos - IVec3::Y) {
        bottom_pos -= IVec3::Y;
        down_steps += 1;
    }
    if !is_obsidian(world.get_node(bottom_pos - IVec3::Y).0, reg) {
        return None;
    }

    // 2. Walk left along axis to left interior edge beside obsidian pillar
    let mut origin = bottom_pos;
    let mut left_steps = 0;
    while left_steps < MAX_PORTAL_INNER_SIZE && check_int(origin - h_dir) {
        origin -= h_dir;
        left_steps += 1;
    }
    if !is_obsidian(world.get_node(origin - h_dir).0, reg) {
        return None;
    }

    // 3. Walk right along axis to measure inner width
    let mut width = 0;
    while width < MAX_PORTAL_INNER_SIZE && check_int(origin + h_dir * width) {
        width += 1;
    }
    if width < MIN_PORTAL_INNER_WIDTH || !is_obsidian(world.get_node(origin + h_dir * width).0, reg)
    {
        return None;
    }

    // 4. Verify bottom obsidian bar across full inner width
    for dw in 0..width {
        if !is_obsidian(world.get_node(origin + h_dir * dw - IVec3::Y).0, reg) {
            return None;
        }
    }

    // 5. Walk upward row by row checking interior + left/right obsidian pillars + top obsidian bar
    let mut height = 0;
    while height <= MAX_PORTAL_INNER_SIZE {
        let row_y = origin + IVec3::Y * height;
        let top_bar = (0..width).all(|dw| is_obsidian(world.get_node(row_y + h_dir * dw).0, reg));
        if top_bar && height >= MIN_PORTAL_INNER_HEIGHT {
            return Some(PortalFrame {
                axis,
                origin,
                width,
                height,
            });
        }
        if height == MAX_PORTAL_INNER_SIZE {
            return None;
        }
        if !is_obsidian(world.get_node(row_y - h_dir).0, reg)
            || !is_obsidian(world.get_node(row_y + h_dir * width).0, reg)
        {
            return None;
        }
        if !(0..width).all(|dw| check_int(row_y + h_dir * dw)) {
            return None;
        }
        height += 1;
    }
    None
}

/// Detects a valid obsidian Nether portal frame (`4x5` .. `23x23`, optional corners) containing `seed_pos`
/// (or directly adjacent if `seed_pos` is a clicked frame face).
pub fn detect_portal_frame(
    world: &WorldStore,
    reg: &EngineRegistries,
    seed_pos: IVec3,
) -> Option<PortalFrame> {
    let offsets = [
        IVec3::ZERO,
        IVec3::Y,
        -IVec3::Y,
        IVec3::X,
        -IVec3::X,
        IVec3::Z,
        -IVec3::Z,
    ];
    for off in offsets {
        let p = seed_pos + off;
        for axis in [PortalAxis::X, PortalAxis::Z] {
            if let Some(frame) = validate_frame_for_axis(world, reg, p, axis, false) {
                return Some(frame);
            }
        }
    }
    None
}

/// Ignites an obsidian portal frame at or adjacent to `seed_pos`, placing `cubex:nether_portal` blocks.
pub fn try_ignite_portal(
    world: &WorldStore,
    reg: &EngineRegistries,
    seed_pos: IVec3,
) -> Option<PortalFrame> {
    let portal_id = reg.get_content_id(NETHER_PORTAL_NODE_ID);
    if portal_id == crate::registry::CONTENT_UNKNOWN {
        return None;
    }
    let frame = detect_portal_frame(world, reg, seed_pos)?;
    let p2 = frame.axis.param2();
    for p in frame.interior_positions() {
        world.set_node(p, portal_id, 0, p2);
        world.mark_priority_dirty_with_neighbors(p);
        relight_around_pos(world, p, reg);
    }
    Some(frame)
}

/// Checks portals connected to a broken block at `broken_pos` (obsidian or portal node) and extinguishes
/// any connected portal whose obsidian perimeter or interior pane was breached. Returns blocks cleared.
pub fn extinguish_portal_around(
    world: &WorldStore,
    reg: &EngineRegistries,
    broken_pos: IVec3,
) -> usize {
    let dirs = [
        IVec3::X,
        -IVec3::X,
        IVec3::Y,
        -IVec3::Y,
        IVec3::Z,
        -IVec3::Z,
    ];
    let mut visited_all = HashSet::new();
    let mut cleared = 0usize;

    for d in dirs {
        let start = broken_pos + d;
        if visited_all.contains(&start) {
            continue;
        }
        let (nid, _, p2) = world.get_node(start);
        if !is_nether_portal_node(nid, reg) {
            continue;
        }

        // BFS connected portal cells
        let mut comp = Vec::new();
        let mut queue = VecDeque::new();
        visited_all.insert(start);
        queue.push_back(start);
        while let Some(cur) = queue.pop_front() {
            comp.push(cur);
            for nd in dirs {
                let np = cur + nd;
                if visited_all.contains(&np) {
                    continue;
                }
                if is_nether_portal_node(world.get_node(np).0, reg) {
                    visited_all.insert(np);
                    queue.push_back(np);
                }
            }
        }

        let axis = PortalAxis::from_param2(p2);
        let still_valid = validate_frame_for_axis(world, reg, start, axis, true)
            .is_some_and(|frame| frame.width * frame.height == comp.len() as i32);

        if !still_valid {
            for p in comp {
                world.set_node(p, CONTENT_AIR, 0, 0);
                world.mark_priority_dirty_with_neighbors(p);
                relight_around_pos(world, p, reg);
                cleared += 1;
            }
        }
    }
    cleared
}

/// Searches loaded terrain around `target` for an existing active `cubex:nether_portal`, returning its bottom interior cell.
pub fn find_existing_portal_near(
    world: &WorldStore,
    reg: &EngineRegistries,
    target: IVec3,
    radius_xz: i32,
) -> Option<IVec3> {
    let mut best: Option<(i32, IVec3)> = None;
    let r2_max = radius_xz * radius_xz;
    let min_y: i32 = if world.dimension == DimensionId::Nether {
        NETHER_LAVA_LEVEL + 2
    } else {
        8
    };
    let max_y: i32 = if world.dimension == DimensionId::Nether {
        121
    } else {
        192
    };

    let cmin_x = (target.x - radius_xz).div_euclid(16);
    let cmax_x = (target.x + radius_xz).div_euclid(16);
    let cmin_z = (target.z - radius_xz).div_euclid(16);
    let cmax_z = (target.z + radius_xz).div_euclid(16);
    let cmin_y = min_y.div_euclid(16);
    let cmax_y = max_y.div_euclid(16);

    for cy in cmin_y..=cmax_y {
        for cz in cmin_z..=cmax_z {
            for cx in cmin_x..=cmax_x {
                let Some(ch) = world.get_chunk(IVec3::new(cx, cy, cz)) else {
                    continue;
                };
                let cg = ch.read();
                for ly in 0..16usize {
                    let wy = cy * 16 + ly as i32;
                    if wy < min_y || wy > max_y {
                        continue;
                    }
                    for lz in 0..16usize {
                        let wz = cz * 16 + lz as i32;
                        let dz = wz - target.z;
                        if dz.abs() > radius_xz {
                            continue;
                        }
                        for lx in 0..16usize {
                            let wx = cx * 16 + lx as i32;
                            let dx = wx - target.x;
                            let d2 = dx * dx + dz * dz;
                            if d2 > r2_max {
                                continue;
                            }
                            let (id, _, _) = cg.get(lx, ly, lz);
                            if is_nether_portal_node(id, reg) {
                                let mut stand = IVec3::new(wx, wy, wz);
                                while stand.y > min_y
                                    && is_nether_portal_node(world.get_node(stand - IVec3::Y).0, reg)
                                {
                                    stand.y -= 1;
                                }
                                let dist = d2 + (stand.y - target.y).abs() * 4;
                                if best.map_or(true, |(bd, _)| dist < bd) {
                                    best = Some((dist, stand));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    best.map(|(_, p)| p)
}

fn is_liquid_or_air(id: u16, reg: &EngineRegistries) -> bool {
    if id == CONTENT_AIR || id == CONTENT_IGNORE {
        return true;
    }
    reg.get_node_def(id)
        .map(|d| !d.walkable || d.drawtype.is_transparent() && !is_obsidian(id, reg))
        .unwrap_or(true)
}

fn find_safe_platform_anchor(world: &WorldStore, reg: &EngineRegistries, target: IVec3) -> IVec3 {
    let y_range: Vec<i32> = if world.dimension == DimensionId::Nether {
        (36..=104).rev().collect()
    } else {
        (56..=156).rev().collect()
    };
    let mut best: Option<(i32, IVec3)> = None;
    for r in 0..=10i32 {
        for dz in -r..=r {
            for dx in -r..=r {
                if dx.abs() != r && dz.abs() != r {
                    continue;
                }
                let wx = target.x + dx;
                let wz = target.z + dz;
                for &wy in &y_range {
                    let floor = world.get_node(IVec3::new(wx, wy - 1, wz)).0;
                    let a0 = world.get_node(IVec3::new(wx, wy, wz)).0;
                    let a1 = world.get_node(IVec3::new(wx, wy + 1, wz)).0;
                    let a2 = world.get_node(IVec3::new(wx, wy + 2, wz)).0;
                    if !is_liquid_or_air(floor, reg)
                        && a0 == CONTENT_AIR
                        && a1 == CONTENT_AIR
                        && a2 == CONTENT_AIR
                    {
                        let score = dx * dx + dz * dz + (wy - target.y).abs() * 2;
                        if best.map_or(true, |(bs, _)| score < bs) {
                            best = Some((score, IVec3::new(wx, wy, wz)));
                        }
                        break;
                    }
                }
            }
        }
        if best.is_some() {
            break;
        }
    }
    best.map(|(_, p)| p).unwrap_or_else(|| {
        let fallback_y = if world.dimension == DimensionId::Nether {
            target.y.clamp(38, 96)
        } else {
            target.y.clamp(65, 118)
        };
        IVec3::new(target.x, fallback_y, target.z)
    })
}

/// Finds a nearby matching portal in `dest_world`, or carves a safe pocket + obsidian bridge/platform +
/// lit `4x5` return portal and returns the player's standing destination cell.
pub fn find_or_build_destination_portal(
    dest_world: &WorldStore,
    reg: &EngineRegistries,
    scaled_target: IVec3,
    axis: PortalAxis,
) -> IVec3 {
    if let Some(existing) = find_existing_portal_near(dest_world, reg, scaled_target, 28) {
        return existing;
    }

    let obsidian_id = reg.get_content_id("mcl_core:obsidian");
    let portal_id = reg.get_content_id(NETHER_PORTAL_NODE_ID);
    let anchor = find_safe_platform_anchor(dest_world, reg, scaled_target);
    let h_dir = axis.step_horiz();
    let d_dir = axis.step_depth();

    // 1. Build 4x3 obsidian landing platform (`w: -1..=2`, `d: -1..=1` around origin = anchor)
    if obsidian_id != crate::registry::CONTENT_UNKNOWN {
        for dd in -1..=1 {
            for dw in -1..=2 {
                let p = anchor + h_dir * dw + d_dir * dd - IVec3::Y;
                dest_world.set_node(p, obsidian_id, 0, 0);
                dest_world.mark_priority_dirty_with_neighbors(p);
            }
        }
    }

    // 2. Carve clean air pocket so arrival never suffocates inside netherrack/stone
    for dy in 0..=3 {
        for dd in -1..=1 {
            for dw in -1..=2 {
                let p = anchor + h_dir * dw + d_dir * dd + IVec3::Y * dy;
                dest_world.set_node(p, CONTENT_AIR, 0, 0);
                dest_world.mark_priority_dirty_with_neighbors(p);
            }
        }
    }

    // 3. Build 4x5 obsidian portal frame (`w: -1..=2`, `y: -1..=3` at `d = 0`)
    if obsidian_id != crate::registry::CONTENT_UNKNOWN {
        for dw in -1..=2 {
            for &dy in &[-1, 3] {
                let p = anchor + h_dir * dw + IVec3::Y * dy;
                dest_world.set_node(p, obsidian_id, 0, 0);
                dest_world.mark_priority_dirty_with_neighbors(p);
            }
        }
        for dy in 0..=2 {
            for &dw in &[-1, 2] {
                let p = anchor + h_dir * dw + IVec3::Y * dy;
                dest_world.set_node(p, obsidian_id, 0, 0);
                dest_world.mark_priority_dirty_with_neighbors(p);
            }
        }
    }

    // 4. Fill 2x3 interior with lit portal blocks and relight
    let p2 = axis.param2();
    if portal_id != crate::registry::CONTENT_UNKNOWN {
        for dy in 0..=2 {
            for dw in 0..=1 {
                let p = anchor + h_dir * dw + IVec3::Y * dy;
                dest_world.set_node(p, portal_id, 0, p2);
                dest_world.mark_priority_dirty_with_neighbors(p);
                relight_around_pos(dest_world, p, reg);
            }
        }
    }

    anchor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{DrawType, NodeDef};
    use crate::world::dimension::scale_portal_coords;
    use crate::world::nether::NETHER_LAVA_LEVEL;

    fn test_reg_with_portal() -> (EngineRegistries, u16, u16) {
        let mut reg = EngineRegistries::new();
        let obsidian = reg.register_node(NodeDef {
            name: "mcl_core:obsidian".into(),
            drawtype: DrawType::Normal,
            walkable: true,
            ..Default::default()
        });
        let lava = reg.register_node(NodeDef {
            name: "mcl_nether:nether_lava_source".into(),
            drawtype: DrawType::Liquid,
            walkable: false,
            light_source: 15,
            ..Default::default()
        });
        let portal = reg.get_content_id(NETHER_PORTAL_NODE_ID);
        assert_ne!(portal, crate::registry::CONTENT_UNKNOWN);
        (reg, obsidian, lava)
    }

    #[test]
    fn nether_portal_detection_ignition_extinguish_and_1_to_8_scaling() {
        let (reg, obsidian, lava) = test_reg_with_portal();
        let dir = std::env::temp_dir().join("cubex_rs_portal_test");
        let _ = std::fs::remove_dir_all(&dir);
        let ow = WorldStore::new(&dir, 100);

        // 1. Standard 4x5 portal along X axis without corners
        let origin = IVec3::new(10, 65, 10);
        for dx in 0..=1 {
            ow.set_node(origin + IVec3::new(dx, -1, 0), obsidian, 15, 0);
            ow.set_node(origin + IVec3::new(dx, 3, 0), obsidian, 15, 0);
        }
        for dy in 0..=2 {
            ow.set_node(origin + IVec3::new(-1, dy, 0), obsidian, 15, 0);
            ow.set_node(origin + IVec3::new(2, dy, 0), obsidian, 15, 0);
        }
        let detected = detect_portal_frame(&ow, &reg, origin).expect("4x5 frame must detect");
        assert_eq!(detected.axis, PortalAxis::X);
        assert_eq!(detected.width, 2);
        assert_eq!(detected.height, 3);

        // Incomplete frame must reject
        let broken = IVec3::new(40, 65, 40);
        for dx in 0..=1 {
            ow.set_node(broken + IVec3::new(dx, -1, 0), obsidian, 15, 0);
        }
        assert!(detect_portal_frame(&ow, &reg, broken).is_none());

        // Ignite fills all 6 cells with cubex:nether_portal
        let ignited = try_ignite_portal(&ow, &reg, origin).expect("portal must ignite");
        let portal_id = reg.get_content_id(NETHER_PORTAL_NODE_ID);
        for p in ignited.interior_positions() {
            assert_eq!(ow.get_node(p).0, portal_id);
            assert_eq!(ow.get_node(p).2, PortalAxis::X.param2());
        }

        // Breaking bottom obsidian extinguishes all 6 portal blocks
        let break_pos = origin + IVec3::new(0, -1, 0);
        ow.set_node(break_pos, CONTENT_AIR, 15, 0);
        let cleared = extinguish_portal_around(&ow, &reg, break_pos);
        assert_eq!(cleared, 6);
        for p in ignited.interior_positions() {
            assert_eq!(ow.get_node(p).0, CONTENT_AIR);
        }

        // Larger 6x7 portal along Z axis
        let z_orig = IVec3::new(-8, 66, 20);
        for dz in 0..4 {
            ow.set_node(z_orig + IVec3::new(0, -1, dz), obsidian, 15, 0);
            ow.set_node(z_orig + IVec3::new(0, 5, dz), obsidian, 15, 0);
        }
        for dy in 0..5 {
            ow.set_node(z_orig + IVec3::new(0, dy, -1), obsidian, 15, 0);
            ow.set_node(z_orig + IVec3::new(0, dy, 4), obsidian, 15, 0);
        }
        let z_ign = try_ignite_portal(&ow, &reg, z_orig).expect("larger Z portal must ignite");
        assert_eq!(z_ign.axis, PortalAxis::Z);
        assert_eq!(z_ign.width, 4);
        assert_eq!(z_ign.height, 5);
        assert_eq!(ow.get_node(z_orig).2, PortalAxis::Z.param2());

        // 1:8 coordinate scaling roundtrip
        let scaled_nether = scale_portal_coords(
            IVec3::new(800, 72, -1600),
            DimensionId::Overworld,
            DimensionId::Nether,
        );
        assert_eq!(scaled_nether.x, 100);
        assert_eq!(scaled_nether.z, -200);
        let scaled_back = scale_portal_coords(
            scaled_nether,
            DimensionId::Nether,
            DimensionId::Overworld,
        );
        assert_eq!(scaled_back.x, 800);
        assert_eq!(scaled_back.z, -1600);

        // Build safe arrival portal over open Nether lava ocean
        let nw = WorldStore::new_for_dimension(
            &dir,
            100,
            DimensionId::Nether,
            crate::world::WorldGenType::Default,
            crate::world::Difficulty::Normal,
            crate::world::Gamerules::default(),
            "Survival",
        );
        for dz in -6..=6 {
            for dx in -6..=6 {
                for y in 28..=NETHER_LAVA_LEVEL {
                    nw.set_node(IVec3::new(scaled_nether.x + dx, y, scaled_nether.z + dz), lava, 0, 0);
                }
            }
        }
        let dest = find_or_build_destination_portal(&nw, &reg, scaled_nether, PortalAxis::X);
        assert_eq!(nw.get_node(dest).0, portal_id);
        assert_eq!(nw.get_node(dest - IVec3::Y).0, obsidian);
        assert_eq!(nw.get_node(dest + IVec3::new(0, -1, 1)).0, obsidian);
        assert_eq!(nw.get_node(dest + IVec3::new(0, -1, -1)).0, obsidian);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
