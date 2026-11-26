use glam::{IVec3, Mat4, Vec3};

use crate::config::EngineSettings;
use crate::physics::collision::world_collision_boxes;
use crate::physics::raycast::box_line_collision;
use crate::physics::PlayerController;
use crate::registry::{EngineRegistries, CONTENT_AIR, CONTENT_IGNORE};
use crate::world::WorldStore;

pub const IDEAL_THIRD_PERSON_DIST: f32 = 4.0;
pub const CAMERA_WALL_MARGIN: f32 = 0.30;
pub const MIN_THIRD_PERSON_DIST: f32 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CameraMode {
    #[default]
    FirstPerson,
    ThirdPersonBack,
    ThirdPersonFront,
}

impl CameraMode {
    pub fn cycle(self) -> Self {
        match self {
            Self::FirstPerson => Self::ThirdPersonBack,
            Self::ThirdPersonBack => Self::ThirdPersonFront,
            Self::ThirdPersonFront => Self::FirstPerson,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::FirstPerson => "First Person",
            Self::ThirdPersonBack => "Third Person Back",
            Self::ThirdPersonFront => "Third Person Front",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CameraRig {
    pub eye_pos: Vec3,
    pub look_dir: Vec3,
    pub view_proj: Mat4,
    pub show_viewmodel: bool,
    pub show_player_model: bool,
    pub effective_distance: f32,
}

/// Raycast a single segment from `origin` along `unit_dir` up to `max_dist` against `walkable`
/// terrain collision boxes, returning the closest hit distance if obstructed.
fn sweep_walkable_ray(
    origin: Vec3,
    unit_dir: Vec3,
    max_dist: f32,
    world: &WorldStore,
    registries: &EngineRegistries,
) -> Option<f32> {
    let d = unit_dir.normalize_or_zero();
    if d.length_squared() < 1e-8 || max_dist <= 0.0 {
        return None;
    }
    let ray_vec = d * max_dist;

    let mut cell = origin.floor().as_ivec3();
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
        if s > 0 { (c + 1) as f32 - o } else { o - c as f32 }
    };
    let mut t_max = Vec3::new(
        next_bound(origin.x, step.x, cell.x) * t_delta.x,
        next_bound(origin.y, step.y, cell.y) * t_delta.y,
        next_bound(origin.z, step.z, cell.z) * t_delta.z,
    );
    let mut dist = 0.0f32;
    let mut best_hit: Option<f32> = None;

    for _ in 0..64 {
        if dist > max_dist {
            break;
        }
        let (id, _, p2) = world.get_node(cell);
        if id != CONTENT_AIR && id != CONTENT_IGNORE {
            for b in world_collision_boxes(cell, id, p2, world, registries) {
                if let Some((ipt, _)) = box_line_collision(b.min, b.max, origin, ray_vec) {
                    let hit_d = ipt.distance(origin);
                    if hit_d > 1e-4 && hit_d <= max_dist {
                        best_hit = Some(match best_hit {
                            Some(prev) => prev.min(hit_d),
                            None => hit_d,
                        });
                    }
                }
            }
            if best_hit.is_some() {
                return best_hit;
            }
        }

        if t_max.x < t_max.y && t_max.x < t_max.z {
            dist = t_max.x;
            t_max.x += t_delta.x;
            cell.x += step.x;
        } else if t_max.y < t_max.z {
            dist = t_max.y;
            t_max.y += t_delta.y;
            cell.y += step.y;
        } else {
            dist = t_max.z;
            t_max.z += t_delta.z;
            cell.z += step.z;
        }
    }
    best_hit
}

/// Sweep central camera ray + small near-plane corner offsets against `walkable` terrain voxels.
fn resolve_third_person_distance(
    pivot: Vec3,
    offset_dir: Vec3,
    ideal_dist: f32,
    world: &WorldStore,
    registries: &EngineRegistries,
) -> f32 {
    let dir = offset_dir.normalize_or_zero();
    if dir.length_squared() < 1e-8 {
        return ideal_dist;
    }

    let mut right = Vec3::Y.cross(dir);
    if right.length_squared() < 1e-6 {
        right = Vec3::X;
    } else {
        right = right.normalize();
    }
    let up = dir.cross(right).normalize();
    let r_off = 0.12;
    let offsets = [
        Vec3::ZERO,
        right * r_off + up * r_off,
        -right * r_off + up * r_off,
        right * r_off - up * r_off,
        -right * r_off - up * r_off,
    ];

    let mut min_dist = ideal_dist;
    for off in offsets {
        if let Some(hit_dist) = sweep_walkable_ray(pivot + off, dir, ideal_dist, world, registries) {
            let pulled = (hit_dist - CAMERA_WALL_MARGIN).clamp(MIN_THIRD_PERSON_DIST, ideal_dist);
            if pulled < min_dist {
                min_dist = pulled;
            }
        }
    }
    min_dist
}

/// Compute camera eye position, view direction, projection matrix, and model visibility flags
/// for `FirstPerson`, `ThirdPersonBack`, and `ThirdPersonFront` modes.
pub fn compute_camera_rig(
    mode: CameraMode,
    player: &PlayerController,
    aspect: f32,
    settings: &EngineSettings,
    world: &WorldStore,
    registries: &EngineRegistries,
) -> CameraRig {
    let fov_rad = player.current_fov.to_radians();
    let far = (settings.render_distance_chunks as f32) * 16.0 + 48.0;
    let proj = Mat4::perspective_rh(fov_rad, aspect.max(0.1), 0.1, far);
    let anchor = player.eye_pos();
    let player_look = player.look_dir();

    match mode {
        CameraMode::FirstPerson => {
            let view = Mat4::look_to_rh(anchor, player_look, Vec3::Y);
            CameraRig {
                eye_pos: anchor,
                look_dir: player_look,
                view_proj: proj * view,
                show_viewmodel: true,
                show_player_model: false,
                effective_distance: 0.0,
            }
        }
        CameraMode::ThirdPersonBack => {
            let offset_dir = -player_look;
            let eff_dist = resolve_third_person_distance(
                anchor,
                offset_dir,
                IDEAL_THIRD_PERSON_DIST,
                world,
                registries,
            );
            let eye_pos = anchor + offset_dir * eff_dist;
            let look_dir = (anchor - eye_pos).normalize_or_zero();
            let view = Mat4::look_to_rh(eye_pos, look_dir, Vec3::Y);
            CameraRig {
                eye_pos,
                look_dir,
                view_proj: proj * view,
                show_viewmodel: false,
                show_player_model: true,
                effective_distance: eff_dist,
            }
        }
        CameraMode::ThirdPersonFront => {
            let offset_dir = player_look;
            let eff_dist = resolve_third_person_distance(
                anchor,
                offset_dir,
                IDEAL_THIRD_PERSON_DIST,
                world,
                registries,
            );
            let eye_pos = anchor + offset_dir * eff_dist;
            let look_dir = (anchor - eye_pos).normalize_or_zero();
            let view = Mat4::look_to_rh(eye_pos, look_dir, Vec3::Y);
            CameraRig {
                eye_pos,
                look_dir,
                view_proj: proj * view,
                show_viewmodel: false,
                show_player_model: true,
                effective_distance: eff_dist,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{DrawType, NodeDef};

    #[test]
    fn camera_mode_cycles_and_pulls_in_on_walkable_obstruction() {
        let m0 = CameraMode::FirstPerson;
        let m1 = m0.cycle();
        let m2 = m1.cycle();
        let m3 = m2.cycle();
        assert_eq!(m1, CameraMode::ThirdPersonBack);
        assert_eq!(m2, CameraMode::ThirdPersonFront);
        assert_eq!(m3, CameraMode::FirstPerson);
        assert_eq!(m0.label(), "First Person");
        assert_eq!(m1.label(), "Third Person Back");
        assert_eq!(m2.label(), "Third Person Front");

        let mut reg = EngineRegistries::new();
        let stone = reg.register_node(NodeDef {
            name: "mcl_core:stone".into(),
            drawtype: DrawType::Normal,
            walkable: true,
            pointable: true,
            ..Default::default()
        });
        let grass = reg.register_node(NodeDef {
            name: "mcl_flowers:tallgrass".into(),
            drawtype: DrawType::Plantlike,
            walkable: false,
            pointable: true,
            ..Default::default()
        });

        let tmp = std::env::temp_dir().join("cubex_cam_rig_test");
        let _ = std::fs::remove_dir_all(&tmp);
        let world = WorldStore::new(&tmp, 42);

        let mut player = PlayerController::default();
        player.pos = Vec3::new(0.5, 1.0, 0.5);
        player.yaw = 0.0; // look_dir = +Z
        player.pitch = 0.0;
        let settings = EngineSettings::default();

        // Non-walkable tall grass behind player does NOT pull camera in
        world.set_node(IVec3::new(0, 2, -1), grass, 15, 0);
        let open_rig =
            compute_camera_rig(CameraMode::ThirdPersonBack, &player, 16.0 / 9.0, &settings, &world, &reg);
        assert!(!open_rig.show_viewmodel);
        assert!(open_rig.show_player_model);
        assert!((open_rig.effective_distance - 4.0).abs() < 1e-4);

        // Solid stone wall at z = -2 (face at z = -1.0 -> hit_dist = 1.5 from z = 0.5) pulls camera to ~1.20
        world.set_node(IVec3::new(0, 2, -2), stone, 15, 0);
        let blocked_rig =
            compute_camera_rig(CameraMode::ThirdPersonBack, &player, 16.0 / 9.0, &settings, &world, &reg);
        assert!(
            blocked_rig.effective_distance >= 0.5 && blocked_rig.effective_distance < 1.5,
            "Expected pull-in in [0.5, 1.5), got {}",
            blocked_rig.effective_distance
        );
        assert!(blocked_rig.eye_pos.z > -1.0, "Camera clipped behind wall: {:?}", blocked_rig.eye_pos);

        // Front wall at z = 2 (face at z = 2.0 -> hit_dist = 1.5) pulls ThirdPersonFront camera inward too
        world.set_node(IVec3::new(0, 2, 2), stone, 15, 0);
        let front_rig =
            compute_camera_rig(CameraMode::ThirdPersonFront, &player, 16.0 / 9.0, &settings, &world, &reg);
        assert!(front_rig.show_player_model && !front_rig.show_viewmodel);
        assert!(
            front_rig.effective_distance >= 0.5 && front_rig.effective_distance < 1.5,
            "Front camera pull-in got {}",
            front_rig.effective_distance
        );
        assert!(front_rig.eye_pos.z < 2.0);

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
