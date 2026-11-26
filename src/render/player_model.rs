use glam::{EulerRot, Mat4, Quat, Vec3};
use image::RgbaImage;

use crate::registry::EngineRegistries;
use crate::render::mesher::VoxelVertex;
use crate::render::viewmodel::build_viewmodel_mesh;
use crate::textures::TextureAtlas;

/// Total standing height of the voxel character in world blocks (matches `PlayerController`).
pub const PLAYER_MODEL_HEIGHT: f32 = 1.80;
/// World blocks per skin pixel (`1.80 / 32 = 0.05625`).
pub const PIXEL_SCALE: f32 = PLAYER_MODEL_HEIGHT / 32.0;

#[derive(Debug, Clone)]
pub struct PlayerModelState {
    pub body_yaw: f32,
    pub head_yaw: f32,
    pub head_pitch: f32,
    pub walk_phase: f32,
    pub walk_blend: f32,
    pub swing_progress: f32,
    pub is_sneaking: bool,
    pub held_item: Option<String>,
    pub is_swinging: bool,
}

impl Default for PlayerModelState {
    fn default() -> Self {
        Self::new()
    }
}

impl PlayerModelState {
    pub fn new() -> Self {
        Self {
            body_yaw: 0.0,
            head_yaw: 0.0,
            head_pitch: 0.0,
            walk_phase: 0.0,
            walk_blend: 0.0,
            swing_progress: 0.0,
            is_sneaking: false,
            held_item: None,
            is_swinging: false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        dt: f32,
        target_yaw: f32,
        target_pitch: f32,
        horizontal_speed: f32,
        is_sneaking: bool,
        is_swinging: bool,
        swing_progress: f32,
        held_item: Option<String>,
    ) {
        let dt = dt.clamp(0.0, 0.1);
        self.head_yaw = target_yaw;
        self.head_pitch = target_pitch.clamp(-1.55, 1.55);
        self.is_sneaking = is_sneaking;
        self.is_swinging = is_swinging;
        self.swing_progress = if is_swinging {
            swing_progress.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.held_item = held_item;

        // Smoothly pull body_yaw toward head_yaw when moving or when head twists past ~35 deg
        let diff = wrap_angle_rad(self.head_yaw - self.body_yaw);
        let max_twist = 35.0f32.to_radians();
        if diff.abs() > max_twist {
            let excess = diff.abs() - max_twist;
            self.body_yaw += diff.signum() * excess;
        }
        let follow_diff = wrap_angle_rad(self.head_yaw - self.body_yaw);
        let follow_rate = if horizontal_speed > 0.2 { 10.0 } else { 2.0 };
        self.body_yaw += follow_diff * (dt * follow_rate).clamp(0.0, 1.0);

        // Smooth walking/sprinting pendulum blend
        let moving = horizontal_speed > 0.15;
        let target_blend = if moving {
            (horizontal_speed / 4.3).clamp(0.35, 1.15)
        } else {
            0.0
        };
        self.walk_blend += (target_blend - self.walk_blend) * (dt * 12.0).clamp(0.0, 1.0);
        if moving {
            let freq = (horizontal_speed * 1.9).clamp(3.2, 14.0);
            self.walk_phase = (self.walk_phase + dt * freq) % std::f32::consts::TAU;
        } else if self.walk_blend < 0.01 {
            self.walk_blend = 0.0;
        }
    }

    /// Current leg pendulum swing angle in radians (bounded within ~+-0.8 rad when sprinting, +-0.7 walking).
    pub fn leg_swing_rad(&self) -> f32 {
        self.walk_phase.sin() * 0.7 * self.walk_blend.clamp(0.0, 1.15)
    }
}

#[inline]
pub fn wrap_angle_rad(mut a: f32) -> f32 {
    while a > std::f32::consts::PI {
        a -= std::f32::consts::TAU;
    }
    while a < -std::f32::consts::PI {
        a += std::f32::consts::TAU;
    }
    a
}

#[inline]
pub fn is_64x64_skin(skin_w: u32, skin_h: u32) -> bool {
    skin_w > 0 && skin_h >= skin_w
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlayerPartCounts {
    pub head_verts: usize,
    pub torso_verts: usize,
    pub left_arm_verts: usize,
    pub right_arm_verts: usize,
    pub left_leg_verts: usize,
    pub right_leg_verts: usize,
    pub overlay_verts: usize,
}

#[derive(Debug, Clone)]
pub struct PlayerModelMesh {
    pub body_verts: Vec<VoxelVertex>,
    pub body_indices: Vec<u32>,
    pub held_verts: Vec<VoxelVertex>,
    pub held_indices: Vec<u32>,
    pub held_texture: Option<RgbaImage>,
    pub part_counts: PlayerPartCounts,
}

#[derive(Copy, Clone)]
struct BoxUvSpec {
    ox: f32,
    oy: f32,
    w: f32,
    h: f32,
    d: f32,
    mirror_x: bool,
}

impl BoxUvSpec {
    /// Returns per-face `(u0, v0, u1, v1)` normalized UVs ordered `[+X, -X, +Y, -Y, +Z, -Z]`.
    fn face_uvs(self, tex_w: f32, tex_h: f32) -> [[f32; 4]; 6] {
        let rect = |x0: f32, y0: f32, x1: f32, y1: f32, flip_u: bool| -> [f32; 4] {
            let u0 = x0 / tex_w;
            let u1 = x1 / tex_w;
            let v0 = y0 / tex_h;
            let v1 = y1 / tex_h;
            if flip_u {
                [u1, v0, u0, v1]
            } else {
                [u0, v0, u1, v1]
            }
        };

        let (ox, oy, w, h, d) = (self.ox, self.oy, self.w, self.h, self.d);
        // Classic Minecraft skin net layout:
        // Right (-X) : [ox, oy+d .. ox+d, oy+d+h]
        // Front (+Z) : [ox+d, oy+d .. ox+d+w, oy+d+h]
        // Left  (+X) : [ox+d+w, oy+d .. ox+2d+w, oy+d+h]
        // Back  (-Z) : [ox+2d+w, oy+d .. ox+2d+2w, oy+d+h]
        // Top   (+Y) : [ox+d, oy .. ox+d+w, oy+d]
        // Bot   (-Y) : [ox+d+w, oy .. ox+d+2w, oy+d]
        let uv_right = rect(ox, oy + d, ox + d, oy + d + h, self.mirror_x);
        let uv_front = rect(ox + d, oy + d, ox + d + w, oy + d + h, self.mirror_x);
        let uv_left = rect(ox + d + w, oy + d, ox + 2.0 * d + w, oy + d + h, self.mirror_x);
        let uv_back = rect(
            ox + 2.0 * d + w,
            oy + d,
            ox + 2.0 * d + 2.0 * w,
            oy + d + h,
            self.mirror_x,
        );
        let uv_top = rect(ox + d, oy, ox + d + w, oy + d, self.mirror_x);
        let uv_bot = rect(ox + d + w, oy, ox + d + 2.0 * w, oy + d, self.mirror_x);

        if self.mirror_x {
            [uv_right, uv_left, uv_top, uv_bot, uv_front, uv_back]
        } else {
            [uv_left, uv_right, uv_top, uv_bot, uv_front, uv_back]
        }
    }
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
fn emit_part_box(
    verts: &mut Vec<VoxelVertex>,
    indices: &mut Vec<u32>,
    transform: Mat4,
    min: Vec3,
    max: Vec3,
    uv_spec: BoxUvSpec,
    tex_w: f32,
    tex_h: f32,
    sky: u8,
    block: u8,
    cutout: bool,
) -> usize {
    let start_verts = verts.len();
    let [x0, y0, z0] = min.to_array();
    let [x1, y1, z1] = max.to_array();
    let face_uvs = uv_spec.face_uvs(tex_w, tex_h);

    // Outward CCW quads ordered [+X, -X, +Y, -Y, +Z, -Z] matching shader & culling.
    let faces: [(Vec3, [[f32; 3]; 4]); 6] = [
        (Vec3::X, [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]]),
        (Vec3::NEG_X, [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]]),
        (Vec3::Y, [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]]),
        (Vec3::NEG_Y, [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]),
        (Vec3::Z, [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]),
        (Vec3::NEG_Z, [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]]),
    ];

    for (fidx, (local_n, pts)) in faces.into_iter().enumerate() {
        let [u0, v0, u1, v1] = face_uvs[fidx];
        let uvs = [[u0, v1], [u1, v1], [u1, v0], [u0, v0]];
        let world_n = transform.transform_vector3(local_n).normalize_or_zero();
        let fdir = dominant_face_dir(world_n);
        let packed = VoxelVertex::pack(0, fdir, sky, block, 3, false, cutout);
        let base = verts.len() as u32;
        for (p, uv) in pts.into_iter().zip(uvs) {
            let wp = transform.transform_point3(Vec3::from_array(p));
            verts.push(VoxelVertex {
                pos: wp.to_array(),
                uv,
                packed,
                tint: [255, 255, 255, 255],
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    verts.len() - start_verts
}

/// Build an articulated 6-part UV-mapped 3D character mesh (plus optional overlays & held item)
/// with feet at `pos.y`, eye at `pos.y + 1.625`, and top at `pos.y + 1.80`.
pub fn build_player_model_vertices(
    state: &PlayerModelState,
    pos: Vec3,
    skin_w: u32,
    skin_h: u32,
    registries: &EngineRegistries,
    atlas: &TextureAtlas,
) -> PlayerModelMesh {
    build_player_model_vertices_lit(state, pos, skin_w, skin_h, 15, 0, registries, atlas)
}

#[allow(clippy::too_many_arguments)]
pub fn build_player_model_vertices_lit(
    state: &PlayerModelState,
    pos: Vec3,
    skin_w: u32,
    skin_h: u32,
    sky: u8,
    block: u8,
    registries: &EngineRegistries,
    atlas: &TextureAtlas,
) -> PlayerModelMesh {
    let p = PIXEL_SCALE;
    let is_64 = is_64x64_skin(skin_w, skin_h);
    let tex_w = 64.0f32;
    let tex_h = if is_64 { 64.0f32 } else { 32.0f32 };

    let mut body_verts = Vec::with_capacity(288);
    let mut body_indices = Vec::with_capacity(432);
    let mut counts = PlayerPartCounts::default();

    // Root world placement (local +Z aligns with look_dir = (-sin(yaw), 0, cos(yaw)))
    let swing_arc = if state.is_swinging || state.swing_progress > 0.001 {
        (state.swing_progress.clamp(0.0, 1.0) * std::f32::consts::PI).sin()
    } else {
        0.0
    };
    let torso_twist = 0.18 * swing_arc;
    let body_root = Mat4::from_translation(pos) * Mat4::from_rotation_y(-state.body_yaw);

    // Sneaking bends torso forward by 0.4 rad and crouches legs while keeping feet >= y=0
    let (hip_y, hip_z, torso_pitch, leg_sneak_pitch) = if state.is_sneaking {
        (11.30 * p, -3.4 * p, 0.40f32, -0.55f32)
    } else {
        (12.0 * p, 0.0, 0.0, 0.0)
    };

    let leg_swing = state.leg_swing_rad();

    // 1. Torso (8x12x4) pivoted at hip
    let torso_mat = body_root
        * Mat4::from_translation(Vec3::new(0.0, hip_y, hip_z))
        * Mat4::from_rotation_y(torso_twist)
        * Mat4::from_rotation_x(torso_pitch);
    counts.torso_verts += emit_part_box(
        &mut body_verts,
        &mut body_indices,
        torso_mat,
        Vec3::new(-4.0 * p, 0.0, -2.0 * p),
        Vec3::new(4.0 * p, 12.0 * p, 2.0 * p),
        BoxUvSpec {
            ox: 16.0,
            oy: 16.0,
            w: 8.0,
            h: 12.0,
            d: 4.0,
            mirror_x: false,
        },
        tex_w,
        tex_h,
        sky,
        block,
        false,
    );

    // 2. Head (8x8x8) pivoted at top of torso (neck), yaw/pitch relative to body
    let rel_head_yaw = wrap_angle_rad(state.head_yaw - state.body_yaw);
    let neck_pos = torso_mat.transform_point3(Vec3::new(0.0, 12.0 * p, 0.0));
    let head_mat = Mat4::from_translation(neck_pos)
        * Mat4::from_rotation_y(-(state.body_yaw + rel_head_yaw))
        * Mat4::from_rotation_x(-state.head_pitch);
    counts.head_verts += emit_part_box(
        &mut body_verts,
        &mut body_indices,
        head_mat,
        Vec3::new(-4.0 * p, 0.0, -4.0 * p),
        Vec3::new(4.0 * p, 8.0 * p, 4.0 * p),
        BoxUvSpec {
            ox: 0.0,
            oy: 0.0,
            w: 8.0,
            h: 8.0,
            d: 8.0,
            mirror_x: false,
        },
        tex_w,
        tex_h,
        sky,
        block,
        false,
    );

    // 3. Right Arm (4x12x4) pivoted at (-6P, 10P, 0) in torso space
    let has_held = state
        .held_item
        .as_deref()
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);
    let r_arm_idle = if has_held { -0.22 } else { 0.0 };
    let r_arm_pitch = leg_swing + r_arm_idle - 1.35 * swing_arc;
    let r_arm_yaw = 0.35 * swing_arc;
    let r_arm_roll = -0.04 - 0.22 * swing_arc;
    let r_arm_mat = torso_mat
        * Mat4::from_translation(Vec3::new(-6.0 * p, 10.0 * p, 0.0))
        * Mat4::from_euler(EulerRot::YXZ, r_arm_yaw, r_arm_pitch, r_arm_roll);
    counts.right_arm_verts += emit_part_box(
        &mut body_verts,
        &mut body_indices,
        r_arm_mat,
        Vec3::new(-2.0 * p, -10.0 * p, -2.0 * p),
        Vec3::new(2.0 * p, 2.0 * p, 2.0 * p),
        BoxUvSpec {
            ox: 40.0,
            oy: 16.0,
            w: 4.0,
            h: 12.0,
            d: 4.0,
            mirror_x: false,
        },
        tex_w,
        tex_h,
        sky,
        block,
        false,
    );

    // 4. Left Arm (4x12x4) pivoted at (+6P, 10P, 0) in torso space
    let l_arm_pitch = -leg_swing;
    let l_arm_mat = torso_mat
        * Mat4::from_translation(Vec3::new(6.0 * p, 10.0 * p, 0.0))
        * Mat4::from_euler(EulerRot::YXZ, 0.0, l_arm_pitch, 0.04);
    let l_arm_uv = if is_64 {
        BoxUvSpec {
            ox: 32.0,
            oy: 48.0,
            w: 4.0,
            h: 12.0,
            d: 4.0,
            mirror_x: false,
        }
    } else {
        BoxUvSpec {
            ox: 40.0,
            oy: 16.0,
            w: 4.0,
            h: 12.0,
            d: 4.0,
            mirror_x: true,
        }
    };
    counts.left_arm_verts += emit_part_box(
        &mut body_verts,
        &mut body_indices,
        l_arm_mat,
        Vec3::new(-2.0 * p, -10.0 * p, -2.0 * p),
        Vec3::new(2.0 * p, 2.0 * p, 2.0 * p),
        l_arm_uv,
        tex_w,
        tex_h,
        sky,
        block,
        false,
    );

    // 5. Right Leg (4x12x4) pivoted at (-2P, hip_y, hip_z)
    let r_leg_mat = body_root
        * Mat4::from_translation(Vec3::new(-2.0 * p, hip_y, hip_z))
        * Mat4::from_rotation_x(leg_sneak_pitch - leg_swing);
    counts.right_leg_verts += emit_part_box(
        &mut body_verts,
        &mut body_indices,
        r_leg_mat,
        Vec3::new(-2.0 * p, -12.0 * p, -2.0 * p),
        Vec3::new(2.0 * p, 0.0, 2.0 * p),
        BoxUvSpec {
            ox: 0.0,
            oy: 16.0,
            w: 4.0,
            h: 12.0,
            d: 4.0,
            mirror_x: false,
        },
        tex_w,
        tex_h,
        sky,
        block,
        false,
    );

    // 6. Left Leg (4x12x4) pivoted at (+2P, hip_y, hip_z)
    let l_leg_mat = body_root
        * Mat4::from_translation(Vec3::new(2.0 * p, hip_y, hip_z))
        * Mat4::from_rotation_x(leg_sneak_pitch + leg_swing);
    let l_leg_uv = if is_64 {
        BoxUvSpec {
            ox: 16.0,
            oy: 48.0,
            w: 4.0,
            h: 12.0,
            d: 4.0,
            mirror_x: false,
        }
    } else {
        BoxUvSpec {
            ox: 0.0,
            oy: 16.0,
            w: 4.0,
            h: 12.0,
            d: 4.0,
            mirror_x: true,
        }
    };
    counts.left_leg_verts += emit_part_box(
        &mut body_verts,
        &mut body_indices,
        l_leg_mat,
        Vec3::new(-2.0 * p, -12.0 * p, -2.0 * p),
        Vec3::new(2.0 * p, 0.0, 2.0 * p),
        l_leg_uv,
        tex_w,
        tex_h,
        sky,
        block,
        false,
    );

    // Hat overlay (present in both 64x32 and 64x64 at UV (32, 0), alpha-cutout)
    counts.overlay_verts += emit_part_box(
        &mut body_verts,
        &mut body_indices,
        head_mat,
        Vec3::new(-4.45 * p, -0.45 * p, -4.45 * p),
        Vec3::new(4.45 * p, 8.45 * p, 4.45 * p),
        BoxUvSpec {
            ox: 32.0,
            oy: 0.0,
            w: 8.0,
            h: 8.0,
            d: 8.0,
            mirror_x: false,
        },
        tex_w,
        tex_h,
        sky,
        block,
        true,
    );

    // 64x64 second-layer overlays (Jacket, Left/Right Sleeves, Left/Right Pants)
    if is_64 {
        let ov = 0.25 * p;
        counts.overlay_verts += emit_part_box(
            &mut body_verts,
            &mut body_indices,
            torso_mat,
            Vec3::new(-4.0 * p - ov, -ov, -2.0 * p - ov),
            Vec3::new(4.0 * p + ov, 12.0 * p + ov, 2.0 * p + ov),
            BoxUvSpec {
                ox: 16.0,
                oy: 32.0,
                w: 8.0,
                h: 12.0,
                d: 4.0,
                mirror_x: false,
            },
            tex_w,
            tex_h,
            sky,
            block,
            true,
        );
        counts.overlay_verts += emit_part_box(
            &mut body_verts,
            &mut body_indices,
            r_arm_mat,
            Vec3::new(-2.0 * p - ov, -10.0 * p - ov, -2.0 * p - ov),
            Vec3::new(2.0 * p + ov, 2.0 * p + ov, 2.0 * p + ov),
            BoxUvSpec {
                ox: 40.0,
                oy: 32.0,
                w: 4.0,
                h: 12.0,
                d: 4.0,
                mirror_x: false,
            },
            tex_w,
            tex_h,
            sky,
            block,
            true,
        );
        counts.overlay_verts += emit_part_box(
            &mut body_verts,
            &mut body_indices,
            l_arm_mat,
            Vec3::new(-2.0 * p - ov, -10.0 * p - ov, -2.0 * p - ov),
            Vec3::new(2.0 * p + ov, 2.0 * p + ov, 2.0 * p + ov),
            BoxUvSpec {
                ox: 48.0,
                oy: 48.0,
                w: 4.0,
                h: 12.0,
                d: 4.0,
                mirror_x: false,
            },
            tex_w,
            tex_h,
            sky,
            block,
            true,
        );
        counts.overlay_verts += emit_part_box(
            &mut body_verts,
            &mut body_indices,
            r_leg_mat,
            Vec3::new(-2.0 * p - ov, -12.0 * p - ov, -2.0 * p - ov),
            Vec3::new(2.0 * p + ov, ov, 2.0 * p + ov),
            BoxUvSpec {
                ox: 0.0,
                oy: 32.0,
                w: 4.0,
                h: 12.0,
                d: 4.0,
                mirror_x: false,
            },
            tex_w,
            tex_h,
            sky,
            block,
            true,
        );
        counts.overlay_verts += emit_part_box(
            &mut body_verts,
            &mut body_indices,
            l_leg_mat,
            Vec3::new(-2.0 * p - ov, -12.0 * p - ov, -2.0 * p - ov),
            Vec3::new(2.0 * p + ov, ov, 2.0 * p + ov),
            BoxUvSpec {
                ox: 0.0,
                oy: 48.0,
                w: 4.0,
                h: 12.0,
                d: 4.0,
                mirror_x: false,
            },
            tex_w,
            tex_h,
            sky,
            block,
            true,
        );
    }

    // 7. Held Item / Block attached to Right Hand in 3rd-person
    let mut held_verts = Vec::new();
    let mut held_indices = Vec::new();
    let mut held_texture = None;

    if let Some(ref item_raw) = state.held_item {
        let item_key = item_raw.split_whitespace().next().unwrap_or("").trim();
        if !item_key.is_empty() {
            let vm = build_viewmodel_mesh(Some(item_key), registries, atlas);
            if !vm.is_arm && !vm.verts.is_empty() && !vm.indices.is_empty() {
                let item_local = if vm.is_node_block {
                    Mat4::from_scale_rotation_translation(
                        Vec3::splat(0.32),
                        Quat::from_euler(EulerRot::YXZ, 0.45, 0.25, 0.0),
                        Vec3::new(0.0, -9.5 * p, 3.2 * p),
                    )
                } else {
                    Mat4::from_scale_rotation_translation(
                        Vec3::splat(0.56),
                        Quat::from_euler(EulerRot::YXZ, -std::f32::consts::FRAC_PI_2, 0.35, 0.0),
                        Vec3::new(0.0, -8.5 * p + 0.14, 1.5 * p + 0.16),
                    )
                };
                let item_world = r_arm_mat * item_local;
                held_verts.reserve(vm.verts.len());
                for v in &vm.verts {
                    let wp = item_world.transform_point3(Vec3::from_array(v.pos));
                    let layer = (v.packed & 0x7FF) as u16;
                    let old_fdir = ((v.packed >> 11) & 0x7) as u8;
                    let local_n = match old_fdir {
                        0 => Vec3::X,
                        1 => Vec3::NEG_X,
                        2 => Vec3::Y,
                        3 => Vec3::NEG_Y,
                        4 => Vec3::Z,
                        _ => Vec3::NEG_Z,
                    };
                    let world_n = item_world.transform_vector3(local_n).normalize_or_zero();
                    let new_fdir = dominant_face_dir(world_n);
                    let cutout = ((v.packed >> 25) & 0x1) != 0;
                    let packed = VoxelVertex::pack(layer, new_fdir, sky, block, 3, false, cutout);
                    held_verts.push(VoxelVertex {
                        pos: wp.to_array(),
                        uv: v.uv,
                        packed,
                        tint: v.tint,
                    });
                }
                held_indices = vm.indices;
                held_texture = vm.texture;
            }
        }
    }

    PlayerModelMesh {
        body_verts,
        body_indices,
        held_verts,
        held_indices,
        held_texture,
        part_counts: counts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{DrawType, ItemDef, NodeDef};
    use std::collections::HashMap;
    use std::path::Path;

    #[test]
    fn six_part_player_model_handles_64x32_and_64x64_skins_and_held_items() {
        let mut reg = EngineRegistries::new();
        reg.register_node(NodeDef {
            name: "mcl_core:stone".into(),
            drawtype: DrawType::Normal,
            ..Default::default()
        });
        reg.register_item(ItemDef {
            name: "mcl_core:stone".into(),
            is_node: true,
            ..Default::default()
        });
        let atlas = TextureAtlas::build(&HashMap::new(), Path::new(""), None, &reg);

        let mut state = PlayerModelState::new();
        state.held_item = Some("mcl_core:stone".into());

        // 1. 64x32 skin layout
        let m32 = build_player_model_vertices(&state, Vec3::ZERO, 64, 32, &reg, &atlas);
        assert_eq!(m32.part_counts.head_verts, 24);
        assert_eq!(m32.part_counts.torso_verts, 24);
        assert_eq!(m32.part_counts.left_arm_verts, 24);
        assert_eq!(m32.part_counts.right_arm_verts, 24);
        assert_eq!(m32.part_counts.left_leg_verts, 24);
        assert_eq!(m32.part_counts.right_leg_verts, 24);
        assert_eq!(m32.part_counts.overlay_verts, 24); // Hat only on 64x32
        assert!(!m32.held_verts.is_empty(), "Expected 3P held block mesh");

        // Verify proportions: feet=0.0, top=1.80 (excluding hat slightly above 1.80)
        let min_y = m32.body_verts[..144]
            .iter()
            .map(|v| v.pos[1])
            .fold(f32::INFINITY, f32::min);
        let max_y = m32.body_verts[..144]
            .iter()
            .map(|v| v.pos[1])
            .fold(f32::NEG_INFINITY, f32::max);
        assert!((min_y - 0.0).abs() < 1e-4, "feet y={min_y}");
        assert!((max_y - 1.80).abs() < 1e-4, "top y={max_y}");

        // All UVs stay valid inside [0, 1]
        for v in &m32.body_verts {
            assert!((0.0..=1.0).contains(&v.uv[0]) && (0.0..=1.0).contains(&v.uv[1]));
        }

        // 2. 64x64 skin layout adds all second-layer overlays & uses bottom-half UVs (v > 0.5)
        let m64 = build_player_model_vertices(&state, Vec3::ZERO, 64, 64, &reg, &atlas);
        assert_eq!(m64.part_counts.overlay_verts, 24 * 6);
        assert!(
            m64.body_verts.iter().any(|v| v.uv[1] > 0.7),
            "64x64 left arm/leg should use y=48..64 UV region (v > 0.7)"
        );
    }

    #[test]
    fn player_model_walk_sneak_swing_and_body_yaw_follow_work() {
        let reg = EngineRegistries::new();
        let atlas = TextureAtlas::build(&HashMap::new(), Path::new(""), None, &reg);
        let mut state = PlayerModelState::new();

        // Turn head > 35 deg: body_yaw follows so twist never exceeds ~35 deg
        state.update(0.05, 1.2, 0.25, 0.0, false, false, 0.0, None);
        let twist = wrap_angle_rad(state.head_yaw - state.body_yaw).abs();
        assert!(twist <= 35.1f32.to_radians(), "Head twist={twist}");

        // Walk cycle drives smooth +-0.7 rad swing
        for _ in 0..10 {
            state.update(0.05, 1.2, 0.0, 4.3, false, false, 0.0, None);
        }
        assert!(state.walk_blend > 0.5);
        assert!(state.leg_swing_rad().abs() <= 0.85);

        // Sneak drops head below standing 1.80 and bends torso forward without sinking feet
        state.is_sneaking = true;
        state.walk_blend = 0.0;
        let ms = build_player_model_vertices(&state, Vec3::ZERO, 64, 32, &reg, &atlas);
        let min_y_sneak = ms.body_verts[..144]
            .iter()
            .map(|v| v.pos[1])
            .fold(f32::INFINITY, f32::min);
        let max_y_sneak = ms.body_verts[..144]
            .iter()
            .map(|v| v.pos[1])
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(min_y_sneak >= -0.02, "Sneak feet sank: {min_y_sneak}");
        assert!(max_y_sneak < 1.72, "Sneak should drop head height, got {max_y_sneak}");

        // Right-arm swing moves arm vertices differently than rest
        state.is_sneaking = false;
        let rest = build_player_model_vertices(&state, Vec3::ZERO, 64, 32, &reg, &atlas);
        state.is_swinging = true;
        state.swing_progress = 0.5;
        let swinging = build_player_model_vertices(&state, Vec3::ZERO, 64, 32, &reg, &atlas);
        assert_ne!(rest.body_verts[48..72][0].pos, swinging.body_verts[48..72][0].pos);
    }
}
