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
    pub skin_id: String,
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
            skin_id: "default".into(),
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
    pub hair_crown_verts: usize,
}

#[derive(Debug, Clone)]
pub struct PlayerModelMesh {
    pub body_verts: Vec<VoxelVertex>,
    pub body_indices: Vec<u32>,
    pub overlay_verts: Vec<VoxelVertex>,
    pub overlay_indices: Vec<u32>,
    pub crown_verts: Vec<VoxelVertex>,
    pub crown_indices: Vec<u32>,
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

/// Rich espresso-brown sRGB palette (`#1E1511`, `#2C1F19`, `#3E2C23`) for CarterPCs 3D curly crown.
pub const CARTER_HAIR_PALETTE: [[u8; 4]; 3] = [
    [0x1E, 0x15, 0x11, 255],
    [0x2C, 0x1F, 0x19, 255],
    [0x3E, 0x2C, 0x23, 255],
];

/// Build a 3x1 sRGB palette texture so GPU sampling preserves deep espresso tones on sRGB swapchains.
pub fn carter_hair_palette_image() -> RgbaImage {
    let mut img = RgbaImage::new(CARTER_HAIR_PALETTE.len() as u32, 1);
    for (i, col) in CARTER_HAIR_PALETTE.iter().enumerate() {
        img.put_pixel(i as u32, 0, image::Rgba(*col));
    }
    img
}

#[allow(clippy::too_many_arguments)]
fn emit_palette_part_box(
    verts: &mut Vec<VoxelVertex>,
    indices: &mut Vec<u32>,
    transform: Mat4,
    min: Vec3,
    max: Vec3,
    shade_idx: usize,
    sky: u8,
    block: u8,
) -> usize {
    let start_verts = verts.len();
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
    let u = (shade_idx.min(CARTER_HAIR_PALETTE.len() - 1) as f32 + 0.5)
        / CARTER_HAIR_PALETTE.len() as f32;
    let uv = [u, 0.5f32];
    for (local_n, pts) in faces {
        let world_n = transform.transform_vector3(local_n).normalize_or_zero();
        let fdir = dominant_face_dir(world_n);
        let packed = VoxelVertex::pack(0, fdir, sky, block, 3, false, false);
        let base = verts.len() as u32;
        for p in pts {
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

/// Build an articulated 6-part UV-mapped 3D character mesh (plus optional overlays, crown & held item)
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

    let mut body_verts = Vec::with_capacity(160);
    let mut body_indices = Vec::with_capacity(240);
    let mut overlay_verts = Vec::with_capacity(144);
    let mut overlay_indices = Vec::with_capacity(216);
    let mut crown_verts = Vec::new();
    let mut crown_indices = Vec::new();
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

    // Hat / Hair outer overlay (1.125x expanded shell around 8x8x8 head, UV (32, 0) = u 0.5..1.0, v 0.0..0.5)
    counts.overlay_verts += emit_part_box(
        &mut overlay_verts,
        &mut overlay_indices,
        head_mat,
        Vec3::new(-4.5 * p, -0.5 * p, -4.5 * p),
        Vec3::new(4.5 * p, 8.5 * p, 4.5 * p),
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
        false,
    );

    // Sculpted 3D curly hair crown: 20 dome curl cubes (0.11..0.16 blocks, y=+0.46..+0.62)
    // plus 5 subtle top-edge forehead curl tips (y=+0.34..+0.42) leaving eyebrows & eyes uncovered.
    if crate::textures::is_carterpcs_skin(&state.skin_id) {
        // (center_xyz, size_xyz, shade_idx) in head_mat local block coordinates
        const CURLS: [([f32; 3], [f32; 3], usize); 25] = [
            // Outer perimeter dome ring (10 staggered curls, y bottom ~0.43..0.46 into top ~0.55..0.58)
            ([-0.165, 0.500, -0.165], [0.145, 0.130, 0.140], 0),
            ([0.000, 0.510, -0.175], [0.155, 0.140, 0.140], 1),
            ([0.165, 0.500, -0.165], [0.145, 0.130, 0.140], 2),
            ([-0.185, 0.498, -0.045], [0.135, 0.125, 0.130], 1),
            ([-0.180, 0.492, 0.080], [0.125, 0.120, 0.130], 2),
            ([0.185, 0.498, -0.045], [0.135, 0.125, 0.130], 0),
            ([0.180, 0.492, 0.080], [0.125, 0.120, 0.130], 1),
            ([-0.130, 0.498, 0.165], [0.125, 0.118, 0.118], 0),
            ([0.000, 0.505, 0.175], [0.145, 0.128, 0.116], 2),
            ([0.130, 0.498, 0.165], [0.125, 0.118, 0.118], 1),
            // Mid-crown staggered dome ring (6 curls, y top ~0.58..0.605)
            ([-0.105, 0.532, -0.105], [0.150, 0.138, 0.150], 2),
            ([0.020, 0.542, -0.115], [0.160, 0.140, 0.140], 0),
            ([0.115, 0.532, -0.090], [0.140, 0.130, 0.140], 1),
            ([-0.115, 0.528, 0.040], [0.140, 0.130, 0.145], 0),
            ([0.110, 0.528, 0.048], [0.148, 0.130, 0.140], 2),
            ([-0.010, 0.534, 0.108], [0.150, 0.130, 0.130], 1),
            // Top dome apex cluster (4 curls, rising to y=0.62)
            ([-0.058, 0.548, -0.038], [0.130, 0.138, 0.130], 1),
            ([0.058, 0.550, -0.028], [0.140, 0.132, 0.138], 2),
            ([-0.040, 0.548, 0.058], [0.120, 0.132, 0.120], 0),
            ([0.040, 0.552, 0.038], [0.130, 0.136, 0.128], 1),
            // 5 tiny subtle curl tips along upper forehead hairline (y strictly 0.345..0.417)
            ([-0.175, 0.376, 0.228], [0.070, 0.058, 0.056], 0),
            ([-0.090, 0.382, 0.234], [0.068, 0.056, 0.056], 2),
            ([0.005, 0.386, 0.236], [0.072, 0.060, 0.056], 1),
            ([0.095, 0.378, 0.232], [0.066, 0.054, 0.054], 0),
            ([0.176, 0.382, 0.226], [0.070, 0.056, 0.056], 2),
        ];
        for (center, size, shade) in CURLS {
            let c = Vec3::from_array(center);
            let half = Vec3::from_array(size) * 0.5;
            counts.hair_crown_verts += emit_palette_part_box(
                &mut crown_verts,
                &mut crown_indices,
                head_mat,
                c - half,
                c + half,
                shade,
                sky,
                block,
            );
        }
    }

    // 64x64 second-layer overlays (Jacket, Left/Right Sleeves, Left/Right Pants)
    if is_64 {
        let ov = 0.25 * p;
        counts.overlay_verts += emit_part_box(
            &mut overlay_verts,
            &mut overlay_indices,
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
            false,
        );
        counts.overlay_verts += emit_part_box(
            &mut overlay_verts,
            &mut overlay_indices,
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
            false,
        );
        counts.overlay_verts += emit_part_box(
            &mut overlay_verts,
            &mut overlay_indices,
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
            false,
        );
        counts.overlay_verts += emit_part_box(
            &mut overlay_verts,
            &mut overlay_indices,
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
            false,
        );
        counts.overlay_verts += emit_part_box(
            &mut overlay_verts,
            &mut overlay_indices,
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
            false,
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
                let item_local = if vm.is_custom_prop {
                    Mat4::from_scale_rotation_translation(
                        Vec3::splat(0.92),
                        Quat::from_euler(EulerRot::YXZ, 0.25, -0.35, 0.0),
                        Vec3::new(0.0, -9.0 * p, 3.0 * p),
                    )
                } else if vm.is_node_block {
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
                    let emissive = (v.packed & VoxelVertex::FLAG_EMISSIVE) != 0;
                    let mut packed = VoxelVertex::pack(
                        layer,
                        new_fdir,
                        sky,
                        if emissive { 15 } else { block },
                        3,
                        false,
                        cutout,
                    );
                    if emissive {
                        packed |= VoxelVertex::FLAG_EMISSIVE;
                    }
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
        overlay_verts,
        overlay_indices,
        crown_verts,
        crown_indices,
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

    #[test]
    fn carterpcs_head_overlay_1_125x_curly_hair_crown_and_ram_mic() {
        let reg = EngineRegistries::new();
        let atlas = TextureAtlas::placeholder();

        let mut state = PlayerModelState::default();
        state.skin_id = "carterpcs.png".into();
        state.held_item = Some(crate::registry::RAM_MIC_ITEM_ID.into());

        let mesh = build_player_model_vertices(&state, Vec3::ZERO, 128, 64, &reg, &atlas);
        assert_eq!(mesh.part_counts.overlay_verts, 24);
        assert_eq!(mesh.overlay_verts.len(), 24);
        assert_eq!(mesh.part_counts.hair_crown_verts, 25 * 24);
        assert_eq!(mesh.crown_verts.len(), 25 * 24);
        assert_eq!(mesh.crown_indices.len(), 25 * 36);
        assert!(!mesh.held_verts.is_empty());
        assert!(!mesh.held_indices.is_empty());

        // Check espresso sRGB palette & head-clearance vertical bounds (eyebrows/eyes stay uncovered)
        assert_eq!(
            CARTER_HAIR_PALETTE,
            [[0x1E, 0x15, 0x11, 255], [0x2C, 0x1F, 0x19, 255], [0x3E, 0x2C, 0x23, 255]]
        );
        let hair_tex = carter_hair_palette_image();
        assert_eq!(hair_tex.dimensions(), (3, 1));
        let neck_y = 24.0 * PIXEL_SCALE;
        let crown_min_y = mesh
            .crown_verts
            .iter()
            .map(|v| v.pos[1] - neck_y)
            .fold(f32::INFINITY, f32::min);
        let crown_max_y = mesh
            .crown_verts
            .iter()
            .map(|v| v.pos[1] - neck_y)
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            crown_min_y >= 0.34 - 1e-4,
            "Hairline curls must stay above eyebrows (y >= 0.34), got min_y={crown_min_y}"
        );
        assert!(
            (0.60..=0.63).contains(&crown_max_y),
            "Dome top must reach ~0.62 above head base, got max_y={crown_max_y}"
        );

        let (mut min_v, mut max_v) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for v in &mesh.overlay_verts {
            let p = Vec3::from_array(v.pos);
            min_v = min_v.min(p);
            max_v = max_v.max(p);
            assert!(
                v.uv[0] >= 0.5 - 1e-4 && v.uv[0] <= 1.0 + 1e-4,
                "Head overlay U must lie in [0.5, 1.0], got {}",
                v.uv[0]
            );
            assert!(
                v.uv[1] >= -1e-4 && v.uv[1] <= 0.5 + 1e-4,
                "Head overlay V must lie in [0.0, 0.5], got {}",
                v.uv[1]
            );
        }
        let span = max_v - min_v;
        let expected_span = 9.0 * PIXEL_SCALE;
        assert!((span.x - expected_span).abs() < 1e-4, "span.x={}", span.x);
        assert!((span.y - expected_span).abs() < 1e-4, "span.y={}", span.y);
        assert!((span.z - expected_span).abs() < 1e-4, "span.z={}", span.z);

        let mut default_state = PlayerModelState::default();
        default_state.skin_id = "default".into();
        let default_mesh =
            build_player_model_vertices(&default_state, Vec3::ZERO, 64, 32, &reg, &atlas);
        assert!(default_mesh.crown_verts.is_empty());

        let vm = crate::render::viewmodel::build_viewmodel_mesh(
            Some(crate::registry::RAM_MIC_ITEM_ID),
            &reg,
            &atlas,
        );
        assert!(vm.is_custom_prop);
        assert!(!vm.verts.is_empty());
        assert_eq!(vm.texture.as_ref().map(|t| t.dimensions()), Some((8, 1)));
        assert!(
            vm.verts
                .iter()
                .any(|v| (v.packed & crate::render::mesher::VoxelVertex::FLAG_EMISSIVE) != 0),
            "RAM mic prop must include emissive status LED vertices"
        );
        let pcb_verts = &vm.verts[0..24];
        let pmin = pcb_verts
            .iter()
            .fold(Vec3::splat(f32::INFINITY), |a, v| a.min(Vec3::from_array(v.pos)));
        let pmax = pcb_verts
            .iter()
            .fold(Vec3::splat(f32::NEG_INFINITY), |a, v| a.max(Vec3::from_array(v.pos)));
        let pspan = pmax - pmin;
        assert!((pspan.x - 0.15).abs() < 1e-4 && (pspan.y - 0.52).abs() < 1e-4 && (pspan.z - 0.035).abs() < 1e-4);
    }
}
