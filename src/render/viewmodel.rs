use glam::{EulerRot, Mat4, Quat, Vec2, Vec3};
use image::{Rgba, RgbaImage};

use crate::registry::{DrawType, EngineRegistries, NodeDef, CONTENT_UNKNOWN, RAM_MIC_ITEM_ID};
use crate::render::mesher::{emit_box, VoxelVertex};
use crate::textures::TextureAtlas;

#[derive(Debug, Clone)]
pub struct ViewmodelMesh {
    pub verts: Vec<VoxelVertex>,
    pub indices: Vec<u32>,
    /// `Some(img)` renders with a dedicated 1-layer crisp pixel texture (tools, items, player arm).
    /// `None` renders directly from the world `TextureAtlas` (3D held blocks/nodeboxes).
    pub texture: Option<RgbaImage>,
    pub is_node_block: bool,
    pub is_arm: bool,
    pub is_custom_prop: bool,
}

/// First-person viewmodel animation state machine (port of Luanti `camera.cpp` & `wieldmesh.cpp`).
#[derive(Debug, Clone)]
pub struct ViewmodelState {
    pub walk_phase: f32,
    pub bob_amount: f32,
    pub swing_progress: f32,
    pub is_swinging: bool,
    pub digging_held: bool,
    pub slot_switch_timer: f32,
    pub arm_inertia: Vec2,
    pub sway_enabled: bool,
    pub eat_timer: f32,
    pub last_item_name: String,
    pub last_slot: usize,
}

impl Default for ViewmodelState {
    fn default() -> Self {
        Self::new()
    }
}

impl ViewmodelState {
    pub fn new() -> Self {
        Self {
            walk_phase: 0.0,
            bob_amount: 0.0,
            swing_progress: 0.0,
            is_swinging: false,
            digging_held: false,
            slot_switch_timer: 0.125,
            arm_inertia: Vec2::ZERO,
            sway_enabled: true,
            eat_timer: 0.0,
            last_item_name: String::new(),
            last_slot: usize::MAX,
        }
    }

    pub fn update(
        &mut self,
        dt: f32,
        horizontal_speed: f32,
        on_ground: bool,
        is_sprinting: bool,
        is_sneaking: bool,
    ) {
        let dt = dt.clamp(0.0, 0.1);

        // Slot switch dip/raise timer (-0.125 -> 0.0 -> +0.125)
        if self.slot_switch_timer < 0.125 {
            self.slot_switch_timer = (self.slot_switch_timer + dt).min(0.125);
        }

        // Sinusoidal walk/sprint view-bobbing
        let moving = horizontal_speed > 0.25 && on_ground;
        let target_bob = if moving {
            let speed_norm = (horizontal_speed / 4.3).clamp(0.3, 1.8);
            let sneak_mul = if is_sneaking { 0.55 } else { 1.0 };
            let sprint_mul = if is_sprinting { 1.25 } else { 1.0 };
            (speed_norm * sneak_mul * sprint_mul).clamp(0.0, 1.8)
        } else {
            0.0
        };
        let blend = (dt * 12.0).clamp(0.0, 1.0);
        self.bob_amount += (target_bob - self.bob_amount) * blend;
        if moving {
            let freq = (horizontal_speed * 0.42).clamp(0.8, 3.6);
            self.walk_phase = (self.walk_phase + dt * freq) % 1.0;
        } else if self.bob_amount < 0.01 {
            self.walk_phase = 0.0;
            self.bob_amount = 0.0;
        }

        // Dig / attack swing arc
        if self.is_swinging || self.digging_held {
            self.is_swinging = true;
            self.swing_progress += dt * 3.6;
            if self.swing_progress >= 1.0 {
                if self.digging_held {
                    self.swing_progress %= 1.0;
                } else {
                    self.swing_progress = 0.0;
                    self.is_swinging = false;
                }
            }
        }

        // Decay camera inertia back to rest
        let decay = (-dt * 14.0).exp();
        self.arm_inertia *= decay;
        if self.eat_timer > 0.0 {
            self.eat_timer = (self.eat_timer - dt).max(0.0);
        }
    }

    pub fn trigger_eat(&mut self) {
        self.eat_timer = 0.45;
    }

    pub fn add_look_inertia(&mut self, dx: f32, dy: f32) {
        if !self.sway_enabled {
            self.arm_inertia = Vec2::ZERO;
            return;
        }
        self.arm_inertia.x = (self.arm_inertia.x - dx * 0.00055).clamp(-0.06, 0.06);
        self.arm_inertia.y = (self.arm_inertia.y + dy * 0.00055).clamp(-0.06, 0.06);
    }

    pub fn trigger_swing(&mut self) {
        if !self.is_swinging || self.swing_progress > 0.72 {
            self.is_swinging = true;
            self.swing_progress = 0.0;
        }
    }

    pub fn set_digging(&mut self, digging: bool) {
        self.digging_held = digging;
        if digging && !self.is_swinging {
            self.is_swinging = true;
            self.swing_progress = 0.0;
        }
    }

    pub fn on_slot_changed(&mut self) {
        self.slot_switch_timer = -0.125;
    }

    pub fn swing_progress(&self) -> f32 {
        self.swing_progress
    }

    pub fn is_swinging(&self) -> bool {
        self.is_swinging
    }

    pub fn slot_dip_offset(&self) -> f32 {
        let frac = (self.slot_switch_timer.abs() / 0.125).clamp(0.0, 1.0);
        -0.42 * (1.0 - frac)
    }

    pub fn bob_offset(&self) -> Vec3 {
        let ph = self.walk_phase * std::f32::consts::TAU;
        let bx = -(ph.sin()) * 0.032 * self.bob_amount;
        let by = (ph * 2.0).sin().abs() * 0.032 * self.bob_amount;
        Vec3::new(bx, by, 0.0)
    }

    /// Compute the view-space transform for rendering the wielded arm, block, or extruded item.
    pub fn model_matrix(&self, mesh: &ViewmodelMesh) -> Mat4 {
        let mut pos = if mesh.is_arm {
            Vec3::new(0.56, -0.52, -0.78)
        } else if mesh.is_custom_prop {
            Vec3::new(0.52, -0.34, -0.82)
        } else if mesh.is_node_block {
            Vec3::new(0.58, -0.48, -0.92)
        } else {
            Vec3::new(0.56, -0.44, -0.84)
        };

        // Slot switch dip/raise
        pos.y += self.slot_dip_offset();

        // Walk/sprint bobbing + subtle camera inertia (gated on `sway_enabled`)
        if self.sway_enabled {
            pos += self.bob_offset();
            pos.x += self.arm_inertia.x;
            pos.y += self.arm_inertia.y;
        }

        let base_rot = if mesh.is_arm {
            Quat::from_euler(EulerRot::YXZ, -0.35, -0.65, 0.30)
        } else if mesh.is_custom_prop {
            Quat::from_euler(EulerRot::YXZ, -0.42, -0.14, 0.08)
        } else if mesh.is_node_block {
            Quat::from_euler(EulerRot::YXZ, 0.785, -0.15, 0.0)
        } else {
            // Extruded tool/item oriented naturally like Minecraft right-hand hold
            Quat::from_euler(EulerRot::YXZ, -1.25, 0.25, -0.22)
        };

        let mut rot = base_rot;
        if self.is_swinging {
            let p = self.swing_progress.clamp(0.0, 1.0);
            let s0 = (p * std::f32::consts::PI).sin();
            let s1 = (p.powf(0.8) * std::f32::consts::PI).sin();
            let s2 = (p * std::f32::consts::PI * 1.6).sin();

            pos.x -= 0.36 * s1;
            pos.y += 0.14 * s2;
            pos.z -= 0.18 * s0;

            let chop_rot = Quat::from_euler(EulerRot::YXZ, 0.55, -1.35, -0.55);
            rot = base_rot.slerp(chop_rot, s0);
        }
        if self.eat_timer > 0.0 {
            let p = (1.0 - self.eat_timer / 0.45).clamp(0.0, 1.0);
            let chomp = (p * std::f32::consts::PI * 5.0).sin().abs();
            let env = (p * std::f32::consts::PI).sin();
            pos.x -= 0.34 * env;
            pos.y += 0.16 * env - 0.05 * chomp;
            pos.z -= 0.20 * env;
            let eat_rot = Quat::from_euler(EulerRot::YXZ, 0.35, -0.40 - 0.18 * chomp, 0.25);
            rot = rot.slerp(eat_rot, env);
        }

        let scale = if mesh.is_arm {
            Vec3::splat(1.0)
        } else if mesh.is_custom_prop {
            Vec3::splat(0.95)
        } else if mesh.is_node_block {
            Vec3::splat(0.38)
        } else {
            Vec3::splat(0.62)
        };

        Mat4::from_scale_rotation_translation(scale, rot, pos)
    }
}

/// Generate a crisp 3D extruded mesh with front face, back face, and 1-pixel perimeter side
/// thickness quads from `RgbaImage` pixels where `alpha >= 128` (zero blurry filtering).
pub fn create_extruded_mesh(img: &RgbaImage) -> ViewmodelMesh {
    let w = img.width().max(1);
    let h = img.height().max(1);
    let wf = w as f32;
    let hf = h as f32;
    let half_d = (1.0 / (w.max(h) as f32).max(16.0)) * 0.55;
    let z0 = -half_d;
    let z1 = half_d;

    let mut verts = Vec::with_capacity((w * h * 8) as usize);
    let mut indices = Vec::with_capacity((w * h * 12) as usize);

    let is_solid = |x: i32, y: i32| -> bool {
        if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
            return false;
        }
        img.get_pixel(x as u32, y as u32).0[3] >= 128
    };

    for py in 0..h {
        for px in 0..w {
            let rgba = img.get_pixel(px, py).0;
            if rgba[3] < 128 {
                continue;
            }
            let x0 = (px as f32) / wf - 0.5;
            let x1 = ((px + 1) as f32) / wf - 0.5;
            let y1 = 0.5 - (py as f32) / hf;
            let y0 = 0.5 - ((py + 1) as f32) / hf;

            let u0 = (px as f32) / wf;
            let u1 = ((px + 1) as f32) / wf;
            let v0 = (py as f32) / hf;
            let v1 = ((py + 1) as f32) / hf;
            let uc = (px as f32 + 0.5) / wf;
            let vc = (py as f32 + 0.5) / hf;

            let tint = [255, 255, 255, 255];

            let mut push_quad = |pts: [[f32; 3]; 4], uvs: [[f32; 2]; 4], face_dir: u8| {
                let packed = VoxelVertex::pack(0, face_dir, 15, 0, 3, false, true);
                let base = verts.len() as u32;
                for (pos, uv) in pts.into_iter().zip(uvs) {
                    verts.push(VoxelVertex {
                        pos,
                        uv,
                        packed,
                        tint,
                    });
                }
                indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            };

            // Front face (+Z)
            push_quad(
                [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
                [[u0, v1], [u1, v1], [u1, v0], [u0, v0]],
                4,
            );
            // Back face (-Z)
            push_quad(
                [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]],
                [[u1, v1], [u0, v1], [u0, v0], [u1, v0]],
                5,
            );

            // Exposed 1-pixel perimeter side thickness quads
            let ix = px as i32;
            let iy = py as i32;
            if !is_solid(ix + 1, iy) {
                push_quad(
                    [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]],
                    [[uc, vc]; 4],
                    0,
                );
            }
            if !is_solid(ix - 1, iy) {
                push_quad(
                    [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]],
                    [[uc, vc]; 4],
                    1,
                );
            }
            if !is_solid(ix, iy - 1) {
                push_quad(
                    [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]],
                    [[uc, vc]; 4],
                    2,
                );
            }
            if !is_solid(ix, iy + 1) {
                push_quad(
                    [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],
                    [[uc, vc]; 4],
                    3,
                );
            }
        }
    }

    ViewmodelMesh {
        verts,
        indices,
        texture: Some(img.clone()),
        is_node_block: false,
        is_arm: false,
        is_custom_prop: false,
    }
}

/// Lightweight 2-sided flat quad item mesh used when `extruded_held_items` is disabled.
pub fn create_flat_item_mesh(img: &RgbaImage) -> ViewmodelMesh {
    let tint = [255, 255, 255, 255];
    let mut verts = Vec::with_capacity(8);
    let mut indices = Vec::with_capacity(12);
    let front_pts = [
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.5, 0.5, 0.0],
        [-0.5, 0.5, 0.0],
    ];
    let front_uvs = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
    let pf = VoxelVertex::pack(0, 4, 15, 0, 3, false, true);
    for (pos, uv) in front_pts.into_iter().zip(front_uvs) {
        verts.push(VoxelVertex { pos, uv, packed: pf, tint });
    }
    indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);

    let back_pts = [
        [0.5, -0.5, 0.0],
        [-0.5, -0.5, 0.0],
        [-0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
    ];
    let back_uvs = [[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]];
    let pb = VoxelVertex::pack(0, 5, 15, 0, 3, false, true);
    for (pos, uv) in back_pts.into_iter().zip(back_uvs) {
        verts.push(VoxelVertex { pos, uv, packed: pb, tint });
    }
    indices.extend_from_slice(&[4, 5, 6, 4, 6, 7]);

    ViewmodelMesh {
        verts,
        indices,
        texture: Some(img.clone()),
        is_node_block: false,
        is_arm: false,
        is_custom_prop: false,
    }
}

/// Build a 3D first-person empty right arm textured from the player's skin (supports 64x32, 128x64, and 64x64).
pub fn create_arm_mesh(skin: &RgbaImage) -> ViewmodelMesh {
    let w = 64.0f32;
    let h = if skin.width() > 0 && skin.height() >= skin.width() {
        64.0f32
    } else {
        32.0f32
    };
    let hw = 0.14f32;
    let hh = 0.44f32;
    let hd = 0.14f32;

    let x0 = -hw;
    let x1 = hw;
    let y0 = -hh;
    let y1 = hh;
    let z0 = -hd;
    let z1 = hd;

    let uv_rect = |px0: f32, py0: f32, px1: f32, py1: f32| -> [[f32; 2]; 4] {
        let u0 = px0 / w;
        let u1 = px1 / w;
        let v0 = py0 / h;
        let v1 = py1 / h;
        [[u0, v1], [u1, v1], [u1, v0], [u0, v0]]
    };

    // Classic Minecraft right-arm UV coordinates (x: 40..56, y: 16..32)
    let faces: [(u8, [[f32; 3]; 4], [[f32; 2]; 4]); 6] = [
        // +X outer
        (0, [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]], uv_rect(40.0, 20.0, 44.0, 32.0)),
        // -X inner
        (1, [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]], uv_rect(48.0, 20.0, 52.0, 32.0)),
        // +Y shoulder cap
        (2, [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]], uv_rect(44.0, 16.0, 48.0, 20.0)),
        // -Y hand tip cap
        (3, [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]], uv_rect(48.0, 16.0, 52.0, 20.0)),
        // +Z front (facing toward camera)
        (4, [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], uv_rect(44.0, 20.0, 48.0, 32.0)),
        // -Z back
        (5, [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]], uv_rect(52.0, 20.0, 56.0, 32.0)),
    ];

    let mut verts = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (fdir, pts, uvs) in faces {
        let packed = VoxelVertex::pack(0, fdir, 15, 0, 3, false, false);
        let base = verts.len() as u32;
        for (pos, uv) in pts.into_iter().zip(uvs) {
            verts.push(VoxelVertex {
                pos,
                uv,
                packed,
                tint: [255, 255, 255, 255],
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    ViewmodelMesh {
        verts,
        indices,
        texture: Some(skin.clone()),
        is_node_block: false,
        is_arm: true,
        is_custom_prop: false,
    }
}

/// sRGB palette colors for `cubex:ram_mic` (`#1C9846` PCB, `#16181C` DRAM, `#F2CA36` gold, `#20232A` mic).
pub const RAM_MIC_PALETTE: [[u8; 4]; 8] = [
    [0x1C, 0x98, 0x46, 255], // 0: Green DDR4 PCB body
    [0x16, 0x18, 0x1C, 255], // 1: Dark DRAM memory chips
    [0xF2, 0xCA, 0x36, 255], // 2: Bright gold edge-connector pins
    [0x20, 0x23, 0x2A, 255], // 3: Matte-black wireless mic body
    [0x2E, 0x33, 0x3D, 255], // 4: Rear clip & top mic grille
    [0xF5, 0xF7, 0xFA, 255], // 5: Crisp white square logo
    [0x30, 0xF8, 0x60, 255], // 6: Glowing green LED status pixel
    [0x13, 0x6E, 0x32, 255], // 7: PCB dark green edge trace
];

/// Build an 8x1 sRGB palette texture so hardware texture sampling preserves rich colors without gamma washout.
pub fn ram_mic_palette_image() -> RgbaImage {
    let mut img = RgbaImage::new(RAM_MIC_PALETTE.len() as u32, 1);
    for (i, col) in RAM_MIC_PALETTE.iter().enumerate() {
        img.put_pixel(i as u32, 0, Rgba(*col));
    }
    img
}

fn push_palette_box(
    verts: &mut Vec<VoxelVertex>,
    indices: &mut Vec<u32>,
    min: [f32; 3],
    max: [f32; 3],
    palette_idx: u8,
    emissive: bool,
) {
    let [x0, y0, z0] = min;
    let [x1, y1, z1] = max;
    let faces: [(u8, [[f32; 3]; 4]); 6] = [
        (0, [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]]),
        (1, [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]]),
        (2, [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]]),
        (3, [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]),
        (4, [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]),
        (5, [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]]),
    ];
    let u = (palette_idx as f32 + 0.5) / RAM_MIC_PALETTE.len() as f32;
    let uv = [u, 0.5f32];
    let tint = [255u8, 255, 255, 255];
    for (fdir, pts) in faces {
        let mut packed = VoxelVertex::pack(0, fdir, 15, if emissive { 15 } else { 0 }, 3, false, false);
        if emissive {
            packed |= VoxelVertex::FLAG_EMISSIVE;
        }
        let base = verts.len() as u32;
        for pos in pts {
            verts.push(VoxelVertex { pos, uv, packed, tint });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

/// Build the 3D compound mesh for `cubex:ram_mic`: a green DDR4 PCB stick (`0.15 x 0.52 x 0.035`)
/// with dark DRAM chips on the lower half, bright gold edge-connector pins along one edge, and a
/// matte-black square wireless clip-on mic (`0.19 x 0.19 x 0.085`) with white logo and green LED.
pub fn create_ram_mic_mesh() -> ViewmodelMesh {
    let mut verts = Vec::with_capacity(24 * 16);
    let mut indices = Vec::with_capacity(36 * 16);

    // 1. Green DDR4 PCB handle (0.15 wide x 0.52 tall x 0.035 thick)
    push_palette_box(
        &mut verts,
        &mut indices,
        [-0.075, -0.26, -0.0175],
        [0.075, 0.26, 0.0175],
        0,
        false,
    );
    push_palette_box(
        &mut verts,
        &mut indices,
        [-0.076, -0.24, -0.010],
        [-0.064, 0.24, 0.010],
        7,
        false,
    );

    // 2. Gold edge-connector pin banks along +X edge split around the DDR4 key notch
    push_palette_box(
        &mut verts,
        &mut indices,
        [0.068, -0.25, -0.019],
        [0.086, -0.03, 0.019],
        2,
        false,
    );
    push_palette_box(
        &mut verts,
        &mut indices,
        [0.068, 0.01, -0.019],
        [0.086, 0.25, 0.019],
        2,
        false,
    );

    // 3. Dark DRAM memory ICs on lower half (3 front +Z, 3 back -Z)
    for cy in [-0.205f32, -0.125, -0.045] {
        push_palette_box(
            &mut verts,
            &mut indices,
            [-0.048, cy - 0.030, 0.016],
            [0.032, cy + 0.030, 0.027],
            1,
            false,
        );
        push_palette_box(
            &mut verts,
            &mut indices,
            [-0.048, cy - 0.030, -0.027],
            [0.032, cy + 0.030, -0.016],
            1,
            false,
        );
    }

    // 4. Matte-black square wireless clip-on mic (0.19 x 0.19 x 0.085) clamped onto upper half
    push_palette_box(
        &mut verts,
        &mut indices,
        [-0.095, 0.03, -0.022],
        [0.095, 0.22, 0.063],
        3,
        false,
    );
    push_palette_box(
        &mut verts,
        &mut indices,
        [-0.052, 0.05, -0.042],
        [0.052, 0.19, -0.018],
        4,
        false,
    );
    push_palette_box(
        &mut verts,
        &mut indices,
        [-0.042, 0.22, -0.008],
        [0.042, 0.240, 0.048],
        4,
        false,
    );

    // 5. White logo square & glowing green status LED pixel on front face
    push_palette_box(
        &mut verts,
        &mut indices,
        [-0.042, 0.082, 0.062],
        [0.042, 0.166, 0.069],
        5,
        false,
    );
    push_palette_box(
        &mut verts,
        &mut indices,
        [0.052, 0.174, 0.062],
        [0.076, 0.198, 0.070],
        6,
        true,
    );

    ViewmodelMesh {
        verts,
        indices,
        texture: Some(ram_mic_palette_image()),
        is_node_block: false,
        is_arm: false,
        is_custom_prop: true,
    }
}

/// Build a 3D shaded mini-block mesh for a held node (supporting `Normal`, `NodeBox`, `Allfaces`, `Glasslike`, etc.).
pub fn create_held_node_mesh(def: &NodeDef, registries: &EngineRegistries, atlas: &TextureAtlas) -> ViewmodelMesh {
    // Flat/sprite-like nodes use extruded 3D item geometry if wield/inventory/tile image exists
    if matches!(
        def.drawtype,
        DrawType::Plantlike | DrawType::Torchlike | DrawType::Signlike | DrawType::Raillike | DrawType::Firelike
    ) || !def.wield_image.trim().is_empty()
        || !def.inventory_image.trim().is_empty()
    {
        if let Some(img) = atlas.get_item_rgba(&def.name, registries) {
            return create_extruded_mesh(&img);
        }
    }

    let layers = atlas
        .node_face_layers
        .get(&def.id)
        .copied()
        .unwrap_or([atlas.fallback_layer; 6]);
    let tint = def
        .palette
        .as_deref()
        .and_then(|p| atlas.sample_palette_color(p, 0))
        .or(def.color)
        .unwrap_or_else(|| {
            if matches!(def.drawtype, DrawType::Allfaces | DrawType::AllfacesOptional)
                && !def.name.contains("cherry")
            {
                [119, 171, 47, 255]
            } else {
                [255, 255, 255, 255]
            }
        });

    let mut verts = Vec::new();
    let mut indices = Vec::new();

    if def.drawtype == DrawType::NodeBox && !def.node_boxes.is_empty() {
        for b in &def.node_boxes {
            emit_box(
                &mut verts,
                &mut indices,
                [b[0], b[1], b[2]],
                [b[3], b[4], b[5]],
                layers,
                15,
                0,
                false,
                true,
                tint,
                None::<fn(usize) -> bool>,
            );
        }
    } else {
        emit_box(
            &mut verts,
            &mut indices,
            [-0.5, -0.5, -0.5],
            [0.5, 0.5, 0.5],
            layers,
            15,
            0,
            false,
            true,
            tint,
            None::<fn(usize) -> bool>,
        );
    }

    ViewmodelMesh {
        verts,
        indices,
        texture: None,
        is_node_block: true,
        is_arm: false,
        is_custom_prop: false,
    }
}

/// Build the appropriate first-person viewmodel mesh for whatever item the player currently holds.
pub fn build_viewmodel_mesh(
    held_item_name: Option<&str>,
    registries: &EngineRegistries,
    atlas: &TextureAtlas,
) -> ViewmodelMesh {
    build_viewmodel_mesh_ex(held_item_name, true, registries, atlas)
}

pub fn build_viewmodel_mesh_ex(
    held_item_name: Option<&str>,
    extruded: bool,
    registries: &EngineRegistries,
    atlas: &TextureAtlas,
) -> ViewmodelMesh {
    build_viewmodel_mesh_with_skin(held_item_name, extruded, None, registries, atlas)
}

pub fn build_viewmodel_mesh_with_skin(
    held_item_name: Option<&str>,
    extruded: bool,
    custom_skin: Option<&RgbaImage>,
    registries: &EngineRegistries,
    atlas: &TextureAtlas,
) -> ViewmodelMesh {
    let fallback_arm = || match custom_skin {
        Some(skin) => create_arm_mesh(skin),
        None => create_arm_mesh(&atlas.get_player_skin_rgba()),
    };
    let raw = held_item_name.unwrap_or("").trim();
    let base = raw.split_whitespace().next().unwrap_or("");
    if base.is_empty() {
        return fallback_arm();
    }

    let resolved = registries.resolve_alias(base);
    if resolved == RAM_MIC_ITEM_ID {
        return create_ram_mic_mesh();
    }
    let cid = registries.get_content_id(resolved);
    let is_node = registries
        .get_item_def(resolved)
        .map(|i| i.is_node)
        .unwrap_or(cid != CONTENT_UNKNOWN);

    if is_node && cid != CONTENT_UNKNOWN {
        if let Some(ndef) = registries.get_node_def(cid) {
            if !extruded
                && (matches!(
                    ndef.drawtype,
                    DrawType::Plantlike
                        | DrawType::Torchlike
                        | DrawType::Signlike
                        | DrawType::Raillike
                        | DrawType::Firelike
                ) || !ndef.wield_image.trim().is_empty()
                    || !ndef.inventory_image.trim().is_empty())
            {
                if let Some(img) = atlas.get_item_rgba(&ndef.name, registries) {
                    return create_flat_item_mesh(&img);
                }
            }
            return create_held_node_mesh(ndef, registries, atlas);
        }
    }

    if let Some(img) = atlas.get_item_rgba(resolved, registries) {
        return if extruded {
            create_extruded_mesh(&img)
        } else {
            create_flat_item_mesh(&img)
        };
    }

    fallback_arm()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    #[test]
    fn viewmodel_state_and_extruded_mesh_work() {
        let mut vm = ViewmodelState::new();
        assert!(!vm.is_swinging());
        vm.trigger_swing();
        assert!(vm.is_swinging());
        vm.update(0.05, 4.3, true, true, false);
        assert!(vm.swing_progress() > 0.0);
        assert!(vm.bob_offset().length() > 0.0);

        vm.on_slot_changed();
        vm.update(0.10, 0.0, true, false, false);
        assert!(vm.slot_dip_offset() < -0.25, "Expected slot dip, got {}", vm.slot_dip_offset());
        for _ in 0..3 {
            vm.update(0.08, 0.0, true, false, false);
        }
        assert!(vm.slot_dip_offset().abs() < 1e-5, "Expected settled slot dip 0, got {}", vm.slot_dip_offset());

        // 4x4 cross shape with alpha >= 128 in center 2x2
        let mut img = RgbaImage::from_pixel(4, 4, Rgba([0, 0, 0, 0]));
        for y in 1..3 {
            for x in 1..3 {
                img.put_pixel(x, y, Rgba([80, 220, 240, 255]));
            }
        }
        let mesh = create_extruded_mesh(&img);
        // 4 solid pixels: each has front + back + 2 exposed outer perimeter edges = 4 * 4 = 16 quads = 64 verts
        assert_eq!(mesh.verts.len(), 64);
        assert_eq!(mesh.indices.len(), 96);

        let skin = RgbaImage::from_pixel(64, 32, Rgba([200, 150, 120, 255]));
        let arm = create_arm_mesh(&skin);
        assert_eq!(arm.verts.len(), 24);
        assert_eq!(arm.indices.len(), 36);
    }
}
