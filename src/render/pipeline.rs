use bytemuck::{Pod, Zeroable};
use glam::{IVec3, Vec3};
use rand::Rng;
use wgpu::util::DeviceExt;

use crate::registry::{DrawType, EngineRegistries, CONTENT_AIR, CONTENT_IGNORE};
use crate::render::mesher::{emit_box, VoxelVertex};
use crate::textures::TextureAtlas;
use crate::world::WorldStore;

/// Reusable GPU vertex/index buffer pair for dynamic meshes (viewmodel, 3P character, held items).
pub struct IndexedMeshBuffers {
    pub vb: Option<wgpu::Buffer>,
    pub ib: Option<wgpu::Buffer>,
    pub index_count: u32,
}

impl IndexedMeshBuffers {
    pub const fn empty() -> Self {
        Self {
            vb: None,
            ib: None,
            index_count: 0,
        }
    }

    pub fn from_mesh(
        device: &wgpu::Device,
        label: &str,
        verts: &[VoxelVertex],
        indices: &[u32],
    ) -> Self {
        if verts.is_empty() || indices.is_empty() {
            return Self::empty();
        }
        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self {
            vb: Some(vb),
            ib: Some(ib),
            index_count: indices.len() as u32,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct LineVertex {
    pub pos: [f32; 3],
    pub color: [f32; 4],
}

impl LineVertex {
    pub const ATTRS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x4,
    ];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LineVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }
}

/// Build 12 wireframe edge lines around each targeted selection AABB (slightly expanded to avoid z-fighting).
pub fn build_selection_wireframe(boxes: &[([f32; 3], [f32; 3])]) -> Vec<LineVertex> {
    let mut out = Vec::with_capacity(boxes.len() * 24);
    let col = [0.02, 0.02, 0.02, 0.88];
    let pad = 0.0025;
    for &(mn, mx) in boxes {
        let x0 = mn[0] - pad;
        let y0 = mn[1] - pad;
        let z0 = mn[2] - pad;
        let x1 = mx[0] + pad;
        let y1 = mx[1] + pad;
        let z1 = mx[2] + pad;

        let edges = [
            ([x0, y0, z0], [x1, y0, z0]),
            ([x1, y0, z0], [x1, y0, z1]),
            ([x1, y0, z1], [x0, y0, z1]),
            ([x0, y0, z1], [x0, y0, z0]),
            ([x0, y1, z0], [x1, y1, z0]),
            ([x1, y1, z0], [x1, y1, z1]),
            ([x1, y1, z1], [x0, y1, z1]),
            ([x0, y1, z1], [x0, y1, z0]),
            ([x0, y0, z0], [x0, y1, z0]),
            ([x1, y0, z0], [x1, y1, z0]),
            ([x1, y0, z1], [x1, y1, z1]),
            ([x0, y0, z1], [x0, y1, z1]),
        ];
        for (a, b) in edges {
            out.push(LineVertex { pos: a, color: col });
            out.push(LineVertex { pos: b, color: col });
        }
    }
    out
}

/// Build progressive block-break crack overlay quads (`stage` in `0..=9`) drawn slightly proud of target AABBs.
pub fn build_crack_overlay(
    boxes: &[([f32; 3], [f32; 3])],
    stage: u8,
    crack_layers: [u16; 10],
) -> (Vec<VoxelVertex>, Vec<u32>) {
    let mut verts = Vec::with_capacity(boxes.len() * 24);
    let mut indices = Vec::with_capacity(boxes.len() * 36);
    let layer = crack_layers[(stage as usize).min(9)];
    let pad = 0.003;
    for &(mn, mx) in boxes {
        emit_box(
            &mut verts,
            &mut indices,
            [mn[0] - pad, mn[1] - pad, mn[2] - pad],
            [mx[0] + pad, mx[1] + pad, mx[2] + pad],
            [layer; 6],
            15,
            0,
            false,
            false,
            [255, 255, 255, 220],
            None::<fn(usize) -> bool>,
        );
    }
    (verts, indices)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParticleKind {
    #[default]
    Chip,
    ShatterCube,
    TorchEmber,
}

#[derive(Debug, Clone)]
pub struct Particle {
    pub kind: ParticleKind,
    pub pos: Vec3,
    pub vel: Vec3,
    pub rot: Vec3,
    pub ang_vel: Vec3,
    pub size: f32,
    pub age: f32,
    pub max_age: f32,
    pub layer: u16,
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    pub tint: [u8; 4],
    pub sky: u8,
    pub block: u8,
    pub bounces: u8,
}

/// 3D block-break particle burst system spawning and simulating falling textured block chips.
#[derive(Debug, Default, Clone)]
pub struct ParticleSystem {
    pub particles: Vec<Particle>,
}

impl ParticleSystem {
    pub fn new() -> Self {
        Self {
            particles: Vec::with_capacity(256),
        }
    }

    pub fn clear(&mut self) {
        self.particles.clear();
    }

    pub fn spawn_dig_chips(
        &mut self,
        block_pos: IVec3,
        node_id: u16,
        count: usize,
        atlas: &TextureAtlas,
        registries: &EngineRegistries,
        sky: u8,
        block: u8,
    ) {
        if node_id == CONTENT_AIR || node_id == CONTENT_IGNORE {
            return;
        }
        let layers = atlas
            .node_face_layers
            .get(&node_id)
            .copied()
            .unwrap_or([atlas.fallback_layer; 6]);
        let def = registries.get_node_def(node_id);
        let tint = def
            .and_then(|d| {
                d.palette
                    .as_deref()
                    .and_then(|p| atlas.sample_palette_color(p, 0))
                    .or(d.color)
            })
            .unwrap_or([255, 255, 255, 255]);

        let mut rng = rand::thread_rng();
        let center = block_pos.as_vec3() + Vec3::splat(0.5);

        for _ in 0..count {
            if self.particles.len() >= 512 {
                self.particles.remove(0);
            }
            let off = Vec3::new(
                rng.gen_range(-0.38..0.38),
                rng.gen_range(-0.38..0.38),
                rng.gen_range(-0.38..0.38),
            );
            let vel = Vec3::new(
                rng.gen_range(-2.2..2.2),
                rng.gen_range(1.4..4.0),
                rng.gen_range(-2.2..2.2),
            );
            let u0 = rng.gen_range(0.0..0.75);
            let v0 = rng.gen_range(0.0..0.75);
            let face_idx = rng.gen_range(0..6);
            self.particles.push(Particle {
                kind: ParticleKind::Chip,
                pos: center + off,
                vel,
                rot: Vec3::ZERO,
                ang_vel: Vec3::ZERO,
                size: rng.gen_range(0.05..0.11),
                age: 0.0,
                max_age: rng.gen_range(0.35..0.75),
                layer: layers[face_idx],
                uv_min: [u0, v0],
                uv_max: [u0 + 0.25, v0 + 0.25],
                tint,
                sky,
                block,
                bounces: 0,
            });
        }
    }

    pub fn spawn_break_burst(
        &mut self,
        block_pos: IVec3,
        node_id: u16,
        atlas: &TextureAtlas,
        registries: &EngineRegistries,
    ) {
        self.spawn_dig_chips(block_pos, node_id, 24, atlas, registries, 15, 0);
    }

    /// Spawn 8..=14 textured 3D mini-cube debris particles with velocity, spin, gravity, and ground bounce.
    pub fn spawn_shatter_debris(
        &mut self,
        block_pos: IVec3,
        node_id: u16,
        atlas: &TextureAtlas,
        registries: &EngineRegistries,
        sky: u8,
        block: u8,
    ) {
        if node_id == CONTENT_AIR || node_id == CONTENT_IGNORE {
            return;
        }
        let layers = atlas
            .node_face_layers
            .get(&node_id)
            .copied()
            .unwrap_or([atlas.fallback_layer; 6]);
        let def = registries.get_node_def(node_id);
        let tint = def
            .and_then(|d| {
                d.palette
                    .as_deref()
                    .and_then(|p| atlas.sample_palette_color(p, 0))
                    .or(d.color)
            })
            .unwrap_or([255, 255, 255, 255]);

        let mut rng = rand::thread_rng();
        let center = block_pos.as_vec3() + Vec3::splat(0.5);
        let count = rng.gen_range(8..=14);

        for _ in 0..count {
            if self.particles.len() >= 512 {
                self.particles.remove(0);
            }
            let off = Vec3::new(
                rng.gen_range(-0.34..0.34),
                rng.gen_range(-0.28..0.34),
                rng.gen_range(-0.34..0.34),
            );
            let vel = Vec3::new(
                rng.gen_range(-2.6..2.6),
                rng.gen_range(1.6..4.2),
                rng.gen_range(-2.6..2.6),
            );
            let ang_vel = Vec3::new(
                rng.gen_range(-8.0..8.0),
                rng.gen_range(-8.0..8.0),
                rng.gen_range(-8.0..8.0),
            );
            let u0 = rng.gen_range(0.0..0.68);
            let v0 = rng.gen_range(0.0..0.68);
            let face_idx = rng.gen_range(0..6);
            self.particles.push(Particle {
                kind: ParticleKind::ShatterCube,
                pos: center + off,
                vel,
                rot: Vec3::new(
                    rng.gen_range(0.0..std::f32::consts::TAU),
                    rng.gen_range(0.0..std::f32::consts::TAU),
                    rng.gen_range(0.0..std::f32::consts::TAU),
                ),
                ang_vel,
                size: rng.gen_range(0.11..0.20),
                age: 0.0,
                max_age: rng.gen_range(0.95..1.55),
                layer: layers[face_idx],
                uv_min: [u0, v0],
                uv_max: [u0 + 0.30, v0 + 0.30],
                tint,
                sky,
                block,
                bounces: 0,
            });
        }
    }

    /// Spawn a rising warm orange-gold ember speck for held/world lit torches.
    pub fn spawn_torch_ember(&mut self, world_pos: Vec3) {
        if self.particles.len() >= 512 {
            self.particles.remove(0);
        }
        let mut rng = rand::thread_rng();
        self.particles.push(Particle {
            kind: ParticleKind::TorchEmber,
            pos: world_pos
                + Vec3::new(
                    rng.gen_range(-0.08..0.08),
                    rng.gen_range(-0.02..0.06),
                    rng.gen_range(-0.08..0.08),
                ),
            vel: Vec3::new(
                rng.gen_range(-0.18..0.18),
                rng.gen_range(0.55..1.05),
                rng.gen_range(-0.18..0.18),
            ),
            rot: Vec3::ZERO,
            ang_vel: Vec3::new(2.0, 3.0, 2.0),
            size: rng.gen_range(0.035..0.060),
            age: 0.0,
            max_age: rng.gen_range(0.35..0.65),
            layer: 0,
            uv_min: [0.0, 0.0],
            uv_max: [1.0, 1.0],
            tint: [255, rng.gen_range(145..215), 55, 255],
            sky: 15,
            block: 15,
            bounces: 0,
        });
    }

    pub fn update(&mut self, dt: f32) {
        self.update_with_world(dt, None, None);
    }

    pub fn update_with_world(
        &mut self,
        dt: f32,
        world: Option<&WorldStore>,
        registries: Option<&EngineRegistries>,
    ) {
        let dt = dt.clamp(0.0, 0.05);
        for p in &mut self.particles {
            p.age += dt;
            match p.kind {
                ParticleKind::TorchEmber => {
                    p.vel.y += 0.35 * dt;
                    p.rot += p.ang_vel * dt;
                    p.pos += p.vel * dt;
                }
                ParticleKind::Chip | ParticleKind::ShatterCube => {
                    p.vel.y -= 18.5 * dt;
                    p.vel.x *= 1.0 - (1.8 * dt).min(0.5);
                    p.vel.z *= 1.0 - (1.8 * dt).min(0.5);
                    p.rot += p.ang_vel * dt;
                    p.pos += p.vel * dt;

                    if p.kind == ParticleKind::ShatterCube {
                        if let (Some(w), Some(reg)) = (world, registries) {
                            let hs = p.size * 0.5;
                            let foot_y = (p.pos.y - hs).floor() as i32;
                            let bx = p.pos.x.floor() as i32;
                            let bz = p.pos.z.floor() as i32;
                            let (nid, _, _) = w.get_node(IVec3::new(bx, foot_y, bz));
                            let is_solid = nid != CONTENT_AIR
                                && nid != CONTENT_IGNORE
                                && reg
                                    .get_node_def(nid)
                                    .map(|d| {
                                        d.walkable
                                            && !matches!(
                                                d.drawtype,
                                                DrawType::Liquid | DrawType::FlowingLiquid
                                            )
                                    })
                                    .unwrap_or(true);
                            if is_solid && p.vel.y <= 0.0 {
                                p.pos.y = (foot_y as f32) + 1.0 + hs;
                                if p.vel.y.abs() > 1.35 && p.bounces < 3 {
                                    p.vel.y = -p.vel.y * 0.42;
                                    p.vel.x *= 0.72;
                                    p.vel.z *= 0.72;
                                    p.ang_vel *= 0.65;
                                    p.bounces += 1;
                                } else {
                                    p.vel.y = 0.0;
                                    p.vel.x *= 0.55;
                                    p.vel.z *= 0.55;
                                    p.ang_vel *= 0.25;
                                }
                            }
                        }
                    }
                }
            }
        }
        self.particles.retain(|p| p.age < p.max_age);
    }

    pub fn build_vertices(&self, cam_pos: Vec3) -> (Vec<VoxelVertex>, Vec<u32>) {
        let mut verts = Vec::with_capacity(self.particles.len() * 8);
        let mut indices = Vec::with_capacity(self.particles.len() * 12);
        for p in &self.particles {
            let [u0, v0] = p.uv_min;
            let [u1, v1] = p.uv_max;
            let uvs = [[u0, v1], [u1, v1], [u1, v0], [u0, v0]];

            if p.kind == ParticleKind::Chip {
                let to_cam = (cam_pos - p.pos).normalize_or_zero();
                let fwd = if to_cam.length_squared() > 0.001 {
                    to_cam
                } else {
                    Vec3::Z
                };
                let right = Vec3::Y.cross(fwd).normalize_or_zero();
                let right = if right.length_squared() > 0.001 {
                    right
                } else {
                    Vec3::X
                };
                let up = fwd.cross(right).normalize_or_zero();

                let hs = p.size * 0.5;
                let p0 = p.pos - right * hs - up * hs;
                let p1 = p.pos + right * hs - up * hs;
                let p2 = p.pos + right * hs + up * hs;
                let p3 = p.pos - right * hs + up * hs;

                let packed = VoxelVertex::pack(p.layer, 2, p.sky, p.block, 3, false, true);
                let base = verts.len() as u32;
                for (pos, uv) in [p0, p1, p2, p3].into_iter().zip(uvs) {
                    verts.push(VoxelVertex {
                        pos: pos.to_array(),
                        uv,
                        packed,
                        tint: p.tint,
                    });
                }
                indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            } else {
                // 3D rotated mini-cube for Block Shatter Debris & Emissive Torch Embers
                let life_frac = (p.age / p.max_age.max(0.01)).clamp(0.0, 1.0);
                let shrink = if life_frac > 0.78 {
                    ((1.0 - life_frac) / 0.22).clamp(0.15, 1.0)
                } else {
                    1.0
                };
                let hs = p.size * 0.5 * shrink;
                let rot_m = glam::Mat3::from_euler(glam::EulerRot::YXZ, p.rot.y, p.rot.x, p.rot.z);
                let extra = if p.kind == ParticleKind::TorchEmber {
                    VoxelVertex::FLAG_EMISSIVE | (1u32 << 27)
                } else {
                    1u32 << 27
                };
                let corners = [
                    Vec3::new(-hs, -hs, -hs),
                    Vec3::new(hs, -hs, -hs),
                    Vec3::new(hs, hs, -hs),
                    Vec3::new(-hs, hs, -hs),
                    Vec3::new(-hs, -hs, hs),
                    Vec3::new(hs, -hs, hs),
                    Vec3::new(hs, hs, hs),
                    Vec3::new(-hs, hs, hs),
                ]
                .map(|c| p.pos + rot_m * c);

                let faces: [(u8, [usize; 4]); 6] = [
                    (0, [5, 6, 2, 1]), // +X
                    (1, [0, 3, 7, 4]), // -X
                    (2, [7, 6, 2, 3]), // +Y
                    (3, [0, 1, 5, 4]), // -Y
                    (4, [4, 7, 6, 5]), // +Z
                    (5, [1, 2, 3, 0]), // -Z
                ];
                for (fdir, idx4) in faces {
                    let packed = VoxelVertex::pack(p.layer, fdir, p.sky, p.block, 3, false, false)
                        | extra;
                    let base = verts.len() as u32;
                    for (&ci, uv) in idx4.iter().zip(uvs) {
                        verts.push(VoxelVertex {
                            pos: corners[ci].to_array(),
                            uv,
                            packed,
                            tint: p.tint,
                        });
                    }
                    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                }
            }
        }
        (verts, indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{DrawType, NodeDef};
    use std::collections::HashMap;
    use std::path::Path;

    #[test]
    fn selection_wireframe_cracks_and_particles_work() {
        let boxes = vec![([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])];
        let lines = build_selection_wireframe(&boxes);
        assert_eq!(lines.len(), 24);

        let mut reg = EngineRegistries::new();
        let stone = reg.register_node(NodeDef {
            name: "mcl_core:stone".into(),
            drawtype: DrawType::Normal,
            ..Default::default()
        });
        let atlas = TextureAtlas::build(&HashMap::new(), Path::new(""), None, &reg);
        let (cv, ci) = build_crack_overlay(&boxes, 5, atlas.get_crack_layers());
        assert_eq!(cv.len(), 24);
        assert_eq!(ci.len(), 36);

        let mut ps = ParticleSystem::new();
        ps.spawn_break_burst(IVec3::new(1, 2, 3), stone, &atlas, &reg);
        assert_eq!(ps.particles.len(), 24);
        ps.update(0.05);
        let (pv, pi) = ps.build_vertices(Vec3::new(1.5, 4.0, 6.0));
        assert_eq!(pv.len(), 96);
        assert_eq!(pi.len(), 144);
    }

    #[test]
    fn shatter_debris_cubes_bounce_and_torch_embers_rise() {
        let mut reg = EngineRegistries::new();
        let stone = reg.register_node(NodeDef {
            name: "mcl_core:stone".into(),
            drawtype: DrawType::Normal,
            walkable: true,
            ..Default::default()
        });
        let atlas = TextureAtlas::build(&HashMap::new(), Path::new(""), None, &reg);

        let dir = std::env::temp_dir().join("cubex_shatter_bounce_test");
        let _ = std::fs::remove_dir_all(&dir);
        let world = WorldStore::new(&dir, 42);
        // Solid stone floor at y = 0 (top surface at y = 1.0)
        for x in -2..=4 {
            for z in -2..=4 {
                world.set_node(IVec3::new(x, 0, z), stone, 15, 0);
            }
        }

        let mut ps = ParticleSystem::new();
        ps.spawn_shatter_debris(IVec3::new(1, 2, 1), stone, &atlas, &reg, 15, 0);
        assert!(
            (8..=14).contains(&ps.particles.len()),
            "Expected 8..=14 shatter cubes, got {}",
            ps.particles.len()
        );
        let n_cubes = ps.particles.len();
        let (sv, si) = ps.build_vertices(Vec3::new(1.5, 4.0, 6.0));
        assert_eq!(sv.len(), n_cubes * 24);
        assert_eq!(si.len(), n_cubes * 36);

        // Simulate falling onto floor at y = 1.0 and bouncing
        for _ in 0..30 {
            ps.update_with_world(0.03, Some(&world), Some(&reg));
        }
        assert!(
            ps.particles
                .iter()
                .all(|p| p.kind == ParticleKind::ShatterCube && p.pos.y >= 1.0),
            "Shatter cubes must collide and bounce above y=1.0 floor"
        );

        let mut embers = ParticleSystem::new();
        embers.spawn_torch_ember(Vec3::new(0.5, 2.0, 0.5));
        let y0 = embers.particles[0].pos.y;
        embers.update_with_world(0.05, None, None);
        assert!(embers.particles[0].pos.y > y0, "Torch ember must rise");
        let (ev, _) = embers.build_vertices(Vec3::ZERO);
        assert!(ev.iter().all(|v| (v.packed & VoxelVertex::FLAG_EMISSIVE) != 0));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
