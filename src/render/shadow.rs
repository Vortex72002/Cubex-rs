use bytemuck::{Pod, Zeroable};
use glam::{IVec3, Mat4, Vec3};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

use crate::render::mesher::VoxelVertex;
use crate::render::GpuChunkMesh;

pub const SHADOW_MAP_SIZE: u32 = 2048;
pub const SHADOW_HALF_EXTENT: f32 = 64.0;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ShadowUniform {
    pub light_view_proj: [[f32; 4]; 4],
    pub params: [f32; 4], // x=enabled, y=bias, z=strength, w=texel_size
}

impl Default for ShadowUniform {
    fn default() -> Self {
        Self {
            light_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            params: [0.0, 0.0012, 0.0, 1.0 / SHADOW_MAP_SIZE as f32],
        }
    }
}

pub struct ShadowMapRenderer {
    pub uniform_buffer: wgpu::Buffer,
    pub pass_bind_group: wgpu::BindGroup,
    pub sample_bgl: wgpu::BindGroupLayout,
    pub sample_bg: wgpu::BindGroup,
    pub depth_view: wgpu::TextureView,
    pub pipeline: wgpu::RenderPipeline,
    _depth_tex: wgpu::Texture,
    _sampler: wgpu::Sampler,
}

impl ShadowMapRenderer {
    pub fn new(device: &wgpu::Device) -> Self {
        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SunShadowDepth2048"),
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SunShadowCmpSampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let init_u = ShadowUniform::default();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SunShadowUniformBuffer"),
            contents: bytemuck::bytes_of(&init_u),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let pass_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ShadowPassBGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pass_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ShadowPassBG"),
            layout: &pass_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let sample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ShadowSampleBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });

        let sample_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ShadowSampleBG"),
            layout: &sample_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ShadowShader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shadow.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ShadowPipelineLayout"),
            bind_group_layouts: &[&pass_bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SunShadowDepthPipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_shadow"),
                buffers: &[VoxelVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            uniform_buffer,
            pass_bind_group,
            sample_bgl,
            sample_bg,
            depth_view,
            pipeline,
            _depth_tex: depth_tex,
            _sampler: sampler,
        }
    }

    /// Texel-snapped directional sun orthographic view-projection matrix centered around `focus_pos`.
    pub fn compute_light_view_proj(focus_pos: Vec3, sun_dir: Vec3) -> Mat4 {
        let mut ldir = sun_dir.normalize_or_zero();
        if ldir.length_squared() < 1e-4 || ldir.y < 0.05 {
            ldir = Vec3::new(0.45, 0.82, 0.35).normalize();
        }
        let up = if ldir.y.abs() > 0.96 { Vec3::Z } else { Vec3::Y };
        let light_dist = 140.0;
        let center_view = Mat4::look_to_rh(focus_pos + ldir * light_dist, -ldir, up);

        // Snap focus to shadow-map texel grid to prevent edge shimmer while walking.
        let world_units_per_texel = (SHADOW_HALF_EXTENT * 2.0) / SHADOW_MAP_SIZE as f32;
        let origin_ls = center_view.transform_point3(focus_pos);
        let snap_x = (origin_ls.x / world_units_per_texel).round() * world_units_per_texel;
        let snap_y = (origin_ls.y / world_units_per_texel).round() * world_units_per_texel;
        let snapped_center = focus_pos + center_view.inverse().transform_vector3(Vec3::new(
            snap_x - origin_ls.x,
            snap_y - origin_ls.y,
            0.0,
        ));

        let view = Mat4::look_to_rh(snapped_center + ldir * light_dist, -ldir, up);
        let proj = Mat4::orthographic_rh(
            -SHADOW_HALF_EXTENT,
            SHADOW_HALF_EXTENT,
            -SHADOW_HALF_EXTENT,
            SHADOW_HALF_EXTENT,
            1.0,
            light_dist * 2.0 + 48.0,
        );
        proj * view
    }

    pub fn update_uniform(
        &self,
        queue: &wgpu::Queue,
        focus_pos: Vec3,
        sun_dir: Vec3,
        enabled: bool,
    ) {
        let strength = if enabled && sun_dir.y > 0.04 {
            ((sun_dir.y - 0.04) / 0.18).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let u = ShadowUniform {
            light_view_proj: Self::compute_light_view_proj(focus_pos, sun_dir).to_cols_array_2d(),
            params: [
                if enabled && strength > 0.001 { 1.0 } else { 0.0 },
                0.0011,
                strength,
                1.0 / SHADOW_MAP_SIZE as f32,
            ],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&u));
    }

    pub fn encode_shadow_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        chunk_meshes: &HashMap<IVec3, GpuChunkMesh>,
        focus_pos: Vec3,
    ) {
        let max_dist_sq = (SHADOW_HALF_EXTENT + 28.0).powi(2);
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SunShadowDepthPass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_viewport(0.0, 0.0, SHADOW_MAP_SIZE as f32, SHADOW_MAP_SIZE as f32, 0.0, 1.0);
        rpass.set_scissor_rect(0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.pass_bind_group, &[]);

        for m in chunk_meshes.values() {
            let center = (m.chunk_pos * 16).as_vec3() + Vec3::splat(8.0);
            let dx = center.x - focus_pos.x;
            let dz = center.z - focus_pos.z;
            if dx * dx + dz * dz > max_dist_sq {
                continue;
            }
            if let (Some(ref vb), Some(ref ib)) = (&m.opaque_vb, &m.opaque_ib) {
                if m.opaque_index_count > 0 {
                    rpass.set_vertex_buffer(0, vb.slice(..));
                    rpass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    rpass.draw_indexed(0..m.opaque_index_count, 0, 0..1);
                }
            }
        }
    }
}
