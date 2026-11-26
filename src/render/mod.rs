pub mod bloom;
pub mod camera;
pub mod entity_model;
pub mod frustum;
pub mod mesher;
pub mod pipeline;
pub mod player_model;
pub mod shadow;
pub mod sky;
pub mod viewmodel;

pub use self::bloom::{BloomPostProcess, BloomTargets};

pub use self::camera::{compute_camera_rig, CameraMode, CameraRig};
pub use self::entity_model::{
    build_dropped_ram_mic_mesh, build_entity_rig_mesh, build_falling_node_mesh,
    canonical_mob_texture_for_entity, classify_entity_rig, fill_transparent_mob_pixels,
    hurt_tint_for_flash, make_fallback_mob_texture, resolve_entity_block_layers,
    EntityRenderSnapshot, EntityRigKind,
};
pub use self::pipeline::{
    build_crack_overlay, build_selection_wireframe, IndexedMeshBuffers, LineVertex, ParticleSystem,
};
pub use self::player_model::{
    build_player_model_vertices, build_player_model_vertices_lit, carter_hair_palette_image,
    PlayerModelMesh, PlayerModelState, PlayerPartCounts,
};
pub use self::shadow::ShadowMapRenderer;
pub use self::sky::{
    compute_nether_sky_palette, compute_sky_palette, SkyPalette, SkyRenderer, SkyUniform,
};
pub use self::viewmodel::{
    build_viewmodel_mesh, build_viewmodel_mesh_ex, build_viewmodel_mesh_with_skin,
    create_ram_mic_mesh, ram_mic_palette_image, ViewmodelMesh, ViewmodelState,
};

use bytemuck::{Pod, Zeroable};
use glam::{IVec3, Mat4, Vec3};
use image::RgbaImage;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::config::{AnisotropyQuality, EngineSettings};
use crate::registry::EngineRegistries;
use crate::textures::TextureAtlas;
use crate::world::light::relight_around_pos;
use crate::world::{DimensionId, WorldStore};

use self::frustum::Frustum;
use self::mesher::{mesh_chunks_parallel_with_atlas, ChunkMeshData, VoxelVertex};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub sun_dir_time: [f32; 4],
    pub fog_color_dist: [f32; 4],
    pub flags: [f32; 4],
    pub hand_light_pos_intensity: [f32; 4],
    pub enhance_flags: [f32; 4],
    pub sky_zenith_clouds: [f32; 4],
    pub sky_sun_color_atmo: [f32; 4],
}

pub fn held_item_light_intensity(item_key: &str, registries: &EngineRegistries) -> f32 {
    if item_key.is_empty() {
        return 0.0;
    }
    let resolved = registries.resolve_alias(item_key);
    let cid = registries.get_content_id(resolved);
    let lower = resolved.to_ascii_lowercase();
    let emit = registries
        .get_node_def(cid)
        .map(|d| d.light_source)
        .filter(|&l| l > 0)
        .unwrap_or_else(|| {
            if lower.contains("torch")
                || lower.contains("lantern")
                || lower.contains("glowstone")
                || lower.contains("lava")
                || lower.contains("shroomlight")
                || lower.contains("blaze")
            {
                14
            } else {
                0
            }
        });
    if emit > 0 {
        (emit as f32 / 15.0).clamp(0.45, 1.0)
    } else {
        0.0
    }
}

pub struct GpuChunkMesh {
    pub chunk_pos: IVec3,
    pub opaque_vb: Option<wgpu::Buffer>,
    pub opaque_ib: Option<wgpu::Buffer>,
    pub opaque_index_count: u32,
    pub trans_vb: Option<wgpu::Buffer>,
    pub trans_ib: Option<wgpu::Buffer>,
    pub trans_index_count: u32,
}

/// Camera and visibility descriptor for rendering one viewport (full-screen, split-screen, quad-split, or multi-monitor).
#[derive(Debug, Clone, Copy)]
pub struct ViewportCamera {
    pub cam_pos: Vec3,
    pub view_proj: Mat4,
    /// `[x, y, width, height]` in physical target pixels.
    pub viewport_px: [u32; 4],
    /// Local player index (`0..=3`) owning this viewport's uniform/viewmodel/selection/crack overlay.
    pub player_slot: usize,
    /// Bitmask of local player character bodies (`1 << slot`, bits `0..=3`) to draw in this viewport.
    pub draw_local_player_mask: u8,
    pub draw_viewmodel: bool,
    pub dimension: DimensionId,
}

/// Snapshot for updating an animated 3D remote LAN player mesh.
#[derive(Debug, Clone)]
pub struct RemotePlayerRenderSnapshot {
    pub name: String,
    pub pos: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub horizontal_speed: f32,
    pub is_sneaking: bool,
    pub is_swinging: bool,
    pub swing_progress: f32,
    pub held_item: Option<String>,
    pub hurt_flash: f32,
}

/// Cached GPU buffers and skeletal state for one remote LAN player.
pub struct RemotePlayerRenderSlot {
    pub world_pos: Vec3,
    pub model: PlayerModelState,
    body_buffers: IndexedMeshBuffers,
    overlay_buffers: IndexedMeshBuffers,
    crown_buffers: IndexedMeshBuffers,
    held_buffers: IndexedMeshBuffers,
    held_tex_bg: Option<wgpu::BindGroup>,
    held_uses_atlas: bool,
    held_cached_key: String,
}

impl RemotePlayerRenderSlot {
    fn new() -> Self {
        Self {
            world_pos: Vec3::ZERO,
            model: PlayerModelState::new(),
            body_buffers: IndexedMeshBuffers::empty(),
            overlay_buffers: IndexedMeshBuffers::empty(),
            crown_buffers: IndexedMeshBuffers::empty(),
            held_buffers: IndexedMeshBuffers::empty(),
            held_tex_bg: None,
            held_uses_atlas: true,
            held_cached_key: String::new(),
        }
    }
}

struct ViewportCamBuffers {
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    viewmodel_cam_buffer: wgpu::Buffer,
    viewmodel_cam_bind_group: wgpu::BindGroup,
    sky_buffer: wgpu::Buffer,
    sky_bind_group: wgpu::BindGroup,
}

/// Per-player viewmodel, selection box, crack overlay, and articulated 3rd-person buffers for Player 2.
pub struct PlayerRenderSlot {
    selection_vb: Option<wgpu::Buffer>,
    selection_vert_count: u32,
    last_selection_boxes: Vec<([f32; 3], [f32; 3])>,
    crack_vb: Option<wgpu::Buffer>,
    crack_ib: Option<wgpu::Buffer>,
    crack_index_count: u32,
    last_crack_stage: Option<u8>,
    last_crack_boxes: Vec<([f32; 3], [f32; 3])>,

    pub viewmodel: ViewmodelState,
    pub show_viewmodel: bool,
    viewmodel_mesh: Option<ViewmodelMesh>,
    viewmodel_vb: Option<wgpu::Buffer>,
    viewmodel_ib: Option<wgpu::Buffer>,
    viewmodel_index_count: u32,
    viewmodel_tex_bg: Option<wgpu::BindGroup>,
    viewmodel_uses_atlas: bool,
    viewmodel_cached_key: String,

    pub player_model: PlayerModelState,
    pub show_player_model: bool,
    pub player_world_pos: Vec3,
    player_body_buffers: IndexedMeshBuffers,
    player_overlay_buffers: IndexedMeshBuffers,
    player_crown_buffers: IndexedMeshBuffers,
    player_held_buffers: IndexedMeshBuffers,
    player_held_tex_bg: Option<wgpu::BindGroup>,
    player_held_uses_atlas: bool,
    player_held_cached_key: String,

    pub hand_light_intensity: f32,
    pub hurt_flash: f32,
}

impl Default for PlayerRenderSlot {
    fn default() -> Self {
        Self {
            selection_vb: None,
            selection_vert_count: 0,
            last_selection_boxes: Vec::new(),
            crack_vb: None,
            crack_ib: None,
            crack_index_count: 0,
            last_crack_stage: None,
            last_crack_boxes: Vec::new(),
            viewmodel: ViewmodelState::new(),
            show_viewmodel: false,
            viewmodel_mesh: None,
            viewmodel_vb: None,
            viewmodel_ib: None,
            viewmodel_index_count: 0,
            viewmodel_tex_bg: None,
            viewmodel_uses_atlas: true,
            viewmodel_cached_key: String::new(),
            player_model: PlayerModelState::new(),
            show_player_model: false,
            player_world_pos: Vec3::ZERO,
            player_body_buffers: IndexedMeshBuffers::empty(),
            player_overlay_buffers: IndexedMeshBuffers::empty(),
            player_crown_buffers: IndexedMeshBuffers::empty(),
            player_held_buffers: IndexedMeshBuffers::empty(),
            player_held_tex_bg: None,
            player_held_uses_atlas: true,
            player_held_cached_key: String::new(),
            hand_light_intensity: 0.0,
            hurt_flash: 0.0,
        }
    }
}

/// Secondary swapchain surface + depth + egui renderer for Player 2's monitor window.
pub struct SecondaryRenderTarget {
    pub surface: wgpu::Surface<'static>,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub depth_view: wgpu::TextureView,
    pub bloom_targets: Option<BloomTargets>,
    pub egui_renderer: egui_wgpu::Renderer,
}

pub struct WgpuRenderer {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub surface: Option<wgpu::Surface<'static>>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub format: wgpu::TextureFormat,
    pub adapter_name: String,
    pub no_vsync_mode: wgpu::PresentMode,

    pub depth_view: wgpu::TextureView,
    pub offscreen_color: Option<(wgpu::Texture, wgpu::TextureView)>,

    viewport_cams: [ViewportCamBuffers; 4],
    pub tex_bind_group_layout: wgpu::BindGroupLayout,
    atlas_tex_view: wgpu::TextureView,
    anisotropy_quality: AnisotropyQuality,
    pub tex_bind_group: wgpu::BindGroup,

    pub opaque_pipeline: wgpu::RenderPipeline,
    pub trans_pipeline: wgpu::RenderPipeline,
    pub crack_pipeline: wgpu::RenderPipeline,
    pub line_pipeline: wgpu::RenderPipeline,
    hdr_opaque_pipeline: wgpu::RenderPipeline,
    hdr_trans_pipeline: wgpu::RenderPipeline,
    hdr_crack_pipeline: wgpu::RenderPipeline,
    hdr_line_pipeline: wgpu::RenderPipeline,
    sky_renderer: SkyRenderer,
    shadow: ShadowMapRenderer,
    bloom: BloomPostProcess,
    bloom_targets: Option<BloomTargets>,

    pub egui_renderer: egui_wgpu::Renderer,
    pub chunk_meshes: HashMap<IVec3, GpuChunkMesh>,
    pub nether_chunk_meshes: HashMap<IVec3, GpuChunkMesh>,

    // Selection wireframe & progressive break crack overlay
    selection_vb: Option<wgpu::Buffer>,
    selection_vert_count: u32,
    last_selection_boxes: Vec<([f32; 3], [f32; 3])>,
    crack_vb: Option<wgpu::Buffer>,
    crack_ib: Option<wgpu::Buffer>,
    crack_index_count: u32,
    last_crack_stage: Option<u8>,
    last_crack_boxes: Vec<([f32; 3], [f32; 3])>,

    // 3D block-break & atmospheric particle systems per dimension
    pub particles: ParticleSystem,
    pub nether_particles: ParticleSystem,

    // First-person viewmodel state & GPU buffers
    pub viewmodel: ViewmodelState,
    pub show_viewmodel: bool,
    viewmodel_mesh: Option<ViewmodelMesh>,
    viewmodel_vb: Option<wgpu::Buffer>,
    viewmodel_ib: Option<wgpu::Buffer>,
    viewmodel_index_count: u32,
    viewmodel_tex_bg: Option<wgpu::BindGroup>,
    viewmodel_uses_atlas: bool,
    viewmodel_cached_key: String,
    viewmodel_extruded: bool,

    // 3rd-person articulated player character state & GPU buffers
    pub player_model: PlayerModelState,
    pub show_player_model: bool,
    player_world_pos: Vec3,
    player_body_buffers: IndexedMeshBuffers,
    player_overlay_buffers: IndexedMeshBuffers,
    player_crown_buffers: IndexedMeshBuffers,
    player_skin_tex_bg: Option<wgpu::BindGroup>,
    player_skin_dims: Option<(u32, u32)>,
    crown_palette_tex_bg: wgpu::BindGroup,
    skin_tex_cache: HashMap<String, (wgpu::BindGroup, (u32, u32))>,
    skin_rgba_cache: HashMap<String, RgbaImage>,
    player_held_buffers: IndexedMeshBuffers,
    player_held_tex_bg: Option<wgpu::BindGroup>,
    player_held_uses_atlas: bool,
    player_held_cached_key: String,

    // Handheld dynamic light intensity (0.0..=1.0)
    pub hand_light_intensity: f32,
    pub player_hurt_flash: f32,

    // Local co-op Player 2..4 GPU buffers & peer rendering toggle
    pub peers: [PlayerRenderSlot; 3],
    pub coop_render_peer: bool,
    pub remote_players: HashMap<String, RemotePlayerRenderSlot>,

    /// Optional atmospheric fog end-distance override (used by Title Panorama horizon blending).
    pub fog_dist_override: Option<f32>,

    // Active 3D LuaEntity/mob GPU batches & cached texture bind groups
    entity_tex_cache: HashMap<String, (wgpu::BindGroup, u32, u32)>,
    entity_draw_batches: Vec<(String, IndexedMeshBuffers)>,
    nether_entity_draw_batches: Vec<(String, IndexedMeshBuffers)>,
    laser_vb: Option<wgpu::Buffer>,
    laser_vert_count: u32,

    // Offscreen cloud-streaming render target & readback buffer (for mobile LAN streaming)
    stream_target: Option<(wgpu::Texture, wgpu::TextureView, wgpu::TextureView, u32, u32)>,
    stream_readback_buf: Option<(wgpu::Buffer, u32, u32, u32)>,

    pub last_draw_calls: u32,
    pub last_visible_chunks: u32,
    pub start_time: Instant,
}

impl WgpuRenderer {
    pub fn new_windowed(window: Arc<Window>, atlas: &TextureAtlas) -> anyhow::Result<Self> {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            ..Default::default()
        });

        let surface = instance.create_surface(window)?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| anyhow::anyhow!("No suitable GPU adapter found"))?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let no_vsync_mode = if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else if caps.present_modes.contains(&wgpu::PresentMode::Immediate) {
            wgpu::PresentMode::Immediate
        } else {
            wgpu::PresentMode::AutoNoVsync
        };

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: no_vsync_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let renderer = Self::init_core(
            instance,
            Some(surface),
            adapter,
            format,
            surface_config,
            no_vsync_mode,
            atlas,
        )?;
        if let Some(ref surf) = renderer.surface {
            surf.configure(&renderer.device, &renderer.surface_config);
        }
        Ok(renderer)
    }

    pub fn new_headless(width: u32, height: u32, atlas: &TextureAtlas) -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| anyhow::anyhow!("No suitable headless GPU adapter found"))?;

        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::Immediate,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let mut renderer = Self::init_core(
            instance,
            None,
            adapter,
            format,
            surface_config,
            wgpu::PresentMode::Immediate,
            atlas,
        )?;
        let (tex, view) = Self::create_color_target(&renderer.device, format, width, height);
        renderer.offscreen_color = Some((tex, view));
        Ok(renderer)
    }

    fn init_core(
        instance: wgpu::Instance,
        surface: Option<wgpu::Surface<'static>>,
        adapter: wgpu::Adapter,
        format: wgpu::TextureFormat,
        surface_config: wgpu::SurfaceConfiguration,
        no_vsync_mode: wgpu::PresentMode,
        atlas: &TextureAtlas,
    ) -> anyhow::Result<Self> {
        let adapter_name = adapter.get_info().name.clone();
        let adapter_limits = adapter.limits();
        let mut limits = wgpu::Limits::default();
        limits.max_texture_array_layers = adapter_limits.max_texture_array_layers.min(2048).max(256);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Cubex-RS GPU Device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))?;

        let depth_view = Self::create_depth_target(&device, surface_config.width, surface_config.height);

        let cam_init = CameraUniform {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0, 24.0, 0.0, 0.0],
            sun_dir_time: [0.4, 0.85, 0.3, 0.0],
            fog_color_dist: [0.46, 0.72, 0.96, 160.0],
            flags: [1.0, 0.0, 1.0, 1.0],
            hand_light_pos_intensity: [0.0, 0.0, 0.0, 0.0],
            enhance_flags: [0.0, 0.0, 0.0, 0.35],
            sky_zenith_clouds: [0.13, 0.38, 0.88, 2.0],
            sky_sun_color_atmo: [1.35, 1.24, 0.98, 1.0],
        };
        let hdr_format = BloomPostProcess::choose_hdr_format(&adapter, format);
        let sky_renderer = SkyRenderer::new(&device, format, hdr_format);
        let sky_init = SkyUniform::from_camera(
            Mat4::IDENTITY,
            0.35,
            0.0,
            &EngineSettings::default(),
            false,
        );
        let cam_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("CameraBGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let make_cam_buffers = |idx: usize| -> ViewportCamBuffers {
            let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("CameraUniformBuffer{idx}")),
                contents: bytemuck::bytes_of(&cam_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let viewmodel_cam_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("ViewmodelCameraUniformBuffer{idx}")),
                contents: bytemuck::bytes_of(&cam_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let sky_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("SkyUniformBuffer{idx}")),
                contents: bytemuck::bytes_of(&sky_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("CameraBG{idx}")),
                layout: &cam_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                }],
            });
            let viewmodel_cam_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("ViewmodelCameraBG{idx}")),
                layout: &cam_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: viewmodel_cam_buffer.as_entire_binding(),
                }],
            });
            let sky_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("SkyBG{idx}")),
                layout: &sky_renderer.bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sky_buffer.as_entire_binding(),
                }],
            });
            ViewportCamBuffers {
                camera_buffer,
                camera_bind_group,
                viewmodel_cam_buffer,
                viewmodel_cam_bind_group,
                sky_buffer,
                sky_bind_group,
            }
        };
        let viewport_cams = [
            make_cam_buffers(0),
            make_cam_buffers(1),
            make_cam_buffers(2),
            make_cam_buffers(3),
        ];

        let tex_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TexArrayBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let anisotropy_quality = AnisotropyQuality::Off;
        let (atlas_tex_view, tex_bind_group) = Self::upload_atlas_internal(
            &device,
            &queue,
            &tex_bind_group_layout,
            atlas,
            anisotropy_quality,
        );

        let shadow = ShadowMapRenderer::new(&device);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VoxelShader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("VoxelPipelineLayout"),
            bind_group_layouts: &[&cam_bgl, &tex_bind_group_layout, &shadow.sample_bgl],
            push_constant_ranges: &[],
        });

        let line_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("LinePipelineLayout"),
            bind_group_layouts: &[&cam_bgl],
            push_constant_ranges: &[],
        });

        let bloom = BloomPostProcess::new(&device, format, hdr_format);

        let build_voxel_pipelines = |target_fmt: wgpu::TextureFormat, prefix: &str| {
            let opaque = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("{prefix}OpaqueVoxelPipeline")),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[VoxelVertex::desc()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_fmt,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
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
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

            let trans = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("{prefix}TransVoxelPipeline")),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[VoxelVertex::desc()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_trans"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_fmt,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

            let crack = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("{prefix}CrackOverlayPipeline")),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[VoxelVertex::desc()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_trans"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_fmt,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

            let line = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("{prefix}SelectionLinePipeline")),
                layout: Some(&line_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_line"),
                    buffers: &[LineVertex::desc()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_line"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_fmt,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

            (opaque, trans, crack, line)
        };

        let (opaque_pipeline, trans_pipeline, crack_pipeline, line_pipeline) =
            build_voxel_pipelines(format, "Sdr");
        let (hdr_opaque_pipeline, hdr_trans_pipeline, hdr_crack_pipeline, hdr_line_pipeline) =
            build_voxel_pipelines(hdr_format, "Hdr");

        let egui_renderer = egui_wgpu::Renderer::new(&device, format, None, 1, false);
        let crown_palette_img = carter_hair_palette_image();
        let crown_palette_tex_bg = Self::upload_single_layer_bind_group(
            &device,
            &queue,
            &tex_bind_group_layout,
            &crown_palette_img,
        );

        Ok(Self {
            instance,
            adapter,
            surface,
            device,
            queue,
            surface_config,
            format,
            adapter_name,
            no_vsync_mode,
            depth_view,
            offscreen_color: None,
            viewport_cams,
            tex_bind_group_layout,
            atlas_tex_view,
            anisotropy_quality,
            tex_bind_group,
            opaque_pipeline,
            trans_pipeline,
            crack_pipeline,
            line_pipeline,
            hdr_opaque_pipeline,
            hdr_trans_pipeline,
            hdr_crack_pipeline,
            hdr_line_pipeline,
            sky_renderer,
            shadow,
            bloom,
            bloom_targets: None,
            egui_renderer,
            chunk_meshes: HashMap::new(),
            nether_chunk_meshes: HashMap::new(),
            selection_vb: None,
            selection_vert_count: 0,
            last_selection_boxes: Vec::new(),
            crack_vb: None,
            crack_ib: None,
            crack_index_count: 0,
            last_crack_stage: None,
            last_crack_boxes: Vec::new(),
            particles: ParticleSystem::new(),
            nether_particles: ParticleSystem::new(),
            viewmodel: ViewmodelState::new(),
            show_viewmodel: false,
            viewmodel_mesh: None,
            viewmodel_vb: None,
            viewmodel_ib: None,
            viewmodel_index_count: 0,
            viewmodel_tex_bg: None,
            viewmodel_uses_atlas: true,
            viewmodel_cached_key: String::new(),
            viewmodel_extruded: true,
            player_model: PlayerModelState::new(),
            show_player_model: false,
            player_world_pos: Vec3::ZERO,
            player_body_buffers: IndexedMeshBuffers::empty(),
            player_overlay_buffers: IndexedMeshBuffers::empty(),
            player_crown_buffers: IndexedMeshBuffers::empty(),
            player_skin_tex_bg: None,
            player_skin_dims: None,
            crown_palette_tex_bg,
            skin_tex_cache: HashMap::new(),
            skin_rgba_cache: HashMap::new(),
            player_held_buffers: IndexedMeshBuffers::empty(),
            player_held_tex_bg: None,
            player_held_uses_atlas: true,
            player_held_cached_key: String::new(),
            hand_light_intensity: 0.0,
            player_hurt_flash: 0.0,
            peers: std::array::from_fn(|_| PlayerRenderSlot::default()),
            coop_render_peer: false,
            remote_players: HashMap::new(),
            fog_dist_override: None,
            entity_tex_cache: HashMap::new(),
            entity_draw_batches: Vec::new(),
            nether_entity_draw_batches: Vec::new(),
            laser_vb: None,
            laser_vert_count: 0,
            stream_target: None,
            stream_readback_buf: None,
            last_draw_calls: 0,
            last_visible_chunks: 0,
            start_time: Instant::now(),
        })
    }

    fn create_depth_target(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DepthTexture"),
            size: wgpu::Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        tex.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn create_color_target(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        w: u32,
        h: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("HeadlessColorTarget"),
            size: wgpu::Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    fn upload_single_layer_bind_group(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        img: &RgbaImage,
    ) -> wgpu::BindGroup {
        let w = img.width().max(1);
        let h = img.height().max(1);
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ViewmodelTexture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            img.as_raw(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * w),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ViewmodelNearestClampSampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ViewmodelTexBG"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        })
    }

    fn create_atlas_sampler(device: &wgpu::Device, quality: AnisotropyQuality) -> wgpu::Sampler {
        let aniso = quality.clamp_u16();
        if aniso > 1 {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("HybridAnisoAtlasSampler"),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                lod_min_clamp: 0.0,
                lod_max_clamp: 4.0,
                anisotropy_clamp: aniso,
                ..Default::default()
            })
        } else {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("NearestRepeatSampler"),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::FilterMode::Nearest,
                lod_min_clamp: 0.0,
                lod_max_clamp: 0.0,
                anisotropy_clamp: 1,
                ..Default::default()
            })
        }
    }

    fn upload_atlas_internal(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        atlas: &TextureAtlas,
        quality: AnisotropyQuality,
    ) -> (wgpu::TextureView, wgpu::BindGroup) {
        let (w, h, layers, mips, per_mip_bytes) = atlas.pack_mipmapped_rgba_bytes();
        let layers = layers.max(1);
        let mips = mips.max(1);

        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("VoxelTextureArray"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: layers,
            },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        for (m_idx, mip_bytes) in per_mip_bytes.iter().enumerate() {
            let mw = (w >> m_idx).max(1);
            let mh = (h >> m_idx).max(1);
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex,
                    mip_level: m_idx as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                mip_bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * mw),
                    rows_per_image: Some(mh),
                },
                wgpu::Extent3d {
                    width: mw,
                    height: mh,
                    depth_or_array_layers: layers,
                },
            );
        }

        let view = tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let sampler = Self::create_atlas_sampler(device, quality);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TexArrayBG"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        (view, bg)
    }

    pub fn set_anisotropy_quality(&mut self, quality: AnisotropyQuality) {
        if self.anisotropy_quality == quality {
            return;
        }
        self.anisotropy_quality = quality;
        let sampler = Self::create_atlas_sampler(&self.device, quality);
        self.tex_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TexArrayBG"),
            layout: &self.tex_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.atlas_tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
    }

    pub fn anisotropy_quality(&self) -> AnisotropyQuality {
        self.anisotropy_quality
    }

    pub fn wait_idle(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }

    pub fn reload_atlas(&mut self, atlas: &TextureAtlas) {
        let (view, bg) = Self::upload_atlas_internal(
            &self.device,
            &self.queue,
            &self.tex_bind_group_layout,
            atlas,
            self.anisotropy_quality,
        );
        self.atlas_tex_view = view;
        self.tex_bind_group = bg;
        self.viewmodel_cached_key.clear();
        self.player_skin_tex_bg = None;
        self.player_skin_dims = None;
        self.skin_tex_cache.remove("default");
        self.skin_rgba_cache.remove("default");
        self.player_held_cached_key.clear();
        for peer in &mut self.peers {
            peer.viewmodel_cached_key.clear();
            peer.player_held_cached_key.clear();
        }
        for remote in self.remote_players.values_mut() {
            remote.held_cached_key.clear();
        }
        self.entity_tex_cache.clear();
    }

    fn pick_surface_present_mode(
        caps: &wgpu::SurfaceCapabilities,
        preferred_no_vsync: wgpu::PresentMode,
        vsync: bool,
    ) -> wgpu::PresentMode {
        if vsync {
            wgpu::PresentMode::Fifo
        } else if caps.present_modes.contains(&preferred_no_vsync) {
            preferred_no_vsync
        } else if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else if caps.present_modes.contains(&wgpu::PresentMode::Immediate) {
            wgpu::PresentMode::Immediate
        } else if caps.present_modes.contains(&wgpu::PresentMode::AutoNoVsync) {
            wgpu::PresentMode::AutoNoVsync
        } else {
            wgpu::PresentMode::Fifo
        }
    }

    /// Create a secondary swapchain target on another monitor window sharing the same GPU `Device`, chunks, and atlas.
    pub fn create_secondary_target(
        &self,
        window: Arc<Window>,
        vsync: bool,
    ) -> anyhow::Result<SecondaryRenderTarget> {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);
        let surface = self.instance.create_surface(window)?;
        let caps = surface.get_capabilities(&self.adapter);
        if !caps.formats.contains(&self.format) {
            anyhow::bail!(
                "Secondary monitor surface does not support primary pipeline format {:?}",
                self.format
            );
        }
        let present_mode = Self::pick_surface_present_mode(&caps, self.no_vsync_mode, vsync);
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.format,
            width,
            height,
            present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&self.device, &surface_config);
        let depth_view = Self::create_depth_target(&self.device, width, height);
        let egui_renderer = egui_wgpu::Renderer::new(&self.device, self.format, None, 1, false);
        Ok(SecondaryRenderTarget {
            surface,
            surface_config,
            depth_view,
            bloom_targets: None,
            egui_renderer,
        })
    }

    pub fn resize_secondary(
        &self,
        target: &mut SecondaryRenderTarget,
        width: u32,
        height: u32,
        vsync: bool,
    ) {
        let w = width.max(1);
        let h = height.max(1);
        target.surface_config.width = w;
        target.surface_config.height = h;
        let caps = target.surface.get_capabilities(&self.adapter);
        target.surface_config.present_mode =
            Self::pick_surface_present_mode(&caps, self.no_vsync_mode, vsync);
        target.surface.configure(&self.device, &target.surface_config);
        target.depth_view = Self::create_depth_target(&self.device, w, h);
        target.bloom_targets = None;
    }

    pub fn set_secondary_vsync(&self, target: &mut SecondaryRenderTarget, vsync: bool) {
        let caps = target.surface.get_capabilities(&self.adapter);
        let mode = Self::pick_surface_present_mode(&caps, self.no_vsync_mode, vsync);
        if target.surface_config.present_mode != mode {
            target.surface_config.present_mode = mode;
            target.surface.configure(&self.device, &target.surface_config);
        }
    }

    pub fn resize(&mut self, width: u32, height: u32, vsync: bool) {
        let w = width.max(1);
        let h = height.max(1);
        self.surface_config.width = w;
        self.surface_config.height = h;
        self.surface_config.present_mode = if vsync {
            wgpu::PresentMode::Fifo
        } else {
            self.no_vsync_mode
        };
        if let Some(ref surf) = self.surface {
            surf.configure(&self.device, &self.surface_config);
        }
        if self.offscreen_color.is_some() {
            self.offscreen_color = Some(Self::create_color_target(&self.device, self.format, w, h));
        }
        self.depth_view = Self::create_depth_target(&self.device, w, h);
        self.bloom_targets = None;
    }

    pub fn bloom_targets_allocated(&self) -> bool {
        self.bloom_targets.is_some()
    }

    pub fn set_vsync(&mut self, vsync: bool) {
        let mode = if vsync {
            wgpu::PresentMode::Fifo
        } else {
            self.no_vsync_mode
        };
        if self.surface_config.present_mode != mode {
            self.surface_config.present_mode = mode;
            if let Some(ref surf) = self.surface {
                surf.configure(&self.device, &self.surface_config);
            }
        }
    }

    fn build_gpu_chunk_mesh(&self, mesh: ChunkMeshData) -> GpuChunkMesh {
        let opaque_vb = if !mesh.opaque_verts.is_empty() {
            Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ChunkOpaqueVB"),
                contents: bytemuck::cast_slice(&mesh.opaque_verts),
                usage: wgpu::BufferUsages::VERTEX,
            }))
        } else {
            None
        };
        let opaque_ib = if !mesh.opaque_indices.is_empty() {
            Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ChunkOpaqueIB"),
                contents: bytemuck::cast_slice(&mesh.opaque_indices),
                usage: wgpu::BufferUsages::INDEX,
            }))
        } else {
            None
        };
        let trans_vb = if !mesh.trans_verts.is_empty() {
            Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ChunkTransVB"),
                contents: bytemuck::cast_slice(&mesh.trans_verts),
                usage: wgpu::BufferUsages::VERTEX,
            }))
        } else {
            None
        };
        let trans_ib = if !mesh.trans_indices.is_empty() {
            Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ChunkTransIB"),
                contents: bytemuck::cast_slice(&mesh.trans_indices),
                usage: wgpu::BufferUsages::INDEX,
            }))
        } else {
            None
        };

        GpuChunkMesh {
            chunk_pos: mesh.chunk_pos,
            opaque_vb,
            opaque_ib,
            opaque_index_count: mesh.opaque_indices.len() as u32,
            trans_vb,
            trans_ib,
            trans_index_count: mesh.trans_indices.len() as u32,
        }
    }

    /// Upload a single `ChunkMeshData` directly to Overworld GPU buffers.
    pub fn upload_chunk_mesh_now(&mut self, mesh: ChunkMeshData) {
        self.upload_chunk_mesh_for_dimension(mesh, DimensionId::Overworld);
    }

    pub fn upload_chunk_mesh_for_dimension(&mut self, mesh: ChunkMeshData, dim: DimensionId) {
        let cpos = mesh.chunk_pos;
        let gpu_mesh = self.build_gpu_chunk_mesh(mesh);
        if dim.is_nether() {
            self.nether_chunk_meshes.insert(cpos, gpu_mesh);
        } else {
            self.chunk_meshes.insert(cpos, gpu_mesh);
        }
    }

    /// Upload pending background `ChunkMeshData` buffers within a strict `budget_ms` GPU budget.
    pub fn upload_meshes_budgeted(
        &mut self,
        pending: &mut VecDeque<ChunkMeshData>,
        max_chunks: usize,
        budget_ms: f32,
    ) -> usize {
        self.upload_meshes_budgeted_for_dimension(
            pending,
            max_chunks,
            budget_ms,
            DimensionId::Overworld,
        )
    }

    pub fn upload_meshes_budgeted_for_dimension(
        &mut self,
        pending: &mut VecDeque<ChunkMeshData>,
        max_chunks: usize,
        budget_ms: f32,
        dim: DimensionId,
    ) -> usize {
        let start = Instant::now();
        let budget_secs = (budget_ms * 0.001).max(0.0);
        let mut uploaded = 0usize;
        while uploaded < max_chunks {
            if uploaded > 0 && start.elapsed().as_secs_f32() > budget_secs {
                break;
            }
            let Some(mesh) = pending.pop_front() else {
                break;
            };
            self.upload_chunk_mesh_for_dimension(mesh, dim);
            uploaded += 1;
        }
        uploaded
    }

    /// Prune GPU chunk buffers outside `keep_radius` chunk columns around `(center_cx, center_cz)`.
    pub fn prune_distant_chunks(&mut self, center_cx: i32, center_cz: i32, keep_radius: i32) -> usize {
        self.prune_distant_chunks_multi(&[(center_cx, center_cz)], keep_radius)
    }

    /// Prune GPU chunk buffers outside `keep_radius` columns of ALL active player centers.
    pub fn prune_distant_chunks_multi(
        &mut self,
        centers: &[(i32, i32)],
        keep_radius: i32,
    ) -> usize {
        self.prune_distant_chunks_for_dimension(DimensionId::Overworld, centers, keep_radius)
    }

    pub fn prune_distant_chunks_for_dimension(
        &mut self,
        dim: DimensionId,
        centers: &[(i32, i32)],
        keep_radius: i32,
    ) -> usize {
        let map = if dim.is_nether() {
            &mut self.nether_chunk_meshes
        } else {
            &mut self.chunk_meshes
        };
        if centers.is_empty() {
            let cleared = map.len();
            map.clear();
            return cleared;
        }
        let before = map.len();
        map.retain(|cpos, _| {
            centers.iter().any(|&(cx, cz)| {
                (cpos.x - cx).abs() <= keep_radius && (cpos.z - cz).abs() <= keep_radius
            })
        });
        before - map.len()
    }

    pub fn particles_for_mut(&mut self, dim: DimensionId) -> &mut ParticleSystem {
        if dim.is_nether() {
            &mut self.nether_particles
        } else {
            &mut self.particles
        }
    }

    /// Priority same-frame synchronous remeshing for player block break/place edits (`<= 0ms` queue delay).
    pub fn remesh_priority_chunks_now(
        &mut self,
        world: &WorldStore,
        registries: &EngineRegistries,
        atlas: &TextureAtlas,
        smooth_lighting: bool,
    ) -> usize {
        let dirty = world.take_priority_dirty_chunks();
        if dirty.is_empty() {
            return 0;
        }
        let snaps: Vec<_> = dirty
            .into_iter()
            .filter_map(|cpos| world.snapshot_chunk(cpos))
            .collect();
        if snaps.is_empty() {
            return 0;
        }
        let dim = world.dimension;
        let meshes = mesh_chunks_parallel_with_atlas(snaps, registries, atlas, smooth_lighting);
        let count = meshes.len();
        for mesh in meshes {
            self.upload_chunk_mesh_for_dimension(mesh, dim);
        }
        count
    }

    /// Convenience helper: incrementally relight around `edited_pos`, mark priority dirty, and remesh synchronously.
    pub fn relight_and_remesh_at(
        &mut self,
        world: &WorldStore,
        edited_pos: IVec3,
        registries: &EngineRegistries,
        atlas: &TextureAtlas,
        smooth_lighting: bool,
    ) -> usize {
        let light_chunks = relight_around_pos(world, edited_pos, registries);
        {
            let mut pq = world.priority_dirty_queue.write();
            for cpos in light_chunks {
                pq.insert(cpos);
            }
        }
        world.mark_priority_dirty_with_neighbors(edited_pos);
        self.remesh_priority_chunks_now(world, registries, atlas, smooth_lighting)
    }

    fn write_selection_slot(
        device: &wgpu::Device,
        boxes: &[([f32; 3], [f32; 3])],
        vb: &mut Option<wgpu::Buffer>,
        vert_count: &mut u32,
        cache: &mut Vec<([f32; 3], [f32; 3])>,
    ) {
        if boxes.is_empty() {
            if vb.is_some() || !cache.is_empty() {
                *vb = None;
                *vert_count = 0;
                cache.clear();
            }
            return;
        }
        if vb.is_some() && cache.as_slice() == boxes {
            return;
        }
        *cache = boxes.to_vec();
        let verts = build_selection_wireframe(boxes);
        *vert_count = verts.len() as u32;
        *vb = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SelectionWireframeVB"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        }));
    }

    fn write_crack_slot(
        device: &wgpu::Device,
        boxes: &[([f32; 3], [f32; 3])],
        stage: Option<u8>,
        atlas: &TextureAtlas,
        vb: &mut Option<wgpu::Buffer>,
        ib: &mut Option<wgpu::Buffer>,
        index_count: &mut u32,
        last_stage: &mut Option<u8>,
        last_boxes: &mut Vec<([f32; 3], [f32; 3])>,
    ) {
        let Some(st) = stage else {
            if vb.is_some() || last_stage.is_some() || !last_boxes.is_empty() {
                *vb = None;
                *ib = None;
                *index_count = 0;
                *last_stage = None;
                last_boxes.clear();
            }
            return;
        };
        if boxes.is_empty() {
            if vb.is_some() || last_stage.is_some() || !last_boxes.is_empty() {
                *vb = None;
                *ib = None;
                *index_count = 0;
                *last_stage = None;
                last_boxes.clear();
            }
            return;
        }
        if vb.is_some() && *last_stage == Some(st) && last_boxes.as_slice() == boxes {
            return;
        }
        *last_stage = Some(st);
        *last_boxes = boxes.to_vec();
        let (verts, indices) = build_crack_overlay(boxes, st, atlas.get_crack_layers());
        *index_count = indices.len() as u32;
        *vb = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CrackOverlayVB"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        }));
        *ib = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CrackOverlayIB"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        }));
    }

    /// Update wireframe black selection box around targeted node AABBs (`selection_boxes`).
    pub fn set_selection_boxes(&mut self, boxes: &[([f32; 3], [f32; 3])]) {
        Self::write_selection_slot(
            &self.device,
            boxes,
            &mut self.selection_vb,
            &mut self.selection_vert_count,
            &mut self.last_selection_boxes,
        );
    }

    /// Borrow a local co-op peer slot (`slot` in `1..=3` for P2..P4).
    pub fn peer_slot(&self, slot: usize) -> Option<&PlayerRenderSlot> {
        if (1..=3).contains(&slot) {
            Some(&self.peers[slot - 1])
        } else {
            None
        }
    }

    /// Mutably borrow a local co-op peer slot (`slot` in `1..=3` for P2..P4).
    pub fn peer_slot_mut(&mut self, slot: usize) -> Option<&mut PlayerRenderSlot> {
        if (1..=3).contains(&slot) {
            Some(&mut self.peers[slot - 1])
        } else {
            None
        }
    }

    /// Update wireframe black selection box for player `slot` (`0` = P1, `1..=3` = P2..P4).
    pub fn set_peer_selection_boxes(&mut self, slot: usize, boxes: &[([f32; 3], [f32; 3])]) {
        if slot == 0 {
            self.set_selection_boxes(boxes);
            return;
        }
        if let Some(peer) = self.peers.get_mut(slot - 1) {
            Self::write_selection_slot(
                &self.device,
                boxes,
                &mut peer.selection_vb,
                &mut peer.selection_vert_count,
                &mut peer.last_selection_boxes,
            );
        }
    }

    /// Update Player 2 wireframe black selection box.
    pub fn set_p2_selection_boxes(&mut self, boxes: &[([f32; 3], [f32; 3])]) {
        self.set_peer_selection_boxes(1, boxes);
    }

    /// Update progressive block-break crack overlay (`stage` in `0..=9`).
    pub fn set_crack_overlay(
        &mut self,
        boxes: &[([f32; 3], [f32; 3])],
        stage: Option<u8>,
        atlas: &TextureAtlas,
    ) {
        Self::write_crack_slot(
            &self.device,
            boxes,
            stage,
            atlas,
            &mut self.crack_vb,
            &mut self.crack_ib,
            &mut self.crack_index_count,
            &mut self.last_crack_stage,
            &mut self.last_crack_boxes,
        );
    }

    /// Update crack overlay for player `slot` (`0` = P1, `1..=3` = P2..P4).
    pub fn set_peer_crack_overlay(
        &mut self,
        slot: usize,
        boxes: &[([f32; 3], [f32; 3])],
        stage: Option<u8>,
        atlas: &TextureAtlas,
    ) {
        if slot == 0 {
            self.set_crack_overlay(boxes, stage, atlas);
            return;
        }
        if let Some(peer) = self.peers.get_mut(slot - 1) {
            Self::write_crack_slot(
                &self.device,
                boxes,
                stage,
                atlas,
                &mut peer.crack_vb,
                &mut peer.crack_ib,
                &mut peer.crack_index_count,
                &mut peer.last_crack_stage,
                &mut peer.last_crack_boxes,
            );
        }
    }

    /// Update Player 2 progressive block-break crack overlay (`stage` in `0..=9`).
    pub fn set_p2_crack_overlay(
        &mut self,
        boxes: &[([f32; 3], [f32; 3])],
        stage: Option<u8>,
        atlas: &TextureAtlas,
    ) {
        self.set_peer_crack_overlay(1, boxes, stage, atlas);
    }

    /// Update first-person viewmodel mesh, handheld torch light intensity, and walk/swing/slot animations.
    #[allow(clippy::too_many_arguments)]
    pub fn set_extruded_held_items(&mut self, enabled: bool) {
        self.viewmodel.sway_enabled = enabled;
        for peer in &mut self.peers {
            peer.viewmodel.sway_enabled = enabled;
        }
        if self.viewmodel_extruded != enabled {
            self.viewmodel_extruded = enabled;
            self.viewmodel_cached_key.clear();
            for peer in &mut self.peers {
                peer.viewmodel_cached_key.clear();
            }
        }
    }

    pub fn update_viewmodel(
        &mut self,
        held_item_name: Option<&str>,
        selected_slot: usize,
        dt: f32,
        horizontal_speed: f32,
        on_ground: bool,
        is_sprinting: bool,
        is_sneaking: bool,
        registries: &EngineRegistries,
        atlas: &TextureAtlas,
    ) {
        let item_key = held_item_name.unwrap_or("").split_whitespace().next().unwrap_or("");
        if self.viewmodel.last_slot != usize::MAX
            && (self.viewmodel.last_slot != selected_slot || self.viewmodel.last_item_name != item_key)
        {
            self.viewmodel.on_slot_changed();
        }
        self.viewmodel.last_slot = selected_slot;

        if self.viewmodel.last_item_name != item_key || self.viewmodel_cached_key != item_key || self.viewmodel_mesh.is_none() {
            self.viewmodel.last_item_name = item_key.to_string();
            self.viewmodel_cached_key = item_key.to_string();

            let skin_opt = self.skin_rgba_cache.get(&self.player_model.skin_id);
            let mesh = build_viewmodel_mesh_with_skin(
                Some(item_key),
                self.viewmodel_extruded,
                skin_opt,
                registries,
                atlas,
            );
            self.viewmodel_uses_atlas = mesh.texture.is_none();
            if let Some(ref img) = mesh.texture {
                self.viewmodel_tex_bg = Some(Self::upload_single_layer_bind_group(
                    &self.device,
                    &self.queue,
                    &self.tex_bind_group_layout,
                    img,
                ));
            } else {
                self.viewmodel_tex_bg = None;
            }
            if !mesh.verts.is_empty() && !mesh.indices.is_empty() {
                self.viewmodel_index_count = mesh.indices.len() as u32;
                self.viewmodel_vb = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("ViewmodelVB"),
                    contents: bytemuck::cast_slice(&mesh.verts),
                    usage: wgpu::BufferUsages::VERTEX,
                }));
                self.viewmodel_ib = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("ViewmodelIB"),
                    contents: bytemuck::cast_slice(&mesh.indices),
                    usage: wgpu::BufferUsages::INDEX,
                }));
            } else {
                self.viewmodel_index_count = 0;
                self.viewmodel_vb = None;
                self.viewmodel_ib = None;
            }
            self.viewmodel_mesh = Some(mesh);

            self.hand_light_intensity = held_item_light_intensity(item_key, registries);
        }

        self.viewmodel
            .update(dt, horizontal_speed, on_ground, is_sprinting, is_sneaking);
    }

    /// Update Player 2 first-person viewmodel mesh and animations.
    /// Update first-person viewmodel for local player `slot` (`1..=3` for P2..P4, `0` = P1).
    #[allow(clippy::too_many_arguments)]
    pub fn update_peer_viewmodel(
        &mut self,
        slot: usize,
        held_item_name: Option<&str>,
        selected_slot: usize,
        dt: f32,
        horizontal_speed: f32,
        on_ground: bool,
        is_sprinting: bool,
        is_sneaking: bool,
        registries: &EngineRegistries,
        atlas: &TextureAtlas,
    ) {
        if slot == 0 {
            self.update_viewmodel(
                held_item_name,
                selected_slot,
                dt,
                horizontal_speed,
                on_ground,
                is_sprinting,
                is_sneaking,
                registries,
                atlas,
            );
            return;
        }
        let extruded = self.viewmodel_extruded;
        let slot_skin_id = self
            .peers
            .get(slot - 1)
            .map(|p| p.player_model.skin_id.clone())
            .unwrap_or_else(|| "default".into());
        let slot_skin_img = self.skin_rgba_cache.get(&slot_skin_id).cloned();
        let Some(peer) = self.peers.get_mut(slot - 1) else {
            return;
        };
        let item_key = held_item_name.unwrap_or("").split_whitespace().next().unwrap_or("");
        if peer.viewmodel.last_slot != usize::MAX
            && (peer.viewmodel.last_slot != selected_slot || peer.viewmodel.last_item_name != item_key)
        {
            peer.viewmodel.on_slot_changed();
        }
        peer.viewmodel.last_slot = selected_slot;

        if peer.viewmodel.last_item_name != item_key
            || peer.viewmodel_cached_key != item_key
            || peer.viewmodel_mesh.is_none()
        {
            peer.viewmodel.last_item_name = item_key.to_string();
            peer.viewmodel_cached_key = item_key.to_string();

            let mesh = build_viewmodel_mesh_with_skin(
                Some(item_key),
                extruded,
                slot_skin_img.as_ref(),
                registries,
                atlas,
            );
            peer.viewmodel_uses_atlas = mesh.texture.is_none();
            if let Some(ref img) = mesh.texture {
                peer.viewmodel_tex_bg = Some(Self::upload_single_layer_bind_group(
                    &self.device,
                    &self.queue,
                    &self.tex_bind_group_layout,
                    img,
                ));
            } else {
                peer.viewmodel_tex_bg = None;
            }
            if !mesh.verts.is_empty() && !mesh.indices.is_empty() {
                peer.viewmodel_index_count = mesh.indices.len() as u32;
                peer.viewmodel_vb = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Peer{slot}ViewmodelVB")),
                    contents: bytemuck::cast_slice(&mesh.verts),
                    usage: wgpu::BufferUsages::VERTEX,
                }));
                peer.viewmodel_ib = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Peer{slot}ViewmodelIB")),
                    contents: bytemuck::cast_slice(&mesh.indices),
                    usage: wgpu::BufferUsages::INDEX,
                }));
            } else {
                peer.viewmodel_index_count = 0;
                peer.viewmodel_vb = None;
                peer.viewmodel_ib = None;
            }
            peer.viewmodel_mesh = Some(mesh);

            peer.hand_light_intensity = held_item_light_intensity(item_key, registries);
        }

        peer.viewmodel
            .update(dt, horizontal_speed, on_ground, is_sprinting, is_sneaking);
    }

    /// Update Player 2 viewmodel & animation (forwards to `update_peer_viewmodel(1, ...)`).
    #[allow(clippy::too_many_arguments)]
    pub fn update_p2_viewmodel(
        &mut self,
        held_item_name: Option<&str>,
        selected_slot: usize,
        dt: f32,
        horizontal_speed: f32,
        on_ground: bool,
        is_sprinting: bool,
        is_sneaking: bool,
        registries: &EngineRegistries,
        atlas: &TextureAtlas,
    ) {
        self.update_peer_viewmodel(
            1,
            held_item_name,
            selected_slot,
            dt,
            horizontal_speed,
            on_ground,
            is_sprinting,
            is_sneaking,
            registries,
            atlas,
        );
    }

    /// Update 3D articulated character geometry for local player `slot` (`1..=3` for P2..P4, `0` = P1).
    #[allow(clippy::too_many_arguments)]
    pub fn set_player_skin(&mut self, slot: usize, skin_id: &str, skin_rgba: &RgbaImage) {
        let key = if skin_id.trim().is_empty() {
            "default".to_string()
        } else {
            skin_id.trim().to_string()
        };
        let dims = (skin_rgba.width().max(1), skin_rgba.height().max(1));
        let bg = Self::upload_single_layer_bind_group(
            &self.device,
            &self.queue,
            &self.tex_bind_group_layout,
            skin_rgba,
        );
        self.skin_tex_cache.insert(key.clone(), (bg, dims));
        self.skin_rgba_cache.insert(key.clone(), skin_rgba.clone());
        if key == "default" {
            self.player_skin_tex_bg = Some(Self::upload_single_layer_bind_group(
                &self.device,
                &self.queue,
                &self.tex_bind_group_layout,
                skin_rgba,
            ));
            self.player_skin_dims = Some(dims);
        }
        if slot == 0 {
            self.player_model.skin_id = key;
            self.viewmodel_cached_key.clear();
            self.viewmodel_mesh = None;
        } else if let Some(peer) = self.peers.get_mut(slot - 1) {
            peer.player_model.skin_id = key;
            peer.viewmodel_cached_key.clear();
            peer.viewmodel_mesh = None;
        }
    }

    fn skin_bind_group_for(&self, skin_id: &str) -> Option<&wgpu::BindGroup> {
        let key = if skin_id.trim().is_empty() {
            "default"
        } else {
            skin_id.trim()
        };
        self.skin_tex_cache
            .get(key)
            .map(|(bg, _)| bg)
            .or(self.player_skin_tex_bg.as_ref())
    }

    fn skin_dims_for(&mut self, skin_id: &str, atlas: &TextureAtlas) -> (u32, u32) {
        let key = if skin_id.trim().is_empty() {
            "default"
        } else {
            skin_id.trim()
        };
        if let Some((_, dims)) = self.skin_tex_cache.get(key) {
            return *dims;
        }
        self.ensure_player_skin_uploaded(atlas)
    }

    /// Update 3rd-person articulated character geometry & GPU buffers for any local player slot (`0..=3`).
    #[allow(clippy::too_many_arguments)]
    pub fn update_peer_player_model(
        &mut self,
        slot: usize,
        player_pos: Vec3,
        head_yaw: f32,
        head_pitch: f32,
        horizontal_speed: f32,
        is_sneaking: bool,
        held_item_name: Option<&str>,
        dt: f32,
        registries: &EngineRegistries,
        atlas: &TextureAtlas,
    ) {
        if slot == 0 {
            self.update_player_model(
                player_pos,
                head_yaw,
                head_pitch,
                horizontal_speed,
                is_sneaking,
                held_item_name,
                dt,
                registries,
                atlas,
            );
            return;
        }
        if slot > 3 {
            return;
        }
        let skin_id = self.peers[slot - 1].player_model.skin_id.clone();
        let (skin_w, skin_h) = self.skin_dims_for(&skin_id, atlas);
        let peer = &mut self.peers[slot - 1];
        peer.player_world_pos = player_pos;
        let held_key = held_item_name
            .unwrap_or("")
            .split_whitespace()
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        let held_opt = if held_key.is_empty() {
            None
        } else {
            Some(held_key.clone())
        };
        let is_swinging = peer.viewmodel.is_swinging();
        let swing_progress = peer.viewmodel.swing_progress();
        peer.player_model.update(
            dt,
            head_yaw,
            head_pitch,
            horizontal_speed,
            is_sneaking,
            is_swinging,
            swing_progress,
            held_opt,
        );

        let mut mesh = build_player_model_vertices_lit(
            &peer.player_model,
            player_pos,
            skin_w,
            skin_h,
            15,
            0,
            registries,
            atlas,
        );
        apply_player_hurt_flash(&mut mesh.body_verts, peer.hurt_flash);
        apply_player_hurt_flash(&mut mesh.overlay_verts, peer.hurt_flash);
        apply_player_hurt_flash(&mut mesh.crown_verts, peer.hurt_flash);
        peer.player_body_buffers = IndexedMeshBuffers::from_mesh(
            &self.device,
            &format!("Peer{slot}ModelBody"),
            &mesh.body_verts,
            &mesh.body_indices,
        );
        peer.player_overlay_buffers = IndexedMeshBuffers::from_mesh(
            &self.device,
            &format!("Peer{slot}ModelOverlay"),
            &mesh.overlay_verts,
            &mesh.overlay_indices,
        );
        peer.player_crown_buffers = IndexedMeshBuffers::from_mesh(
            &self.device,
            &format!("Peer{slot}ModelCrown"),
            &mesh.crown_verts,
            &mesh.crown_indices,
        );
        peer.player_held_buffers = IndexedMeshBuffers::from_mesh(
            &self.device,
            &format!("Peer{slot}ModelHeld"),
            &mesh.held_verts,
            &mesh.held_indices,
        );
        if peer.player_held_cached_key != held_key {
            peer.player_held_cached_key = held_key;
            peer.player_held_uses_atlas = mesh.held_texture.is_none();
            if let Some(ref himg) = mesh.held_texture {
                peer.player_held_tex_bg = Some(Self::upload_single_layer_bind_group(
                    &self.device,
                    &self.queue,
                    &self.tex_bind_group_layout,
                    himg,
                ));
            } else {
                peer.player_held_tex_bg = None;
            }
        }
    }

    fn ensure_player_skin_uploaded(&mut self, atlas: &TextureAtlas) -> (u32, u32) {
        if let (Some(_), Some(dims)) = (&self.player_skin_tex_bg, self.player_skin_dims) {
            dims
        } else {
            let skin_img = atlas.get_player_skin_rgba();
            let dims = (skin_img.width(), skin_img.height());
            self.player_skin_tex_bg = Some(Self::upload_single_layer_bind_group(
                &self.device,
                &self.queue,
                &self.tex_bind_group_layout,
                &skin_img,
            ));
            self.player_skin_dims = Some(dims);
            dims
        }
    }

    /// Update Player 2 3D articulated character geometry (drawn in peers' views and 3rd-person view).
    #[allow(clippy::too_many_arguments)]
    pub fn update_p2_player_model(
        &mut self,
        player_pos: Vec3,
        head_yaw: f32,
        head_pitch: f32,
        horizontal_speed: f32,
        is_sneaking: bool,
        held_item_name: Option<&str>,
        dt: f32,
        registries: &EngineRegistries,
        atlas: &TextureAtlas,
    ) {
        self.update_peer_player_model(
            1,
            player_pos,
            head_yaw,
            head_pitch,
            horizontal_speed,
            is_sneaking,
            held_item_name,
            dt,
            registries,
            atlas,
        );
    }

    /// Update 3D articulated character meshes for all connected remote LAN players.
    pub fn update_remote_players(
        &mut self,
        remotes: &[RemotePlayerRenderSnapshot],
        dt: f32,
        registries: &EngineRegistries,
        atlas: &TextureAtlas,
    ) {
        self.remote_players
            .retain(|name, _| remotes.iter().any(|r| &r.name == name));
        if remotes.is_empty() {
            return;
        }
        let (skin_w, skin_h) = self.ensure_player_skin_uploaded(atlas);
        for snap in remotes {
            let entry = self
                .remote_players
                .entry(snap.name.clone())
                .or_insert_with(RemotePlayerRenderSlot::new);
            entry.world_pos = snap.pos;
            let held_key = snap
                .held_item
                .as_deref()
                .unwrap_or("")
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
            let held_opt = if held_key.is_empty() {
                None
            } else {
                Some(held_key.clone())
            };
            entry.model.update(
                dt,
                snap.yaw,
                snap.pitch,
                snap.horizontal_speed,
                snap.is_sneaking,
                snap.is_swinging,
                snap.swing_progress,
                held_opt,
            );
            let mut mesh = build_player_model_vertices_lit(
                &entry.model,
                snap.pos,
                skin_w,
                skin_h,
                15,
                0,
                registries,
                atlas,
            );
            apply_player_hurt_flash(&mut mesh.body_verts, snap.hurt_flash);
            apply_player_hurt_flash(&mut mesh.overlay_verts, snap.hurt_flash);
            apply_player_hurt_flash(&mut mesh.crown_verts, snap.hurt_flash);
            entry.body_buffers = IndexedMeshBuffers::from_mesh(
                &self.device,
                "RemotePlayerBody",
                &mesh.body_verts,
                &mesh.body_indices,
            );
            entry.overlay_buffers = IndexedMeshBuffers::from_mesh(
                &self.device,
                "RemotePlayerOverlay",
                &mesh.overlay_verts,
                &mesh.overlay_indices,
            );
            entry.crown_buffers = IndexedMeshBuffers::from_mesh(
                &self.device,
                "RemotePlayerCrown",
                &mesh.crown_verts,
                &mesh.crown_indices,
            );
            entry.held_buffers = IndexedMeshBuffers::from_mesh(
                &self.device,
                "RemotePlayerHeld",
                &mesh.held_verts,
                &mesh.held_indices,
            );
            if entry.held_cached_key != held_key {
                entry.held_cached_key = held_key;
                entry.held_uses_atlas = mesh.held_texture.is_none();
                if let Some(ref himg) = mesh.held_texture {
                    entry.held_tex_bg = Some(Self::upload_single_layer_bind_group(
                        &self.device,
                        &self.queue,
                        &self.tex_bind_group_layout,
                        himg,
                    ));
                } else {
                    entry.held_tex_bg = None;
                }
            }
        }
    }

    /// Update 3rd-person articulated character animation & GPU buffers (synced with `self.viewmodel` swing).
    #[allow(clippy::too_many_arguments)]
    pub fn update_player_model(
        &mut self,
        player_pos: Vec3,
        head_yaw: f32,
        head_pitch: f32,
        horizontal_speed: f32,
        is_sneaking: bool,
        held_item_name: Option<&str>,
        dt: f32,
        registries: &EngineRegistries,
        atlas: &TextureAtlas,
    ) {
        let is_swinging = self.viewmodel.is_swinging();
        let swing_progress = self.viewmodel.swing_progress();
        self.update_player_model_with_swing(
            player_pos,
            head_yaw,
            head_pitch,
            horizontal_speed,
            is_sneaking,
            is_swinging,
            swing_progress,
            held_item_name,
            dt,
            15,
            0,
            registries,
            atlas,
        );
    }

    /// Update 3rd-person articulated character with explicit swing & light parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn update_player_model_with_swing(
        &mut self,
        player_pos: Vec3,
        head_yaw: f32,
        head_pitch: f32,
        horizontal_speed: f32,
        is_sneaking: bool,
        is_swinging: bool,
        swing_progress: f32,
        held_item_name: Option<&str>,
        dt: f32,
        sky_light: u8,
        block_light: u8,
        registries: &EngineRegistries,
        atlas: &TextureAtlas,
    ) {
        self.player_world_pos = player_pos;
        let held_key = held_item_name
            .unwrap_or("")
            .split_whitespace()
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        let held_opt = if held_key.is_empty() {
            None
        } else {
            Some(held_key.clone())
        };

        self.player_model.update(
            dt,
            head_yaw,
            head_pitch,
            horizontal_speed,
            is_sneaking,
            is_swinging,
            swing_progress,
            held_opt,
        );

        let skin_id = self.player_model.skin_id.clone();
        let (skin_w, skin_h) = self.skin_dims_for(&skin_id, atlas);

        let mut mesh = build_player_model_vertices_lit(
            &self.player_model,
            player_pos,
            skin_w,
            skin_h,
            sky_light,
            block_light,
            registries,
            atlas,
        );
        apply_player_hurt_flash(&mut mesh.body_verts, self.player_hurt_flash);
        apply_player_hurt_flash(&mut mesh.overlay_verts, self.player_hurt_flash);
        apply_player_hurt_flash(&mut mesh.crown_verts, self.player_hurt_flash);
        self.player_body_buffers = IndexedMeshBuffers::from_mesh(
            &self.device,
            "PlayerModelBody",
            &mesh.body_verts,
            &mesh.body_indices,
        );
        self.player_overlay_buffers = IndexedMeshBuffers::from_mesh(
            &self.device,
            "PlayerModelOverlay",
            &mesh.overlay_verts,
            &mesh.overlay_indices,
        );
        self.player_crown_buffers = IndexedMeshBuffers::from_mesh(
            &self.device,
            "PlayerModelCrown",
            &mesh.crown_verts,
            &mesh.crown_indices,
        );
        self.player_held_buffers = IndexedMeshBuffers::from_mesh(
            &self.device,
            "PlayerModelHeld",
            &mesh.held_verts,
            &mesh.held_indices,
        );

        if self.player_held_cached_key != held_key {
            self.player_held_cached_key = held_key;
            self.player_held_uses_atlas = mesh.held_texture.is_none();
            if let Some(ref himg) = mesh.held_texture {
                self.player_held_tex_bg = Some(Self::upload_single_layer_bind_group(
                    &self.device,
                    &self.queue,
                    &self.tex_bind_group_layout,
                    himg,
                ));
            } else {
                self.player_held_tex_bg = None;
            }
        }
    }

    /// Build batched articulated 3D voxel meshes and cache textures for Overworld world entities.
    pub fn update_entities(
        &mut self,
        entities: &[EntityRenderSnapshot],
        registries: &EngineRegistries,
        atlas: &TextureAtlas,
    ) {
        self.entity_draw_batches = self.build_entity_batches(entities, registries, atlas);
    }

    /// Build batched articulated 3D voxel meshes and cache textures for Nether world entities.
    pub fn update_nether_entities(
        &mut self,
        entities: &[EntityRenderSnapshot],
        registries: &EngineRegistries,
        atlas: &TextureAtlas,
    ) {
        self.nether_entity_draw_batches = self.build_entity_batches(entities, registries, atlas);
    }

    /// Upload world-space charging laser beams for Overworld Ocean Monument Guardians.
    pub fn update_guardian_lasers(&mut self, beams: &[crate::world::GuardianLaserBeam]) {
        if beams.is_empty() {
            self.laser_vert_count = 0;
            return;
        }
        let mut verts: Vec<LineVertex> = Vec::with_capacity(beams.len() * 10);
        for b in beams {
            let p = b.progress.clamp(0.0, 1.0);
            let col = if p < 0.5 {
                let t = p * 2.0;
                [
                    0.25 + (1.0 - 0.25) * t,
                    0.95 + (0.92 - 0.95) * t,
                    1.0 + (0.25 - 1.0) * t,
                    0.95,
                ]
            } else {
                let t = (p - 0.5) * 2.0;
                [
                    1.0,
                    0.92 + (0.22 - 0.92) * t,
                    0.25 + (0.88 - 0.25) * t,
                    1.0,
                ]
            };
            let offsets = [
                Vec3::ZERO,
                Vec3::new(0.035, 0.0, 0.0),
                Vec3::new(-0.035, 0.0, 0.0),
                Vec3::new(0.0, 0.035, 0.0),
                Vec3::new(0.0, -0.035, 0.0),
            ];
            for off in offsets {
                verts.push(LineVertex {
                    pos: (b.start + off).to_array(),
                    color: col,
                });
                verts.push(LineVertex {
                    pos: (b.end + off).to_array(),
                    color: col,
                });
            }
        }
        self.laser_vert_count = verts.len() as u32;
        self.laser_vb = Some(
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("GuardianLaserVB"),
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
        );
    }

    fn build_entity_batches(
        &mut self,
        entities: &[EntityRenderSnapshot],
        registries: &EngineRegistries,
        atlas: &TextureAtlas,
    ) -> Vec<(String, IndexedMeshBuffers)> {
        if entities.is_empty() {
            return Vec::new();
        }

        const ATLAS_ENTITY_BATCH_KEY: &str = "__atlas__";
        let mut grouped: HashMap<String, (Vec<VoxelVertex>, Vec<u32>)> = HashMap::new();
        for ent in entities {
            if matches!(ent.visual.as_str(), "wielditem" | "item" | "cube")
                && ent.entity_name != "__builtin:item"
            {
                if let Some((layers, tints)) = resolve_entity_block_layers(ent, registries, atlas) {
                    let (ev, ei) = build_falling_node_mesh(ent, layers, tints);
                    if !ev.is_empty() && !ei.is_empty() {
                        let entry = grouped
                            .entry(ATLAS_ENTITY_BATCH_KEY.to_string())
                            .or_default();
                        let base = entry.0.len() as u32;
                        entry.0.extend(ev);
                        entry.1.extend(ei.into_iter().map(|idx| base + idx));
                    }
                    continue;
                }
            }

            let canon_tex = canonical_mob_texture_for_entity(&ent.entity_name);
            let raw_tex_token = ent
                .texture_name
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim();
            let resolved_item_spec = if ent.entity_name == "__builtin:item" {
                if !ent.wield_item.is_empty() {
                    Some(ent.wield_item.as_str())
                } else if !ent.texture_name.is_empty() {
                    Some(ent.texture_name.as_str())
                } else {
                    None
                }
            } else if matches!(ent.visual.as_str(), "wielditem" | "item" | "cube")
                && !raw_tex_token.is_empty()
                && !raw_tex_token.ends_with(".png")
                && atlas.get_item_rgba(raw_tex_token, registries).is_some()
            {
                Some(raw_tex_token)
            } else {
                None
            };
            let item_id = resolved_item_spec
                .and_then(|s| s.split_whitespace().next())
                .map(|tok| registries.resolve_alias(tok.trim()))
                .unwrap_or("");
            if item_id == crate::registry::RAM_MIC_ITEM_ID {
                const RAM_MIC_BATCH_KEY: &str = "prop:cubex:ram_mic";
                if !self.entity_tex_cache.contains_key(RAM_MIC_BATCH_KEY) {
                    let pal = ram_mic_palette_image();
                    let pw = pal.width().max(1);
                    let ph = pal.height().max(1);
                    let bg = Self::upload_single_layer_bind_group(
                        &self.device,
                        &self.queue,
                        &self.tex_bind_group_layout,
                        &pal,
                    );
                    self.entity_tex_cache
                        .insert(RAM_MIC_BATCH_KEY.to_string(), (bg, pw, ph));
                }
                let (ev, ei) = build_dropped_ram_mic_mesh(ent);
                if !ev.is_empty() && !ei.is_empty() {
                    let entry = grouped.entry(RAM_MIC_BATCH_KEY.to_string()).or_default();
                    let base = entry.0.len() as u32;
                    entry.0.extend(ev);
                    entry.1.extend(ei.into_iter().map(|idx| base + idx));
                }
                continue;
            }
            let tex_key = if let Some(item_spec) = resolved_item_spec {
                format!("item:{item_spec}")
            } else if !ent.texture_name.is_empty() {
                format!("tex:{}", ent.texture_name)
            } else if let Some(canon) = canon_tex {
                format!("canon:{canon}")
            } else {
                format!("mob:{}", ent.entity_name)
            };

            if !self.entity_tex_cache.contains_key(&tex_key) {
                let is_item_sprite = tex_key.starts_with("item:");
                let mut img = if let Some(item_spec) = tex_key.strip_prefix("item:") {
                    atlas
                        .get_item_rgba(item_spec, registries)
                        .unwrap_or_else(|| make_fallback_mob_texture(&ent.entity_name))
                } else if let Some(tex_spec) = tex_key.strip_prefix("tex:") {
                    let base = tex_spec
                        .split('^')
                        .next()
                        .unwrap_or(tex_spec)
                        .trim_matches(|c| c == '(' || c == ')' || c == ' ');
                    atlas
                        .resolve_spec_immut(tex_spec)
                        .or_else(|| atlas.get_rgba(base).cloned())
                        .or_else(|| {
                            base.rsplit('/')
                                .next()
                                .and_then(|short| atlas.get_rgba(short).cloned())
                        })
                        .or_else(|| canon_tex.and_then(|c| atlas.get_rgba(c).cloned()))
                        .unwrap_or_else(|| make_fallback_mob_texture(&ent.entity_name))
                } else if let Some(canon) = tex_key.strip_prefix("canon:") {
                    atlas
                        .get_rgba(canon)
                        .cloned()
                        .unwrap_or_else(|| make_fallback_mob_texture(&ent.entity_name))
                } else {
                    canon_tex
                        .and_then(|c| atlas.get_rgba(c).cloned())
                        .unwrap_or_else(|| make_fallback_mob_texture(&ent.entity_name))
                };
                if !is_item_sprite {
                    fill_transparent_mob_pixels(&mut img);
                }
                let dims = (img.width().max(1), img.height().max(1));
                let bg = Self::upload_single_layer_bind_group(
                    &self.device,
                    &self.queue,
                    &self.tex_bind_group_layout,
                    &img,
                );
                self.entity_tex_cache.insert(tex_key.clone(), (bg, dims.0, dims.1));
            }

            let &(_, tw, th) = self.entity_tex_cache.get(&tex_key).unwrap();
            let (ev, ei) = build_entity_rig_mesh(ent, tw, th);
            if ev.is_empty() || ei.is_empty() {
                continue;
            }
            let entry = grouped.entry(tex_key).or_default();
            let base = entry.0.len() as u32;
            entry.0.extend(ev);
            entry.1.extend(ei.into_iter().map(|idx| base + idx));
        }

        let mut batches = Vec::with_capacity(grouped.len());
        for (tex_key, (verts, indices)) in grouped {
            if verts.is_empty() || indices.is_empty() {
                continue;
            }
            let bufs = IndexedMeshBuffers::from_mesh(&self.device, "EntityBatch", &verts, &indices);
            batches.push((tex_key, bufs));
        }
        batches
    }

    pub fn compute_panorama_camera(time_sec: f32, aspect: f32, center: Vec3) -> (Vec3, Mat4) {
        let angle = time_sec * 0.14;
        let eye = Vec3::new(
            center.x + angle.cos() * 14.0,
            center.y + 10.0 + (time_sec * 0.4).sin() * 1.5,
            center.z + angle.sin() * 14.0,
        );
        let proj = Mat4::perspective_rh(72.0f32.to_radians(), aspect.max(0.1), 0.1, 220.0);
        let view = Mat4::look_at_rh(eye, center, Vec3::Y);
        (eye, proj * view)
    }

    pub fn render_to_headless_target(
        &mut self,
        cam_pos: Vec3,
        view_proj: Mat4,
        time_of_day: f32,
        settings: &EngineSettings,
    ) -> anyhow::Result<RgbaImage> {
        self.render_to_headless_target_with_egui(
            cam_pos,
            view_proj,
            time_of_day,
            settings,
            None,
            None,
            1.0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_to_headless_target_with_egui(
        &mut self,
        cam_pos: Vec3,
        view_proj: Mat4,
        time_of_day: f32,
        settings: &EngineSettings,
        egui_ctx: Option<&egui::Context>,
        egui_output: Option<egui::FullOutput>,
        pixels_per_point: f32,
    ) -> anyhow::Result<RgbaImage> {
        self.render_frame(
            cam_pos,
            view_proj,
            time_of_day,
            settings,
            egui_ctx,
            egui_output,
            pixels_per_point,
        )?;
        self.readback_offscreen_image()
    }

    /// Render a split-screen frame to the offscreen target and read back an RGBA screenshot (for headless tests).
    #[allow(clippy::too_many_arguments)]
    pub fn render_split_to_headless_target(
        &mut self,
        p1_cam: ViewportCamera,
        p2_cam: ViewportCamera,
        time_of_day: f32,
        settings: &EngineSettings,
        egui_ctx: Option<&egui::Context>,
        egui_output: Option<egui::FullOutput>,
        pixels_per_point: f32,
    ) -> anyhow::Result<RgbaImage> {
        self.render_split_frame(
            p1_cam,
            p2_cam,
            time_of_day,
            settings,
            egui_ctx,
            egui_output,
            pixels_per_point,
        )?;
        self.readback_offscreen_image()
    }

    /// Render 1..=4 arbitrary viewports (split-screen, quad-split, etc.) and read back an RGBA screenshot.
    #[allow(clippy::too_many_arguments)]
    pub fn render_viewports_to_headless_target(
        &mut self,
        viewports: &[ViewportCamera],
        time_of_day: f32,
        settings: &EngineSettings,
        egui_ctx: Option<&egui::Context>,
        egui_output: Option<egui::FullOutput>,
        pixels_per_point: f32,
    ) -> anyhow::Result<RgbaImage> {
        self.render_viewports_frame(
            viewports,
            time_of_day,
            settings,
            egui_ctx,
            egui_output,
            pixels_per_point,
        )?;
        self.readback_offscreen_image()
    }

    fn readback_offscreen_image(&self) -> anyhow::Result<RgbaImage> {
        let Some((ref tex, _)) = self.offscreen_color else {
            anyhow::bail!("Renderer was not initialized with an offscreen target");
        };
        let w = self.surface_config.width.max(1);
        let h = self.surface_config.height.max(1);
        let unpadded_bpr = w * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bpr = ((unpadded_bpr + align - 1) / align) * align;
        let buf_size = (padded_bpr * h) as u64;

        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("HeadlessReadbackBuffer"),
            size: buf_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("HeadlessReadbackEncoder"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()??;

        let data = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((w * h * 4) as usize);
        for row in 0..h as usize {
            let start = row * padded_bpr as usize;
            pixels.extend_from_slice(&data[start..start + unpadded_bpr as usize]);
        }
        drop(data);
        readback.unmap();

        RgbaImage::from_raw(w, h, pixels)
            .ok_or_else(|| anyhow::anyhow!("Failed to construct RgbaImage from headless readback"))
    }

    /// Render a single player's 3D viewport to an offscreen `RgbaImage` for PC-to-phone cloud streaming.
    pub fn render_stream_rgba(
        &mut self,
        cam: ViewportCamera,
        width: u32,
        height: u32,
        time_of_day: f32,
        settings: &EngineSettings,
    ) -> anyhow::Result<RgbaImage> {
        self.capture_viewport_with_egui_rgba(
            cam,
            width,
            height,
            time_of_day,
            settings,
            None,
            None,
            1.0,
        )
    }

    /// Render a single 3D viewport with optional `egui` overlay into an offscreen target and read back an `RgbaImage`.
    #[allow(clippy::too_many_arguments)]
    pub fn capture_viewport_with_egui_rgba(
        &mut self,
        mut cam: ViewportCamera,
        width: u32,
        height: u32,
        time_of_day: f32,
        settings: &EngineSettings,
        egui_ctx: Option<&egui::Context>,
        egui_output: Option<egui::FullOutput>,
        pixels_per_point: f32,
    ) -> anyhow::Result<RgbaImage> {
        let w = width.clamp(16, 2560);
        let h = height.clamp(16, 2560);
        cam.viewport_px = [0, 0, w, h];

        let recreate_target = match self.stream_target {
            Some((_, _, _, tw, th)) => tw != w || th != h,
            None => true,
        };
        if recreate_target {
            let (tex, color_view) = Self::create_color_target(&self.device, self.format, w, h);
            let depth_view = Self::create_depth_target(&self.device, w, h);
            self.stream_target = Some((tex, color_view, depth_view, w, h));
        }

        let unpadded_bpr = w * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bpr = ((unpadded_bpr + align - 1) / align) * align;
        let recreate_buf = match self.stream_readback_buf {
            Some((_, _, bw, bh)) => bw != w || bh != h,
            None => true,
        };
        if recreate_buf {
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("CloudStreamReadbackBuffer"),
                size: (padded_bpr * h) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            self.stream_readback_buf = Some((buf, padded_bpr, w, h));
        }

        self.write_viewport_uniforms(0, &cam, time_of_day, settings, false);
        let aspect_sky = sky_color_for_dimension(time_of_day, cam.dimension);
        let fog_dist = self.effective_fog_dist(settings);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("OffscreenCaptureEncoder"),
            });
        if !cam.dimension.is_nether() {
            self.update_and_encode_shadows(&mut encoder, cam.cam_pos, time_of_day, settings);
        }

        let mut vis = 0u32;
        let mut draws = 0u32;
        {
            let (_, ref target_view, ref target_depth, _, _) =
                *self.stream_target.as_ref().unwrap();
            let part_buf = if egui_ctx.is_some() {
                if cam.dimension.is_nether() {
                    self.build_particle_buffers_from(&self.nether_particles, cam.cam_pos)
                } else {
                    self.build_particle_buffers_from(&self.particles, cam.cam_pos)
                }
            } else {
                None
            };
            self.encode_viewport_3d(
                &mut encoder,
                target_view,
                target_depth,
                w,
                h,
                0,
                &cam,
                true,
                aspect_sky,
                fog_dist,
                part_buf.as_ref(),
                false,
                &mut vis,
                &mut draws,
            );
            Self::encode_egui_pass(
                &self.device,
                &self.queue,
                &mut self.egui_renderer,
                &mut encoder,
                target_view,
                w,
                h,
                egui_ctx,
                egui_output,
                pixels_per_point.max(0.5),
            );
        }

        let (ref tex, _, _, _, _) = *self.stream_target.as_ref().unwrap();
        let (ref readback, buf_bpr, _, _) = *self.stream_readback_buf.as_ref().unwrap();
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(buf_bpr),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()??;

        let is_bgra = matches!(
            self.format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        );
        let data = slice.get_mapped_range();
        let mut pixels = vec![0u8; (w * h * 4) as usize];
        for row in 0..h as usize {
            let src_start = row * buf_bpr as usize;
            let src = &data[src_start..src_start + unpadded_bpr as usize];
            let dst = &mut pixels[row * unpadded_bpr as usize..(row + 1) * unpadded_bpr as usize];
            if is_bgra {
                for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
                    d[0] = s[2];
                    d[1] = s[1];
                    d[2] = s[0];
                    d[3] = s[3];
                }
            } else {
                dst.copy_from_slice(src);
            }
        }
        drop(data);
        readback.unmap();

        RgbaImage::from_raw(w, h, pixels)
            .ok_or_else(|| anyhow::anyhow!("Failed to construct RgbaImage from viewport readback"))
    }

    #[inline]
    fn effective_fog_dist(&self, settings: &EngineSettings) -> f32 {
        self.fog_dist_override
            .unwrap_or_else(|| (settings.render_distance_chunks as f32) * 16.0)
    }

    fn update_and_encode_shadows(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        focus_pos: Vec3,
        time_of_day: f32,
        settings: &EngineSettings,
    ) {
        let palette = compute_sky_palette(time_of_day);
        let sun_dir = Vec3::from_array(palette.sun_dir);
        let enabled = settings.dynamic_shadows
            && settings.warm_sun_and_atmosphere
            && sun_dir.y > 0.06
            && !self.chunk_meshes.is_empty();
        self.shadow
            .update_uniform(&self.queue, focus_pos, sun_dir, enabled);
        if enabled {
            self.shadow
                .encode_shadow_pass(encoder, &self.chunk_meshes, focus_pos);
        }
    }

    fn write_viewport_uniforms(
        &self,
        cam_buf_idx: usize,
        cam: &ViewportCamera,
        time_of_day: f32,
        settings: &EngineSettings,
        bloom_hdr_active: bool,
    ) {
        let is_nether = cam.dimension.is_nether();
        let palette = if is_nether {
            compute_nether_sky_palette()
        } else {
            compute_sky_palette(time_of_day)
        };
        let aspect_sky = palette.horizon;
        let fog_dist = self.effective_fog_dist(settings);
        let sun_dir = palette.sun_dir;
        let elapsed_sec = self.start_time.elapsed().as_secs_f32();
        let peer_opt = if cam.player_slot >= 1 {
            self.peers.get(cam.player_slot - 1)
        } else {
            None
        };
        let slot_intensity = peer_opt
            .map(|p| p.hand_light_intensity)
            .unwrap_or(self.hand_light_intensity);
        let hand_intensity = if settings.handheld_torch_light {
            slot_intensity
        } else {
            0.0
        };
        let light_pos = if let Some(peer) = peer_opt {
            if peer.show_player_model {
                peer.player_world_pos + Vec3::new(0.0, 1.1, 0.0)
            } else {
                cam.cam_pos
            }
        } else if self.show_player_model {
            self.player_world_pos + Vec3::new(0.0, 1.1, 0.0)
        } else {
            cam.cam_pos
        };
        let warm_flag = if settings.warm_sun_and_atmosphere { 1.0 } else { 0.0 };
        let wind_flag = if settings.wind_swaying_foliage { 1.0 } else { 0.0 };
        let bloom_flag = if bloom_hdr_active { 1.0 } else { 0.0 };
        let water_flag = if settings.waving_water { 1.0 } else { 0.0 };
        let cloud_oct = if is_nether {
            0.0
        } else {
            settings.cloud_octaves() as f32
        };
        let proc_sky_flag = if settings.procedural_sky_clouds { 1.0 } else { 0.0 };
        let dim_w = if is_nether { 1.0 } else { 0.0 };
        let uniform = CameraUniform {
            view_proj: cam.view_proj.to_cols_array_2d(),
            camera_pos: [cam.cam_pos.x, cam.cam_pos.y, cam.cam_pos.z, dim_w],
            sun_dir_time: [sun_dir[0], sun_dir[1], sun_dir[2], elapsed_sec],
            fog_color_dist: [aspect_sky[0], aspect_sky[1], aspect_sky[2], fog_dist],
            flags: [
                if settings.smooth_lighting { 1.0 } else { 0.0 },
                if settings.dynamic_shadows && !is_nether {
                    1.0
                } else {
                    0.0
                },
                water_flag,
                settings.brightness,
            ],
            hand_light_pos_intensity: [light_pos.x, light_pos.y, light_pos.z, hand_intensity],
            enhance_flags: [warm_flag, wind_flag, bloom_flag, time_of_day],
            sky_zenith_clouds: [palette.zenith[0], palette.zenith[1], palette.zenith[2], cloud_oct],
            sky_sun_color_atmo: [
                palette.sun_color[0],
                palette.sun_color[1],
                palette.sun_color[2],
                proc_sky_flag,
            ],
        };
        let cam_buffers = &self.viewport_cams[cam_buf_idx.min(3)];
        self.queue
            .write_buffer(&cam_buffers.camera_buffer, 0, bytemuck::bytes_of(&uniform));
        let sky_uniform = SkyUniform::from_camera_for_dimension(
            cam.view_proj,
            time_of_day,
            elapsed_sec,
            settings,
            bloom_hdr_active,
            is_nether,
        );
        self.queue
            .write_buffer(&cam_buffers.sky_buffer, 0, bytemuck::bytes_of(&sky_uniform));

        let (show_vm, vm_count, vm_mesh_opt, vm_mat_opt) = if let Some(peer) = peer_opt {
            (
                cam.draw_viewmodel && peer.show_viewmodel,
                peer.viewmodel_index_count,
                peer.viewmodel_mesh.as_ref(),
                peer.viewmodel_mesh
                    .as_ref()
                    .map(|m| peer.viewmodel.model_matrix(m)),
            )
        } else {
            (
                cam.draw_viewmodel && self.show_viewmodel,
                self.viewmodel_index_count,
                self.viewmodel_mesh.as_ref(),
                self.viewmodel_mesh
                    .as_ref()
                    .map(|m| self.viewmodel.model_matrix(m)),
            )
        };

        if show_vm && vm_count > 0 {
            if let (Some(_), Some(model_mat)) = (vm_mesh_opt, vm_mat_opt) {
                let aspect = cam.viewport_px[2].max(1) as f32 / cam.viewport_px[3].max(1) as f32;
                let vm_proj =
                    Mat4::perspective_rh(70.0f32.to_radians(), aspect.max(0.1), 0.05, 16.0);
                let vm_vp = vm_proj * model_mat;
                let vm_uniform = CameraUniform {
                    view_proj: vm_vp.to_cols_array_2d(),
                    camera_pos: [0.0, 0.0, 0.0, dim_w],
                    sun_dir_time: [sun_dir[0], sun_dir[1].max(0.25), sun_dir[2], elapsed_sec],
                    fog_color_dist: [aspect_sky[0], aspect_sky[1], aspect_sky[2], 10000.0],
                    flags: [0.0, 0.0, 0.0, settings.brightness],
                    hand_light_pos_intensity: [
                        0.0,
                        0.0,
                        0.0,
                        if hand_intensity > 0.0 { 0.85 } else { 0.0 },
                    ],
                    enhance_flags: [warm_flag, 0.0, bloom_flag, time_of_day],
                    sky_zenith_clouds: [palette.zenith[0], palette.zenith[1], palette.zenith[2], 0.0],
                    sky_sun_color_atmo: [
                        palette.sun_color[0],
                        palette.sun_color[1],
                        palette.sun_color[2],
                        0.0,
                    ],
                };
                self.queue.write_buffer(
                    &cam_buffers.viewmodel_cam_buffer,
                    0,
                    bytemuck::bytes_of(&vm_uniform),
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_player_mesh_buffers<'a>(
        &'a self,
        rpass: &mut wgpu::RenderPass<'a>,
        opaque_pipe: &'a wgpu::RenderPipeline,
        cam_bg: &'a wgpu::BindGroup,
        skin_bg: Option<&'a wgpu::BindGroup>,
        body_buffers: &'a IndexedMeshBuffers,
        crown_buffers: &'a IndexedMeshBuffers,
        held_buffers: &'a IndexedMeshBuffers,
        held_uses_atlas: bool,
        held_tex_bg: Option<&'a wgpu::BindGroup>,
        draw_calls: &mut u32,
    ) {
        if body_buffers.index_count == 0 {
            return;
        }
        if let (Some(ref vb), Some(ref ib), Some(sbg)) =
            (&body_buffers.vb, &body_buffers.ib, skin_bg)
        {
            rpass.set_pipeline(opaque_pipe);
            rpass.set_bind_group(0, cam_bg, &[]);
            rpass.set_bind_group(1, sbg, &[]);
            rpass.set_vertex_buffer(0, vb.slice(..));
            rpass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..body_buffers.index_count, 0, 0..1);
            *draw_calls += 1;
        }
        if crown_buffers.index_count > 0 {
            if let (Some(ref cvb), Some(ref cib)) = (&crown_buffers.vb, &crown_buffers.ib) {
                rpass.set_pipeline(opaque_pipe);
                rpass.set_bind_group(0, cam_bg, &[]);
                rpass.set_bind_group(1, &self.crown_palette_tex_bg, &[]);
                rpass.set_vertex_buffer(0, cvb.slice(..));
                rpass.set_index_buffer(cib.slice(..), wgpu::IndexFormat::Uint32);
                rpass.draw_indexed(0..crown_buffers.index_count, 0, 0..1);
                *draw_calls += 1;
            }
        }
        if held_buffers.index_count > 0 {
            if let (Some(ref hvb), Some(ref hib)) = (&held_buffers.vb, &held_buffers.ib) {
                rpass.set_pipeline(opaque_pipe);
                rpass.set_bind_group(0, cam_bg, &[]);
                if held_uses_atlas {
                    rpass.set_bind_group(1, &self.tex_bind_group, &[]);
                } else if let Some(hbg) = held_tex_bg {
                    rpass.set_bind_group(1, hbg, &[]);
                } else {
                    rpass.set_bind_group(1, &self.tex_bind_group, &[]);
                }
                rpass.set_vertex_buffer(0, hvb.slice(..));
                rpass.set_index_buffer(hib.slice(..), wgpu::IndexFormat::Uint32);
                rpass.draw_indexed(0..held_buffers.index_count, 0, 0..1);
                *draw_calls += 1;
            }
        }
        rpass.set_bind_group(1, &self.tex_bind_group, &[]);
    }

    fn draw_player_overlay_buffers<'a>(
        &'a self,
        rpass: &mut wgpu::RenderPass<'a>,
        trans_pipe: &'a wgpu::RenderPipeline,
        cam_bg: &'a wgpu::BindGroup,
        skin_bg: Option<&'a wgpu::BindGroup>,
        overlay_buffers: &'a IndexedMeshBuffers,
        draw_calls: &mut u32,
    ) {
        if overlay_buffers.index_count == 0 {
            return;
        }
        if let (Some(ref ovb), Some(ref oib), Some(sbg)) =
            (&overlay_buffers.vb, &overlay_buffers.ib, skin_bg)
        {
            rpass.set_pipeline(trans_pipe);
            rpass.set_bind_group(0, cam_bg, &[]);
            rpass.set_bind_group(1, sbg, &[]);
            rpass.set_vertex_buffer(0, ovb.slice(..));
            rpass.set_index_buffer(oib.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..overlay_buffers.index_count, 0, 0..1);
            *draw_calls += 1;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_viewport_3d(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        target_depth_view: &wgpu::TextureView,
        target_w: u32,
        target_h: u32,
        cam_buf_idx: usize,
        cam: &ViewportCamera,
        clear_color: bool,
        aspect_sky: [f32; 3],
        fog_dist: f32,
        particle_buffers: Option<&(wgpu::Buffer, wgpu::Buffer, u32)>,
        use_hdr_pipelines: bool,
        out_visible: &mut u32,
        out_draws: &mut u32,
    ) {
        let opaque_pipe = if use_hdr_pipelines {
            &self.hdr_opaque_pipeline
        } else {
            &self.opaque_pipeline
        };
        let trans_pipe = if use_hdr_pipelines {
            &self.hdr_trans_pipeline
        } else {
            &self.trans_pipeline
        };
        let crack_pipe = if use_hdr_pipelines {
            &self.hdr_crack_pipeline
        } else {
            &self.crack_pipeline
        };
        let line_pipe = if use_hdr_pipelines {
            &self.hdr_line_pipeline
        } else {
            &self.line_pipeline
        };
        let frustum = Frustum::from_view_proj(cam.view_proj);
        let max_dist_sq = (fog_dist + 24.0).powi(2);

        let chunk_map = if cam.dimension.is_nether() {
            &self.nether_chunk_meshes
        } else {
            &self.chunk_meshes
        };
        let mut visible: Vec<&GpuChunkMesh> = Vec::with_capacity(chunk_map.len());
        for m in chunk_map.values() {
            let min = (m.chunk_pos * 16).as_vec3();
            let max = min + Vec3::splat(16.0);
            let center = min + Vec3::splat(8.0);
            if center.distance_squared(cam.cam_pos) <= max_dist_sq && frustum.intersects_aabb(min, max) {
                visible.push(m);
            }
        }
        *out_visible += visible.len() as u32;

        let vx = cam.viewport_px[0].min(target_w.saturating_sub(1));
        let vy = cam.viewport_px[1].min(target_h.saturating_sub(1));
        let vw = cam.viewport_px[2].clamp(1, (target_w - vx).max(1));
        let vh = cam.viewport_px[3].clamp(1, (target_h - vy).max(1));

        let cam_buffers = &self.viewport_cams[cam_buf_idx.min(3)];
        let cam_bg = &cam_buffers.camera_bind_group;
        let vm_bg = &cam_buffers.viewmodel_cam_bind_group;
        let peer_opt = if cam.player_slot >= 1 {
            self.peers.get(cam.player_slot - 1)
        } else {
            None
        };

        let mut draw_calls = 0u32;
        {
            let color_ops = wgpu::Operations {
                load: if clear_color {
                    wgpu::LoadOp::Clear(wgpu::Color {
                        r: aspect_sky[0] as f64,
                        g: aspect_sky[1] as f64,
                        b: aspect_sky[2] as f64,
                        a: 1.0,
                    })
                } else {
                    wgpu::LoadOp::Load
                },
                store: wgpu::StoreOp::Store,
            };
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("VoxelWorldPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: color_ops,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: target_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_viewport(vx as f32, vy as f32, vw as f32, vh as f32, 0.0, 1.0);
            rpass.set_scissor_rect(vx, vy, vw, vh);

            rpass.set_bind_group(0, cam_bg, &[]);
            rpass.set_bind_group(1, &self.tex_bind_group, &[]);
            rpass.set_bind_group(2, &self.shadow.sample_bg, &[]);

            rpass.set_pipeline(opaque_pipe);
            for m in &visible {
                if let (Some(ref vb), Some(ref ib)) = (&m.opaque_vb, &m.opaque_ib) {
                    if m.opaque_index_count > 0 {
                        rpass.set_vertex_buffer(0, vb.slice(..));
                        rpass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        rpass.draw_indexed(0..m.opaque_index_count, 0, 0..1);
                        draw_calls += 1;
                    }
                }
            }

            // Draw local player character bodies requested in `draw_local_player_mask`
            if (cam.draw_local_player_mask & (1u8 << 0)) != 0 {
                self.draw_player_mesh_buffers(
                    &mut rpass,
                    opaque_pipe,
                    cam_bg,
                    self.skin_bind_group_for(&self.player_model.skin_id),
                    &self.player_body_buffers,
                    &self.player_crown_buffers,
                    &self.player_held_buffers,
                    self.player_held_uses_atlas,
                    self.player_held_tex_bg.as_ref(),
                    &mut draw_calls,
                );
            }
            for (peer_idx, peer) in self.peers.iter().enumerate() {
                if (cam.draw_local_player_mask & (1u8 << (peer_idx + 1))) != 0 {
                    self.draw_player_mesh_buffers(
                        &mut rpass,
                        opaque_pipe,
                        cam_bg,
                        self.skin_bind_group_for(&peer.player_model.skin_id),
                        &peer.player_body_buffers,
                        &peer.player_crown_buffers,
                        &peer.player_held_buffers,
                        peer.player_held_uses_atlas,
                        peer.player_held_tex_bg.as_ref(),
                        &mut draw_calls,
                    );
                }
            }
            if !cam.dimension.is_nether() {
                // Draw all connected remote LAN players
                for remote in self.remote_players.values() {
                    self.draw_player_mesh_buffers(
                        &mut rpass,
                        opaque_pipe,
                        cam_bg,
                        self.skin_bind_group_for(&remote.model.skin_id),
                        &remote.body_buffers,
                        &remote.crown_buffers,
                        &remote.held_buffers,
                        remote.held_uses_atlas,
                        remote.held_tex_bg.as_ref(),
                        &mut draw_calls,
                    );
                }
            }

            // Draw active 3D LuaEntity / NativeMob / Projectile batches for this dimension
            let entity_batches = if cam.dimension.is_nether() {
                &self.nether_entity_draw_batches
            } else {
                &self.entity_draw_batches
            };
            for (tex_key, bufs) in entity_batches {
                if bufs.index_count == 0 {
                    continue;
                }
                if let (Some(ref vb), Some(ref ib)) = (&bufs.vb, &bufs.ib) {
                    rpass.set_pipeline(opaque_pipe);
                    rpass.set_bind_group(0, cam_bg, &[]);
                    if let Some((ref bg, _, _)) = self.entity_tex_cache.get(tex_key) {
                        rpass.set_bind_group(1, bg, &[]);
                    } else {
                        rpass.set_bind_group(1, &self.tex_bind_group, &[]);
                    }
                    rpass.set_vertex_buffer(0, vb.slice(..));
                    rpass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    rpass.draw_indexed(0..bufs.index_count, 0, 0..1);
                    draw_calls += 1;
                }
            }
            if !entity_batches.is_empty() {
                rpass.set_bind_group(1, &self.tex_bind_group, &[]);
            }

            if let Some((ref pvb, ref pib, pcnt)) = particle_buffers {
                rpass.set_vertex_buffer(0, pvb.slice(..));
                rpass.set_index_buffer(pib.slice(..), wgpu::IndexFormat::Uint32);
                rpass.draw_indexed(0..*pcnt, 0, 0..1);
                draw_calls += 1;
            }

            let (crack_vb, crack_ib, crack_cnt) = if let Some(peer) = peer_opt {
                (&peer.crack_vb, &peer.crack_ib, peer.crack_index_count)
            } else {
                (&self.crack_vb, &self.crack_ib, self.crack_index_count)
            };
            if let (Some(ref cvb), Some(ref cib)) = (crack_vb, crack_ib) {
                if crack_cnt > 0 {
                    rpass.set_pipeline(crack_pipe);
                    rpass.set_vertex_buffer(0, cvb.slice(..));
                    rpass.set_index_buffer(cib.slice(..), wgpu::IndexFormat::Uint32);
                    rpass.draw_indexed(0..crack_cnt, 0, 0..1);
                    draw_calls += 1;
                }
            }

            // Procedural atmospheric sky dome & FBM clouds (Early-Z rejects occluded terrain pixels)
            self.sky_renderer
                .draw(&mut rpass, &cam_buffers.sky_bind_group, use_hdr_pipelines);
            draw_calls += 1;

            rpass.set_bind_group(0, cam_bg, &[]);
            rpass.set_pipeline(trans_pipe);
            rpass.set_bind_group(1, &self.tex_bind_group, &[]);
            rpass.set_bind_group(2, &self.shadow.sample_bg, &[]);
            for m in &visible {
                if let (Some(ref vb), Some(ref ib)) = (&m.trans_vb, &m.trans_ib) {
                    if m.trans_index_count > 0 {
                        rpass.set_vertex_buffer(0, vb.slice(..));
                        rpass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        rpass.draw_indexed(0..m.trans_index_count, 0, 0..1);
                        draw_calls += 1;
                    }
                }
            }

            // Alpha-blended player skin overlays (1.125x head overlay & 64x64 outer layers)
            if (cam.draw_local_player_mask & (1u8 << 0)) != 0 {
                self.draw_player_overlay_buffers(
                    &mut rpass,
                    trans_pipe,
                    cam_bg,
                    self.skin_bind_group_for(&self.player_model.skin_id),
                    &self.player_overlay_buffers,
                    &mut draw_calls,
                );
            }
            for (peer_idx, peer) in self.peers.iter().enumerate() {
                if (cam.draw_local_player_mask & (1u8 << (peer_idx + 1))) != 0 {
                    self.draw_player_overlay_buffers(
                        &mut rpass,
                        trans_pipe,
                        cam_bg,
                        self.skin_bind_group_for(&peer.player_model.skin_id),
                        &peer.player_overlay_buffers,
                        &mut draw_calls,
                    );
                }
            }
            if !cam.dimension.is_nether() {
                for remote in self.remote_players.values() {
                    self.draw_player_overlay_buffers(
                        &mut rpass,
                        trans_pipe,
                        cam_bg,
                        self.skin_bind_group_for(&remote.model.skin_id),
                        &remote.overlay_buffers,
                        &mut draw_calls,
                    );
                }
            }
            rpass.set_bind_group(1, &self.tex_bind_group, &[]);

            let (sel_vb, sel_cnt) = if let Some(peer) = peer_opt {
                (&peer.selection_vb, peer.selection_vert_count)
            } else {
                (&self.selection_vb, self.selection_vert_count)
            };
            if let Some(ref svb) = sel_vb {
                if sel_cnt > 0 {
                    rpass.set_pipeline(line_pipe);
                    rpass.set_bind_group(0, cam_bg, &[]);
                    rpass.set_vertex_buffer(0, svb.slice(..));
                    rpass.draw(0..sel_cnt, 0..1);
                    draw_calls += 1;
                }
            }
            if !cam.dimension.is_nether() && self.laser_vert_count > 0 {
                if let Some(ref lvb) = self.laser_vb {
                    rpass.set_pipeline(line_pipe);
                    rpass.set_bind_group(0, cam_bg, &[]);
                    rpass.set_vertex_buffer(0, lvb.slice(..));
                    rpass.draw(0..self.laser_vert_count, 0..1);
                    draw_calls += 1;
                }
            }
        }

        let (show_vm, vm_cnt, vm_vb, vm_ib, vm_atlas, vm_tbg) = if let Some(peer) = peer_opt {
            (
                cam.draw_viewmodel && peer.show_viewmodel,
                peer.viewmodel_index_count,
                &peer.viewmodel_vb,
                &peer.viewmodel_ib,
                peer.viewmodel_uses_atlas,
                &peer.viewmodel_tex_bg,
            )
        } else {
            (
                cam.draw_viewmodel && self.show_viewmodel,
                self.viewmodel_index_count,
                &self.viewmodel_vb,
                &self.viewmodel_ib,
                self.viewmodel_uses_atlas,
                &self.viewmodel_tex_bg,
            )
        };
        if show_vm && vm_cnt > 0 {
            if let (Some(ref vb), Some(ref ib)) = (vm_vb, vm_ib) {
                let mut vpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("ViewmodelPass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: target_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                vpass.set_viewport(vx as f32, vy as f32, vw as f32, vh as f32, 0.0, 1.0);
                vpass.set_scissor_rect(vx, vy, vw, vh);

                vpass.set_pipeline(opaque_pipe);
                vpass.set_bind_group(0, vm_bg, &[]);
                if vm_atlas {
                    vpass.set_bind_group(1, &self.tex_bind_group, &[]);
                } else if let Some(ref bg) = vm_tbg {
                    vpass.set_bind_group(1, bg, &[]);
                } else {
                    vpass.set_bind_group(1, &self.tex_bind_group, &[]);
                }
                vpass.set_bind_group(2, &self.shadow.sample_bg, &[]);
                vpass.set_vertex_buffer(0, vb.slice(..));
                vpass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                vpass.draw_indexed(0..vm_cnt, 0, 0..1);
                draw_calls += 1;
            }
        }
        *out_draws += draw_calls;
    }

    fn build_particle_buffers_from(
        &self,
        sys: &ParticleSystem,
        cam_pos: Vec3,
    ) -> Option<(wgpu::Buffer, wgpu::Buffer, u32)> {
        if sys.particles.is_empty() {
            return None;
        }
        let (pverts, pindices) = sys.build_vertices(cam_pos);
        if pindices.is_empty() {
            return None;
        }
        let vb = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ParticleVB"),
            contents: bytemuck::cast_slice(&pverts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ib = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ParticleIB"),
            contents: bytemuck::cast_slice(&pindices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Some((vb, ib, pindices.len() as u32))
    }

    fn encode_egui_pass(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        egui_renderer: &mut egui_wgpu::Renderer,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        egui_ctx: Option<&egui::Context>,
        egui_output: Option<egui::FullOutput>,
        pixels_per_point: f32,
    ) {
        if let (Some(ctx), Some(output)) = (egui_ctx, egui_output) {
            let paint_jobs = ctx.tessellate(output.shapes, pixels_per_point);
            let screen_desc = egui_wgpu::ScreenDescriptor {
                size_in_pixels: [width.max(1), height.max(1)],
                pixels_per_point,
            };

            for (id, delta) in &output.textures_delta.set {
                egui_renderer.update_texture(device, queue, *id, delta);
            }
            egui_renderer.update_buffers(device, queue, encoder, &paint_jobs, &screen_desc);

            {
                let rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("EguiOverlayPass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                let mut rpass_static = rpass.forget_lifetime();
                egui_renderer.render(&mut rpass_static, &paint_jobs, &screen_desc);
            }

            for id in &output.textures_delta.free {
                egui_renderer.free_texture(id);
            }
        }
    }

    pub fn render_frame(
        &mut self,
        cam_pos: Vec3,
        view_proj: Mat4,
        time_of_day: f32,
        settings: &EngineSettings,
        egui_ctx: Option<&egui::Context>,
        egui_output: Option<egui::FullOutput>,
        pixels_per_point: f32,
    ) -> anyhow::Result<()> {
        self.render_frame_in_dimension(
            cam_pos,
            view_proj,
            DimensionId::Overworld,
            time_of_day,
            settings,
            egui_ctx,
            egui_output,
            pixels_per_point,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_frame_in_dimension(
        &mut self,
        cam_pos: Vec3,
        view_proj: Mat4,
        dimension: DimensionId,
        time_of_day: f32,
        settings: &EngineSettings,
        egui_ctx: Option<&egui::Context>,
        egui_output: Option<egui::FullOutput>,
        pixels_per_point: f32,
    ) -> anyhow::Result<()> {
        let w = self.surface_config.width.max(1);
        let h = self.surface_config.height.max(1);
        let mut mask = if self.show_player_model { 1u8 } else { 0u8 };
        if self.coop_render_peer {
            for (idx, peer) in self.peers.iter().enumerate() {
                if peer.show_player_model {
                    mask |= 1u8 << (idx + 1);
                }
            }
        }
        let cam = ViewportCamera {
            cam_pos,
            view_proj,
            viewport_px: [0, 0, w, h],
            player_slot: 0,
            draw_local_player_mask: mask,
            draw_viewmodel: self.show_viewmodel,
            dimension,
        };
        self.render_viewports_frame(
            &[cam],
            time_of_day,
            settings,
            egui_ctx,
            egui_output,
            pixels_per_point,
        )
    }

    /// Render a local 2-player Split-Screen frame into the primary window or offscreen surface.
    #[allow(clippy::too_many_arguments)]
    pub fn render_split_frame(
        &mut self,
        p1_cam: ViewportCamera,
        p2_cam: ViewportCamera,
        time_of_day: f32,
        settings: &EngineSettings,
        egui_ctx: Option<&egui::Context>,
        egui_output: Option<egui::FullOutput>,
        pixels_per_point: f32,
    ) -> anyhow::Result<()> {
        self.render_viewports_frame(
            &[p1_cam, p2_cam],
            time_of_day,
            settings,
            egui_ctx,
            egui_output,
            pixels_per_point,
        )
    }

    /// Render 1..=4 player viewports into the primary window or offscreen surface.
    #[allow(clippy::too_many_arguments)]
    pub fn render_viewports_frame(
        &mut self,
        viewports: &[ViewportCamera],
        time_of_day: f32,
        settings: &EngineSettings,
        egui_ctx: Option<&egui::Context>,
        egui_output: Option<egui::FullOutput>,
        pixels_per_point: f32,
    ) -> anyhow::Result<()> {
        let w = self.surface_config.width.max(1);
        let h = self.surface_config.height.max(1);
        let use_bloom = settings.bloom_glow && !viewports.is_empty();
        for (idx, cam) in viewports.iter().take(4).enumerate() {
            self.write_viewport_uniforms(idx, cam, time_of_day, settings, use_bloom);
        }
        let ow_particles = viewports
            .iter()
            .find(|c| !c.dimension.is_nether())
            .and_then(|c| self.build_particle_buffers_from(&self.particles, c.cam_pos));
        let nether_particles = viewports
            .iter()
            .find(|c| c.dimension.is_nether())
            .and_then(|c| self.build_particle_buffers_from(&self.nether_particles, c.cam_pos));

        let surface_tex = if let Some(ref surf) = self.surface {
            match surf.get_current_texture() {
                Ok(t) => Some(t),
                Err(wgpu::SurfaceError::Lost) => {
                    surf.configure(&self.device, &self.surface_config);
                    return Ok(());
                }
                Err(wgpu::SurfaceError::Outdated) => return Ok(()),
                Err(e) => return Err(e.into()),
            }
        } else {
            None
        };

        let target_view_owned;
        let target_view = if let Some(ref st) = surface_tex {
            target_view_owned = st.texture.create_view(&wgpu::TextureViewDescriptor::default());
            &target_view_owned
        } else if let Some((_, ref view)) = self.offscreen_color {
            view
        } else {
            return Ok(());
        };

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("CubexMultiViewportEncoder"),
        });
        if let Some(first_ow) = viewports.iter().find(|c| !c.dimension.is_nether()) {
            self.update_and_encode_shadows(&mut encoder, first_ow.cam_pos, time_of_day, settings);
        }

        let clear_dim = viewports
            .first()
            .map(|c| c.dimension)
            .unwrap_or(DimensionId::Overworld);
        let aspect_sky = sky_color_for_dimension(time_of_day, clear_dim);
        let fog_dist = self.effective_fog_dist(settings);
        let mut vis = 0u32;
        let mut draws = 0u32;

        if viewports.is_empty() {
            self.bloom_targets = None;
            let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("CubexClearPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: aspect_sky[0] as f64,
                            g: aspect_sky[1] as f64,
                            b: aspect_sky[2] as f64,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        } else if use_bloom {
            self.bloom
                .ensure_targets(&self.device, &mut self.bloom_targets, w, h);
            let scene_view = &self.bloom_targets.as_ref().unwrap().scene_view;
            for (idx, cam) in viewports.iter().take(4).enumerate() {
                let vp_sky = sky_color_for_dimension(time_of_day, cam.dimension);
                let vp_parts = if cam.dimension.is_nether() {
                    nether_particles.as_ref()
                } else {
                    ow_particles.as_ref()
                };
                self.encode_viewport_3d(
                    &mut encoder,
                    scene_view,
                    &self.depth_view,
                    w,
                    h,
                    idx,
                    cam,
                    idx == 0,
                    vp_sky,
                    fog_dist,
                    vp_parts,
                    true,
                    &mut vis,
                    &mut draws,
                );
            }
            if let Some(ref targets) = self.bloom_targets {
                self.bloom
                    .encode_bloom_and_composite(&mut encoder, targets, target_view);
            }
        } else {
            self.bloom_targets = None;
            for (idx, cam) in viewports.iter().take(4).enumerate() {
                let vp_sky = sky_color_for_dimension(time_of_day, cam.dimension);
                let vp_parts = if cam.dimension.is_nether() {
                    nether_particles.as_ref()
                } else {
                    ow_particles.as_ref()
                };
                self.encode_viewport_3d(
                    &mut encoder,
                    target_view,
                    &self.depth_view,
                    w,
                    h,
                    idx,
                    cam,
                    idx == 0,
                    vp_sky,
                    fog_dist,
                    vp_parts,
                    false,
                    &mut vis,
                    &mut draws,
                );
            }
        }
        self.last_visible_chunks = vis;
        self.last_draw_calls = draws;

        Self::encode_egui_pass(
            &self.device,
            &self.queue,
            &mut self.egui_renderer,
            &mut encoder,
            target_view,
            w,
            h,
            egui_ctx,
            egui_output,
            pixels_per_point,
        );

        self.queue.submit(std::iter::once(encoder.finish()));
        if let Some(st) = surface_tex {
            st.present();
        }
        Ok(())
    }

    /// Render a peer viewport on a secondary monitor window sharing GPU chunks & atlas.
    #[allow(clippy::too_many_arguments)]
    pub fn render_secondary_frame(
        &mut self,
        target: &mut SecondaryRenderTarget,
        p2_cam: ViewportCamera,
        time_of_day: f32,
        settings: &EngineSettings,
        egui_ctx: Option<&egui::Context>,
        egui_output: Option<egui::FullOutput>,
        pixels_per_point: f32,
    ) -> anyhow::Result<()> {
        self.render_secondary_viewports_frame(
            target,
            &[p2_cam],
            time_of_day,
            settings,
            egui_ctx,
            egui_output,
            pixels_per_point,
        )
    }

    /// Render 1..=2 viewports onto a secondary monitor window sharing GPU chunks, mobs, and atlas.
    #[allow(clippy::too_many_arguments)]
    pub fn render_secondary_viewports_frame(
        &mut self,
        target: &mut SecondaryRenderTarget,
        viewports: &[ViewportCamera],
        time_of_day: f32,
        settings: &EngineSettings,
        egui_ctx: Option<&egui::Context>,
        egui_output: Option<egui::FullOutput>,
        pixels_per_point: f32,
    ) -> anyhow::Result<()> {
        let w = target.surface_config.width.max(1);
        let h = target.surface_config.height.max(1);
        let use_bloom = settings.bloom_glow && !viewports.is_empty();
        for (idx, cam) in viewports.iter().take(4).enumerate() {
            self.write_viewport_uniforms(idx, cam, time_of_day, settings, use_bloom);
        }
        let ow_particles = viewports
            .iter()
            .find(|c| !c.dimension.is_nether())
            .and_then(|c| self.build_particle_buffers_from(&self.particles, c.cam_pos));
        let nether_particles = viewports
            .iter()
            .find(|c| c.dimension.is_nether())
            .and_then(|c| self.build_particle_buffers_from(&self.nether_particles, c.cam_pos));

        let surface_tex = match target.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Lost) => {
                target.surface.configure(&self.device, &target.surface_config);
                return Ok(());
            }
            Err(wgpu::SurfaceError::Outdated) => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        let target_view = surface_tex
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("CubexSecondaryMonitorEncoder"),
        });
        if let Some(first_ow) = viewports.iter().find(|c| !c.dimension.is_nether()) {
            self.update_and_encode_shadows(&mut encoder, first_ow.cam_pos, time_of_day, settings);
        }

        let fog_dist = self.effective_fog_dist(settings);
        let mut vis = 0u32;
        let mut draws = 0u32;
        if use_bloom {
            self.bloom
                .ensure_targets(&self.device, &mut target.bloom_targets, w, h);
            let scene_view = &target.bloom_targets.as_ref().unwrap().scene_view;
            for (idx, cam) in viewports.iter().take(4).enumerate() {
                let vp_sky = sky_color_for_dimension(time_of_day, cam.dimension);
                let vp_parts = if cam.dimension.is_nether() {
                    nether_particles.as_ref()
                } else {
                    ow_particles.as_ref()
                };
                self.encode_viewport_3d(
                    &mut encoder,
                    scene_view,
                    &target.depth_view,
                    w,
                    h,
                    idx,
                    cam,
                    idx == 0,
                    vp_sky,
                    fog_dist,
                    vp_parts,
                    true,
                    &mut vis,
                    &mut draws,
                );
            }
            if let Some(ref btargets) = target.bloom_targets {
                self.bloom
                    .encode_bloom_and_composite(&mut encoder, btargets, &target_view);
            }
        } else {
            target.bloom_targets = None;
            for (idx, cam) in viewports.iter().take(4).enumerate() {
                let vp_sky = sky_color_for_dimension(time_of_day, cam.dimension);
                let vp_parts = if cam.dimension.is_nether() {
                    nether_particles.as_ref()
                } else {
                    ow_particles.as_ref()
                };
                self.encode_viewport_3d(
                    &mut encoder,
                    &target_view,
                    &target.depth_view,
                    w,
                    h,
                    idx,
                    cam,
                    idx == 0,
                    vp_sky,
                    fog_dist,
                    vp_parts,
                    false,
                    &mut vis,
                    &mut draws,
                );
            }
        }

        Self::encode_egui_pass(
            &self.device,
            &self.queue,
            &mut target.egui_renderer,
            &mut encoder,
            &target_view,
            w,
            h,
            egui_ctx,
            egui_output,
            pixels_per_point,
        );

        self.queue.submit(std::iter::once(encoder.finish()));
        surface_tex.present();
        Ok(())
    }
}

pub fn sky_color_for_tod(tod: f32) -> [f32; 3] {
    compute_sky_palette(tod).horizon
}

pub fn sky_color_for_dimension(tod: f32, dim: DimensionId) -> [f32; 3] {
    if dim.is_nether() {
        compute_nether_sky_palette().horizon
    } else {
        compute_sky_palette(tod).horizon
    }
}

fn apply_player_hurt_flash(verts: &mut [mesher::VoxelVertex], hurt_flash: f32) {
    if hurt_flash > 0.01 {
        let tint = hurt_tint_for_flash(hurt_flash);
        for v in verts {
            v.tint = tint;
            v.packed |= 1u32 << 26;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{DrawType, NodeDef};
    use crate::world::Chunk;
    use std::path::Path;

    #[test]
    fn headless_wgpu_pipeline_priority_remesh_and_viewmodel_pass() {
        let mut reg = EngineRegistries::new();
        let stone = reg.register_node(NodeDef {
            name: "mcl_core:stone".into(),
            drawtype: DrawType::Normal,
            ..Default::default()
        });
        let atlas = TextureAtlas::build(&HashMap::new(), Path::new(""), None, &reg);
        let mut renderer = WgpuRenderer::new_headless(160, 90, &atlas).expect("headless wgpu");

        let tmp = std::env::temp_dir().join("cubex_headless_render_test");
        let world = WorldStore::new(&tmp, 123);
        let mut ch = Chunk::new(IVec3::ZERO);
        ch.set(8, 2, 8, stone, 15, 0);
        world.insert_chunk(ch);
        world.mark_priority_dirty_with_neighbors(IVec3::new(8, 2, 8));

        let remeshed = renderer.remesh_priority_chunks_now(&world, &reg, &atlas, true);
        assert!(remeshed >= 1);
        assert!(renderer.chunk_meshes.contains_key(&IVec3::ZERO));

        renderer.show_viewmodel = true;
        renderer.update_viewmodel(None, 0, 0.016, 2.0, true, false, false, &reg, &atlas);
        renderer.set_selection_boxes(&[([8.0, 2.0, 8.0], [9.0, 3.0, 9.0])]);
        renderer.set_crack_overlay(&[([8.0, 2.0, 8.0], [9.0, 3.0, 9.0])], Some(4), &atlas);
        renderer.particles.spawn_break_burst(IVec3::new(8, 2, 8), stone, &atlas, &reg);

        let mut settings = EngineSettings::default();
        settings.handheld_torch_light = true;
        let eye = Vec3::new(8.5, 4.0, 12.0);
        let vp = Mat4::perspective_rh(75.0f32.to_radians(), 160.0 / 90.0, 0.1, 100.0)
            * Mat4::look_at_rh(eye, Vec3::new(8.5, 2.5, 8.5), Vec3::Y);
        let img = renderer
            .render_to_headless_target(eye, vp, 0.35, &settings)
            .expect("headless render readback");
        assert_eq!(img.dimensions(), (160, 90));
        // Confirm non-black sky/scene rendered
        assert!(img.pixels().any(|p| p.0[2] > 50));

        // Verify 3rd-person camera rig + articulated character + held item pass
        let mut p = crate::physics::PlayerController::default();
        p.pos = Vec3::new(8.5, 3.0, 8.5);
        p.yaw = 0.0;
        p.pitch = -0.15;
        let rig = compute_camera_rig(
            CameraMode::ThirdPersonBack,
            &p,
            160.0 / 90.0,
            &settings,
            &world,
            &reg,
        );
        renderer.show_viewmodel = rig.show_viewmodel;
        renderer.show_player_model = false;
        let img_no_3p = renderer
            .render_to_headless_target(rig.eye_pos, rig.view_proj, 0.35, &settings)
            .expect("headless render without 3p");

        renderer.show_player_model = rig.show_player_model;
        renderer.update_player_model(
            p.pos,
            p.yaw,
            p.pitch,
            4.2,
            false,
            Some("mcl_core:stone"),
            0.05,
            &reg,
            &atlas,
        );
        let img_3p = renderer
            .render_to_headless_target(rig.eye_pos, rig.view_proj, 0.35, &settings)
            .expect("headless 3p render");
        assert_eq!(img_3p.dimensions(), (160, 90));
        let diff_pixels = img_no_3p
            .pixels()
            .zip(img_3p.pixels())
            .filter(|(a, b)| a.0 != b.0)
            .count();
        assert!(
            diff_pixels > 50,
            "Expected visible 3P player model pixels in VoxelWorldPass, got {diff_pixels} changed pixels"
        );

        // Verify selection box cache + crack cache + distant chunk pruning
        renderer.set_selection_boxes(&[([8.0, 2.0, 8.0], [9.0, 3.0, 9.0])]);
        assert!(renderer.selection_vb.is_some());
        assert_eq!(renderer.last_selection_boxes.len(), 1);
        renderer.set_selection_boxes(&[]);
        assert!(renderer.selection_vb.is_none());
        assert!(renderer.last_selection_boxes.is_empty());

        let mut extra = Chunk::new(IVec3::new(100, 0, 100));
        extra.set(0, 0, 0, stone, 15, 0);
        world.insert_chunk(extra);
        world.mark_priority_dirty_with_neighbors(IVec3::new(100 * 16, 0, 100 * 16));
        renderer.remesh_priority_chunks_now(&world, &reg, &atlas, false);
        assert!(renderer.chunk_meshes.contains_key(&IVec3::new(100, 0, 100)));
        let pruned = renderer.prune_distant_chunks(0, 0, 4);
        assert_eq!(pruned, 1);
        assert!(!renderer.chunk_meshes.contains_key(&IVec3::new(100, 0, 100)));
        assert!(renderer.chunk_meshes.contains_key(&IVec3::ZERO));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn headless_split_screen_renders_both_viewports_and_peer_characters() {
        let mut reg = EngineRegistries::new();
        let stone = reg.register_node(NodeDef {
            name: "mcl_core:stone".into(),
            drawtype: DrawType::Normal,
            ..Default::default()
        });
        let atlas = TextureAtlas::build(&HashMap::new(), Path::new(""), None, &reg);
        let mut renderer = WgpuRenderer::new_headless(320, 180, &atlas).expect("headless wgpu");

        let tmp = std::env::temp_dir().join("cubex_headless_split_test");
        let _ = std::fs::remove_dir_all(&tmp);
        let world = WorldStore::new(&tmp, 99);
        let mut ch = Chunk::new(IVec3::ZERO);
        for z in 4..=12 {
            for x in 4..=12 {
                ch.set(x, 1, z, stone, 15, 0);
            }
        }
        world.insert_chunk(ch);
        world.mark_priority_dirty_with_neighbors(IVec3::new(8, 1, 8));
        renderer.remesh_priority_chunks_now(&world, &reg, &atlas, false);

        renderer.update_player_model(
            Vec3::new(6.0, 2.0, 8.0),
            0.0,
            0.0,
            0.0,
            false,
            None,
            0.016,
            &reg,
            &atlas,
        );
        renderer.update_p2_player_model(
            Vec3::new(8.5, 2.0, 8.5),
            std::f32::consts::PI,
            0.0,
            1.5,
            false,
            Some("mcl_core:stone"),
            0.016,
            &reg,
            &atlas,
        );
        renderer.peers[0].show_viewmodel = true;
        renderer.update_p2_viewmodel(
            Some("mcl_core:stone"),
            0,
            0.016,
            0.0,
            true,
            false,
            false,
            &reg,
            &atlas,
        );

        let p1_eye = Vec3::new(8.5, 3.6, 5.0);
        let p1_vp = Mat4::perspective_rh(75.0f32.to_radians(), 160.0 / 180.0, 0.1, 100.0)
            * Mat4::look_at_rh(p1_eye, Vec3::new(8.5, 2.9, 8.5), Vec3::Y);
        let p2_eye = Vec3::new(11.0, 4.5, 11.0);
        let p2_vp = Mat4::perspective_rh(75.0f32.to_radians(), 160.0 / 180.0, 0.1, 100.0)
            * Mat4::look_at_rh(p2_eye, Vec3::new(6.0, 2.5, 8.0), Vec3::Y);

        let img_no_p2 = renderer
            .render_split_to_headless_target(
                ViewportCamera {
                    cam_pos: p1_eye,
                    view_proj: p1_vp,
                    viewport_px: [0, 0, 160, 180],
                    player_slot: 0,
                    draw_local_player_mask: 0,
                    draw_viewmodel: false,
                    dimension: DimensionId::Overworld,
                },
                ViewportCamera {
                    cam_pos: p2_eye,
                    view_proj: p2_vp,
                    viewport_px: [160, 0, 160, 180],
                    player_slot: 1,
                    draw_local_player_mask: 1u8 << 0,
                    draw_viewmodel: true,
                    dimension: DimensionId::Overworld,
                },
                0.35,
                &EngineSettings::default(),
                None,
                None,
                1.0,
            )
            .expect("split render without p2 body");

        let img_with_p2 = renderer
            .render_split_to_headless_target(
                ViewportCamera {
                    cam_pos: p1_eye,
                    view_proj: p1_vp,
                    viewport_px: [0, 0, 160, 180],
                    player_slot: 0,
                    draw_local_player_mask: 1u8 << 1,
                    draw_viewmodel: false,
                    dimension: DimensionId::Overworld,
                },
                ViewportCamera {
                    cam_pos: p2_eye,
                    view_proj: p2_vp,
                    viewport_px: [160, 0, 160, 180],
                    player_slot: 1,
                    draw_local_player_mask: 1u8 << 0,
                    draw_viewmodel: true,
                    dimension: DimensionId::Overworld,
                },
                0.35,
                &EngineSettings::default(),
                None,
                None,
                1.0,
            )
            .expect("split render with p2 body");

        assert_eq!(img_with_p2.dimensions(), (320, 180));
        // Left viewport (P1 looking at P2) must visibly contain P2's rendered character
        let mut left_diff = 0usize;
        for y in 0..180 {
            for x in 0..160 {
                if img_no_p2.get_pixel(x, y).0 != img_with_p2.get_pixel(x, y).0 {
                    left_diff += 1;
                }
            }
        }
        assert!(
            left_diff > 40,
            "Expected P2 3D character rendered in P1 split viewport, got {left_diff} changed pixels"
        );
        // Right viewport (P2) must render non-uniform world + viewmodel
        let p0 = img_with_p2.get_pixel(160, 0).0;
        let has_right_content = (160..320).any(|x| (0..180).any(|y| img_with_p2.get_pixel(x, y).0 != p0));
        assert!(has_right_content);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn headless_entity_rig_renders_in_singleplayer_and_split_screen() {
        let mut reg = EngineRegistries::new();
        let stone = reg.register_node(NodeDef {
            name: "mcl_core:stone".into(),
            drawtype: DrawType::Normal,
            ..Default::default()
        });
        let atlas = TextureAtlas::build(&HashMap::new(), Path::new(""), None, &reg);
        let mut renderer = WgpuRenderer::new_headless(320, 180, &atlas).expect("headless wgpu");

        let tmp = std::env::temp_dir().join("cubex_headless_entity_test");
        let _ = std::fs::remove_dir_all(&tmp);
        let world = WorldStore::new(&tmp, 123);
        let mut ch = Chunk::new(IVec3::ZERO);
        for z in 2..=14 {
            for x in 2..=14 {
                ch.set(x, 1, z, stone, 15, 0);
            }
        }
        world.insert_chunk(ch);
        world.mark_priority_dirty_with_neighbors(IVec3::new(8, 1, 8));
        renderer.remesh_priority_chunks_now(&world, &reg, &atlas, false);

        let mut zombie = EntityRenderSnapshot {
            id: 42,
            entity_name: "mobs_mc:zombie".into(),
            pos: Vec3::new(8.5, 2.0, 8.5),
            yaw: 0.35,
            collisionbox: [-0.3, 0.0, -0.3, 0.3, 1.9, 0.3],
            visual: "mesh".into(),
            visual_size: Vec3::ONE,
            mesh: "mobs_mc_zombie.b3d".into(),
            texture_name: "mobs_mc_zombie.png".into(),
            wield_item: String::new(),
            walk_phase: 1.0,
            walk_blend: 0.85,
            anim_time: 1.0,
            vel_y: 0.0,
            hurt_flash: 0.0,
            sky_light: 15,
            block_light: 0,
        };

        let sp_eye = Vec3::new(8.5, 3.8, 4.8);
        let sp_vp = Mat4::perspective_rh(72.0f32.to_radians(), 320.0 / 180.0, 0.1, 100.0)
            * Mat4::look_at_rh(sp_eye, Vec3::new(8.5, 2.9, 8.5), Vec3::Y);
        renderer.update_entities(&[], &reg, &atlas);
        let img_empty = renderer
            .render_to_headless_target(sp_eye, sp_vp, 0.35, &EngineSettings::default())
            .expect("singleplayer empty");

        renderer.update_entities(&[zombie.clone()], &reg, &atlas);
        let img_zombie = renderer
            .render_to_headless_target(sp_eye, sp_vp, 0.35, &EngineSettings::default())
            .expect("singleplayer zombie");
        let sp_diff: usize = img_empty
            .pixels()
            .zip(img_zombie.pixels())
            .filter(|(a, b)| a.0 != b.0)
            .count();
        assert!(
            sp_diff > 40,
            "Expected zombie entity rendered in singleplayer, got {sp_diff} changed pixels"
        );

        zombie.hurt_flash = 0.25;
        renderer.update_entities(&[zombie.clone()], &reg, &atlas);
        let img_hurt = renderer
            .render_to_headless_target(sp_eye, sp_vp, 0.35, &EngineSettings::default())
            .expect("singleplayer hurt zombie");
        let hurt_red_pixels: usize = img_zombie
            .pixels()
            .zip(img_hurt.pixels())
            .filter(|(a, b)| a.0 != b.0 && b.0[0] > b.0[1].saturating_add(40))
            .count();
        assert!(
            hurt_red_pixels > 20,
            "Expected red hurt_flash pixels on damaged entity, got {hurt_red_pixels}"
        );

        // Split-Screen P1 & P2 both see the entity in their own viewports
        let p1_eye = Vec3::new(8.5, 3.6, 5.0);
        let p1_vp = Mat4::perspective_rh(72.0f32.to_radians(), 160.0 / 180.0, 0.1, 100.0)
            * Mat4::look_at_rh(p1_eye, Vec3::new(8.5, 2.9, 8.5), Vec3::Y);
        let p2_eye = Vec3::new(12.0, 3.6, 8.5);
        let p2_vp = Mat4::perspective_rh(72.0f32.to_radians(), 160.0 / 180.0, 0.1, 100.0)
            * Mat4::look_at_rh(p2_eye, Vec3::new(8.5, 2.9, 8.5), Vec3::Y);

        let cam1 = ViewportCamera {
            cam_pos: p1_eye,
            view_proj: p1_vp,
            viewport_px: [0, 0, 160, 180],
            player_slot: 0,
            draw_local_player_mask: 0,
            draw_viewmodel: false,
            dimension: DimensionId::Overworld,
        };
        let cam2 = ViewportCamera {
            cam_pos: p2_eye,
            view_proj: p2_vp,
            viewport_px: [160, 0, 160, 180],
            player_slot: 1,
            draw_local_player_mask: 0,
            draw_viewmodel: false,
            dimension: DimensionId::Overworld,
        };

        renderer.update_entities(&[], &reg, &atlas);
        let split_empty = renderer
            .render_split_to_headless_target(
                cam1.clone(),
                cam2.clone(),
                0.35,
                &EngineSettings::default(),
                None,
                None,
                1.0,
            )
            .expect("split empty");

        renderer.update_entities(&[zombie], &reg, &atlas);
        let split_ent = renderer
            .render_split_to_headless_target(
                cam1,
                cam2,
                0.35,
                &EngineSettings::default(),
                None,
                None,
                1.0,
            )
            .expect("split entity");

        let left_diff = (0..180)
            .flat_map(|y| (0..160).map(move |x| (x, y)))
            .filter(|&(x, y)| split_empty.get_pixel(x, y).0 != split_ent.get_pixel(x, y).0)
            .count();
        let right_diff = (0..180)
            .flat_map(|y| (160..320).map(move |x| (x, y)))
            .filter(|&(x, y)| split_empty.get_pixel(x, y).0 != split_ent.get_pixel(x, y).0)
            .count();
        assert!(left_diff > 40, "Expected entity in P1 viewport, got {left_diff}");
        assert!(right_diff > 40, "Expected entity in P2 viewport, got {right_diff}");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn headless_four_viewports_and_remote_lan_player_render() {
        let mut reg = EngineRegistries::new();
        let stone = reg.register_node(NodeDef {
            name: "mcl_core:stone".into(),
            drawtype: DrawType::Normal,
            ..Default::default()
        });
        let atlas = TextureAtlas::build(&HashMap::new(), Path::new(""), None, &reg);
        let mut renderer = WgpuRenderer::new_headless(320, 180, &atlas).expect("headless wgpu");

        let tmp = std::env::temp_dir().join("cubex_headless_4p_lan_test");
        let _ = std::fs::remove_dir_all(&tmp);
        let world = WorldStore::new(&tmp, 777);
        let mut ch = Chunk::new(IVec3::ZERO);
        for z in 2..=14 {
            for x in 2..=14 {
                ch.set(x, 1, z, stone, 15, 0);
            }
        }
        world.insert_chunk(ch);
        world.mark_priority_dirty_with_neighbors(IVec3::new(8, 1, 8));
        renderer.remesh_priority_chunks_now(&world, &reg, &atlas, false);

        renderer.update_player_model(
            Vec3::new(6.0, 2.0, 6.0),
            0.0,
            0.0,
            0.0,
            false,
            None,
            0.016,
            &reg,
            &atlas,
        );
        for (slot, pos) in [
            (1usize, Vec3::new(10.0, 2.0, 6.0)),
            (2usize, Vec3::new(6.0, 2.0, 10.0)),
            (3usize, Vec3::new(10.0, 2.0, 10.0)),
        ] {
            renderer.update_peer_player_model(
                slot,
                pos,
                std::f32::consts::PI,
                0.0,
                1.0,
                false,
                Some("mcl_core:stone"),
                0.016,
                &reg,
                &atlas,
            );
        }

        let make_cam = |eye: Vec3, vp: [u32; 4], slot: usize| -> ViewportCamera {
            let proj = Mat4::perspective_rh(
                72.0f32.to_radians(),
                vp[2] as f32 / vp[3].max(1) as f32,
                0.1,
                100.0,
            ) * Mat4::look_at_rh(eye, Vec3::new(8.5, 2.8, 8.5), Vec3::Y);
            ViewportCamera {
                cam_pos: eye,
                view_proj: proj,
                viewport_px: vp,
                player_slot: slot,
                draw_local_player_mask: 0b1111 & !(1u8 << slot),
                draw_viewmodel: false,
                dimension: DimensionId::Overworld,
            }
        };
        let cams = [
            make_cam(Vec3::new(8.5, 3.8, 4.5), [0, 0, 160, 90], 0),
            make_cam(Vec3::new(12.5, 3.8, 8.5), [160, 0, 160, 90], 1),
            make_cam(Vec3::new(8.5, 3.8, 12.5), [0, 90, 160, 90], 2),
            make_cam(Vec3::new(4.5, 3.8, 8.5), [160, 90, 160, 90], 3),
        ];

        renderer.update_entities(&[], &reg, &atlas);
        renderer.update_remote_players(&[], 0.016, &reg, &atlas);
        let empty_img = renderer
            .render_viewports_to_headless_target(
                &cams,
                0.35,
                &EngineSettings::default(),
                None,
                None,
                1.0,
            )
            .expect("quad empty");

        let cow = EntityRenderSnapshot {
            id: 1,
            entity_name: "mobs_mc:cow".into(),
            pos: Vec3::new(7.4, 2.0, 8.5),
            yaw: 0.4,
            collisionbox: [-0.45, 0.0, -0.45, 0.45, 1.3, 0.45],
            visual: "mesh".into(),
            visual_size: Vec3::ONE,
            mesh: "mobs_mc_cow.b3d".into(),
            texture_name: String::new(),
            wield_item: String::new(),
            walk_phase: 0.5,
            walk_blend: 0.8,
            anim_time: 0.5,
            vel_y: 0.0,
            hurt_flash: 0.0,
            sky_light: 15,
            block_light: 0,
        };
        let zombie = EntityRenderSnapshot {
            id: 2,
            entity_name: "mobs_mc:zombie".into(),
            pos: Vec3::new(9.6, 2.0, 8.5),
            yaw: -0.4,
            collisionbox: [-0.3, 0.0, -0.3, 0.3, 1.9, 0.3],
            visual: "mesh".into(),
            visual_size: Vec3::ONE,
            mesh: "mobs_mc_zombie.b3d".into(),
            texture_name: "mobs_mc_zombie.png".into(),
            wield_item: String::new(),
            walk_phase: 1.0,
            walk_blend: 0.9,
            anim_time: 1.0,
            vel_y: 0.0,
            hurt_flash: 0.0,
            sky_light: 15,
            block_light: 0,
        };
        renderer.update_entities(&[cow, zombie], &reg, &atlas);
        renderer.update_remote_players(
            &[RemotePlayerRenderSnapshot {
                name: "LanFriend".into(),
                pos: Vec3::new(8.5, 2.0, 8.5),
                yaw: 0.0,
                pitch: 0.0,
                horizontal_speed: 2.0,
                is_sneaking: false,
                is_swinging: false,
                swing_progress: 0.0,
                held_item: Some("mcl_core:stone".into()),
                hurt_flash: 0.0,
            }],
            0.016,
            &reg,
            &atlas,
        );

        let full_img = renderer
            .render_viewports_to_headless_target(
                &cams,
                0.35,
                &EngineSettings::default(),
                None,
                None,
                1.0,
            )
            .expect("quad with remote + mobs");
        assert_eq!(full_img.dimensions(), (320, 180));

        for (qname, xr, yr) in [
            ("P1-TL", 0..160u32, 0..90u32),
            ("P2-TR", 160..320u32, 0..90u32),
            ("P3-BL", 0..160u32, 90..180u32),
            ("P4-BR", 160..320u32, 90..180u32),
        ] {
            let diff = xr
                .clone()
                .flat_map(|x| yr.clone().map(move |y| (x, y)))
                .filter(|&(x, y)| empty_img.get_pixel(x, y).0 != full_img.get_pixel(x, y).0)
                .count();
            assert!(
                diff > 30,
                "Expected remote LAN player + cow/zombie visible in {qname}, got {diff} changed pixels"
            );
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn headless_stream_rgba_and_jpeg_encode() {
        let tmp = std::env::temp_dir().join("cubex_stream_rgba_test");
        let _ = std::fs::remove_dir_all(&tmp);
        let reg = crate::registry::EngineRegistries::default();
        let atlas = TextureAtlas::build(&HashMap::new(), &tmp, None, &reg);

        let Ok(mut renderer) = WgpuRenderer::new_headless(320, 180, &atlas) else {
            let _ = std::fs::remove_dir_all(&tmp);
            return;
        };

        let mob = EntityRenderSnapshot {
            id: 99,
            entity_name: "mobs_mc:cow".into(),
            pos: Vec3::new(8.0, 2.0, 8.0),
            yaw: 0.2,
            collisionbox: [-0.35, 0.0, -0.35, 0.35, 1.8, 0.35],
            visual: "mesh".into(),
            visual_size: Vec3::ONE,
            mesh: "mobs_mc_cow.b3d".into(),
            texture_name: "mcl_mobs_cow.png".into(),
            wield_item: String::new(),
            walk_phase: 0.0,
            walk_blend: 0.0,
            anim_time: 0.0,
            vel_y: 0.0,
            hurt_flash: 0.0,
            sky_light: 15,
            block_light: 0,
        };
        renderer.update_entities(&[mob], &reg, &atlas);

        let eye = Vec3::new(8.0, 3.5, 12.5);
        let vp = Mat4::perspective_rh(72f32.to_radians(), 320.0 / 180.0, 0.1, 200.0)
            * Mat4::look_at_rh(eye, Vec3::new(8.0, 2.8, 8.0), Vec3::Y);
        let cam = ViewportCamera {
            cam_pos: eye,
            view_proj: vp,
            viewport_px: [0, 0, 320, 180],
            player_slot: 0,
            draw_local_player_mask: 0,
            draw_viewmodel: false,
            dimension: DimensionId::Overworld,
        };

        let img = renderer
            .render_stream_rgba(cam, 320, 180, 0.35, &EngineSettings::default())
            .expect("render_stream_rgba");
        assert_eq!(img.dimensions(), (320, 180));

        // Ensure frame has varied pixels (sky + 3D entity)
        let first = img.get_pixel(0, 0).0;
        let has_diff = img.pixels().any(|p| p.0 != first);
        assert!(has_diff, "Cloud stream frame must contain rendered geometry, not a flat color");

        // Encode as JPEG using image::codecs::jpeg::JpegEncoder (same path used for mobile WebSocket stream)
        let rgb: Vec<u8> = img
            .as_raw()
            .chunks_exact(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect();
        let mut jpeg_bytes = Vec::new();
        let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg_bytes, 72);
        enc.encode(
            &rgb,
            img.width(),
            img.height(),
            image::ExtendedColorType::Rgb8,
        )
        .expect("jpeg encode");
        assert!(jpeg_bytes.len() > 500, "JPEG too small: {}", jpeg_bytes.len());
        assert_eq!(&jpeg_bytes[0..3], &[0xFF, 0xD8, 0xFF], "Must start with JPEG SOI+APP0 marker");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn enhanced_graphics_presets_and_bloom_live_toggle_singleplayer_splitscreen_and_resize() {
        use crate::config::GraphicsPreset;
        use crate::registry::TileSpec;

        let tmp = std::env::temp_dir().join("cubex_enhanced_gfx_test");
        let _ = std::fs::remove_dir_all(&tmp);
        let mut reg = EngineRegistries::new();
        let stone = reg.register_node(NodeDef {
            name: "mcl_core:stone".into(),
            drawtype: DrawType::Normal,
            tiles: vec![TileSpec { name: "default_stone.png".into(), color: None }],
            walkable: true,
            ..Default::default()
        });
        let torch = reg.register_node(NodeDef {
            name: "mcl_torches:torch".into(),
            drawtype: DrawType::Torchlike,
            tiles: vec![TileSpec { name: "default_torch_on_floor.png".into(), color: None }],
            light_source: 14,
            walkable: false,
            ..Default::default()
        });
        let leaves = reg.register_node(NodeDef {
            name: "mcl_trees:leaves_oak".into(),
            drawtype: DrawType::AllfacesOptional,
            tiles: vec![TileSpec { name: "default_leaves.png".into(), color: None }],
            waving: 1,
            walkable: true,
            ..Default::default()
        });
        let atlas = TextureAtlas::build(&HashMap::new(), &tmp, None, &reg);

        let Ok(mut renderer) = WgpuRenderer::new_headless(240, 135, &atlas) else {
            let _ = std::fs::remove_dir_all(&tmp);
            return;
        };

        let world = WorldStore::new(&tmp, 77);
        let mut ch = Chunk::new(IVec3::ZERO);
        for x in 4..12 {
            for z in 4..12 {
                ch.set(x, 1, z, stone, 15, 0);
            }
        }
        ch.set(8, 2, 8, torch, 15, 14);
        ch.set(7, 3, 7, leaves, 15, 0);
        world.insert_chunk(ch);
        world.mark_priority_dirty_with_neighbors(IVec3::new(8, 2, 8));
        renderer.remesh_priority_chunks_now(&world, &reg, &atlas, true);

        renderer.show_viewmodel = true;
        renderer.update_viewmodel(
            Some("mcl_torches:torch"),
            0,
            0.016,
            2.0,
            true,
            false,
            false,
            &reg,
            &atlas,
        );

        let sp_eye = Vec3::new(8.5, 4.0, 12.0);
        let sp_vp = Mat4::perspective_rh(72.0f32.to_radians(), 240.0 / 135.0, 0.1, 100.0)
            * Mat4::look_at_rh(sp_eye, Vec3::new(8.5, 2.2, 8.0), Vec3::Y);

        // 1. Performance preset: bloom targets must NOT be allocated
        let mut settings = EngineSettings::default();
        settings.apply_graphics_preset(GraphicsPreset::Performance);
        renderer.set_extruded_held_items(settings.extruded_held_items);
        let perf_img = renderer
            .render_to_headless_target(sp_eye, sp_vp, 0.32, &settings)
            .expect("performance singleplayer render");
        assert!(
            !renderer.bloom_targets_allocated(),
            "Performance preset must bypass bloom allocation"
        );

        // 2. Balanced preset (Shadow Map + Dynamic Torch + Warm Sun + ACES/Bloom)
        settings.apply_graphics_preset(GraphicsPreset::Balanced);
        renderer.set_extruded_held_items(settings.extruded_held_items);
        let bal_img = renderer
            .render_to_headless_target(sp_eye, sp_vp, 0.32, &settings)
            .expect("balanced singleplayer render");
        assert!(renderer.bloom_targets_allocated());

        // 3. Cinematic preset (Bloom + HDR + ACES)
        settings.apply_graphics_preset(GraphicsPreset::Cinematic);
        let cin_img = renderer
            .render_to_headless_target(sp_eye, sp_vp, 0.32, &settings)
            .expect("cinematic singleplayer render");
        assert!(
            renderer.bloom_targets_allocated(),
            "Cinematic preset must allocate Bloom targets"
        );

        let diff_perf_cin: usize = perf_img
            .pixels()
            .zip(cin_img.pixels())
            .filter(|(a, b)| a.0 != b.0)
            .count();
        assert!(
            diff_perf_cin > 200,
            "Performance vs Cinematic should differ visibly, got {diff_perf_cin} pixels"
        );
        let _ = bal_img;

        // 4. Split-screen viewports + resize transitions with Bloom ON and OFF -> zero leak/panic
        let cam1 = ViewportCamera {
            cam_pos: sp_eye,
            view_proj: sp_vp,
            viewport_px: [0, 0, 160, 180],
            player_slot: 0,
            draw_local_player_mask: 0,
            draw_viewmodel: true,
            dimension: DimensionId::Overworld,
        };
        let cam2 = ViewportCamera {
            cam_pos: Vec3::new(11.5, 4.0, 8.0),
            view_proj: Mat4::perspective_rh(72.0f32.to_radians(), 160.0 / 180.0, 0.1, 100.0)
                * Mat4::look_at_rh(Vec3::new(11.5, 4.0, 8.0), Vec3::new(8.5, 2.2, 8.0), Vec3::Y),
            viewport_px: [160, 0, 160, 180],
            player_slot: 1,
            draw_local_player_mask: 0,
            draw_viewmodel: false,
            dimension: DimensionId::Overworld,
        };
        renderer.resize(320, 180, false);
        assert!(!renderer.bloom_targets_allocated());
        let split_cin = renderer
            .render_split_to_headless_target(cam1, cam2, 0.32, &settings, None, None, 1.0)
            .expect("split cinematic");
        assert_eq!(split_cin.dimensions(), (320, 180));
        assert!(renderer.bloom_targets_allocated());

        // Switch live to Custom (Bloom OFF, Warm Sun ON, extruded OFF) -> bloom targets immediately released
        settings.bloom_glow = false;
        settings.extruded_held_items = false;
        settings.sync_graphics_preset_from_toggles();
        renderer.set_extruded_held_items(settings.extruded_held_items);
        renderer.update_viewmodel(
            Some("mcl_torches:torch"),
            0,
            0.016,
            2.0,
            true,
            false,
            false,
            &reg,
            &atlas,
        );
        let split_custom = renderer
            .render_split_to_headless_target(cam1, cam2, 0.32, &settings, None, None, 1.0)
            .expect("split custom");
        assert_eq!(split_custom.dimensions(), (320, 180));
        assert!(
            !renderer.bloom_targets_allocated(),
            "Turning Bloom OFF live must free bloom targets"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn procedural_sky_dome_clouds_water_reflections_singleplayer_and_splitscreen() {
        use crate::config::GraphicsPreset;
        use crate::registry::TileSpec;
        use image::{Rgba, RgbaImage};

        let mut reg = EngineRegistries::new();
        let snow = reg.register_node(NodeDef {
            name: "mcl_core:snowblock".into(),
            drawtype: DrawType::Normal,
            tiles: vec![TileSpec {
                name: "snow.png".into(),
                color: None,
            }],
            ..Default::default()
        });
        let water = reg.register_node(NodeDef {
            name: "mcl_core:water_source".into(),
            drawtype: DrawType::Liquid,
            waving: 3,
            tiles: vec![TileSpec {
                name: "water.png".into(),
                color: None,
            }],
            ..Default::default()
        });

        let tmp = std::env::temp_dir().join("cubex_sky_water_snow_render_test");
        let _ = std::fs::remove_dir_all(&tmp);
        let tex_dir = tmp.join("textures");
        let _ = std::fs::create_dir_all(&tex_dir);
        RgbaImage::from_pixel(16, 16, Rgba([244, 248, 255, 255]))
            .save(tex_dir.join("snow.png"))
            .unwrap();
        RgbaImage::from_pixel(16, 16, Rgba([38, 108, 214, 192]))
            .save(tex_dir.join("water.png"))
            .unwrap();

        let atlas = TextureAtlas::build(&HashMap::new(), &tmp, None, &reg);
        let world = WorldStore::new(tmp.join("world"), 51);
        let mut ch = Chunk::new(IVec3::ZERO);
        for z in 0..16 {
            for x in 0..8 {
                ch.set(x, 1, z, snow, 15, 0);
            }
            for x in 8..16 {
                ch.set(x, 1, z, water, 15, 0);
            }
        }
        world.insert_chunk(ch);
        world.mark_priority_dirty_with_neighbors(IVec3::new(8, 1, 8));

        let Ok(mut renderer) = WgpuRenderer::new_headless(320, 180, &atlas) else {
            let _ = std::fs::remove_dir_all(&tmp);
            return;
        };
        renderer.remesh_priority_chunks_now(&world, &reg, &atlas, false);

        let eye = Vec3::new(7.5, 4.5, 13.0);
        let target_horizon = Vec3::new(7.5, 5.2, 0.0);
        let vp = Mat4::perspective_rh(72.0f32.to_radians(), 320.0 / 180.0, 0.1, 200.0)
            * Mat4::look_at_rh(eye, target_horizon, Vec3::Y);

        let mut settings = EngineSettings::default();
        settings.apply_graphics_preset(GraphicsPreset::Balanced);
        settings.render_distance_chunks = 6;

        let noon_img = renderer
            .render_to_headless_target(eye, vp, 0.50, &settings)
            .expect("noon sky render");
        let sunset_img = renderer
            .render_to_headless_target(eye, vp, 0.75, &settings)
            .expect("sunset sky render");
        let night_img = renderer
            .render_to_headless_target(eye, vp, 0.00, &settings)
            .expect("night sky render");

        let band_avg = |img: &RgbaImage, y0: u32, y1: u32| -> (f32, f32, f32) {
            let mut sum = [0.0f32; 3];
            let mut cnt = 0.0f32;
            for y in y0..y1 {
                for x in 20..300 {
                    let p = img.get_pixel(x, y).0;
                    sum[0] += p[0] as f32;
                    sum[1] += p[1] as f32;
                    sum[2] += p[2] as f32;
                    cnt += 1.0;
                }
            }
            (sum[0] / cnt, sum[1] / cnt, sum[2] / cnt)
        };
        let (nr, ng, nb) = band_avg(&noon_img, 4, 28);
        let (sr, _sg, sb) = band_avg(&sunset_img, 55, 85);
        let (dr, dg, db) = band_avg(&night_img, 4, 28);
        assert!(nb > nr + 30.0 && ng > 100.0, "Noon sky should be bright blue: ({nr}, {ng}, {nb})");
        assert!(sr > sb + 10.0, "Sunset sky horizon/band should shift warm amber: R={sr}, B={sb}");
        assert!(dr < 60.0 && dg < 70.0 && db < 110.0, "Night sky should be dark indigo: ({dr}, {dg}, {db})");

        // Snow foreground on left half must stay clean off-white with zero black/atlas-bleed grid seams
        let mut dark_snow_pixels = 0u32;
        for y in 145..170 {
            for x in 30..110 {
                let p = noon_img.get_pixel(x, y).0;
                if p[0] < 140 || p[1] < 140 || p[2] < 140 {
                    dark_snow_pixels += 1;
                }
            }
        }
        assert_eq!(
            dark_snow_pixels, 0,
            "Snow plain rendered {dark_snow_pixels} dark seam/bleed pixels"
        );

        // Toggle clouds OFF and switch to Cinematic (Bloom + 4-oct) in 2P split-screen
        settings.procedural_sky_clouds = false;
        settings.sync_graphics_preset_from_toggles();
        let noon_noclouds = renderer
            .render_to_headless_target(eye, vp, 0.50, &settings)
            .expect("no clouds render");
        assert_ne!(noon_img.as_raw(), noon_noclouds.as_raw());

        settings.apply_graphics_preset(GraphicsPreset::Cinematic);
        let split_vp1 = Mat4::perspective_rh(72.0f32.to_radians(), 160.0 / 180.0, 0.1, 200.0)
            * Mat4::look_at_rh(eye, target_horizon, Vec3::Y);
        let split_vp2 = Mat4::perspective_rh(72.0f32.to_radians(), 160.0 / 180.0, 0.1, 200.0)
            * Mat4::look_at_rh(
                Vec3::new(11.5, 4.5, 13.0),
                Vec3::new(11.5, 4.8, 0.0),
                Vec3::Y,
            );
        let cam1 = ViewportCamera {
            cam_pos: eye,
            view_proj: split_vp1,
            viewport_px: [0, 0, 160, 180],
            player_slot: 0,
            draw_local_player_mask: 0,
            draw_viewmodel: false,
            dimension: DimensionId::Overworld,
        };
        let cam2 = ViewportCamera {
            cam_pos: Vec3::new(11.5, 4.5, 13.0),
            view_proj: split_vp2,
            viewport_px: [160, 0, 160, 180],
            player_slot: 1,
            draw_local_player_mask: 0,
            draw_viewmodel: false,
            dimension: DimensionId::Overworld,
        };
        let split_img = renderer
            .render_split_to_headless_target(cam1, cam2, 0.50, &settings, None, None, 1.0)
            .expect("split screen procedural sky + water reflection");
        assert_eq!(split_img.dimensions(), (320, 180));
        // Both left and right split-screen sky viewports must contain bright sky pixels
        assert!(split_img.get_pixel(80, 10).0[2] > 120);
        assert!(split_img.get_pixel(240, 10).0[2] > 120);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn directional_sun_shadow_map_casts_real_terrain_shadows() {
        use crate::config::GraphicsPreset;
        use crate::registry::TileSpec;
        use image::{Rgba, RgbaImage};

        let mut reg = EngineRegistries::new();
        let stone = reg.register_node(NodeDef {
            name: "mcl_core:stone".into(),
            drawtype: DrawType::Normal,
            tiles: vec![TileSpec {
                name: "stone.png".into(),
                color: None,
            }],
            ..Default::default()
        });
        let tmp = std::env::temp_dir().join("cubex_sun_shadow_cast_test");
        let _ = std::fs::remove_dir_all(&tmp);
        let tex_dir = tmp.join("textures");
        let _ = std::fs::create_dir_all(&tex_dir);
        RgbaImage::from_pixel(16, 16, Rgba([190, 190, 190, 255]))
            .save(tex_dir.join("stone.png"))
            .unwrap();
        let atlas = TextureAtlas::build(&HashMap::new(), &tmp, None, &reg);

        let world = WorldStore::new(tmp.join("world"), 9);
        let mut ch = Chunk::new(IVec3::ZERO);
        for z in 0..16 {
            for x in 0..16 {
                ch.set(x, 1, z, stone, 15, 0);
            }
        }
        // High stone pillar/overhang on the sunward side casting shadow across floor without touching skylight
        for y in 4..10 {
            for z in 5..10 {
                for x in 10..13 {
                    ch.set(x, y, z, stone, 15, 0);
                }
            }
        }
        world.insert_chunk(ch);
        world.mark_priority_dirty_with_neighbors(IVec3::new(8, 2, 8));

        let Ok(mut renderer) = WgpuRenderer::new_headless(320, 180, &atlas) else {
            let _ = std::fs::remove_dir_all(&tmp);
            return;
        };
        renderer.remesh_priority_chunks_now(&world, &reg, &atlas, true);

        let eye = Vec3::new(8.0, 9.0, 13.0);
        let vp = Mat4::perspective_rh(65.0f32.to_radians(), 320.0 / 180.0, 0.1, 160.0)
            * Mat4::look_at_rh(eye, Vec3::new(8.0, 2.0, 7.0), Vec3::Y);

        let mut settings = EngineSettings::default();
        settings.apply_graphics_preset(GraphicsPreset::Balanced);
        settings.render_distance_chunks = 6;
        let with_shadows = renderer
            .render_to_headless_target(eye, vp, 0.34, &settings)
            .expect("render with sun shadows");

        settings.dynamic_shadows = false;
        let without_shadows = renderer
            .render_to_headless_target(eye, vp, 0.34, &settings)
            .expect("render without sun shadows");

        let darker_pixels = with_shadows
            .pixels()
            .zip(without_shadows.pixels())
            .filter(|(a, b)| a.0[0] + 6 < b.0[0] || a.0[1] + 6 < b.0[1] || a.0[2] + 6 < b.0[2])
            .count();
        assert!(
            darker_pixels > 80,
            "Sun shadow map must visibly darken shadowed terrain pixels, got {darker_pixels}"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn headless_carterpcs_skin_crown_and_ram_mic_render() {
        let reg = EngineRegistries::new();
        let atlas = TextureAtlas::build(&HashMap::new(), Path::new(""), None, &reg);
        let skins_dir = crate::paths::EnginePaths::resolve().skins_dir;
        let skins = crate::textures::list_available_skins(&skins_dir, &atlas.player_skin);
        let carter = skins
            .iter()
            .find(|s| crate::textures::is_carterpcs_skin(&s.id))
            .expect("carterpcs.png in assets/skins");

        let Ok(mut renderer) = WgpuRenderer::new_headless(240, 180, &atlas) else {
            return;
        };
        renderer.set_player_skin(0, &carter.id, &carter.image);
        renderer.show_viewmodel = true;
        renderer.update_viewmodel(
            Some(crate::registry::RAM_MIC_ITEM_ID),
            0,
            0.016,
            0.0,
            true,
            false,
            false,
            &reg,
            &atlas,
        );
        assert!(renderer.viewmodel_index_count > 0);
        assert!(
            renderer
                .viewmodel_mesh
                .as_ref()
                .is_some_and(|m| m.is_custom_prop),
            "cubex:ram_mic viewmodel must use custom compound prop mesh"
        );

        renderer.update_player_model(
            Vec3::new(8.5, 2.0, 8.5),
            0.0,
            0.0,
            0.0,
            false,
            Some(crate::registry::RAM_MIC_ITEM_ID),
            0.016,
            &reg,
            &atlas,
        );
        assert!(renderer.player_overlay_buffers.index_count > 0);
        assert!(renderer.player_crown_buffers.index_count > 0);
        assert!(renderer.player_held_buffers.index_count > 0);

        let dropped_mic = EntityRenderSnapshot {
            id: 99,
            entity_name: "__builtin:item".into(),
            pos: Vec3::new(8.0, 2.5, 9.2),
            yaw: 0.0,
            collisionbox: [-0.18, -0.18, -0.18, 0.18, 0.18, 0.18],
            visual: "wielditem".into(),
            visual_size: Vec3::splat(0.67),
            mesh: String::new(),
            texture_name: String::new(),
            wield_item: crate::registry::RAM_MIC_ITEM_ID.into(),
            walk_phase: 0.3,
            walk_blend: 0.0,
            anim_time: 0.3,
            vel_y: 0.0,
            hurt_flash: 0.0,
            sky_light: 15,
            block_light: 0,
        };
        renderer.update_entities(&[dropped_mic], &reg, &atlas);
        assert!(
            renderer
                .entity_draw_batches
                .iter()
                .any(|(k, b)| k == "prop:cubex:ram_mic" && b.index_count > 36),
            "Dropped cubex:ram_mic must batch its dedicated 3D compound mesh"
        );

        let eye = Vec3::new(8.5, 3.2, 11.5);
        let vp = Mat4::perspective_rh(72.0f32.to_radians(), 240.0 / 180.0, 0.1, 100.0)
            * Mat4::look_at_rh(eye, Vec3::new(8.5, 3.0, 8.5), Vec3::Y);
        let cams = [ViewportCamera {
            cam_pos: eye,
            view_proj: vp,
            viewport_px: [0, 0, 240, 180],
            player_slot: 0,
            draw_local_player_mask: 1u8 << 0,
            draw_viewmodel: true,
            dimension: DimensionId::Overworld,
        }];
        let img = renderer
            .render_viewports_to_headless_target(
                &cams,
                0.35,
                &EngineSettings::default(),
                None,
                None,
                1.0,
            )
            .expect("headless render with CarterPCs skin & RAM mic");
        assert_eq!(img.dimensions(), (240, 180));
        assert!(img.pixels().any(|p| p.0[3] == 255 && (p.0[0] > 0 || p.0[1] > 0 || p.0[2] > 0)));
    }

    #[test]
    fn headless_mipmaps_anisotropy_and_splitscreen_overworld_nether_render() {
        let mut reg = EngineRegistries::new();
        let stone = reg.register_node(NodeDef {
            name: "mcl_core:stone".into(),
            drawtype: DrawType::Normal,
            walkable: true,
            ..Default::default()
        });
        let netherrack = reg.register_node(NodeDef {
            name: "mcl_nether:netherrack".into(),
            drawtype: DrawType::Normal,
            walkable: true,
            ..Default::default()
        });
        let portal_id = reg.get_content_id(crate::registry::NETHER_PORTAL_NODE_ID);
        let atlas = TextureAtlas::build(&HashMap::new(), Path::new(""), None, &reg);
        let mut renderer =
            WgpuRenderer::new_headless(320, 180, &atlas).expect("headless wgpu split-dim");

        // Live anisotropy switches (Off -> 4x -> 8x)
        assert_eq!(renderer.anisotropy_quality(), AnisotropyQuality::Off);
        renderer.set_anisotropy_quality(AnisotropyQuality::X4);
        assert_eq!(renderer.anisotropy_quality(), AnisotropyQuality::X4);
        renderer.set_anisotropy_quality(AnisotropyQuality::X8);
        assert_eq!(renderer.anisotropy_quality(), AnisotropyQuality::X8);

        let dir = std::env::temp_dir().join("cubex_headless_split_ow_nether");
        let _ = std::fs::remove_dir_all(&dir);
        let ow = WorldStore::new(&dir, 777);
        let nw = WorldStore::new_for_dimension(
            &dir,
            777,
            DimensionId::Nether,
            crate::world::WorldGenType::Default,
            crate::world::Difficulty::Normal,
            crate::world::Gamerules::default(),
            "Survival",
        );

        let mut ch_ow = Chunk::new(IVec3::ZERO);
        ch_ow.set(8, 2, 8, stone, 15, 0);
        ch_ow.set(8, 3, 8, portal_id, 15, 0);
        ow.insert_chunk(ch_ow);
        ow.mark_priority_dirty_with_neighbors(IVec3::new(8, 2, 8));
        renderer.remesh_priority_chunks_now(&ow, &reg, &atlas, true);

        let mut ch_nw = Chunk::new(IVec3::ZERO);
        ch_nw.param1.fill(0);
        ch_nw.set(8, 2, 8, netherrack, 0xA0, 0);
        ch_nw.set(8, 3, 8, portal_id, 0xB0, 1);
        nw.insert_chunk(ch_nw);
        nw.mark_priority_dirty_with_neighbors(IVec3::new(8, 2, 8));
        renderer.remesh_priority_chunks_now(&nw, &reg, &atlas, true);
        assert!(renderer.chunk_meshes.contains_key(&IVec3::ZERO));
        assert!(renderer.nether_chunk_meshes.contains_key(&IVec3::ZERO));

        let eye = Vec3::new(8.5, 4.5, 13.0);
        let proj = Mat4::perspective_rh(72.0f32.to_radians(), 160.0 / 180.0, 0.1, 100.0);
        let vp = proj * Mat4::look_at_rh(eye, Vec3::new(8.5, 3.0, 8.5), Vec3::Y);
        let cams = [
            ViewportCamera {
                cam_pos: eye,
                view_proj: vp,
                viewport_px: [0, 0, 160, 180],
                player_slot: 0,
                draw_local_player_mask: 0,
                draw_viewmodel: false,
                dimension: DimensionId::Overworld,
            },
            ViewportCamera {
                cam_pos: eye,
                view_proj: vp,
                viewport_px: [160, 0, 160, 180],
                player_slot: 1,
                draw_local_player_mask: 0,
                draw_viewmodel: false,
                dimension: DimensionId::Nether,
            },
        ];
        let mut settings = EngineSettings::default();
        settings.apply_graphics_preset(crate::config::GraphicsPreset::Balanced);
        let img = renderer
            .render_viewports_to_headless_target(&cams, 0.50, &settings, None, None, 1.0)
            .expect("split Overworld+Nether headless render");
        assert_eq!(img.dimensions(), (320, 180));

        // Upper-left sky must be Overworld noon blue (B > R), upper-right must be Nether crimson (R > B)
        let ow_sky = img.get_pixel(40, 12).0;
        let nw_sky = img.get_pixel(240, 12).0;
        assert!(
            ow_sky[2] > ow_sky[0] && ow_sky[2] > 120,
            "Expected Overworld blue sky on left half, got {:?}",
            ow_sky
        );
        assert!(
            nw_sky[0] > nw_sky[2] && nw_sky[0] < 130,
            "Expected dark crimson Nether dome on right half, got {:?}",
            nw_sky
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
