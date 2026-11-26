use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use crate::config::EngineSettings;

/// 3-tier sky, sun, cloud, and fog palette for a given `time_of_day` (`0.0..1.0`).
#[derive(Debug, Clone, Copy)]
pub struct SkyPalette {
    pub zenith: [f32; 3],
    pub horizon: [f32; 3],
    pub ground: [f32; 3],
    pub sun_color: [f32; 3],
    pub cloud_color: [f32; 3],
    pub sun_dir: [f32; 3],
    pub daylight: f32,
    pub sunset: f32,
    pub night: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SkyUniform {
    pub inv_view_proj: [[f32; 4]; 4],
    pub sun_dir_time: [f32; 4],
    pub zenith_color: [f32; 4],
    pub horizon_color: [f32; 4],
    pub ground_color: [f32; 4],
    pub sun_color: [f32; 4],
    pub cloud_color: [f32; 4],
    pub params: [f32; 4],
}

#[inline]
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let k = t.clamp(0.0, 1.0);
    [
        a[0] + (b[0] - a[0]) * k,
        a[1] + (b[1] - a[1]) * k,
        a[2] + (b[2] - a[2]) * k,
    ]
}

#[inline]
fn smoothstep01(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Computes normalized sun direction and dynamic 3-tier sky/cloud/fog colors across Day, Sunset/Sunrise, and Night.
pub fn compute_sky_palette(tod: f32) -> SkyPalette {
    let angle = (tod - 0.25) * std::f32::consts::TAU;
    let sun_dir_v = Vec3::new(angle.cos(), angle.sin(), 0.28).normalize();
    let sun_elev = sun_dir_v.y;

    let daylight = smoothstep01(-0.14, 0.28, sun_elev);
    let sunset = 1.0 - smoothstep01(0.0, 0.26, (sun_elev - 0.04).abs());
    let night = (1.0 - daylight - sunset * 0.55).clamp(0.0, 1.0);

    // Linear-space atmospheric colors (encoded to sRGB on output)
    let night_zenith = [0.006, 0.011, 0.032];
    let night_horizon = [0.025, 0.042, 0.088];
    let night_ground = [0.004, 0.007, 0.016];
    let night_sun = [0.35, 0.42, 0.58];
    let night_cloud = [0.08, 0.11, 0.19];

    let day_zenith = [0.13, 0.38, 0.88];
    let day_horizon = [0.46, 0.72, 0.96];
    let day_ground = [0.24, 0.34, 0.48];
    let day_sun = [1.35, 1.24, 0.98];
    let day_cloud = [0.98, 0.99, 1.00];

    let set_zenith = [0.16, 0.19, 0.42];
    let set_horizon = [0.88, 0.45, 0.24];
    let set_ground = [0.22, 0.14, 0.18];
    let set_sun = [1.65, 0.78, 0.32];
    let set_cloud = [1.05, 0.66, 0.48];

    let zenith = lerp3(lerp3(night_zenith, day_zenith, daylight), set_zenith, sunset * 0.75);
    let horizon = lerp3(lerp3(night_horizon, day_horizon, daylight), set_horizon, sunset * 0.85);
    let ground = lerp3(lerp3(night_ground, day_ground, daylight), set_ground, sunset * 0.75);
    let sun_color = lerp3(lerp3(night_sun, day_sun, daylight), set_sun, sunset * 0.90);
    let cloud_color = lerp3(lerp3(night_cloud, day_cloud, daylight), set_cloud, sunset * 0.85);

    SkyPalette {
        zenith,
        horizon,
        ground,
        sun_color,
        cloud_color,
        sun_dir: [sun_dir_v.x, sun_dir_v.y, sun_dir_v.z],
        daylight,
        sunset,
        night,
    }
}

/// Computes the Nether cavern sky/fog palette (crimson horizon `#3A0B0B` -> ember zenith `#1A0404`).
pub fn compute_nether_sky_palette() -> SkyPalette {
    SkyPalette {
        zenith: [0.011, 0.0014, 0.0014],
        horizon: [0.042, 0.0033, 0.0033],
        ground: [0.085, 0.012, 0.0035],
        sun_color: [0.95, 0.32, 0.09],
        cloud_color: [0.20, 0.035, 0.020],
        sun_dir: [0.0, -1.0, 0.0],
        daylight: 0.0,
        sunset: 0.0,
        night: 0.0,
    }
}

impl SkyUniform {
    pub fn from_camera(
        view_proj: Mat4,
        time_of_day: f32,
        elapsed_sec: f32,
        settings: &EngineSettings,
        bloom_active: bool,
    ) -> Self {
        Self::from_camera_for_dimension(view_proj, time_of_day, elapsed_sec, settings, bloom_active, false)
    }

    pub fn from_camera_for_dimension(
        view_proj: Mat4,
        time_of_day: f32,
        elapsed_sec: f32,
        settings: &EngineSettings,
        bloom_active: bool,
        is_nether: bool,
    ) -> Self {
        let pal = if is_nether {
            compute_nether_sky_palette()
        } else {
            compute_sky_palette(time_of_day)
        };
        let inv_vp = if view_proj.determinant().abs() > 1e-6 {
            view_proj.inverse()
        } else {
            Mat4::IDENTITY
        };
        let warm_flag = if settings.warm_sun_and_atmosphere { 1.0 } else { 0.0 };
        let bloom_flag = if bloom_active { 1.0 } else { 0.0 };
        let octaves = if is_nether { 0.0 } else { settings.cloud_octaves() as f32 };
        let dim_or_tod = if is_nether { -1.0 } else { time_of_day.max(0.0) };

        Self {
            inv_view_proj: inv_vp.to_cols_array_2d(),
            sun_dir_time: [pal.sun_dir[0], pal.sun_dir[1], pal.sun_dir[2], elapsed_sec],
            zenith_color: [pal.zenith[0], pal.zenith[1], pal.zenith[2], pal.daylight],
            horizon_color: [pal.horizon[0], pal.horizon[1], pal.horizon[2], pal.sunset],
            ground_color: [pal.ground[0], pal.ground[1], pal.ground[2], pal.night],
            sun_color: [pal.sun_color[0], pal.sun_color[1], pal.sun_color[2], octaves],
            cloud_color: [pal.cloud_color[0], pal.cloud_color[1], pal.cloud_color[2], 0.46],
            params: [warm_flag, bloom_flag, settings.brightness, dim_or_tod],
        }
    }
}

/// Procedural atmospheric sky dome & FBM drifting cloud renderer (`sky.wgsl`).
pub struct SkyRenderer {
    pub bind_group_layout: wgpu::BindGroupLayout,
    sdr_pipeline: wgpu::RenderPipeline,
    hdr_pipeline: wgpu::RenderPipeline,
}

impl SkyRenderer {
    pub fn new(
        device: &wgpu::Device,
        sdr_format: wgpu::TextureFormat,
        hdr_format: wgpu::TextureFormat,
    ) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SkyUniformBGL"),
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SkyPipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ProceduralSkyShader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("sky.wgsl").into()),
        });

        let build_pipeline = |target_fmt: wgpu::TextureFormat, label: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_sky"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_sky"),
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
            })
        };

        let sdr_pipeline = build_pipeline(sdr_format, "SdrSkyPipeline");
        let hdr_pipeline = build_pipeline(hdr_format, "HdrSkyPipeline");

        Self {
            bind_group_layout,
            sdr_pipeline,
            hdr_pipeline,
        }
    }

    pub fn draw<'a>(
        &'a self,
        rpass: &mut wgpu::RenderPass<'a>,
        bind_group: &'a wgpu::BindGroup,
        use_hdr: bool,
    ) {
        rpass.set_pipeline(if use_hdr {
            &self.hdr_pipeline
        } else {
            &self.sdr_pipeline
        });
        rpass.set_bind_group(0, bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sky_palette_transitions_day_sunset_and_night() {
        let noon = compute_sky_palette(0.50);
        let sunset = compute_sky_palette(0.75);
        let night = compute_sky_palette(0.00);

        // Noon: rich blue sky, brighter horizon, high daylight
        assert!(noon.daylight > 0.95);
        assert!(noon.sunset < 0.05);
        assert!(noon.zenith[2] > noon.zenith[0]);
        assert!(noon.horizon[2] > noon.horizon[0]);

        // Sunset: warm golden-orange horizon dominates blue
        assert!(sunset.sunset > 0.70);
        assert!(sunset.horizon[0] > sunset.horizon[2]);
        assert!(sunset.horizon[0] > noon.horizon[0]);

        // Night: deep dark blues, high night factor
        assert!(night.night > 0.90);
        assert!(night.daylight < 0.05);
        assert!(night.zenith[0] < 0.05 && night.zenith[2] < 0.08);
        assert!(night.horizon[2] < 0.15);
    }
}
