use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BloomParams {
    pub texel_size: [f32; 2],
    pub threshold: f32,
    pub strength: f32,
}

/// Resolution-dependent offscreen HDR + quarter-res ping-pong textures for Bloom.
/// Kept `None` whenever Bloom is disabled so zero extra VRAM/bandwidth is allocated.
pub struct BloomTargets {
    pub width: u32,
    pub height: u32,
    pub scene_tex: wgpu::Texture,
    pub scene_view: wgpu::TextureView,
    pub bloom_a_tex: wgpu::Texture,
    pub bloom_a_view: wgpu::TextureView,
    pub bloom_b_tex: wgpu::Texture,
    pub bloom_b_view: wgpu::TextureView,
    pub bright_params_bg: wgpu::BindGroup,
    pub blur1_params_bg: wgpu::BindGroup,
    pub blur2_params_bg: wgpu::BindGroup,
    pub composite_params_bg: wgpu::BindGroup,
    pub bright_tex_bg: wgpu::BindGroup,
    pub blur_a_to_b_bg: wgpu::BindGroup,
    pub blur_b_to_a_bg: wgpu::BindGroup,
    pub composite_tex_bg: wgpu::BindGroup,
}

/// Optional quarter-resolution dual-Kawase Bloom + ACES filmic composite post-process.
pub struct BloomPostProcess {
    pub hdr_format: wgpu::TextureFormat,
    linear_sampler: wgpu::Sampler,
    params_bgl: wgpu::BindGroupLayout,
    single_tex_bgl: wgpu::BindGroupLayout,
    composite_tex_bgl: wgpu::BindGroupLayout,
    bright_pipeline: wgpu::RenderPipeline,
    blur_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,
}

impl BloomPostProcess {
    pub fn choose_hdr_format(
        adapter: &wgpu::Adapter,
        fallback: wgpu::TextureFormat,
    ) -> wgpu::TextureFormat {
        let f = wgpu::TextureFormat::Rgba16Float;
        let features = adapter.get_texture_format_features(f);
        let needed_usage =
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        if features.allowed_usages.contains(needed_usage)
            && features
                .flags
                .contains(wgpu::TextureFormatFeatureFlags::FILTERABLE)
        {
            f
        } else {
            fallback
        }
    }

    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        hdr_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("BloomPostShader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("bloom.wgsl").into()),
        });

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("BloomLinearSampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let params_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BloomParamsBGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let single_tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BloomSingleTexBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
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

        let composite_tex_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("BloomCompositeTexBGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        let single_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("BloomSingleLayout"),
            bind_group_layouts: &[&params_bgl, &single_tex_bgl],
            push_constant_ranges: &[],
        });

        let composite_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("BloomCompositeLayout"),
            bind_group_layouts: &[&params_bgl, &composite_tex_bgl],
            push_constant_ranges: &[],
        });

        let mk_pipe = |label: &str,
                       layout: &wgpu::PipelineLayout,
                       frag_entry: &str,
                       out_format: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_fullscreen"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(frag_entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: out_format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        };

        let bright_pipeline = mk_pipe(
            "BloomBrightPipeline",
            &single_layout,
            "fs_bright_downsample",
            hdr_format,
        );
        let blur_pipeline =
            mk_pipe("BloomBlurPipeline", &single_layout, "fs_kawase_blur", hdr_format);
        let composite_pipeline = mk_pipe(
            "BloomCompositePipeline",
            &composite_layout,
            "fs_composite",
            surface_format,
        );

        Self {
            hdr_format,
            linear_sampler,
            params_bgl,
            single_tex_bgl,
            composite_tex_bgl,
            bright_pipeline,
            blur_pipeline,
            composite_pipeline,
        }
    }

    pub fn ensure_targets<'a>(
        &self,
        device: &wgpu::Device,
        slot: &'a mut Option<BloomTargets>,
        width: u32,
        height: u32,
    ) -> &'a BloomTargets {
        let w = width.max(1);
        let h = height.max(1);
        let need_rebuild = match slot {
            Some(ref t) => t.width != w || t.height != h,
            None => true,
        };
        if need_rebuild {
            let qw = (w / 4).max(1);
            let qh = (h / 4).max(1);
            let mk_tex = |label: &str, tw: u32, th: u32| {
                let tex = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: tw,
                        height: th,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: self.hdr_format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                (tex, view)
            };

            let (scene_tex, scene_view) = mk_tex("BloomHdrScene", w, h);
            let (bloom_a_tex, bloom_a_view) = mk_tex("BloomQuarterA", qw, qh);
            let (bloom_b_tex, bloom_b_view) = mk_tex("BloomQuarterB", qw, qh);

            let mk_params_bg = |label: &str, params: BloomParams| {
                let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &self.params_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                })
            };

            let full_ts = [1.0 / w as f32, 1.0 / h as f32];
            let q_ts = [1.0 / qw as f32, 1.0 / qh as f32];

            let bright_params_bg = mk_params_bg(
                "BloomBrightParamsBG",
                BloomParams {
                    texel_size: full_ts,
                    threshold: 0.88,
                    strength: 1.0,
                },
            );
            let blur1_params_bg = mk_params_bg(
                "BloomBlur1ParamsBG",
                BloomParams {
                    texel_size: q_ts,
                    threshold: 0.0,
                    strength: 1.15,
                },
            );
            let blur2_params_bg = mk_params_bg(
                "BloomBlur2ParamsBG",
                BloomParams {
                    texel_size: q_ts,
                    threshold: 0.0,
                    strength: 2.15,
                },
            );
            let composite_params_bg = mk_params_bg(
                "BloomCompositeParamsBG",
                BloomParams {
                    texel_size: full_ts,
                    threshold: 0.88,
                    strength: 0.28,
                },
            );

            let mk_single_bg = |label: &str, view: &wgpu::TextureView| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &self.single_tex_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                        },
                    ],
                })
            };

            let bright_tex_bg = mk_single_bg("BloomBrightTexBG", &scene_view);
            let blur_a_to_b_bg = mk_single_bg("BloomBlurAToBBG", &bloom_a_view);
            let blur_b_to_a_bg = mk_single_bg("BloomBlurBToABG", &bloom_b_view);

            let composite_tex_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("BloomCompositeTexBG"),
                layout: &self.composite_tex_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&scene_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&bloom_a_view),
                    },
                ],
            });

            *slot = Some(BloomTargets {
                width: w,
                height: h,
                scene_tex,
                scene_view,
                bloom_a_tex,
                bloom_a_view,
                bloom_b_tex,
                bloom_b_view,
                bright_params_bg,
                blur1_params_bg,
                blur2_params_bg,
                composite_params_bg,
                bright_tex_bg,
                blur_a_to_b_bg,
                blur_b_to_a_bg,
                composite_tex_bg,
            });
        }
        slot.as_ref().unwrap()
    }

    pub fn encode_bloom_and_composite(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        targets: &BloomTargets,
        dst_view: &wgpu::TextureView,
    ) {
        let mut run_pass = |label: &str,
                            pipeline: &wgpu::RenderPipeline,
                            params_bg: &wgpu::BindGroup,
                            tex_bg: &wgpu::BindGroup,
                            out_view: &wgpu::TextureView| {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: out_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(pipeline);
            rpass.set_bind_group(0, params_bg, &[]);
            rpass.set_bind_group(1, tex_bg, &[]);
            rpass.draw(0..3, 0..1);
        };

        // Pass 1: Bright downsample full-res HDR scene -> quarter-res A
        run_pass(
            "BloomBrightPass",
            &self.bright_pipeline,
            &targets.bright_params_bg,
            &targets.bright_tex_bg,
            &targets.bloom_a_view,
        );
        // Pass 2: Dual-Kawase blur A -> B
        run_pass(
            "BloomBlurPass1",
            &self.blur_pipeline,
            &targets.blur1_params_bg,
            &targets.blur_a_to_b_bg,
            &targets.bloom_b_view,
        );
        // Pass 3: Dual-Kawase blur B -> A
        run_pass(
            "BloomBlurPass2",
            &self.blur_pipeline,
            &targets.blur2_params_bg,
            &targets.blur_b_to_a_bg,
            &targets.bloom_a_view,
        );
        // Pass 4: Composite scene_hdr + quarter-res bloom_a -> ACES surface
        run_pass(
            "BloomCompositePass",
            &self.composite_pipeline,
            &targets.composite_params_bg,
            &targets.composite_tex_bg,
            dst_view,
        );
    }
}
