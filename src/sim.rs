use crate::palette::{self, Palette};
use crate::rules::Rule;
use egui::Color32;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use wgpu::util::DeviceExt;

#[allow(dead_code)]
pub struct Sim {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub size: (u32, u32),

    tex_a: wgpu::Texture,
    tex_b: wgpu::Texture,
    view_a: wgpu::TextureView,
    view_b: wgpu::TextureView,
    current_is_a: bool,

    palette_tex: wgpu::Texture,
    palette_view: wgpu::TextureView,
    palette_sampler: wgpu::Sampler,

    rule_buffer: wgpu::Buffer,
    cam_buffer: wgpu::Buffer,

    compute_bind_group_layout: wgpu::BindGroupLayout,
    render_bind_group_layout: wgpu::BindGroupLayout,
    edit_bind_group_layout: wgpu::BindGroupLayout,
    random_bind_group_layout: wgpu::BindGroupLayout,
    clear_bind_group_layout: wgpu::BindGroupLayout,
    compute_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    edit_pipeline: wgpu::ComputePipeline,
    random_pipeline: wgpu::ComputePipeline,
    clear_pipeline: wgpu::ComputePipeline,
    clear_inside_pipeline: wgpu::ComputePipeline,

    compute_bg_ab: wgpu::BindGroup,
    compute_bg_ba: wgpu::BindGroup,
    render_bg_a: wgpu::BindGroup,
    render_bg_b: wgpu::BindGroup,

    pub cpu_state: Vec<u32>,
    pub rule: Rule,
    pub wrap: bool,

    pub palette: Palette,

    readback: Option<Readback>,
    pub last_population: u32,

    pub png_readback: Option<PngReadback>,
}

#[allow(dead_code)]
struct Readback {
    buffer: wgpu::Buffer,
    mapped: Arc<AtomicBool>,
}

/// The outcome of a finished GPU->CPU region readback.
pub enum ReadbackResult {
    /// Written to a PNG file on disk.
    Saved(std::path::PathBuf),
    /// Raw RGBA8 pixels, e.g. for the system clipboard.
    Image {
        width: u32,
        height: u32,
        rgba: Vec<u8>,
    },
}

/// A pending GPU->CPU snapshot of a selected region. When the `path` is set the
/// result is saved as a PNG; otherwise the raw RGBA pixels are returned instead.
pub struct PngReadback {
    buffer: wgpu::Buffer,
    mapped: Arc<AtomicBool>,
    width: u32,
    height: u32,
    padded_row: u32,
    path: Option<std::path::PathBuf>,
    palette: Palette,
}

impl Sim {
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
        rule: Rule,
        palette: &Palette,
    ) -> Self {
        let (tex_a, view_a) = create_state_texture(&device, width, height);
        let (tex_b, view_b) = create_state_texture(&device, width, height);

        let (palette_tex, palette_view, palette_sampler) =
            create_palette_texture(&device, &queue, palette);

        let rule_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rule_buffer"),
            contents: bytemuck::cast_slice(&[rule.birth, rule.survive, 1u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let cam_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cam_buffer"),
            contents: bytemuck::bytes_of(&CamData::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("compute_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::R32Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("render_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::R32Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let edit_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("edit_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let random_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("random_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::R32Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let clear_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("clear_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute_pipeline_layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });        let edit_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("edit_pipeline_layout"),
                bind_group_layouts: &[&edit_bind_group_layout],
                push_constant_ranges: &[],
            });
        let random_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("random_pipeline_layout"),
                bind_group_layouts: &[&random_bind_group_layout],
                push_constant_ranges: &[],
            });
        let clear_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("clear_pipeline_layout"),
                bind_group_layouts: &[&clear_bind_group_layout],
                push_constant_ranges: &[],
            });
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("render_pipeline_layout"),
                bind_group_layouts: &[&render_bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sim_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sim.wgsl").into()),
        });
        let edit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("edit_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/edit.wgsl").into()),
        });
        let random_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("random_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/random.wgsl").into()),
        });
        let clear_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("clear_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/clear_outside.wgsl").into()),
        });
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("render_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/render.wgsl").into()),
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute_pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let edit_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("edit_pipeline"),
            layout: Some(&edit_pipeline_layout),
            module: &edit_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let random_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("random_pipeline"),
            layout: Some(&random_pipeline_layout),
            module: &random_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("clear_pipeline"),
            layout: Some(&clear_pipeline_layout),
            module: &clear_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let clear_inside_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("clear_inside_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/clear_inside.wgsl").into()),
        });
        let clear_inside_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("clear_inside_pipeline"),
            layout: Some(&clear_pipeline_layout),
            module: &clear_inside_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render_pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let compute_bg_ab = make_compute_bind_group(
            &device,
            &compute_bind_group_layout,
            &view_a,
            &view_b,
            &rule_buffer,
        );
        let compute_bg_ba = make_compute_bind_group(
            &device,
            &compute_bind_group_layout,
            &view_b,
            &view_a,
            &rule_buffer,
        );
        let render_bg_a = make_render_bind_group(
            &device,
            &render_bind_group_layout,
            &view_a,
            &palette_view,
            &palette_sampler,
            &cam_buffer,
        );
        let render_bg_b = make_render_bind_group(
            &device,
            &render_bind_group_layout,
            &view_b,
            &palette_view,
            &palette_sampler,
            &cam_buffer,
        );

        let cpu_state = vec![0u32; (width * height) as usize];

        Self {
            device,
            queue,
            size: (width, height),
            tex_a,
            tex_b,
            view_a,
            view_b,
            current_is_a: true,
            palette_tex,
            palette_view,
            palette_sampler,
            rule_buffer,
            cam_buffer,
            compute_bind_group_layout,
            render_bind_group_layout,
            edit_bind_group_layout,
            random_bind_group_layout,
            clear_bind_group_layout,
            compute_pipeline,
            render_pipeline,
            edit_pipeline,
            random_pipeline,
            clear_pipeline,
            clear_inside_pipeline,
            compute_bg_ab,
            compute_bg_ba,
            render_bg_a,
            render_bg_b,
            cpu_state,
            rule,
            wrap: true,
            palette: palette.clone(),
            readback: None,
            last_population: 0,
            png_readback: None,
        }
    }

    #[allow(dead_code)]
    pub fn resize(&mut self, width: u32, height: u32) {
        if self.size == (width, height) {
            return;
        }
        let (old_w, old_h) = self.size;
        self.size = (width, height);
        let (tex_a, view_a) = create_state_texture(&self.device, width, height);
        let (tex_b, view_b) = create_state_texture(&self.device, width, height);

        // Clear the new textures and copy the old current state into the center.
        let zero = vec![0u8; (width * height * 4) as usize];
        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let layout = wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        };
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex_a,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &zero,
            layout,
            extent,
        );
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex_b,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &zero,
            layout,
            extent,
        );

        let old_src = if self.current_is_a {
            &self.tex_a
        } else {
            &self.tex_b
        };
        // Center the old content in the new grid on both grow and shrink. The
        // offset is signed; the copy is clipped to the overlapping region so a
        // negative offset (shrinking) crops from the source, not just the bottom-right.
        let ox = (width as i32 - old_w as i32) / 2;
        let oy = (height as i32 - old_h as i32) / 2;
        let src_x = if ox < 0 { (-ox) as u32 } else { 0 };
        let dst_x = if ox > 0 { ox as u32 } else { 0 };
        let src_y = if oy < 0 { (-oy) as u32 } else { 0 };
        let dst_y = if oy > 0 { oy as u32 } else { 0 };
        let copy_w = (old_w as i32 - src_x as i32).max(0) as u32;
        let copy_w = copy_w.min(width - dst_x);
        let copy_h = (old_h as i32 - src_y as i32).max(0) as u32;
        let copy_h = copy_h.min(height - dst_y);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resize_copy_encoder"),
            });
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: old_src,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: src_x,
                    y: src_y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &tex_a,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: dst_x,
                    y: dst_y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: copy_w,
                height: copy_h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));

        self.tex_a = tex_a;
        self.tex_b = tex_b;
        self.view_a = view_a;
        self.view_b = view_b;
        self.current_is_a = true;

        self.compute_bg_ab = make_compute_bind_group(
            &self.device,
            &self.compute_bind_group_layout,
            &self.view_a,
            &self.view_b,
            &self.rule_buffer,
        );
        self.compute_bg_ba = make_compute_bind_group(
            &self.device,
            &self.compute_bind_group_layout,
            &self.view_b,
            &self.view_a,
            &self.rule_buffer,
        );
        self.render_bg_a = make_render_bind_group(
            &self.device,
            &self.render_bind_group_layout,
            &self.view_a,
            &self.palette_view,
            &self.palette_sampler,
            &self.cam_buffer,
        );
        self.render_bg_b = make_render_bind_group(
            &self.device,
            &self.render_bind_group_layout,
            &self.view_b,
            &self.palette_view,
            &self.palette_sampler,
            &self.cam_buffer,
        );

        self.cpu_state.resize((width * height) as usize, 0);
    }

    pub fn set_rule(&mut self, rule: Rule) {
        self.rule = rule;
        self.write_rule_params();
    }

    pub fn set_wrap(&mut self, wrap: bool) {
        self.wrap = wrap;
        self.write_rule_params();
    }

    fn write_rule_params(&mut self) {
        let wrap = if self.wrap { 1u32 } else { 0u32 };
        self.queue.write_buffer(
            &self.rule_buffer,
            0,
            bytemuck::cast_slice(&[self.rule.birth, self.rule.survive, wrap, 0u32]),
        );
    }

    pub fn set_palette(&mut self, palette: &Palette) {
        self.palette = palette.clone();
        let mut data = vec![0u8; palette::PALETTE_SIZE * 4];
        palette.build_rgba8(&mut data);
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.palette_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some((palette::PALETTE_SIZE * 4) as u32),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: palette::PALETTE_SIZE as u32,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn set_camera(
        &mut self,
        center: [f32; 2],
        viewport_px: [f32; 2],
        scale_px: f32,
        wrap: bool,
    ) {
        let (w, h) = self.size;
        let data = CamData {
            center,
            viewport: viewport_px,
            grid: [w as f32, h as f32],
            scale: scale_px,
            circle_threshold: 8.0,
            wrap: if wrap { 1u32 } else { 0u32 },
            _pad: 0,
        };
        self.queue
            .write_buffer(&self.cam_buffer, 0, bytemuck::bytes_of(&data));
    }

    pub fn upload_cpu_state(&mut self) {
        let (w, h) = self.size;
        let data = bytemuck::cast_slice(&self.cpu_state);
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.tex_a,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.tex_b,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Run `steps` simulation generations and return a single command buffer.
    pub fn step(&mut self, steps: u32) -> wgpu::CommandBuffer {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("step_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sim_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compute_pipeline);
            let (w, h) = self.size;
            let groups_x = (w + 15) / 16;
            let groups_y = (h + 15) / 16;
            for _ in 0..steps {
                if self.current_is_a {
                    pass.set_bind_group(0, &self.compute_bg_ab, &[]);
                } else {
                    pass.set_bind_group(0, &self.compute_bg_ba, &[]);
                }
                pass.dispatch_workgroups(groups_x, groups_y, 1);
                self.current_is_a = !self.current_is_a;
            }
        }
        encoder.finish()
    }

    /// Apply a list of edits directly to the current GPU state.
    pub fn apply_edits(&mut self, edits: &[Edit]) -> wgpu::CommandBuffer {
        if edits.is_empty() {
            let encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("empty_edit_encoder"),
                });
            return encoder.finish();
        }
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("edit_buffer"),
                contents: bytemuck::cast_slice(edits),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let target = if self.current_is_a {
            &self.view_a
        } else {
            &self.view_b
        };
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("edit_bind_group"),
            layout: &self.edit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(target),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("edit_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("edit_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.edit_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((edits.len() as u32 + 63) / 64, 1, 1);
        }
        encoder.finish()
    }

    /// Randomly flip `fraction` of the current live cells on the GPU.
    pub fn randomize_gpu(&mut self, fraction: f32, seed: u32) -> wgpu::CommandBuffer {
        let (w, h) = self.size;
        let params = RandomParams {
            fraction,
            seed,
            grid_x: w,
            grid_y: h,
            _pad: 0,
        };
        let param_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("random_param_buffer"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let (src, dst) = if self.current_is_a {
            (&self.view_a, &self.view_b)
        } else {
            (&self.view_b, &self.view_a)
        };
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("random_bind_group"),
            layout: &self.random_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(dst),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: param_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("random_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("random_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.random_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((w + 15) / 16, (h + 15) / 16, 1);
        }
        self.current_is_a = !self.current_is_a;
        encoder.finish()
    }

    /// Clear every cell outside the inclusive selection rect back to dead.
    pub fn clear_outside(&mut self, x0: u32, y0: u32, x1: u32, y1: u32) -> wgpu::CommandBuffer {
        let pipeline = self.clear_pipeline.clone();
        self.clear_rect(x0, y0, x1, y1, &pipeline)
    }

    /// Clear every cell inside the inclusive selection rect back to dead.
    pub fn clear_inside(&mut self, x0: u32, y0: u32, x1: u32, y1: u32) -> wgpu::CommandBuffer {
        let pipeline = self.clear_inside_pipeline.clone();
        self.clear_rect(x0, y0, x1, y1, &pipeline)
    }

    fn clear_rect(
        &mut self,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
        pipeline: &wgpu::ComputePipeline,
    ) -> wgpu::CommandBuffer {
        let (w, h) = self.size;
        let params = ClearParams {
            x0,
            y0,
            x1,
            y1,
            grid_w: w,
            grid_h: h,
        };
        let param_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("clear_param_buffer"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let target = if self.current_is_a {
            &self.view_a
        } else {
            &self.view_b
        };
        let bind_group = make_clear_bind_group(
            &self.device,
            &self.clear_bind_group_layout,
            target,
            &param_buffer,
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("clear_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clear_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((w + 15) / 16, (h + 15) / 16, 1);
        }
        encoder.finish()
    }

    /// Copy the selected region to a readback buffer; the PNG is written from
    /// `poll_png` once the buffer has been mapped.
    pub fn request_selection_png(
        &mut self,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
        path: std::path::PathBuf,
        palette: Palette,
    ) {
        self.request_readback(x0, y0, x1, y1, Some(path), palette);
    }

    /// Like `request_selection_png`, but yields raw RGBA pixels (for the
    /// clipboard) instead of writing a file.
    pub fn request_selection_clipboard(
        &mut self,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
        palette: Palette,
    ) {
        self.request_readback(x0, y0, x1, y1, None, palette);
    }

    fn request_readback(
        &mut self,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
        path: Option<std::path::PathBuf>,
        palette: Palette,
    ) {
        if self.png_readback.is_some() {
            return;
        }
        let width = x1 - x0 + 1;
        let height = y1 - y0 + 1;
        let row_bytes = (width * 4) as u64;
        let padded_row = ((row_bytes + 255) / 256) * 256;
        let buffer_size = padded_row * (height as u64);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("png_readback_buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let src = if self.current_is_a {
            &self.tex_a
        } else {
            &self.tex_b
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("png_readback_encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: src,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: x0,
                    y: y0,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row as u32),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));
        let mapped = Arc::new(AtomicBool::new(false));
        let mapped_cb = Arc::clone(&mapped);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |_result| {
                mapped_cb.store(true, Ordering::Relaxed);
            });
        self.png_readback = Some(PngReadback {
            buffer,
            mapped,
            width,
            height,
            padded_row: padded_row as u32,
            path,
            palette,
        });
    }

    /// Returns the outcome if a pending readback completed.
    pub fn poll_png(&mut self) -> Option<Result<ReadbackResult, String>> {
        let rb = self.png_readback.take()?;
        self.device.poll(wgpu::Maintain::Poll);
        if !rb.mapped.load(Ordering::Relaxed) {
            self.png_readback = Some(rb);
            return None;
        }
        let result = write_png_from_readback(&rb);
        Some(result)
    }

    /// Draw the current state into the given render pass.
    pub fn render(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        render_pass.set_pipeline(&self.render_pipeline);
        if self.current_is_a {
            render_pass.set_bind_group(0, &self.render_bg_a, &[]);
        } else {
            render_pass.set_bind_group(0, &self.render_bg_b, &[]);
        }
        render_pass.draw(0..3, 0..1);
    }

    pub fn submit(&self, command_buffers: Vec<wgpu::CommandBuffer>) {
        self.queue.submit(command_buffers);
    }

    /// Returns Some(population) if a pending readback has completed.
    #[allow(dead_code)]
    pub fn poll_population(&mut self) -> Option<u32> {
        if self.readback.is_none() {
            return None;
        }
        self.device.poll(wgpu::Maintain::Poll);
        if let Some(rb) = self.readback.take() {
            if rb.mapped.load(Ordering::Relaxed) {
                let view = rb.buffer.slice(..).get_mapped_range();
                let data: &[u32] = bytemuck::cast_slice(&view);
                let count = data.iter().filter(|&&v| v > 0).count() as u32;
                drop(view);
                rb.buffer.unmap();
                self.last_population = count;
                return Some(count);
            } else {
                self.readback = Some(rb);
            }
        }
        None
    }

    pub fn has_pending_readback(&self) -> bool {
        self.readback.is_some()
    }

    #[allow(dead_code)]
    pub fn request_population_readback(&mut self) {
        if self.readback.is_some() {
            return;
        }
        let (w, h) = self.size;
        let row_bytes = (w * 4) as u64;
        let padded_bytes_per_row = ((row_bytes + 255) / 256) * 256;
        let buffer_size = padded_bytes_per_row * (h as u64);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback_encoder"),
            });
        encoder.clear_buffer(&buffer, 0, Some(buffer_size));
        let src = if self.current_is_a {
            &self.tex_a
        } else {
            &self.tex_b
        };
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: src,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row as u32),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));
        let mapped = Arc::new(AtomicBool::new(false));
        let mapped_cb = Arc::clone(&mapped);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |_result| {
                mapped_cb.store(true, Ordering::Relaxed);
            });
        self.readback = Some(Readback { buffer, mapped });
    }
}

fn age_to_color(age: u32, palette: &Palette) -> Color32 {
    if age == 0 {
        return palette.sample(0.0);
    }
    // Same aging curve as render.wgsl: fresh cells use the palette's right end.
    let n = age.ilog2() as f32;
    let t = (1.0 - n * 0.08).clamp(0.0, 1.0);
    palette.sample(t)
}

fn write_png(
    path: &std::path::Path,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut encoder = png::Encoder::new(file, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
    writer.write_image_data(rgba).map_err(|e| e.to_string())
}

fn readback_rgba(rb: &PngReadback) -> Vec<u8> {
    let view = rb.buffer.slice(..).get_mapped_range();
    let data: &[u32] = bytemuck::cast_slice(&view);
    let w = rb.width as usize;
    let h = rb.height as usize;
    let padded_u32s = (rb.padded_row / 4) as usize;
    let mut rgba = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let age = data[y * padded_u32s + x];
            let c = age_to_color(age, &rb.palette);
            let o = (y * w + x) * 4;
            rgba[o] = c.r();
            rgba[o + 1] = c.g();
            rgba[o + 2] = c.b();
            rgba[o + 3] = c.a();
        }
    }
    drop(view);
    rb.buffer.unmap();
    rgba
}

fn write_png_from_readback(rb: &PngReadback) -> Result<ReadbackResult, String> {
    let rgba = readback_rgba(rb);
    match &rb.path {
        Some(path) => {
            write_png(path, rb.width, rb.height, &rgba)?;
            Ok(ReadbackResult::Saved(path.clone()))
        }
        None => Ok(ReadbackResult::Image {
            width: rb.width,
            height: rb.height,
            rgba,
        }),
    }
}

fn create_state_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("state_texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn create_palette_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    palette: &Palette,
) -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler) {
    let mut data = vec![0u8; palette::PALETTE_SIZE * 4];
    palette.build_rgba8(&mut data);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("palette_texture"),
        size: wgpu::Extent3d {
            width: palette::PALETTE_SIZE as u32,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some((palette::PALETTE_SIZE * 4) as u32),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: palette::PALETTE_SIZE as u32,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("palette_sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        ..Default::default()
    });
    (texture, view, sampler)
}

fn make_compute_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    src: &wgpu::TextureView,
    dst: &wgpu::TextureView,
    rule: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute_bind_group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(src),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(dst),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: rule.as_entire_binding(),
            },
        ],
    })
}

fn make_clear_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    target: &wgpu::TextureView,
    param_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("clear_bind_group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(target),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: param_buffer.as_entire_binding(),
            },
        ],
    })
}

fn make_render_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    state: &wgpu::TextureView,
    palette: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    cam: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("render_bind_group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(state),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(palette),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: cam.as_entire_binding(),
            },
        ],
    })
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Edit {
    pub x: u32,
    pub y: u32,
    pub value: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RandomParams {
    fraction: f32,
    seed: u32,
    grid_x: u32,
    grid_y: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ClearParams {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    grid_w: u32,
    grid_h: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CamData {
    center: [f32; 2],
    viewport: [f32; 2],
    grid: [f32; 2],
    scale: f32,
    circle_threshold: f32,
    wrap: u32,
    _pad: u32,
}

impl Default for CamData {
    fn default() -> Self {
        Self {
            center: [0.0, 0.0],
            viewport: [1.0, 1.0],
            grid: [1.0, 1.0],
            scale: 1.0,
            circle_threshold: 8.0,
            wrap: 0,
            _pad: 0,
        }
    }
}
