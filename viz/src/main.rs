// src/main.rs
//#![windows_subsystem = "windows"]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferBindingType, BufferDescriptor, BufferUsages, Color,
    ColorTargetState, CommandEncoderDescriptor, DeviceDescriptor, FragmentState, Instance,
    InstanceDescriptor, LoadOp, MultisampleState, Operations, PipelineLayoutDescriptor, PowerPreference,
    PresentMode, PrimitiveState, RenderPipeline, RequestAdapterOptions, ShaderModuleDescriptor,
    ShaderModuleDescriptorPassthrough, ShaderStages, StoreOp, SurfaceConfiguration, TextureUsages, Trace,
    VertexState,
};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes};

// --- CLI ---------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "wgpu-spv", about = "Tiny wgpu demo that builds a pipeline from SPIR-V", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Use a single SPIR-V module with separate vertex & fragment entry points
    #[command(name = "vf", visible_alias = "vertex-fragment")]
    VertexFragment {
        /// Path to the SPIR-V module containing both entry points
        path: PathBuf,
        /// Vertex shader entry point name (default: main)
        #[arg(long, default_value = "main")]
        vertex: String,
        /// Fragment shader entry point name (default: main)
        #[arg(long, default_value = "main")]
        fragment: String,
    },
    /// Show device and driver information
    #[command(name = "info")]
    Info,
}

// --- Pipeline spec passed to the app -----------------------------------------

enum PipelineSpec {
    VertexFragment {
        path: PathBuf,
        vertex: String,
        fragment: String,
    },
}

// --- App state ---------------------------------------------------------------

/// Debug buffer size: 16KB = 4096 u32s
/// Layout: { write_head: u32, read_head: u32, data: [4094]u32 }
const DEBUG_BUFFER_SIZE: u64 = 16384;

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: SurfaceConfiguration,
    pipeline: RenderPipeline,
    // Debug buffer support (optional - only present when shader uses debug intrinsics)
    debug_buffer: Option<wgpu::Buffer>,
    debug_staging_buffer: Option<wgpu::Buffer>,
    debug_bind_group: Option<BindGroup>,
}

impl State {
    async fn new(window: Arc<Window>, spec: &PipelineSpec) -> Result<Self> {
        let instance = Instance::new(&InstanceDescriptor::default());

        let surface = instance.create_surface(window.clone()).context("failed to create wgpu surface")?;

        // v26: returns Result<Adapter, RequestAdapterError>
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("request_adapter failed")?;

        // Check if SPIRV_SHADER_PASSTHROUGH is supported
        let adapter_features = adapter.features();
        let spirv_passthrough_supported =
            adapter_features.contains(wgpu::Features::SPIRV_SHADER_PASSTHROUGH);

        println!(
            "SPIRV_SHADER_PASSTHROUGH supported: {}",
            spirv_passthrough_supported
        );

        // Build required features
        let mut required_features = wgpu::Features::empty();
        if spirv_passthrough_supported {
            required_features |= wgpu::Features::SPIRV_SHADER_PASSTHROUGH;
        }
        // Need writable storage from vertex shader for debug buffer
        required_features |= wgpu::Features::VERTEX_WRITABLE_STORAGE;

        // v26: request_device takes a single descriptor; trace is in the descriptor
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: None,
                required_features,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: Trace::Off,
            })
            .await
            .context("failed to create logical device")?;

        let caps = surface.get_capabilities(&adapter);
        let format =
            caps.formats.get(0).copied().ok_or_else(|| anyhow!("surface reports no supported formats"))?;
        let size = window.inner_size();

        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: PresentMode::Fifo,
            alpha_mode: caps
                .alpha_modes
                .get(0)
                .copied()
                .ok_or_else(|| anyhow!("surface reports no alpha modes"))?,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // === Create debug buffer (always, shader may or may not use it) =========
        let debug_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("debug_buffer"),
            size: DEBUG_BUFFER_SIZE,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Initialize debug buffer to zeros (write_head=0, read_head=0, data=zeros)
        queue.write_buffer(&debug_buffer, 0, &[0u8; DEBUG_BUFFER_SIZE as usize]);

        let debug_staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("debug_staging_buffer"),
            size: DEBUG_BUFFER_SIZE,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout for debug buffer at set=0, binding=0
        let debug_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("debug_bind_group_layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let debug_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("debug_bind_group"),
            layout: &debug_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &debug_buffer,
                    offset: 0,
                    size: None,
                }),
            }],
        });

        // === Build pipeline from the chosen mode ==============================
        let pipeline = match spec {
            PipelineSpec::VertexFragment {
                path,
                vertex,
                fragment,
            } => {
                let module = load_spirv_module(&device, path)
                    .with_context(|| format!("load SPIR-V module {:?}", path))?;

                let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: Some("layout"),
                    bind_group_layouts: &[&debug_bind_group_layout],
                    push_constant_ranges: &[],
                });

                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("pipeline"),
                    layout: Some(&layout),
                    vertex: VertexState {
                        module: &module,
                        entry_point: Some(vertex.as_str()),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(FragmentState {
                        module: &module,
                        entry_point: Some(fragment.as_str()),
                        targets: &[Some(ColorTargetState {
                            format: config.format,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: MultisampleState::default(),
                    multiview: None,
                    cache: None,
                })
            }
        };

        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            pipeline,
            debug_buffer: Some(debug_buffer),
            debug_staging_buffer: Some(debug_staging_buffer),
            debug_bind_group: Some(debug_bind_group),
        })
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width > 0 && size.height > 0 {
            self.config.width = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn render(&mut self) {
        match self.surface.get_current_texture() {
            Ok(frame) => {
                let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

                let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("encoder"),
                });

                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: Operations {
                                load: LoadOp::Clear(Color {
                                    r: 0.02,
                                    g: 0.02,
                                    b: 0.02,
                                    a: 1.0,
                                }),
                                store: StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        ..Default::default()
                    });

                    rpass.set_pipeline(&self.pipeline);
                    // Set debug bind group if available
                    if let Some(ref bind_group) = self.debug_bind_group {
                        rpass.set_bind_group(0, bind_group, &[]);
                    }
                    rpass.draw(0..3, 0..1);
                }

                // Copy debug buffer to staging buffer for readback
                if let (Some(ref debug_buffer), Some(ref staging_buffer)) =
                    (&self.debug_buffer, &self.debug_staging_buffer)
                {
                    encoder.copy_buffer_to_buffer(debug_buffer, 0, staging_buffer, 0, DEBUG_BUFFER_SIZE);
                }

                self.queue.submit(Some(encoder.finish()));
                frame.present();
            }
            Err(e @ wgpu::SurfaceError::Lost) | Err(e @ wgpu::SurfaceError::Outdated) => {
                eprintln!("surface {e}; reconfiguring");
                let size = self.window.inner_size();
                if size.width > 0 && size.height > 0 {
                    self.config.width = size.width;
                    self.config.height = size.height;
                }
                self.surface.configure(&self.device, &self.config);
            }
            Err(wgpu::SurfaceError::Timeout) => {
                eprintln!("surface timeout; skipping frame");
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                eprintln!("GPU out of memory; exiting");
                std::process::exit(1);
            }
            Err(wgpu::SurfaceError::Other) => {
                // Non-fatal miscellaneous error; skip this frame.
                eprintln!("surface error: Other; skipping frame");
            }
        }
    }

    /// Read and print all debug output from the buffer (call on exit)
    fn print_debug_output(&mut self) {
        let staging_buffer = match &self.debug_staging_buffer {
            Some(b) => b,
            None => return,
        };

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        // Wait for the GPU to finish
        let _ = self.device.poll(wgpu::PollType::Wait);

        if rx.recv().unwrap().is_ok() {
            let data = buffer_slice.get_mapped_range();
            let u32_data: &[u32] = bytemuck::cast_slice(&data);

            // Buffer layout: [write_head, read_head, data[4094]]
            let write_head = u32_data[0] as usize;
            let data_start = 2;
            let data_len = 4094;

            if write_head == 0 {
                eprintln!("[DEBUG] No debug output recorded");
                return;
            }

            eprintln!("[DEBUG] {} values recorded:", write_head);

            // Print all values from the ring buffer (from 0 to write_head, wrapping)
            let count = write_head.min(data_len); // Don't print more than buffer size
            for i in 0..count {
                let index = i % data_len;
                let value = u32_data[data_start + index];
                let as_i32 = value as i32;
                let as_f32 = f32::from_bits(value);
                // Try to decode as 4 ASCII chars
                let bytes = value.to_le_bytes();
                let as_str: String = bytes
                    .iter()
                    .map(|&b| if b.is_ascii_graphic() || b == b' ' { b as char } else { '.' })
                    .collect();
                println!(
                    "[DEBUG {:5}] raw=0x{:08x} i32={:11} f32={:14} str=\"{}\"",
                    i, value, as_i32, as_f32, as_str
                );
            }

            drop(data);
            staging_buffer.unmap();
        }
    }
}

// Load a .spv file and create a ShaderModule using the SPIR-V helper
fn load_spirv_module(device: &wgpu::Device, path: &Path) -> Result<wgpu::ShaderModule> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;

    // Check if SPIRV_SHADER_PASSTHROUGH is supported
    if device.features().contains(wgpu::Features::SPIRV_SHADER_PASSTHROUGH) {
        // Convert bytes to u32 words for SPIR-V passthrough
        let mut spirv_data = Vec::new();
        for chunk in bytes.chunks_exact(4) {
            let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            spirv_data.push(word);
        }

        // Use create_shader_module_passthrough to bypass wgpu's SPIR-V validation
        // This allows loading SPIR-V with unsupported capabilities like Linkage
        device.push_error_scope(wgpu::ErrorFilter::Validation);

        let shader_module = unsafe {
            device.create_shader_module_passthrough(ShaderModuleDescriptorPassthrough::SpirV(
                wgpu::ShaderModuleDescriptorSpirV {
                    label: Some(&format!("{}", path.display())),
                    source: std::borrow::Cow::Borrowed(&spirv_data),
                },
            ))
        };

        // Check for validation errors even with passthrough
        let error_option = pollster::block_on(device.pop_error_scope());
        if let Some(error) = error_option {
            return Err(anyhow::Error::msg(format!(
                "Shader validation failed (passthrough): {}",
                error
            )));
        }

        Ok(shader_module)
    } else {
        // Fall back to regular shader module creation with validation
        let source = wgpu::util::make_spirv(&bytes);

        // Push an error scope to catch shader validation errors
        device.push_error_scope(wgpu::ErrorFilter::Validation);

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{}", path.display())),
            source,
        });

        // Check for validation errors
        let error_option = pollster::block_on(device.pop_error_scope());
        if let Some(error) = error_option {
            return Err(anyhow::Error::msg(format!("Shader validation failed: {}", error)));
        }

        Ok(shader_module)
    }
}

// --- Winit app shell ---------------------------------------------------------

struct App {
    state: Option<State>,
    spec: PipelineSpec,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = match event_loop.create_window(WindowAttributes::default().with_title("wgpu + SPIR-V"))
        {
            Ok(w) => Arc::new(w),
            Err(e) => {
                eprintln!("failed to create window: {e}");
                std::process::exit(1);
            }
        };

        match pollster::block_on(State::new(window, &self.spec)) {
            Ok(state) => self.state = Some(state),
            Err(e) => {
                eprintln!("failed to initialize GPU state: {e:#}");
                std::process::exit(1);
            }
        }
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        if let Some(state) = &mut self.state {
            if state.window.id() == window_id {
                match event {
                    WindowEvent::CloseRequested => {
                        state.print_debug_output();
                        std::process::exit(0);
                    }
                    WindowEvent::Resized(size) => state.resize(size),
                    WindowEvent::RedrawRequested => state.render(),
                    WindowEvent::ScaleFactorChanged {
                        scale_factor: _,
                        mut inner_size_writer,
                    } => {
                        // Request a size; a `Resized` will follow.
                        let _ = inner_size_writer.request_inner_size(state.window.inner_size());
                    }
                    _ => {}
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &mut self.state {
            state.window.request_redraw();
        }
    }
}

async fn show_device_info() -> Result<()> {
    let instance = Instance::new(&InstanceDescriptor::default());

    // Try to get an adapter
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .context("No suitable adapter found")?;

    let info = adapter.get_info();
    let features = adapter.features();
    let limits = adapter.limits();

    println!("GPU Device Information:");
    println!("  Name: {}", info.name);
    println!("  Vendor: {:?}", info.vendor);
    println!("  Device: {}", info.device);
    println!("  Device Type: {:?}", info.device_type);
    println!("  Driver: {}", info.driver);
    println!("  Driver Info: {}", info.driver_info);
    println!("  Backend: {:?}", info.backend);

    println!("\nSupported Features:");
    println!("  {:#?}", features);

    println!("\nDevice Limits:");
    println!("  Max Texture Dimension 1D: {}", limits.max_texture_dimension_1d);
    println!("  Max Texture Dimension 2D: {}", limits.max_texture_dimension_2d);
    println!("  Max Texture Dimension 3D: {}", limits.max_texture_dimension_3d);
    println!("  Max Bind Groups: {}", limits.max_bind_groups);
    println!(
        "  Max Uniform Buffer Binding Size: {}",
        limits.max_uniform_buffer_binding_size
    );
    println!(
        "  Max Storage Buffer Binding Size: {}",
        limits.max_storage_buffer_binding_size
    );

    Ok(())
}

fn main() -> Result<()> {
    // Parse CLI and map to our pipeline spec
    let cli = Cli::parse();

    match cli.command {
        Command::Info => {
            pollster::block_on(show_device_info())?;
            return Ok(());
        }
        Command::VertexFragment {
            path,
            vertex,
            fragment,
        } => {
            let spec = PipelineSpec::VertexFragment {
                path,
                vertex,
                fragment,
            };

            let event_loop = EventLoop::new().context("failed to create event loop")?;
            let mut app = App { state: None, spec };

            if let Err(e) = event_loop.run_app(&mut app) {
                return Err(anyhow!(e)).context("winit event loop errored");
            }
        }
    }

    Ok(())
}
