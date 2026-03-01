use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use bytes::Bytes;
use dssim::Dssim;
use exif::{In, Tag, Value};
use image::{GenericImageView, ImageReader};
use image::imageops::FilterType;
use img_hash::image::GrayImage as HashGrayImage;
use img_hash::{HashAlg, HasherConfig};
use img_parts::ImageEXIF;
use little_exif::exif_tag::ExifTag as LittleExifTag;
use little_exif::metadata::Metadata as LittleExifMetadata;
use little_exif::rational::uR64 as LittleExifUR64;
use rgb::FromSlice;
use serde::Deserialize;
use serde::Serialize;
use slimg_core::{
    ImageData as SlimgImageData, PipelineOptions, ResizeMode, convert, decode_file,
};
use slimg_core::resize::resize as slimg_resize;
use std::fs;
use std::fs::File;
use std::io::{BufReader, Cursor, Read};
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use crossbeam_channel as channel;
use tauri::Emitter;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ImageListItem {
    relative_path: String,
    absolute_path: String,
    size_bytes: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ImageDetails {
    display_data_url: Option<String>,
    size_bytes: Option<u64>,
    width: Option<u32>,
    height: Option<u32>,
    dpi_x: Option<f32>,
    dpi_y: Option<f32>,
    color: Option<String>,
    error: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ImageVerificationResult {
    similarity: f32,
    verdict: String,
    best_transform: String,
    orientation_issue: bool,
    aspect_issue: bool,
    aspect_ratio_delta: f32,
    scale_ratio_delta: f32,
    source_width: u32,
    source_height: u32,
    dest_width: u32,
    dest_height: u32,
    message: String,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct MigrationProgressEvent {
    total: usize,
    processed: usize,
    succeeded: usize,
    failed: usize,
    message: String,
    current_relative_path: Option<String>,
    current_action: Option<String>,
    current_source_size_bytes: Option<u64>,
    current_dest_size_bytes: Option<u64>,
    done: bool,
    canceled: bool,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct MigrationProgressItemUpdate {
    relative_path: String,
    action: String,
    source_size_bytes: Option<u64>,
    dest_size_bytes: Option<u64>,
    message: String,
    fallback_code: Option<String>,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct MigrationProgressBatchEvent {
    total: usize,
    processed: usize,
    succeeded: usize,
    failed: usize,
    message: String,
    updates: Vec<MigrationProgressItemUpdate>,
    done: bool,
    canceled: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ConcurrencyProfile {
    cpu_cores: usize,
    min: usize,
    max: usize,
    default_value: usize,
}

#[derive(Default)]
struct MigrationRuntimeState {
    running: bool,
    cancel_flag: Option<Arc<AtomicBool>>,
}

static MIGRATION_STATE: OnceLock<Mutex<MigrationRuntimeState>> = OnceLock::new();
static GPU_RESIZE_CONTEXT: OnceLock<Result<Arc<GpuResizeContext>, String>> = OnceLock::new();

fn migration_state() -> &'static Mutex<MigrationRuntimeState> {
    MIGRATION_STATE.get_or_init(|| Mutex::new(MigrationRuntimeState::default()))
}

struct GpuResizeContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bgl: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
}

#[derive(Clone)]
struct MigrationTaskFile {
    source_path: PathBuf,
    dest_path: PathBuf,
    relative_path: String,
    real_format: RealImageFormat,
    orientation: Option<u16>,
    dpi_x: Option<f32>,
    dpi_y: Option<f32>,
    width: Option<u32>,
    height: Option<u32>,
}

#[derive(Clone, Copy)]
struct OptimizationCriteria {
    use_dpi: bool,
    target_dpi: f32,
    use_max_width: bool,
    max_width: u32,
    use_max_height: bool,
    max_height: u32,
}

#[derive(Clone, Copy)]
struct MigrationOptions {
    restore_metadata: bool,
    acceleration_mode: AccelerationMode,
}

#[derive(Clone, Copy)]
struct OptimizationPlan {
    scale: f32,
    apply_target_dpi: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AccelerationMode {
    Auto,
    Cpu,
    GpuPreferred,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
enum AccelerationModeArg {
    Auto,
    Cpu,
    Gpu,
}

impl From<AccelerationModeArg> for AccelerationMode {
    fn from(value: AccelerationModeArg) -> Self {
        match value {
            AccelerationModeArg::Auto => Self::Auto,
            AccelerationModeArg::Cpu => Self::Cpu,
            AccelerationModeArg::Gpu => Self::GpuPreferred,
        }
    }
}

fn acceleration_mode_label(mode: AccelerationMode) -> &'static str {
    match mode {
        AccelerationMode::Auto => "Auto",
        AccelerationMode::Cpu => "CPU",
        AccelerationMode::GpuPreferred => "GPU 우선(폴백 가능)",
    }
}

fn resolve_acceleration_backend(mode: AccelerationMode) -> (&'static str, Option<String>) {
    match mode {
        AccelerationMode::Cpu => ("cpu", None),
        AccelerationMode::Auto => match get_gpu_resize_context() {
            Ok(_) => ("auto", Some("Auto 모드: 파일별로 CPU/GPU 동적 선택".to_string())),
            Err(e) => ("auto", Some(format!("Auto 모드: GPU 사용 불가로 CPU 사용 ({})", e))),
        },
        AccelerationMode::GpuPreferred => match get_gpu_resize_context() {
            Ok(_) => ("gpu", Some("GPU 리사이즈 활성화".to_string())),
            Err(e) => ("cpu", Some(format!("GPU 우선 모드 실패로 CPU 폴백: {}", e))),
        },
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RealImageFormat {
    Jpeg,
    Png,
    WebP,
    Avif,
    HeicLike,
    Jxl,
    Qoi,
    Tiff,
    Bmp,
    Gif,
}

fn read_magic_prefix(path: &Path) -> Option<Vec<u8>> {
    let mut file = File::open(path).ok()?;
    let mut buf = vec![0u8; 32];
    let n = file.read(&mut buf).ok()?;
    buf.truncate(n);
    Some(buf)
}

fn detect_real_image_format(path: &Path) -> Option<RealImageFormat> {
    let magic = read_magic_prefix(path)?;
    if magic.len() >= 3 && magic[0..3] == [0xFF, 0xD8, 0xFF] {
        return Some(RealImageFormat::Jpeg);
    }
    if magic.len() >= 4 && magic[0..4] == [0x89, 0x50, 0x4E, 0x47] {
        return Some(RealImageFormat::Png);
    }
    if magic.len() >= 12 && &magic[0..4] == b"RIFF" && &magic[8..12] == b"WEBP" {
        return Some(RealImageFormat::WebP);
    }
    if magic.len() >= 6 && (&magic[0..6] == b"GIF87a" || &magic[0..6] == b"GIF89a") {
        return Some(RealImageFormat::Gif);
    }
    if magic.len() >= 2 && &magic[0..2] == b"BM" {
        return Some(RealImageFormat::Bmp);
    }
    if magic.len() >= 4
        && ((magic[0..4] == [0x49, 0x49, 0x2A, 0x00]) || (magic[0..4] == [0x4D, 0x4D, 0x00, 0x2A]))
    {
        return Some(RealImageFormat::Tiff);
    }
    if magic.len() >= 2 && magic[0..2] == [0xFF, 0x0A] {
        return Some(RealImageFormat::Jxl);
    }
    if magic.len() >= 8 && magic[0..4] == [0x00, 0x00, 0x00, 0x0C] && &magic[4..8] == b"JXL " {
        return Some(RealImageFormat::Jxl);
    }
    if magic.len() >= 4 && &magic[0..4] == b"qoif" {
        return Some(RealImageFormat::Qoi);
    }
    if magic.len() >= 12 && &magic[4..8] == b"ftyp" {
        let brand = &magic[8..12];
        if brand.starts_with(b"avif") || brand.starts_with(b"avis") {
            return Some(RealImageFormat::Avif);
        }
        if brand.starts_with(b"heic")
            || brand.starts_with(b"heix")
            || brand.starts_with(b"hevc")
            || brand.starts_with(b"hevx")
            || brand.starts_with(b"heif")
            || brand.starts_with(b"mif1")
            || brand.starts_with(b"msf1")
        {
            return Some(RealImageFormat::HeicLike);
        }
    }
    None
}

fn is_supported_image(path: &Path) -> bool {
    detect_real_image_format(path).is_some()
}

fn walk_image_files(
    base_path: &Path,
    current_path: &Path,
    output: &mut Vec<ImageListItem>,
) -> Result<(), String> {
    let entries = fs::read_dir(current_path)
        .map_err(|e| format!("디렉터리를 읽을 수 없습니다: {} ({})", current_path.display(), e))?;

    for entry in entries {
        let entry = match entry {
            Ok(value) => value,
            Err(_) => continue,
        };

        let path = entry.path();
        if path.is_dir() {
            walk_image_files(base_path, &path, output)?;
            continue;
        }

        if !path.is_file() || !is_supported_image(&path) {
            continue;
        }

        let relative_path = path
            .strip_prefix(base_path)
            .unwrap_or(path.as_path())
            .to_string_lossy()
            .to_string();
        let absolute_path = path.to_string_lossy().to_string();
        let size_bytes = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

        output.push(ImageListItem {
            relative_path,
            absolute_path,
            size_bytes,
        });
    }

    Ok(())
}

fn walk_migration_files(
    source_base: &Path,
    dest_base: &Path,
    current_path: &Path,
    output: &mut Vec<MigrationTaskFile>,
) -> Result<(), String> {
    let entries = fs::read_dir(current_path)
        .map_err(|e| format!("디렉터리를 읽을 수 없습니다: {} ({})", current_path.display(), e))?;

    for entry in entries {
        let entry = match entry {
            Ok(value) => value,
            Err(_) => continue,
        };

        let path = entry.path();
        if path.is_dir() {
            walk_migration_files(source_base, dest_base, &path, output)?;
            continue;
        }

        if !path.is_file() {
            continue;
        }

        let Some(real_format) = detect_real_image_format(&path) else {
            continue;
        };

        let relative = path
            .strip_prefix(source_base)
            .unwrap_or(path.as_path())
            .to_string_lossy()
            .to_string();
        let dest_path = dest_base.join(Path::new(&relative));
        let (orientation, dpi_x, dpi_y) = read_exif_info(&path);
        let (width, height) = read_image_dimensions_with_orientation(&path, orientation);

        output.push(MigrationTaskFile {
            source_path: path,
            dest_path,
            relative_path: relative,
            real_format,
            orientation,
            dpi_x,
            dpi_y,
            width,
            height,
        });
    }

    Ok(())
}

fn read_image_dimensions_with_orientation(path: &Path, orientation: Option<u16>) -> (Option<u32>, Option<u32>) {
    match image::image_dimensions(path) {
        Ok((w, h)) => {
            if orientation_swaps_dimensions(orientation) {
                (Some(h), Some(w))
            } else {
                (Some(w), Some(h))
            }
        }
        Err(_) => (None, None),
    }
}

fn orientation_swaps_dimensions(orientation: Option<u16>) -> bool {
    matches!(orientation, Some(5 | 6 | 7 | 8))
}

fn rational_to_f32(value: exif::Rational) -> Option<f32> {
    if value.denom == 0 {
        return None;
    }
    Some(value.num as f32 / value.denom as f32)
}

fn read_exif_info(path: &Path) -> (Option<u16>, Option<f32>, Option<f32>) {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(_) => return (None, None, None),
    };

    let mut reader = BufReader::new(file);
    let exif = match exif::Reader::new().read_from_container(&mut reader) {
        Ok(exif) => exif,
        Err(_) => return (None, None, None),
    };

    let orientation = exif
        .get_field(Tag::Orientation, In::PRIMARY)
        .and_then(|field| match &field.value {
            Value::Short(values) => values.first().copied(),
            _ => None,
        });

    let mut dpi_x = None;
    let mut dpi_y = None;

    if let Some(field) = exif.get_field(Tag::XResolution, In::PRIMARY) {
        if let Value::Rational(values) = &field.value {
            dpi_x = values.first().and_then(|r| rational_to_f32(*r));
        }
    }

    if let Some(field) = exif.get_field(Tag::YResolution, In::PRIMARY) {
        if let Value::Rational(values) = &field.value {
            dpi_y = values.first().and_then(|r| rational_to_f32(*r));
        }
    }

    let unit = exif
        .get_field(Tag::ResolutionUnit, In::PRIMARY)
        .and_then(|field| match &field.value {
            Value::Short(values) => values.first().copied(),
            _ => None,
        });

    if unit == Some(3) {
        dpi_x = dpi_x.map(|v| v * 2.54);
        dpi_y = dpi_y.map(|v| v * 2.54);
    }

    (orientation, dpi_x, dpi_y)
}

fn read_exif_dpi(path: &Path) -> (Option<f32>, Option<f32>) {
    let (_, dpi_x, dpi_y) = read_exif_info(path);
    (dpi_x, dpi_y)
}

fn read_exif_orientation(path: &Path) -> Option<u16> {
    let (orientation, _, _) = read_exif_info(path);
    orientation
}

fn read_image_dimensions(path: &Path) -> (Option<u32>, Option<u32>) {
    let orientation = read_exif_orientation(path);
    read_image_dimensions_with_orientation(path, orientation)
}

fn load_image_with_fallback(path: &Path) -> Result<image::DynamicImage, String> {
    let bytes = fs::read(path)
        .map_err(|e| format!("파일 읽기 실패: {} ({})", path.display(), e))?;

    let guessed_reader = ImageReader::new(Cursor::new(bytes.clone()))
        .with_guessed_format()
        .map_err(|e| format!("포맷 추정 실패: {}", e))?;
    if let Ok(img) = guessed_reader.decode() {
        return Ok(img);
    }

    image::open(path).map_err(|e| format!("이미지 디코드 실패: {}", e))
}

fn init_gpu_resize_context() -> Result<Arc<GpuResizeContext>, String> {
    const SHADER: &str = r#"
@group(0) @binding(0) var t_src: texture_2d<f32>;
@group(0) @binding(1) var s_src: sampler;

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(3.0, 1.0)
  );
  var out: VsOut;
  let p = pos[vid];
  out.pos = vec4<f32>(p, 0.0, 1.0);
  out.uv = p * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
  return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  return textureSample(t_src, s_src, in.uv);
}
"#;
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| "GPU 어댑터를 찾지 못했습니다.".to_string())?;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("image-opt-gpu-device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
        },
        None,
    ))
    .map_err(|e| format!("GPU 디바이스 생성 실패: {}", e))?;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("image-opt-resize-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("image-opt-bgl"),
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

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("image-opt-pipeline-layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("image-opt-resize-pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("image-opt-linear-sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    Ok(Arc::new(GpuResizeContext {
        device,
        queue,
        bgl,
        pipeline,
        sampler,
    }))
}

fn get_gpu_resize_context() -> Result<&'static Arc<GpuResizeContext>, String> {
    match GPU_RESIZE_CONTEXT.get_or_init(init_gpu_resize_context) {
        Ok(ctx) => Ok(ctx),
        Err(e) => Err(e.clone()),
    }
}

fn gpu_resize_rgba_wgpu(
    src_rgba: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Result<Vec<u8>, String> {
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return Err("GPU 리사이즈 입력 크기가 유효하지 않습니다.".to_string());
    }

    let expected = (src_w as usize) * (src_h as usize) * 4;
    if src_rgba.len() != expected {
        return Err(format!(
            "GPU 리사이즈 입력 버퍼 길이 불일치: expected {}, got {}",
            expected,
            src_rgba.len()
        ));
    }

    let ctx = get_gpu_resize_context()?;
    let device = &ctx.device;
    let queue = &ctx.queue;
    let max_dim = device.limits().max_texture_dimension_2d;
    if src_w > max_dim || src_h > max_dim || dst_w > max_dim || dst_h > max_dim {
        return Err(format!(
            "GPU 텍스처 한계 초과 (max: {}): src={}x{}, dst={}x{}",
            max_dim, src_w, src_h, dst_w, dst_h
        ));
    }

    let input_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("image-opt-input"),
        size: wgpu::Extent3d {
            width: src_w,
            height: src_h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let output_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("image-opt-output"),
        size: wgpu::Extent3d {
            width: dst_w,
            height: dst_h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &input_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        src_rgba,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(src_w * 4),
            rows_per_image: Some(src_h),
        },
        wgpu::Extent3d {
            width: src_w,
            height: src_h,
            depth_or_array_layers: 1,
        },
    );

    let input_view = input_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let output_view = output_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("image-opt-bind-group"),
        layout: &ctx.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&input_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&ctx.sampler),
            },
        ],
    });

    let bytes_per_pixel = 4u32;
    let unpadded_bpr = dst_w * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bpr = ((unpadded_bpr + align - 1) / align) * align;
    let output_size = padded_bpr as u64 * dst_h as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("image-opt-readback"),
        size: output_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("image-opt-encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("image-opt-render-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &output_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &output_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &output_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr),
                rows_per_image: Some(dst_h),
            },
        },
        wgpu::Extent3d {
            width: dst_w,
            height: dst_h,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let slice = output_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    match rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(format!("GPU readback map 실패: {}", e)),
        Err(_) => return Err("GPU readback 응답 수신 실패".to_string()),
    }

    let mapped = slice.get_mapped_range();
    let mut out = vec![0u8; (dst_w as usize) * (dst_h as usize) * 4];
    let row_len = unpadded_bpr as usize;
    let padded_len = padded_bpr as usize;
    for y in 0..(dst_h as usize) {
        let src_start = y * padded_len;
        let src_end = src_start + row_len;
        let dst_start = y * row_len;
        let dst_end = dst_start + row_len;
        out[dst_start..dst_end].copy_from_slice(&mapped[src_start..src_end]);
    }
    drop(mapped);
    output_buffer.unmap();

    Ok(out)
}

fn guess_mime_from_path(path: &Path) -> &'static str {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "tif" | "tiff" => "image/tiff",
        _ => "application/octet-stream",
    }
}

fn build_display_data_url(path: &Path) -> Option<String> {
    let bytes = fs::read(path).ok()?;
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();

    // WebView2(Chromium) DOES NOT support TIFF/BMP natively.
    // Convert them to PNG for display preview.
    if matches!(ext.as_str(), "tif" | "tiff" | "bmp") {
        if let Ok(img) = image::load_from_memory(&bytes) {
            // Thumbnail for preview performance
            let preview = img.thumbnail(1200, 1200);
            let mut buffer = Cursor::new(Vec::new());
            if preview.write_to(&mut buffer, image::ImageFormat::Png).is_ok() {
                let encoded = BASE64_STANDARD.encode(buffer.into_inner());
                return Some(format!("data:image/png;base64,{}", encoded));
            }
        }
    }

    let encoded = BASE64_STANDARD.encode(bytes);
    Some(format!(
        "data:{};base64,{}",
        guess_mime_from_path(path),
        encoded
    ))
}

fn compute_ncc_similarity(a: &image::GrayImage, b: &image::GrayImage) -> f32 {
    let n = (a.width() as usize) * (a.height() as usize);
    if n == 0 || a.dimensions() != b.dimensions() {
        return 0.0;
    }

    let sum_a: f32 = a.pixels().map(|p| p[0] as f32).sum();
    let sum_b: f32 = b.pixels().map(|p| p[0] as f32).sum();
    let mean_a = sum_a / n as f32;
    let mean_b = sum_b / n as f32;

    let mut num = 0.0_f32;
    let mut den_a = 0.0_f32;
    let mut den_b = 0.0_f32;

    for (pa, pb) in a.pixels().zip(b.pixels()) {
        let da = pa[0] as f32 - mean_a;
        let db = pb[0] as f32 - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }

    if den_a <= f32::EPSILON || den_b <= f32::EPSILON {
        return 0.0;
    }

    let ncc = num / (den_a.sqrt() * den_b.sqrt());
    ((ncc + 1.0) * 0.5).clamp(0.0, 1.0)
}

fn compute_hash_similarity(a: &image::DynamicImage, b: &image::DynamicImage) -> f32 {
    let hasher = HasherConfig::new()
        .hash_alg(HashAlg::Gradient)
        .hash_size(8, 8)
        .to_hasher();

    let a_luma = a.to_luma8();
    let b_luma = b.to_luma8();
    let a_hash_img = match HashGrayImage::from_raw(a_luma.width(), a_luma.height(), a_luma.into_raw()) {
        Some(img) => img,
        None => return 0.0,
    };
    let b_hash_img = match HashGrayImage::from_raw(b_luma.width(), b_luma.height(), b_luma.into_raw()) {
        Some(img) => img,
        None => return 0.0,
    };

    let ah = hasher.hash_image(&a_hash_img);
    let bh = hasher.hash_image(&b_hash_img);
    let bit_len = (ah.as_bytes().len() * 8) as f32;
    if bit_len <= 0.0 {
        return 0.0;
    }
    let dist = ah.dist(&bh) as f32;
    (1.0 - (dist / bit_len)).clamp(0.0, 1.0)
}

fn compute_dssim_similarity(a: &image::DynamicImage, b: &image::DynamicImage) -> Option<f32> {
    let dssim = Dssim::new();
    let a_rgba = a.to_rgba8();
    let b_rgba = b.to_rgba8();
    let (w, h) = a_rgba.dimensions();
    if b_rgba.dimensions() != (w, h) || w == 0 || h == 0 {
        return None;
    }

    let a_pixels = a_rgba.as_raw().as_rgba();
    let b_pixels = b_rgba.as_raw().as_rgba();
    let a_img = dssim.create_image_rgba(a_pixels, w as usize, h as usize)?;
    let b_img = dssim.create_image_rgba(b_pixels, w as usize, h as usize)?;
    let (dssim_value, _) = dssim.compare(&a_img, b_img);
    let dssim_scalar: f64 = dssim_value.into();
    let similarity = (1.0_f64 / (1.0_f64 + dssim_scalar.max(0.0))) as f32;
    Some(similarity.clamp(0.0, 1.0))
}

fn verify_image_pair_sync(source_path: String, dest_path: String) -> Result<ImageVerificationResult, String> {
    let source_raw = load_image_with_fallback(Path::new(&source_path))
        .map_err(|e| format!("원본 이미지 로드 실패: {}", e))?;
    let dest_raw = load_image_with_fallback(Path::new(&dest_path))
        .map_err(|e| format!("대상 이미지 로드 실패: {}", e))?;

    // 검증은 표시 기준과 동일하게 EXIF Orientation을 정규화한 픽셀 기준으로 수행
    let source_orientation = read_exif_orientation(Path::new(&source_path));
    let dest_orientation = read_exif_orientation(Path::new(&dest_path));
    let source = normalize_dynamic_image_orientation(source_raw, source_orientation);
    let dest = normalize_dynamic_image_orientation(dest_raw, dest_orientation);

    // 비율/스케일 역시 정규화된 픽셀 기준으로 판정
    let (sw, sh) = source.dimensions();
    let (dw, dh) = dest.dimensions();

    if dw == 0 || dh == 0 {
        return Err("대상 이미지 해상도가 유효하지 않습니다.".to_string());
    }
    if sw == 0 || sh == 0 {
        return Err("원본 이미지 해상도가 유효하지 않습니다.".to_string());
    }

    const HASH_EARLY_PASS: f32 = 0.985;
    const HASH_EARLY_FAIL: f32 = 0.70;

    let best_transform = "identity";
    let dest_small = dest.resize_exact(256, 256, FilterType::Triangle);
    let dest_luma = dest_small.to_luma8();
    let source_small = source.resize_exact(256, 256, FilterType::Triangle);
    let source_luma = source_small.to_luma8();
    let hash_score = compute_hash_similarity(&source_small, &dest_small);

    let source_ar = sw as f32 / sh as f32;
    let dest_ar = dw as f32 / dh as f32;
    let aspect_ratio_delta = if source_ar > f32::EPSILON {
        ((dest_ar - source_ar).abs() / source_ar).clamp(0.0, 10.0)
    } else {
        1.0
    };

    let sx = dw as f32 / sw as f32;
    let sy = dh as f32 / sh as f32;
    let scale_ratio_delta = if sx.max(sy) > f32::EPSILON {
        ((sx - sy).abs() / sx.max(sy)).clamp(0.0, 10.0)
    } else {
        1.0
    };

    let orientation_issue = false;
    let aspect_issue = aspect_ratio_delta > 0.02 || scale_ratio_delta > 0.02;
    let ncc_score = compute_ncc_similarity(&source_luma, &dest_luma);

    let (best_score, _best_dssim, message) = if aspect_issue {
        (
            (hash_score * 0.6 + ncc_score * 0.4).clamp(0.0, 1.0),
            ncc_score,
            "실패: 비율 왜곡 또는 비정상 리사이즈 가능성이 있습니다.".to_string(),
        )
    } else if hash_score >= HASH_EARLY_PASS {
        (
            hash_score,
            hash_score,
            format!(
                "검증 통과(빠른 판정): 해시 유사도 {:.1}%로 매우 높아 조기 종료했습니다.",
                hash_score * 100.0
            ),
        )
    } else if hash_score <= HASH_EARLY_FAIL {
        (
            hash_score,
            hash_score,
            format!(
                "실패(빠른 판정): 해시 유사도 {:.1}%로 매우 낮아 조기 종료했습니다.",
                hash_score * 100.0
            ),
        )
    } else {
        let dssim_score = compute_dssim_similarity(&source_small, &dest_small).unwrap_or(ncc_score);
        let combined = (hash_score * 0.35) + (dssim_score * 0.45) + (ncc_score * 0.20);
        let verdict_preview = if combined < 0.88 {
            "실패"
        } else if combined < 0.94 {
            "주의"
        } else {
            "검증 통과"
        };
        (
            combined,
            dssim_score,
            format!(
                "{}: 유사도 {:.1}% (해시 {:.1}%, 구조 {:.1}%)",
                verdict_preview,
                combined * 100.0,
                hash_score * 100.0,
                dssim_score * 100.0
            ),
        )
    };

    let verdict = if orientation_issue || aspect_issue || best_score < 0.88 {
        "fail"
    } else if best_score < 0.94 {
        "warn"
    } else {
        "pass"
    };

    Ok(ImageVerificationResult {
        similarity: best_score.max(0.0),
        verdict: verdict.to_string(),
        best_transform: best_transform.to_string(),
        orientation_issue,
        aspect_issue,
        aspect_ratio_delta,
        scale_ratio_delta,
        source_width: sw,
        source_height: sh,
        dest_width: dw,
        dest_height: dh,
        message,
    })
}

fn scan_folder_sync(path: String) -> Result<Vec<ImageListItem>, String> {
    let base_path = PathBuf::from(path);
    if !base_path.exists() {
        return Err(format!("존재하지 않는 경로입니다: {}", base_path.display()));
    }
    if !base_path.is_dir() {
        return Err(format!("폴더 경로가 아닙니다: {}", base_path.display()));
    }

    let mut files = Vec::new();
    walk_image_files(&base_path, &base_path, &mut files)?;
    files.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
    Ok(files)
}

fn get_image_details_sync(path: String) -> Result<ImageDetails, String> {
    let file_path = PathBuf::from(path);
    if !file_path.exists() || !file_path.is_file() {
        return Err(format!("유효한 파일이 아닙니다: {}", file_path.display()));
    }

    let (dpi_x, dpi_y) = read_exif_dpi(&file_path);
    let size_bytes = fs::metadata(&file_path).ok().map(|m| m.len());

    let img = match load_image_with_fallback(&file_path) {
        Ok(img) => img,
        Err(e) => {
            return Ok(ImageDetails {
                display_data_url: build_display_data_url(&file_path),
                size_bytes,
                width: None,
                height: None,
                dpi_x,
                dpi_y,
                color: None,
                error: Some(e),
            });
        }
    };

    let (width, height) = match read_image_dimensions(&file_path) {
        (Some(w), Some(h)) => (w, h),
        _ => img.dimensions(),
    };
    let color = Some(format!("{:?}", img.color()));

    Ok(ImageDetails {
        display_data_url: build_display_data_url(&file_path),
        size_bytes,
        width: Some(width),
        height: Some(height),
        dpi_x,
        dpi_y,
        color,
        error: None,
    })
}

fn compute_required_scale(file: &MigrationTaskFile, criteria: OptimizationCriteria) -> Option<OptimizationPlan> {
    let mut required_scales: Vec<f32> = Vec::new();
    let mut apply_target_dpi = false;

    if criteria.use_dpi {
        let dpi_candidates = [file.dpi_x, file.dpi_y];
        let dpi_exceeded = dpi_candidates
            .iter()
            .flatten()
            .any(|dpi| *dpi > criteria.target_dpi);

        if dpi_exceeded {
            apply_target_dpi = true;
            let sx = file.dpi_x.map(|x| criteria.target_dpi / x);
            let sy = file.dpi_y.map(|y| criteria.target_dpi / y);
            let dpi_scale = match (sx, sy) {
                (Some(x), Some(y)) => x.min(y),
                (Some(x), None) => x,
                (None, Some(y)) => y,
                (None, None) => 1.0,
            };
            required_scales.push(dpi_scale);
        }
    }

    if criteria.use_max_width {
        if let Some(width) = file.width {
            if width > criteria.max_width {
                required_scales.push(criteria.max_width as f32 / width as f32);
            }
        }
    }

    if criteria.use_max_height {
        if let Some(height) = file.height {
            if height > criteria.max_height {
                required_scales.push(criteria.max_height as f32 / height as f32);
            }
        }
    }

    if required_scales.is_empty() {
        return None;
    }

    let scale = required_scales
        .into_iter()
        .fold(1.0_f32, |acc, value| acc.min(value))
        .clamp(0.01, 1.0);

    if scale >= 1.0 {
        None
    } else {
        Some(OptimizationPlan {
            scale,
            apply_target_dpi,
        })
    }
}

fn normalize_slimg_image_orientation(
    image: SlimgImageData,
    orientation: Option<u16>,
) -> Result<SlimgImageData, String> {
    let rgba = image::RgbaImage::from_raw(image.width, image.height, image.data)
        .ok_or_else(|| "RGBA 버퍼를 이미지로 변환할 수 없습니다.".to_string())?;
    let dynamic = image::DynamicImage::ImageRgba8(rgba);

    let oriented = match orientation.unwrap_or(1) {
        2 => dynamic.fliph(),
        3 => dynamic.rotate180(),
        4 => dynamic.flipv(),
        5 => dynamic.rotate90().fliph(),
        6 => dynamic.rotate90(),
        7 => dynamic.rotate90().flipv(),
        8 => dynamic.rotate270(),
        _ => dynamic,
    };

    let rgba_out = oriented.to_rgba8();
    Ok(SlimgImageData::new(
        rgba_out.width(),
        rgba_out.height(),
        rgba_out.into_raw(),
    ))
}

fn normalize_dynamic_image_orientation(
    image: image::DynamicImage,
    orientation: Option<u16>,
) -> image::DynamicImage {
    match orientation.unwrap_or(1) {
        2 => image.fliph(),
        3 => image.rotate180(),
        4 => image.flipv(),
        5 => image.rotate90().fliph(),
        6 => image.rotate90(),
        7 => image.rotate90().flipv(),
        8 => image.rotate270(),
        _ => image,
    }
}

fn copy_metadata_best_effort(
    source_path: &Path,
    dest_path: &Path,
    set_target_dpi: Option<u32>,
) -> String {
    let detected = detect_real_image_format(source_path);
    let mut metadata = match LittleExifMetadata::new_from_path(source_path) {
        Ok(v) => v,
        Err(e) => return format!("메타데이터 복원 일부 실패: 원본 EXIF 읽기 실패 ({})", e),
    };

    // 본문 픽셀은 방향 정규화해서 저장하므로 EXIF Orientation도 1로 맞춘다.
    metadata.set_tag(LittleExifTag::Orientation(vec![1]));

    // DPI 기준 최적화 활성 시, 대상 DPI를 명시적으로 고정 저장한다.
    if let Some(target_dpi) = set_target_dpi {
        let r = LittleExifUR64::from(target_dpi);
        metadata.set_tag(LittleExifTag::XResolution(vec![r.clone()]));
        metadata.set_tag(LittleExifTag::YResolution(vec![r]));
        metadata.set_tag(LittleExifTag::ResolutionUnit(vec![2]));
    }

    if metadata.write_to_file(dest_path).is_ok() {
        return "메타데이터 복원 완료".to_string();
    }

    // little_exif 실패 시 JPEG/PNG/WebP는 기존 EXIF 복사 방식으로 폴백
    let source_bytes = match fs::read(source_path) {
        Ok(v) => v,
        Err(e) => return format!("메타데이터 복원 실패(원본 읽기): {}", e),
    };
    let dest_bytes = match fs::read(dest_path) {
        Ok(v) => v,
        Err(e) => return format!("메타데이터 복원 실패(대상 읽기): {}", e),
    };

    let fallback_result = match detected {
        Some(RealImageFormat::Jpeg) => {
            let src = img_parts::jpeg::Jpeg::from_bytes(Bytes::from(source_bytes));
            let dst = img_parts::jpeg::Jpeg::from_bytes(Bytes::from(dest_bytes));
            match (src, dst) {
                (Ok(src_img), Ok(mut dst_img)) => {
                    dst_img.set_exif(src_img.exif());
                    let mut encoded = Vec::new();
                    dst_img
                        .encoder()
                        .write_to(&mut encoded)
                        .and_then(|_| fs::write(dest_path, encoded))
                        .map_err(|e| e.to_string())
                }
                (Err(e), _) | (_, Err(e)) => Err(e.to_string()),
            }
        }
        Some(RealImageFormat::Png) => {
            let src = img_parts::png::Png::from_bytes(Bytes::from(source_bytes));
            let dst = img_parts::png::Png::from_bytes(Bytes::from(dest_bytes));
            match (src, dst) {
                (Ok(src_img), Ok(mut dst_img)) => {
                    dst_img.set_exif(src_img.exif());
                    let mut encoded = Vec::new();
                    dst_img
                        .encoder()
                        .write_to(&mut encoded)
                        .and_then(|_| fs::write(dest_path, encoded))
                        .map_err(|e| e.to_string())
                }
                (Err(e), _) | (_, Err(e)) => Err(e.to_string()),
            }
        }
        Some(RealImageFormat::WebP) => {
            let src = img_parts::webp::WebP::from_bytes(Bytes::from(source_bytes));
            let dst = img_parts::webp::WebP::from_bytes(Bytes::from(dest_bytes));
            match (src, dst) {
                (Ok(src_img), Ok(mut dst_img)) => {
                    dst_img.set_exif(src_img.exif());
                    let mut encoded = Vec::new();
                    dst_img
                        .encoder()
                        .write_to(&mut encoded)
                        .and_then(|_| fs::write(dest_path, encoded))
                        .map_err(|e| e.to_string())
                }
                (Err(e), _) | (_, Err(e)) => Err(e.to_string()),
            }
        }
        _ => Err("메타데이터 복원 생략(포맷 미지원)".to_string()),
    };

    match fallback_result {
        Ok(_) => "메타데이터 복원 완료(폴백)".to_string(),
        Err(e) if e.is_empty() => "메타데이터 복원 일부 실패".to_string(),
        Err(e) => format!("메타데이터 복원 일부 실패: {}", e),
    }
}

struct ProcessSuccess {
    action: &'static str,
    source_size_bytes: u64,
    dest_size_bytes: u64,
    message: String,
    fallback_code: Option<String>,
}

struct WorkerUpdate {
    item: MigrationProgressItemUpdate,
}

enum ProcessOutcome {
    Success(ProcessSuccess),
    Failed(String),
    Cancelled,
}

fn classify_gpu_fallback_code(error: &str) -> &'static str {
    if error.contains("텍스처 한계 초과") {
        "LIMIT"
    } else if error.contains("어댑터") || error.contains("디바이스 생성") {
        "INIT_FAIL"
    } else {
        "RUNTIME_FAIL"
    }
}

fn should_use_gpu_in_auto_mode(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> bool {
    const AUTO_GPU_MIN_PIXELS: u64 = 2_000_000;
    const AUTO_GPU_MIN_WORK_DELTA: f32 = 0.08;
    let src_px = (src_w as u64).saturating_mul(src_h as u64);
    let dst_px = (dst_w as u64).saturating_mul(dst_h as u64);
    let max_px = src_px.max(dst_px);
    if max_px < AUTO_GPU_MIN_PIXELS {
        return false;
    }

    if src_px == 0 {
        return false;
    }
    let scale = ((dst_px as f64) / (src_px as f64)).sqrt() as f32;
    (1.0 - scale).abs() >= AUTO_GPU_MIN_WORK_DELTA
}

fn run_slimg_resize_with_cancel(
    file: &MigrationTaskFile,
    plan: OptimizationPlan,
    cancel_flag: &AtomicBool,
    criteria: OptimizationCriteria,
    migration_options: MigrationOptions,
) -> ProcessOutcome {
    if cancel_flag.load(Ordering::Relaxed) {
        return ProcessOutcome::Cancelled;
    }

    let real_format = Some(file.real_format);
    if matches!(real_format, Some(RealImageFormat::Bmp) | Some(RealImageFormat::Tiff)) {
        let source_size_bytes = fs::metadata(&file.source_path)
            .map(|m| m.len())
            .unwrap_or(0);

        let image = match load_image_with_fallback(&file.source_path) {
            Ok(v) => v,
            Err(e) => {
                return ProcessOutcome::Failed(format!(
                    "이미지 디코드 실패: {} ({})",
                    file.relative_path, e
                ));
            }
        };

        if cancel_flag.load(Ordering::Relaxed) {
            return ProcessOutcome::Cancelled;
        }

        let oriented = normalize_dynamic_image_orientation(image, file.orientation);
        let new_w = ((oriented.width() as f32) * plan.scale).round().max(1.0) as u32;
        let new_h = ((oriented.height() as f32) * plan.scale).round().max(1.0) as u32;
        let resized = oriented.resize_exact(new_w, new_h, FilterType::Lanczos3);

        let output = if matches!(real_format, Some(RealImageFormat::Tiff)) {
            image::ImageFormat::Tiff
        } else {
            image::ImageFormat::Bmp
        };

        if let Err(e) = resized.save_with_format(&file.dest_path, output) {
            return ProcessOutcome::Failed(format!(
                "이미지 저장 실패: {} ({})",
                file.dest_path.display(),
                e
            ));
        }

        let metadata_message = if migration_options.restore_metadata {
            copy_metadata_best_effort(&file.source_path, &file.dest_path, if plan.apply_target_dpi {
                Some(criteria.target_dpi as u32)
            } else {
                None
            })
        } else {
            "메타데이터 복원 비활성화".to_string()
        };
        let optimized_size_bytes = fs::metadata(&file.dest_path)
            .map(|m| m.len())
            .unwrap_or(0);
        if optimized_size_bytes > source_size_bytes {
            if let Err(e) = fs::copy(&file.source_path, &file.dest_path) {
                return ProcessOutcome::Failed(format!(
                    "원본 복원 실패(최적화 결과가 더 큼): {} -> {} ({})",
                    file.source_path.display(),
                    file.dest_path.display(),
                    e
                ));
            }
            let restored_size_bytes = fs::metadata(&file.dest_path)
                .map(|m| m.len())
                .unwrap_or(source_size_bytes);
            return ProcessOutcome::Success(ProcessSuccess {
                action: "skipped",
                source_size_bytes,
                dest_size_bytes: restored_size_bytes,
                message: format!(
                    "{} 최적화 결과가 원본보다 커서 원본으로 복원 후 스킵 처리",
                    if matches!(real_format, Some(RealImageFormat::Tiff)) {
                        "TIFF"
                    } else {
                        "BMP"
                    }
                ),
                fallback_code: None,
            });
        }

        return ProcessOutcome::Success(ProcessSuccess {
            action: "optimized",
            source_size_bytes,
            dest_size_bytes: optimized_size_bytes,
            message: format!("{} 최적화 완료(방향 정규화 적용) / {}",
              if matches!(real_format, Some(RealImageFormat::Tiff)) { "TIFF" } else { "BMP" },
              metadata_message),
            fallback_code: None,
        });
    }

    let source_size_bytes = fs::metadata(&file.source_path)
        .map(|m| m.len())
        .unwrap_or(0);
    let (decoded, decoded_format) = match decode_file(&file.source_path) {
        Ok(value) => value,
        Err(e) => {
            return ProcessOutcome::Failed(format!(
                "slimg-core 디코드 실패: {} ({})",
                file.relative_path, e
            ));
        }
    };

    if cancel_flag.load(Ordering::Relaxed) {
        return ProcessOutcome::Cancelled;
    }

    let normalized = match normalize_slimg_image_orientation(decoded, file.orientation) {
        Ok(image) => image,
        Err(e) => {
            return ProcessOutcome::Failed(format!(
                "방향 정규화 실패: {} ({})",
                file.relative_path, e
            ));
        }
    };

    if cancel_flag.load(Ordering::Relaxed) {
        return ProcessOutcome::Cancelled;
    }

    let pipeline_options = PipelineOptions {
        format: decoded_format,
        quality: 100,
        resize: None,
        crop: None,
        extend: None,
        fill_color: None,
    };

    let new_w = ((normalized.width as f32) * plan.scale).round().max(1.0) as u32;
    let new_h = ((normalized.height as f32) * plan.scale).round().max(1.0) as u32;
    let effective_mode = match migration_options.acceleration_mode {
        AccelerationMode::Cpu => AccelerationMode::Cpu,
        AccelerationMode::GpuPreferred => AccelerationMode::GpuPreferred,
        AccelerationMode::Auto => {
            if get_gpu_resize_context().is_err() {
                AccelerationMode::Cpu
            } else if should_use_gpu_in_auto_mode(normalized.width, normalized.height, new_w, new_h) {
                AccelerationMode::GpuPreferred
            } else {
                AccelerationMode::Cpu
            }
        }
    };

    let (preprocessed, accel_note, accel_fallback_code) = match effective_mode {
        AccelerationMode::Cpu => {
            match slimg_resize(&normalized, &ResizeMode::Scale(plan.scale as f64)) {
                Ok(img) => {
                    let note = if matches!(migration_options.acceleration_mode, AccelerationMode::Auto) {
                        "가속: Auto->CPU".to_string()
                    } else {
                        "가속: CPU".to_string()
                    };
                    (img, note, None)
                }
                Err(_) => {
                    let note = if matches!(migration_options.acceleration_mode, AccelerationMode::Auto) {
                        "가속: Auto->CPU(리사이즈 실패로 원본 유지)".to_string()
                    } else {
                        "가속: CPU(리사이즈 실패로 원본 유지)".to_string()
                    };
                    (normalized.clone(), note, None)
                }
            }
        },
        _ => match panic::catch_unwind(AssertUnwindSafe(|| {
            gpu_resize_rgba_wgpu(&normalized.data, normalized.width, normalized.height, new_w, new_h)
        })) {
            Ok(Ok(rgba)) => (
                SlimgImageData::new(new_w, new_h, rgba),
                if matches!(migration_options.acceleration_mode, AccelerationMode::Auto) {
                    "가속: Auto->GPU(wgpu)".to_string()
                } else {
                    "가속: GPU(wgpu)".to_string()
                },
                None,
            ),
            Ok(Err(e)) => {
                let resized =
                    slimg_resize(&normalized, &ResizeMode::Scale(plan.scale as f64)).unwrap_or_else(|_| normalized.clone());
                (
                    resized,
                    if matches!(migration_options.acceleration_mode, AccelerationMode::Auto) {
                        format!("가속: Auto(GPU) 실패 -> CPU 폴백 ({})", e)
                    } else {
                        format!("가속: GPU 실패 -> CPU 폴백 ({})", e)
                    },
                    Some(classify_gpu_fallback_code(&e).to_string()),
                )
            }
            Err(_) => {
                let resized =
                    slimg_resize(&normalized, &ResizeMode::Scale(plan.scale as f64)).unwrap_or_else(|_| normalized.clone());
                (
                    resized,
                    if matches!(migration_options.acceleration_mode, AccelerationMode::Auto) {
                        "가속: Auto(GPU) 패닉 -> CPU 폴백".to_string()
                    } else {
                        "가속: GPU 패닉 -> CPU 폴백".to_string()
                    },
                    Some("PANIC".to_string()),
                )
            }
        },
    };

    let result = match convert(&preprocessed, &pipeline_options) {
        Ok(value) => value,
        Err(e) => {
            return ProcessOutcome::Failed(format!(
                "slimg-core 변환 실패: {} ({})",
                file.relative_path, e
            ));
        }
    };

    if cancel_flag.load(Ordering::Relaxed) {
        return ProcessOutcome::Cancelled;
    }

    if let Err(e) = result.save(&file.dest_path) {
        return ProcessOutcome::Failed(format!(
            "대상 저장 실패: {} ({})",
            file.dest_path.display(),
            e
        ));
    }
    let optimized_size_bytes = fs::metadata(&file.dest_path)
        .map(|m| m.len())
        .unwrap_or(0);
    if optimized_size_bytes > source_size_bytes {
        if let Err(e) = fs::copy(&file.source_path, &file.dest_path) {
            return ProcessOutcome::Failed(format!(
                "원본 복원 실패(최적화 결과가 더 큼): {} -> {} ({})",
                file.source_path.display(),
                file.dest_path.display(),
                e
            ));
        }
        let restored_size_bytes = fs::metadata(&file.dest_path)
            .map(|m| m.len())
            .unwrap_or(source_size_bytes);
        return ProcessOutcome::Success(ProcessSuccess {
            action: "skipped",
            source_size_bytes,
            dest_size_bytes: restored_size_bytes,
            message: "최적화 결과가 원본보다 커서 원본으로 복원 후 스킵 처리".to_string(),
            fallback_code: None,
        });
    }

    let metadata_message = if migration_options.restore_metadata {
        copy_metadata_best_effort(
            &file.source_path,
            &file.dest_path,
            if plan.apply_target_dpi {
                Some(criteria.target_dpi as u32)
            } else {
                None
            },
        )
    } else {
        "메타데이터 복원 비활성화".to_string()
    };

    ProcessOutcome::Success(ProcessSuccess {
        action: "optimized",
        source_size_bytes,
        dest_size_bytes: optimized_size_bytes,
        message: format!("최적화 완료(방향 정규화 적용) / {} / {}", metadata_message, accel_note),
        fallback_code: accel_fallback_code,
    })
}

fn process_single_migration_file(
    file: &MigrationTaskFile,
    cancel_flag: &AtomicBool,
    criteria: OptimizationCriteria,
    options: MigrationOptions,
) -> ProcessOutcome {
    if cancel_flag.load(Ordering::Relaxed) {
        return ProcessOutcome::Cancelled;
    }

    if !matches!(
        file.real_format,
        RealImageFormat::Jpeg
            | RealImageFormat::Png
            | RealImageFormat::WebP
            | RealImageFormat::Avif
            | RealImageFormat::Qoi
            | RealImageFormat::Bmp
            | RealImageFormat::Tiff
    ) {
        return ProcessOutcome::Failed(format!(
            "마이그레이션 대상 포맷이 아닙니다: {}",
            file.relative_path
        ));
    }

    if let Some(parent) = file.dest_path.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            return ProcessOutcome::Failed(format!(
                "대상 폴더 생성 실패: {} ({})",
                parent.display(),
                e
            ));
        }
    }

    let source_size_bytes = fs::metadata(&file.source_path).map(|m| m.len()).unwrap_or(0);

    if let Some(plan) = compute_required_scale(file, criteria) {
        return run_slimg_resize_with_cancel(file, plan, cancel_flag, criteria, options);
    }

    if cancel_flag.load(Ordering::Relaxed) {
        return ProcessOutcome::Cancelled;
    }

    if let Err(e) = fs::copy(&file.source_path, &file.dest_path) {
        return ProcessOutcome::Failed(format!(
            "파일 복사 실패: {} -> {} ({})",
            file.source_path.display(),
            file.dest_path.display(),
            e
        ));
    }

    let dest_size_bytes = fs::metadata(&file.dest_path).map(|m| m.len()).unwrap_or(0);
    ProcessOutcome::Success(ProcessSuccess {
        action: "skipped",
        source_size_bytes,
        dest_size_bytes,
        message: "기준 미초과로 복사 처리".to_string(),
        fallback_code: None,
    })
}

fn emit_migration_progress(app: &tauri::AppHandle, event: MigrationProgressEvent) {
    let _ = app.emit("migration-progress", event);
}

fn emit_migration_progress_batch(app: &tauri::AppHandle, event: MigrationProgressBatchEvent) {
    let _ = app.emit("migration-progress-batch", event);
}

fn get_concurrency_profile_sync() -> ConcurrencyProfile {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let min = 1usize;
    let max = cores.max(1);
    let default_value = cores.saturating_sub(2).max(1).min(max);
    ConcurrencyProfile {
        cpu_cores: cores,
        min,
        max,
        default_value,
    }
}

fn run_migration_sync(
    app: tauri::AppHandle,
    source_path: String,
    dest_path: String,
    cancel_flag: Arc<AtomicBool>,
    concurrency_limit_override: Option<usize>,
    criteria: OptimizationCriteria,
    options: MigrationOptions,
) -> Result<(), String> {
    let source_base = PathBuf::from(source_path);
    let dest_base = PathBuf::from(dest_path);

    if !source_base.exists() || !source_base.is_dir() {
        return Err(format!("유효한 원본 폴더가 아닙니다: {}", source_base.display()));
    }
    if !dest_base.exists() {
        fs::create_dir_all(&dest_base)
            .map_err(|e| format!("대상 폴더 생성 실패: {} ({})", dest_base.display(), e))?;
    }
    if !dest_base.is_dir() {
        return Err(format!("유효한 대상 폴더가 아닙니다: {}", dest_base.display()));
    }

    let (resolved_backend, backend_notice) = resolve_acceleration_backend(options.acceleration_mode);
    let mut files = Vec::new();
    walk_migration_files(&source_base, &dest_base, &source_base, &mut files)?;
    files.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));

    let total = files.len();

    emit_migration_progress(
        &app,
        MigrationProgressEvent {
            total,
            processed: 0,
            succeeded: 0,
            failed: 0,
            message: if total == 0 {
                "처리할 파일이 없습니다.".to_string()
            } else {
                match backend_notice {
                    Some(ref notice) => {
                        format!(
                            "총 {}개 파일 처리 시작 / 가속 모드: {} / 백엔드: {} ({})",
                            total,
                            acceleration_mode_label(options.acceleration_mode),
                            resolved_backend,
                            notice
                        )
                    }
                    None => format!(
                        "총 {}개 파일 처리 시작 / 가속 모드: {} / 백엔드: {}",
                        total,
                        acceleration_mode_label(options.acceleration_mode),
                        resolved_backend
                    ),
                }
            },
            current_relative_path: None,
            current_action: None,
            current_source_size_bytes: None,
            current_dest_size_bytes: None,
            done: total == 0,
            canceled: false,
        },
    );

    if total == 0 {
        let _ = app.emit(
            "migration-done",
            MigrationProgressEvent {
                total,
                processed: 0,
                succeeded: 0,
                failed: 0,
                message: "처리할 파일이 없습니다.".to_string(),
                current_relative_path: None,
                current_action: None,
                current_source_size_bytes: None,
                current_dest_size_bytes: None,
                done: true,
                canceled: false,
            },
        );
        return Ok(());
    }

    let profile = get_concurrency_profile_sync();
    let selected_limit = concurrency_limit_override
        .unwrap_or(profile.default_value)
        .clamp(profile.min, profile.max);
    let (update_tx, update_rx) = channel::unbounded::<WorkerUpdate>();
    let (task_tx, task_rx) = channel::unbounded::<MigrationTaskFile>();
    for file in files {
        let _ = task_tx.send(file);
    }
    drop(task_tx);

    let mut workers = Vec::new();
    for _ in 0..selected_limit.max(1) {
        let rx = task_rx.clone();
        let tx = update_tx.clone();
        let cancel_for_worker = cancel_flag.clone();
        let worker = std::thread::spawn(move || {
            loop {
                if cancel_for_worker.load(Ordering::Relaxed) {
                    break;
                }

                let file = match rx.recv() {
                    Ok(file) => file,
                    Err(_) => break,
                };

                if cancel_for_worker.load(Ordering::Relaxed) {
                    break;
                }

                let update = match process_single_migration_file(&file, &cancel_for_worker, criteria, options) {
                    ProcessOutcome::Success(success) => Some(WorkerUpdate {
                        item: MigrationProgressItemUpdate {
                            relative_path: file.relative_path.clone(),
                            action: success.action.to_string(),
                            source_size_bytes: Some(success.source_size_bytes),
                            dest_size_bytes: Some(success.dest_size_bytes),
                            message: success.message,
                            fallback_code: success.fallback_code,
                        },
                    }),
                    ProcessOutcome::Failed(err) => Some(WorkerUpdate {
                        item: MigrationProgressItemUpdate {
                            relative_path: file.relative_path.clone(),
                            action: "failed".to_string(),
                            source_size_bytes: fs::metadata(&file.source_path).ok().map(|m| m.len()),
                            dest_size_bytes: None,
                            message: err,
                            fallback_code: None,
                        },
                    }),
                    ProcessOutcome::Cancelled => None,
                };

                if let Some(update) = update {
                    let _ = tx.send(update);
                }
            }
        });
        workers.push(worker);
    }
    drop(update_tx);

    let mut final_processed = 0usize;
    let mut final_succeeded = 0usize;
    let mut final_failed = 0usize;
    let mut pending_updates: Vec<MigrationProgressItemUpdate> = Vec::new();
    let mut last_emit_at = Instant::now();
    const PROGRESS_EMIT_MAX_ITEMS: usize = 32;
    const PROGRESS_EMIT_MAX_DELAY_MS: u64 = 120;

    while let Ok(update) = update_rx.recv() {
        final_processed += 1;
        if update.item.action == "failed" {
            final_failed += 1;
        } else {
            final_succeeded += 1;
        }
        pending_updates.push(update.item);

        let should_emit = pending_updates.len() >= PROGRESS_EMIT_MAX_ITEMS
            || last_emit_at.elapsed() >= Duration::from_millis(PROGRESS_EMIT_MAX_DELAY_MS);
        if should_emit {
            let message = pending_updates
                .last()
                .map(|v| v.message.clone())
                .unwrap_or_else(|| format!("{}/{} 처리 중", final_processed, total));
            emit_migration_progress_batch(
                &app,
                MigrationProgressBatchEvent {
                    total,
                    processed: final_processed,
                    succeeded: final_succeeded,
                    failed: final_failed,
                    message,
                    updates: std::mem::take(&mut pending_updates),
                    done: false,
                    canceled: false,
                },
            );
            last_emit_at = Instant::now();
        }
    }

    if !pending_updates.is_empty() {
        let message = pending_updates
            .last()
            .map(|v| v.message.clone())
            .unwrap_or_else(|| format!("{}/{} 처리 중", final_processed, total));
        emit_migration_progress_batch(
            &app,
            MigrationProgressBatchEvent {
                total,
                processed: final_processed,
                succeeded: final_succeeded,
                failed: final_failed,
                message,
                updates: std::mem::take(&mut pending_updates),
                done: false,
                canceled: false,
            },
        );
    }

    let mut worker_panic_count = 0usize;
    for worker in workers {
        if worker.join().is_err() {
            worker_panic_count += 1;
        }
    }
    let is_canceled = cancel_flag.load(Ordering::Relaxed);
    let incomplete_count = total.saturating_sub(final_processed);
    let abnormal_finish = !is_canceled && (worker_panic_count > 0 || incomplete_count > 0);
    let final_failed_for_done = if abnormal_finish {
        final_failed.saturating_add(incomplete_count)
    } else {
        final_failed
    };

    let _ = app.emit(
        "migration-done",
        MigrationProgressEvent {
            total,
            processed: final_processed,
            succeeded: final_succeeded,
            failed: final_failed_for_done,
            message: if is_canceled {
                "사용자 요청으로 마이그레이션이 취소되었습니다.".to_string()
            } else if abnormal_finish {
                format!(
                    "마이그레이션이 비정상 종료되었습니다. 처리 {} / {} (미처리: {}, 워커 패닉: {})",
                    final_processed, total, incomplete_count, worker_panic_count
                )
            } else {
                "마이그레이션이 완료되었습니다.".to_string()
            },
            current_relative_path: None,
            current_action: None,
            current_source_size_bytes: None,
            current_dest_size_bytes: None,
            done: true,
            canceled: is_canceled,
        },
    );

    Ok(())
}

#[tauri::command]
async fn scan_folder(path: String) -> Result<Vec<ImageListItem>, String> {
    tauri::async_runtime::spawn_blocking(move || scan_folder_sync(path))
        .await
        .map_err(|e| format!("scan_folder 작업 조인 실패: {}", e))?
}

#[tauri::command]
async fn get_image_details(path: String) -> Result<ImageDetails, String> {
    tauri::async_runtime::spawn_blocking(move || get_image_details_sync(path))
        .await
        .map_err(|e| format!("get_image_details 작업 조인 실패: {}", e))?
}

#[tauri::command]
async fn get_destination_image_details(
    dest_base_path: String,
    relative_path: String,
) -> Result<Option<ImageDetails>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let dest_path = PathBuf::from(dest_base_path).join(Path::new(&relative_path));
        if !dest_path.exists() || !dest_path.is_file() {
            return Ok(None);
        }
        let details = get_image_details_sync(dest_path.to_string_lossy().to_string())?;
        Ok(Some(details))
    })
    .await
    .map_err(|e| format!("get_destination_image_details 작업 조인 실패: {}", e))?
}

#[tauri::command]
async fn verify_image_pair(
    source_path: String,
    dest_path: String,
) -> Result<ImageVerificationResult, String> {
    tauri::async_runtime::spawn_blocking(move || verify_image_pair_sync(source_path, dest_path))
        .await
        .map_err(|e| format!("verify_image_pair 작업 조인 실패: {}", e))?
}

#[tauri::command]
fn get_concurrency_profile() -> ConcurrencyProfile {
    get_concurrency_profile_sync()
}

#[tauri::command]
async fn start_migration(
    app: tauri::AppHandle,
    source_path: String,
    dest_path: String,
    concurrency_limit: Option<usize>,
    use_dpi: Option<bool>,
    target_dpi: Option<u32>,
    use_max_width: Option<bool>,
    max_width: Option<u32>,
    use_max_height: Option<bool>,
    max_height: Option<u32>,
    restore_metadata: Option<bool>,
    acceleration_mode: Option<AccelerationModeArg>,
) -> Result<(), String> {
    let criteria = OptimizationCriteria {
        use_dpi: use_dpi.unwrap_or(true),
        target_dpi: target_dpi.unwrap_or(300).clamp(72, 1200) as f32,
        use_max_width: use_max_width.unwrap_or(false),
        max_width: max_width.unwrap_or(4000).clamp(64, 20000),
        use_max_height: use_max_height.unwrap_or(false),
        max_height: max_height.unwrap_or(4000).clamp(64, 20000),
    };

    if !criteria.use_dpi && !criteria.use_max_width && !criteria.use_max_height {
        return Err("최적화 기준을 하나 이상 선택하세요.".to_string());
    }

    let options = MigrationOptions {
        restore_metadata: restore_metadata.unwrap_or(true),
        acceleration_mode: acceleration_mode
            .map(AccelerationMode::from)
            .unwrap_or(AccelerationMode::Auto),
    };

    let cancel_flag = {
        let mut state = migration_state()
            .lock()
            .map_err(|_| "마이그레이션 상태 잠금 실패".to_string())?;

        if state.running {
            return Err("이미 마이그레이션이 진행 중입니다.".to_string());
        }

        let flag = Arc::new(AtomicBool::new(false));
        state.running = true;
        state.cancel_flag = Some(flag.clone());
        flag
    };

    tauri::async_runtime::spawn(async move {
        let app_for_work = app.clone();
        let result = tauri::async_runtime::spawn_blocking(move || {
            run_migration_sync(
                app_for_work,
                source_path,
                dest_path,
                cancel_flag,
                concurrency_limit,
                criteria,
                options,
            )
        })
        .await;

        match result {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                let _ = app.emit(
                    "migration-done",
                    MigrationProgressEvent {
                        total: 0,
                        processed: 0,
                        succeeded: 0,
                        failed: 0,
                        message: format!("마이그레이션 실패: {}", err),
                        current_relative_path: None,
                        current_action: None,
                        current_source_size_bytes: None,
                        current_dest_size_bytes: None,
                        done: true,
                        canceled: false,
                    },
                );
            }
            Err(join_err) => {
                let _ = app.emit(
                    "migration-done",
                    MigrationProgressEvent {
                        total: 0,
                        processed: 0,
                        succeeded: 0,
                        failed: 0,
                        message: format!("마이그레이션 작업 실패: {}", join_err),
                        current_relative_path: None,
                        current_action: None,
                        current_source_size_bytes: None,
                        current_dest_size_bytes: None,
                        done: true,
                        canceled: false,
                    },
                );
            }
        }

        if let Ok(mut state) = migration_state().lock() {
            state.running = false;
            state.cancel_flag = None;
        }
    });

    Ok(())
}

#[tauri::command]
fn cancel_migration() -> Result<(), String> {
    let state = migration_state()
        .lock()
        .map_err(|_| "마이그레이션 상태 잠금 실패".to_string())?;
    if let Some(flag) = &state.cancel_flag {
        flag.store(true, Ordering::Relaxed);
        return Ok(());
    }
    Err("진행 중인 마이그레이션이 없습니다.".to_string())
}

#[tauri::command]
fn migration_running() -> bool {
    if let Ok(state) = migration_state().lock() {
        return state.running;
    }
    false
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            scan_folder,
            get_image_details,
            get_destination_image_details,
            verify_image_pair,
            get_concurrency_profile,
            start_migration,
            cancel_migration,
            migration_running
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
