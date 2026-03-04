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
    format: Option<String>,
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
    adapter_name: String,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ColorMode {
    Monochrome, // 1-bit black & white
    Grayscale,  // 8-bit gray
    Rgb,        // 24-bit
    Rgba,       // 32-bit
}

#[derive(Clone, Copy)]
struct ColorCriteria {
    enabled: bool,
    target_mode: ColorMode,
}

#[derive(Clone, Copy)]
struct OptimizationCriteria {
    use_dpi: bool,
    target_dpi: f32,
    use_max_width: bool,
    max_width: u32,
    use_max_height: bool,
    max_height: u32,
    color: ColorCriteria,
}

#[derive(Clone, Copy)]
struct MigrationOptions {
    restore_metadata: bool,
    acceleration_mode: AccelerationMode,
    encode_quality: u8,
}

#[derive(Clone, Copy)]
struct OptimizationPlan {
    scale: f32,
    apply_target_dpi: bool,
    /// Resolved color transform (already analyzed against actual pixel data).
    /// `None` means no color transform, OR color needs lazy analysis (see `pending_color`).
    color_transform: Option<ColorMode>,
    /// If `Some`, color analysis is deferred — run `analyze_color_mode` then apply if result <= target.
    pending_color: Option<ColorMode>,
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
            Ok(ctx) => ("auto", Some(format!("Auto 모드: 파일별 CPU/GPU 동적 선택 [{}]", ctx.adapter_name))),
            Err(e) => ("auto", Some(format!("Auto 모드: GPU 사용 불가로 CPU 사용 ({})", e))),
        },
        AccelerationMode::GpuPreferred => match get_gpu_resize_context() {
            Ok(ctx) => ("gpu", Some(format!("GPU 리사이즈 활성화 [{}]", ctx.adapter_name))),
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
        let (orientation, _, _) = read_exif_info(&path);
        let (dpi_x, dpi_y) = read_exif_dpi(&path);
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
    let (_, exif_x, exif_y) = read_exif_info(path);
    // kamadak-exif 로 DPI 를 얻은 경우 그대로 반환
    if exif_x.is_some() && exif_y.is_some() {
        return (exif_x, exif_y);
    }
    // EXIF 가 없거나 부분적인 경우 포맷별 헤더에서 직접 읽어 보충
    // (PNG pHYs, JPEG JFIF APP0, TIFF IFD, BMP XPelsPerMeter 등)
    let (hdr_x, hdr_y) = read_dpi_from_file_header(path);
    (exif_x.or(hdr_x), exif_y.or(hdr_y))
}

fn read_exif_orientation(path: &Path) -> Option<u16> {
    let (orientation, _, _) = read_exif_info(path);
    orientation
}

fn read_image_dimensions(path: &Path) -> (Option<u32>, Option<u32>) {
    let orientation = read_exif_orientation(path);
    read_image_dimensions_with_orientation(path, orientation)
}

// ── 파일 헤더에서 컬러 정보 읽기 ───────────────────────────────────────────

fn read_color_png(bytes: &[u8]) -> Option<String> {
    // PNG: signature(8) + IHDR chunk: Length(4) + Type("IHDR")(4) + Width(4) + Height(4) + BitDepth(1) + ColorType(1)
    if bytes.len() < 26 { return None; }
    if &bytes[0..8] != b"\x89PNG\r\n\x1a\n" { return None; }
    if &bytes[12..16] != b"IHDR" { return None; }
    let bit_depth = bytes[24];
    let color_type = bytes[25];
    let desc = match color_type {
        0 => match bit_depth { // Grayscale
            1 => "흑백 (1비트)",
            2 => "흑백 (2비트)",
            4 => "흑백 (4비트)",
            _ => "흑백 (Grayscale)",
        },
        2 => "RGB",           // Truecolor
        3 => match bit_depth { // Indexed-colour
            1 => "팔레트 (1비트)",
            2 => "팔레트 (2비트)",
            4 => "팔레트 (4비트)",
            _ => "팔레트 (인덱스)",
        },
        4 => "Grayscale+Alpha",
        6 => "RGBA",
        _ => return None,
    };
    Some(desc.to_string())
}

fn read_color_jpeg(bytes: &[u8]) -> Option<String> {
    // Scan JPEG segments for SOF marker (FFCx, not FFC4/FFC8/FFCC)
    let mut pos = 0;
    while pos + 1 < bytes.len() {
        if bytes[pos] != 0xFF { return None; }
        let marker = bytes[pos + 1];
        pos += 2;
        // Markers without a length field
        if marker == 0xD8 || marker == 0xD9 || marker == 0x01 || (0xD0..=0xD7).contains(&marker) {
            continue;
        }
        if pos + 2 > bytes.len() { break; }
        let seg_len = u16::from_be_bytes([bytes[pos], bytes[pos + 1]]) as usize;
        if seg_len < 2 { break; }
        // SOF markers: C0–CF except C4(DHT), C8(JPEG), CC(DAC)
        if (0xC0..=0xCF).contains(&marker) && !matches!(marker, 0xC4 | 0xC8 | 0xCC) {
            // SOF: length(2) + precision(1) + height(2) + width(2) + Nf(1)
            if pos + 8 > bytes.len() { break; }
            let precision = bytes[pos + 2];
            let nf = bytes[pos + 7];
            let desc = match nf {
                1 => format!("흑백 (Grayscale, {}비트)", precision),
                3 => format!("RGB ({}비트)", precision),
                4 => format!("CMYK ({}비트)", precision),
                n => format!("{} 채널 ({}비트)", n, precision),
            };
            return Some(desc);
        }
        pos += seg_len;
    }
    None
}

fn read_color_tiff(bytes: &[u8]) -> Option<String> {
    if bytes.len() < 8 { return None; }
    let is_le = match &bytes[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let read_u16 = |off: usize| -> Option<u16> {
        if off + 2 > bytes.len() { return None; }
        Some(if is_le { u16::from_le_bytes([bytes[off], bytes[off+1]]) }
             else     { u16::from_be_bytes([bytes[off], bytes[off+1]]) })
    };
    let read_u32 = |off: usize| -> Option<u32> {
        if off + 4 > bytes.len() { return None; }
        Some(if is_le { u32::from_le_bytes([bytes[off], bytes[off+1], bytes[off+2], bytes[off+3]]) }
             else     { u32::from_be_bytes([bytes[off], bytes[off+1], bytes[off+2], bytes[off+3]]) })
    };
    // Read a SHORT or LONG scalar from an IFD entry's value field (or offset)
    let read_scalar = |typ: u16, count: u32, val_off: usize| -> Option<u16> {
        let type_size: usize = match typ { 3 => 2, 4 => 4, _ => return None };
        let data_off = if (count as usize) * type_size <= 4 { val_off }
                       else { read_u32(val_off)? as usize };
        match typ {
            3 => read_u16(data_off),
            4 => read_u32(data_off).map(|v| v as u16),
            _ => None,
        }
    };

    let ifd_off = read_u32(4)? as usize;
    if ifd_off + 2 > bytes.len() { return None; }
    let num_entries = read_u16(ifd_off)? as usize;

    let mut photometric: Option<u16> = None;
    let mut bits_per_sample: u16 = 8;
    let mut samples_per_pixel: u16 = 1;

    for i in 0..num_entries {
        let entry_off = ifd_off + 2 + i * 12;
        if entry_off + 12 > bytes.len() { break; }
        let tag   = read_u16(entry_off)?;
        let typ   = read_u16(entry_off + 2)?;
        let count = read_u32(entry_off + 4)?;
        let val_off = entry_off + 8;
        match tag {
            258 => { // BitsPerSample (read first element)
                if let Some(v) = read_scalar(typ, count, val_off) { bits_per_sample = v; }
            }
            262 => { // PhotometricInterpretation
                photometric = read_scalar(typ, count, val_off);
            }
            277 => { // SamplesPerPixel
                if let Some(v) = read_scalar(typ, count, val_off) { samples_per_pixel = v; }
            }
            _ => {}
        }
    }

    let photometric = photometric?;
    let desc = match photometric {
        0 | 1 => match bits_per_sample { // WhiteIsZero / BlackIsZero
            1 => "흑백 (1비트)".to_string(),
            _ => format!("흑백 (Grayscale, {}비트)", bits_per_sample),
        },
        2 => match samples_per_pixel { // RGB
            3 => format!("RGB ({}비트)", bits_per_sample),
            4 => format!("RGBA ({}비트)", bits_per_sample),
            n => format!("RGB {} 채널 ({}비트)", n, bits_per_sample),
        },
        3 => "팔레트 (인덱스)".to_string(),
        4 => "투명 마스크".to_string(),
        5 => "CMYK".to_string(),
        6 => format!("YCbCr ({}비트)", bits_per_sample),
        _ => format!("Photometric {} ({}비트)", photometric, bits_per_sample),
    };
    Some(desc)
}

fn read_color_bmp(bytes: &[u8]) -> Option<String> {
    if bytes.len() < 30 { return None; }
    if &bytes[0..2] != b"BM" { return None; }
    // DIB header size at offset 14 determines header variant
    let dib_size = u32::from_le_bytes([bytes[14], bytes[15], bytes[16], bytes[17]]);
    let bit_count = if dib_size == 12 {
        // BITMAPCOREHEADER: bit count at offset 24
        if bytes.len() < 26 { return None; }
        u16::from_le_bytes([bytes[24], bytes[25]])
    } else {
        // BITMAPINFOHEADER and later: bit count at offset 28
        u16::from_le_bytes([bytes[28], bytes[29]])
    };
    let desc = match bit_count {
        1  => "흑백 (1비트)",
        4  => "팔레트 (4비트)",
        8  => "팔레트 (8비트)",
        16 => "RGB (16비트)",
        24 => "RGB (24비트)",
        32 => "RGBA (32비트)",
        n  => return Some(format!("{}비트", n)),
    };
    Some(desc.to_string())
}

fn read_color_webp(bytes: &[u8]) -> Option<String> {
    if bytes.len() < 12 { return None; }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WEBP" { return None; }
    let mut pos = 12usize;
    while pos + 8 <= bytes.len() {
        let chunk_type = &bytes[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([bytes[pos+4], bytes[pos+5], bytes[pos+6], bytes[pos+7]]) as usize;
        if chunk_type == b"VP8 " {
            return Some("RGB (VP8 손실)".to_string());
        } else if chunk_type == b"VP8L" {
            return Some("RGBA (VP8L 무손실)".to_string());
        } else if chunk_type == b"VP8X" {
            // Extended: flags at pos+8, alpha flag = bit 4 (0x10)
            if pos + 9 < bytes.len() {
                let flags = bytes[pos + 8];
                return Some(if flags & 0x10 != 0 {
                    "RGBA (WebP 확장)".to_string()
                } else {
                    "RGB (WebP 확장)".to_string()
                });
            }
        }
        // Chunks are padded to 2-byte boundary
        let advance = 8 + chunk_size + (chunk_size & 1);
        if advance == 0 { break; }
        pos += advance;
    }
    None
}

fn read_color_qoi(bytes: &[u8]) -> Option<String> {
    // QOI header: magic(4) + width(4) + height(4) + channels(1) + colorspace(1) = 14 bytes
    if bytes.len() < 14 { return None; }
    if &bytes[0..4] != b"qoif" { return None; }
    let desc = match bytes[12] {
        3 => "RGB",
        4 => "RGBA",
        _ => return None,
    };
    Some(desc.to_string())
}

/// 파일 헤더에서 직접 컬러 정보를 읽어 사람이 읽기 좋은 문자열로 반환.
/// 포맷은 magic bytes로 감지하므로 파일 확장자와 무관하게 동작.
fn read_color_from_file_header(path: &Path) -> Option<String> {
    let format = detect_real_image_format(path)?;
    // 파일을 미리 읽어 두고 각 파서에 슬라이스를 넘김
    let bytes = fs::read(path).ok()?;
    match format {
        RealImageFormat::Png      => read_color_png(&bytes),
        RealImageFormat::Jpeg     => read_color_jpeg(&bytes),
        RealImageFormat::Tiff     => read_color_tiff(&bytes),
        RealImageFormat::Bmp      => read_color_bmp(&bytes),
        RealImageFormat::Gif      => Some("팔레트 (인덱스)".to_string()),
        RealImageFormat::WebP     => read_color_webp(&bytes),
        RealImageFormat::Qoi      => read_color_qoi(&bytes),
        // JXL / AVIF / HEIC: 헤더 구조가 복잡하므로 None 반환 (호출 측에서 폴백 처리)
        RealImageFormat::Jxl | RealImageFormat::Avif | RealImageFormat::HeicLike => None,
    }
}

// ── 파일 헤더에서 DPI 정보 읽기 ─────────────────────────────────────────────

fn read_dpi_png(bytes: &[u8]) -> (Option<f32>, Option<f32>) {
    // pHYs 청크: X(4) + Y(4) + Unit(1), unit=1 이면 pixels per metre
    if bytes.len() < 8 || &bytes[0..8] != b"\x89PNG\r\n\x1a\n" {
        return (None, None);
    }
    let mut pos = 8usize;
    while pos + 12 <= bytes.len() {
        let length = u32::from_be_bytes([bytes[pos], bytes[pos+1], bytes[pos+2], bytes[pos+3]]) as usize;
        let ctype  = &bytes[pos+4..pos+8];
        if ctype == b"pHYs" && pos + 17 <= bytes.len() {
            let xppu = u32::from_be_bytes([bytes[pos+8], bytes[pos+9], bytes[pos+10], bytes[pos+11]]);
            let yppu = u32::from_be_bytes([bytes[pos+12], bytes[pos+13], bytes[pos+14], bytes[pos+15]]);
            let unit = bytes[pos+16];
            if unit == 1 && xppu > 0 && yppu > 0 {
                // 1 inch = 0.0254 m  →  DPI = ppm × 0.0254
                return (Some(xppu as f32 * 0.0254), Some(yppu as f32 * 0.0254));
            }
            break; // pHYs 발견했으나 단위 없음
        }
        // pHYs 는 IDAT 이전에 있어야 하므로 IDAT/IEND 이후는 불필요
        if ctype == b"IDAT" || ctype == b"IEND" { break; }
        pos += 8 + length + 4; // chunk header(8) + data + CRC(4)
    }
    (None, None)
}

fn read_dpi_jpeg_jfif(bytes: &[u8]) -> (Option<f32>, Option<f32>) {
    // JFIF APP0 (FF E0): identifier(5) + ver(2) + units(1) + Xdensity(2) + Ydensity(2)
    if bytes.len() < 4 || &bytes[0..3] != b"\xFF\xD8\xFF" { return (None, None); }
    let mut pos = 2usize;
    while pos + 1 < bytes.len() {
        if bytes[pos] != 0xFF { break; }
        let marker = bytes[pos + 1];
        pos += 2;
        if marker == 0xD8 || marker == 0xD9 || marker == 0x01 || (0xD0..=0xD7).contains(&marker) {
            continue;
        }
        if pos + 2 > bytes.len() { break; }
        let seg_len = u16::from_be_bytes([bytes[pos], bytes[pos+1]]) as usize;
        if seg_len < 2 { break; }
        if marker == 0xE0 && seg_len >= 14 && pos + seg_len <= bytes.len() {
            if &bytes[pos+2..pos+7] == b"JFIF\x00" {
                let units    = bytes[pos+9];
                let xdensity = u16::from_be_bytes([bytes[pos+10], bytes[pos+11]]) as f32;
                let ydensity = u16::from_be_bytes([bytes[pos+12], bytes[pos+13]]) as f32;
                if xdensity > 0.0 && ydensity > 0.0 {
                    return match units {
                        1 => (Some(xdensity), Some(ydensity)),                // DPI
                        2 => (Some(xdensity * 2.54), Some(ydensity * 2.54)), // DPCM → DPI
                        _ => (None, None), // 0 = 비율만, 절대 DPI 없음
                    };
                }
            }
        }
        if marker == 0xDA { break; } // SOS — 이후는 이미지 데이터
        pos += seg_len;
    }
    (None, None)
}

fn read_dpi_tiff_ifd(bytes: &[u8]) -> (Option<f32>, Option<f32>) {
    // ? 연산자를 사용하기 위해 Option 반환 내부 함수로 위임
    fn inner(bytes: &[u8]) -> Option<(Option<f32>, Option<f32>)> {
        if bytes.len() < 8 { return None; }
        let is_le = match &bytes[0..2] {
            b"II" => true,
            b"MM" => false,
            _ => return None,
        };
        let ru16 = |off: usize| -> Option<u16> {
            if off + 2 > bytes.len() { return None; }
            Some(if is_le { u16::from_le_bytes([bytes[off], bytes[off+1]]) }
                 else     { u16::from_be_bytes([bytes[off], bytes[off+1]]) })
        };
        let ru32 = |off: usize| -> Option<u32> {
            if off + 4 > bytes.len() { return None; }
            Some(if is_le { u32::from_le_bytes([bytes[off], bytes[off+1], bytes[off+2], bytes[off+3]]) }
                 else     { u32::from_be_bytes([bytes[off], bytes[off+1], bytes[off+2], bytes[off+3]]) })
        };
        let rational = |off: usize| -> Option<f32> {
            let n = ru32(off)?;
            let d = ru32(off + 4)?;
            if d == 0 { None } else { Some(n as f32 / d as f32) }
        };

        let ifd_off = ru32(4)? as usize;
        if ifd_off + 2 > bytes.len() { return None; }
        let nentries = ru16(ifd_off)? as usize;

        let mut xres: Option<f32> = None;
        let mut yres: Option<f32> = None;
        let mut unit: u16 = 2; // 기본값: inch

        for i in 0..nentries {
            let eoff = ifd_off + 2 + i * 12;
            if eoff + 12 > bytes.len() { break; }
            let tag = match ru16(eoff) { Some(v) => v, None => break };
            let typ = match ru16(eoff + 2) { Some(v) => v, None => break };
            let val_off = eoff + 8;
            match (tag, typ) {
                (282, 5) => { // XResolution, RATIONAL (8 bytes → 항상 오프셋 포인터)
                    if let Some(data_off) = ru32(val_off).map(|v| v as usize) {
                        xres = rational(data_off);
                    }
                }
                (283, 5) => { // YResolution, RATIONAL
                    if let Some(data_off) = ru32(val_off).map(|v| v as usize) {
                        yres = rational(data_off);
                    }
                }
                (296, 3) => { // ResolutionUnit, SHORT
                    if let Some(u) = ru16(val_off) { unit = u; }
                }
                _ => {}
            }
        }

        let convert = |v: f32| if unit == 3 { v * 2.54 } else { v }; // cm → inch
        Some((xres.map(convert), yres.map(convert)))
    }
    inner(bytes).unwrap_or((None, None))
}

fn read_dpi_bmp(bytes: &[u8]) -> (Option<f32>, Option<f32>) {
    // BITMAPINFOHEADER(40+): XPelsPerMeter(offset 38), YPelsPerMeter(offset 42)
    if bytes.len() < 46 || &bytes[0..2] != b"BM" { return (None, None); }
    let dib_size = u32::from_le_bytes([bytes[14], bytes[15], bytes[16], bytes[17]]);
    if dib_size < 40 { return (None, None); } // BITMAPCOREHEADER 는 DPI 필드 없음
    let x_ppm = i32::from_le_bytes([bytes[38], bytes[39], bytes[40], bytes[41]]);
    let y_ppm = i32::from_le_bytes([bytes[42], bytes[43], bytes[44], bytes[45]]);
    if x_ppm > 0 && y_ppm > 0 {
        // pixels per metre → DPI
        (Some(x_ppm as f32 * 0.0254), Some(y_ppm as f32 * 0.0254))
    } else {
        (None, None)
    }
}

/// 파일 헤더에서 DPI를 읽어 반환. 포맷은 magic bytes로 감지.
/// kamadak-exif 가 읽지 못하는 PNG pHYs, JPEG JFIF APP0, BMP XPelsPerMeter 등을 커버.
fn read_dpi_from_file_header(path: &Path) -> (Option<f32>, Option<f32>) {
    let format = match detect_real_image_format(path) {
        Some(f) => f,
        None => return (None, None),
    };
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(_) => return (None, None),
    };
    match format {
        RealImageFormat::Png  => read_dpi_png(&bytes),
        RealImageFormat::Jpeg => read_dpi_jpeg_jfif(&bytes),
        RealImageFormat::Tiff => read_dpi_tiff_ifd(&bytes),
        RealImageFormat::Bmp  => read_dpi_bmp(&bytes),
        _ => (None, None),
    }
}

/// FillOrder=2(LSB-first) TIFF를 수정: 스트립 데이터 비트 반전 + FillOrder 태그를 1로 패치.
/// `image` 크레이트의 tiff 디코더(fax 크레이트)가 MSB-first만 지원하기 때문에 필요.
fn try_fix_tiff_lsb_fill_order(bytes: &[u8]) -> Option<Vec<u8>> {
    if bytes.len() < 8 {
        return None;
    }
    let is_le = match &bytes[..2] {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };

    let read_u16 = |b: &[u8], off: usize| -> Option<u16> {
        if off + 2 > b.len() { return None; }
        Some(if is_le { u16::from_le_bytes([b[off], b[off+1]]) }
             else     { u16::from_be_bytes([b[off], b[off+1]]) })
    };
    let read_u32 = |b: &[u8], off: usize| -> Option<u32> {
        if off + 4 > b.len() { return None; }
        Some(if is_le { u32::from_le_bytes([b[off], b[off+1], b[off+2], b[off+3]]) }
             else     { u32::from_be_bytes([b[off], b[off+1], b[off+2], b[off+3]]) })
    };

    // IFD 내 태그 값(SHORT 또는 LONG 배열)을 usize 벡터로 읽는 헬퍼
    let read_values = |b: &[u8], typ: u16, count: usize, val_or_off: usize| -> Option<Vec<usize>> {
        let type_size: usize = match typ { 3 => 2, 4 => 4, _ => return None };
        let total_size = count * type_size;
        let data_off = if total_size <= 4 {
            val_or_off
        } else {
            read_u32(b, val_or_off)? as usize
        };
        if data_off + total_size > b.len() { return None; }
        let mut vals = Vec::with_capacity(count);
        for i in 0..count {
            let off = data_off + i * type_size;
            let v = match typ {
                3 => read_u16(b, off)? as usize,
                4 => read_u32(b, off)? as usize,
                _ => unreachable!(),
            };
            vals.push(v);
        }
        Some(vals)
    };

    let ifd_offset = read_u32(bytes, 4)? as usize;
    if ifd_offset + 2 > bytes.len() { return None; }
    let num_entries = read_u16(bytes, ifd_offset)? as usize;

    let mut fill_order_val_off: Option<usize> = None;
    let mut strip_offsets: Vec<usize> = Vec::new();
    let mut strip_byte_counts: Vec<usize> = Vec::new();

    for i in 0..num_entries {
        let entry_off = ifd_offset + 2 + i * 12;
        if entry_off + 12 > bytes.len() { break; }
        let tag   = read_u16(bytes, entry_off)?;
        let typ   = read_u16(bytes, entry_off + 2)?;
        let count = read_u32(bytes, entry_off + 4)? as usize;
        let val_or_off = entry_off + 8;
        match tag {
            266 => { // FillOrder
                if typ == 3 {
                    let val = read_u16(bytes, val_or_off)?;
                    if val != 2 { return None; } // LSB-first가 아니면 패치 불필요
                    fill_order_val_off = Some(val_or_off);
                }
            }
            273 => { // StripOffsets
                strip_offsets = read_values(bytes, typ, count, val_or_off)?;
            }
            279 => { // StripByteCounts
                strip_byte_counts = read_values(bytes, typ, count, val_or_off)?;
            }
            _ => {}
        }
    }

    let fill_order_val_off = fill_order_val_off?; // FillOrder=2 없으면 Nothing to do
    if strip_offsets.is_empty() || strip_offsets.len() != strip_byte_counts.len() {
        return None;
    }

    let mut result = bytes.to_vec();

    // FillOrder 태그를 1(MSB-first)로 패치
    if is_le { result[fill_order_val_off] = 1; result[fill_order_val_off + 1] = 0; }
    else      { result[fill_order_val_off] = 0; result[fill_order_val_off + 1] = 1; }

    // 각 스트립 데이터의 바이트별 비트 순서 반전
    for (&off, &len) in strip_offsets.iter().zip(strip_byte_counts.iter()) {
        let end = off + len;
        if end > result.len() { return None; }
        for b in &mut result[off..end] {
            *b = b.reverse_bits();
        }
    }

    Some(result)
}

fn is_palette_1bit_source(path: &Path) -> bool {
    read_color_from_file_header(path)
        .map(|v| v.contains("팔레트 (1비트)"))
        .unwrap_or(false)
}

fn read_png_phys_chunk(bytes: &[u8]) -> Option<(u32, u32, u8)> {
    if bytes.len() < 8 || &bytes[0..8] != b"\x89PNG\r\n\x1a\n" {
        return None;
    }
    let mut pos = 8usize;
    while pos + 12 <= bytes.len() {
        let length = u32::from_be_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]) as usize;
        if pos + 12 + length > bytes.len() {
            return None;
        }
        let ctype = &bytes[pos + 4..pos + 8];
        if ctype == b"pHYs" && length == 9 {
            let xppu = u32::from_be_bytes([bytes[pos + 8], bytes[pos + 9], bytes[pos + 10], bytes[pos + 11]]);
            let yppu = u32::from_be_bytes([bytes[pos + 12], bytes[pos + 13], bytes[pos + 14], bytes[pos + 15]]);
            let unit = bytes[pos + 16];
            return Some((xppu, yppu, unit));
        }
        if ctype == b"IEND" {
            break;
        }
        pos += 12 + length;
    }
    None
}

fn crc32_png(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg() & 0xEDB8_8320;
            crc = (crc >> 1) ^ mask;
        }
    }
    !crc
}

fn build_png_phys_chunk(xppu: u32, yppu: u32, unit: u8) -> Vec<u8> {
    let mut chunk = Vec::with_capacity(4 + 4 + 9 + 4);
    chunk.extend_from_slice(&9u32.to_be_bytes());
    chunk.extend_from_slice(b"pHYs");
    chunk.extend_from_slice(&xppu.to_be_bytes());
    chunk.extend_from_slice(&yppu.to_be_bytes());
    chunk.push(unit);
    let mut crc_input = Vec::with_capacity(4 + 9);
    crc_input.extend_from_slice(b"pHYs");
    crc_input.extend_from_slice(&xppu.to_be_bytes());
    crc_input.extend_from_slice(&yppu.to_be_bytes());
    crc_input.push(unit);
    let crc = crc32_png(&crc_input);
    chunk.extend_from_slice(&crc.to_be_bytes());
    chunk
}

fn upsert_png_phys_chunk(bytes: &[u8], xppu: u32, yppu: u32, unit: u8) -> Result<Vec<u8>, String> {
    if bytes.len() < 8 || &bytes[0..8] != b"\x89PNG\r\n\x1a\n" {
        return Err("PNG 시그니처가 아닙니다.".to_string());
    }

    let phys_chunk = build_png_phys_chunk(xppu, yppu, unit);
    let mut out = Vec::with_capacity(bytes.len() + phys_chunk.len());
    out.extend_from_slice(&bytes[0..8]);

    let mut pos = 8usize;
    let mut inserted = false;
    while pos + 12 <= bytes.len() {
        let length = u32::from_be_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]) as usize;
        let chunk_end = pos + 12 + length;
        if chunk_end > bytes.len() {
            return Err("PNG 청크 길이가 잘못되었습니다.".to_string());
        }
        let ctype = &bytes[pos + 4..pos + 8];

        if ctype == b"pHYs" {
            if !inserted {
                out.extend_from_slice(&phys_chunk);
                inserted = true;
            }
            pos = chunk_end;
            continue;
        }

        if !inserted && (ctype == b"IDAT" || ctype == b"IEND") {
            out.extend_from_slice(&phys_chunk);
            inserted = true;
        }

        out.extend_from_slice(&bytes[pos..chunk_end]);
        pos = chunk_end;
        if ctype == b"IEND" {
            break;
        }
    }

    if !inserted {
        out.extend_from_slice(&phys_chunk);
    }

    Ok(out)
}

/// TIFF IFD에서 PhotometricInterpretation 값만 빠르게 읽어 Palette(3) 여부 반환.
/// 파일을 전부 읽지 않고 magic prefix만으로 판별한다.
fn is_palette_tiff(bytes: &[u8]) -> bool {
    if bytes.len() < 8 { return false; }
    let le = match &bytes[0..2] { b"II" => true, b"MM" => false, _ => return false };
    let r16 = |o: usize| -> u16 {
        let s = match bytes.get(o..o+2) { Some(s) => s, None => return 0 };
        if le { u16::from_le_bytes([s[0],s[1]]) } else { u16::from_be_bytes([s[0],s[1]]) }
    };
    let r32 = |o: usize| -> u32 {
        let s = match bytes.get(o..o+4) { Some(s) => s, None => return 0 };
        if le { u32::from_le_bytes([s[0],s[1],s[2],s[3]]) } else { u32::from_be_bytes([s[0],s[1],s[2],s[3]]) }
    };
    if r16(2) != 42 { return false; }
    let ifd = r32(4) as usize;
    let n = r16(ifd) as usize;
    for i in 0..n {
        let e = ifd + 2 + i * 12;
        if r16(e) == 262 { // PhotometricInterpretation
            let ftype = r16(e + 2);
            let voff = e + 8;
            let val = if ftype == 3 { r16(voff) as u32 } else { r32(voff) };
            return val == 3;
        }
    }
    false
}

/// Palette/Indexed color TIFF (PhotometricInterpretation=3)를 수동으로 디코딩하여 RGB DynamicImage로 변환.
/// `image`/`tiff` crate가 RGBPalette를 지원하지 않으므로 TIFF IFD를 직접 파싱하고
/// weezl로 LZW 압축을 해제한 뒤 ColorMap으로 RGB 변환한다.
/// `max_size`: Some(n)이면 인덱스 버퍼를 n px 이내로 다운샘플 후 변환 (썸네일용 고속 경로).
fn try_decode_palette_tiff(bytes: &[u8]) -> Option<image::DynamicImage> {
    try_decode_palette_tiff_inner(bytes, None)
}

fn try_decode_palette_tiff_inner(bytes: &[u8], max_size: Option<u32>) -> Option<image::DynamicImage> {
    // ── TIFF 헤더 ──────────────────────────────────────────────────────────
    if bytes.len() < 8 { return None; }
    let little_endian = match &bytes[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };

    let read_u16 = |off: usize| -> Option<u16> {
        let s = bytes.get(off..off+2)?;
        Some(if little_endian {
            u16::from_le_bytes([s[0], s[1]])
        } else {
            u16::from_be_bytes([s[0], s[1]])
        })
    };
    let read_u32 = |off: usize| -> Option<u32> {
        let s = bytes.get(off..off+4)?;
        Some(if little_endian {
            u32::from_le_bytes([s[0], s[1], s[2], s[3]])
        } else {
            u32::from_be_bytes([s[0], s[1], s[2], s[3]])
        })
    };

    // magic
    if read_u16(2)? != 42 { return None; }
    let ifd_offset = read_u32(4)? as usize;

    // ── IFD 파싱 ──────────────────────────────────────────────────────────
    let num_entries = read_u16(ifd_offset)? as usize;

    let mut width: u32 = 0;
    let mut height: u32 = 0;
    let mut bits_per_sample: u16 = 8;
    let mut compression: u16 = 1;
    let mut photometric: u16 = 0;
    let mut strip_offsets: Vec<u32> = Vec::new();
    let mut strip_byte_counts: Vec<u32> = Vec::new();
    let mut rows_per_strip: u32 = u32::MAX;
    let mut colormap_offset: usize = 0;
    let mut colormap_count: usize = 0;
    let mut predictor: u16 = 1;

    for i in 0..num_entries {
        let entry_off = ifd_offset + 2 + i * 12;
        let tag   = read_u16(entry_off)?;
        let ftype = read_u16(entry_off + 2)?;
        let count = read_u32(entry_off + 4)? as usize;
        let voff  = entry_off + 8;

        // value_or_offset: count*typesize <= 4 이면 value 자체, 아니면 offset
        let type_size: usize = match ftype { 1|2|6|7 => 1, 3|8 => 2, 4|9|11 => 4, 5|10|12 => 8, _ => 1 };
        let inline = count * type_size <= 4;

        let read_u16_val = |idx: usize| -> Option<u16> {
            if inline {
                read_u16(voff + idx * 2)
            } else {
                let base = read_u32(voff)? as usize;
                read_u16(base + idx * 2)
            }
        };
        let read_u32_val = |idx: usize| -> Option<u32> {
            if inline {
                // SHORT(3)가 인라인이면 u16 -> u32
                if ftype == 3 { Some(read_u16(voff + idx * 2)? as u32) }
                else { read_u32(voff + idx * 4) }
            } else {
                let base = read_u32(voff)? as usize;
                if ftype == 3 { Some(read_u16(base + idx * 2)? as u32) }
                else { read_u32(base + idx * 4) }
            }
        };

        match tag {
            256 => width        = read_u32_val(0)?,
            257 => height       = read_u32_val(0)?,
            258 => bits_per_sample = read_u16_val(0)?,
            259 => compression  = read_u16_val(0)?,
            262 => photometric  = read_u16_val(0)?,
            273 => { // StripOffsets
                for j in 0..count { strip_offsets.push(read_u32_val(j)?); }
            }
            278 => rows_per_strip = read_u32_val(0)?,
            317 => predictor = read_u16_val(0)?,
            279 => { // StripByteCounts
                for j in 0..count { strip_byte_counts.push(read_u32_val(j)?); }
            }
            320 => { // ColorMap
                colormap_count = count;
                colormap_offset = if inline { voff } else { read_u32(voff)? as usize };
            }
            _ => {}
        }
    }

    // Palette TIFF 조건 검증
    if photometric != 3 { return None; }                // RGBPalette만 처리
    if bits_per_sample != 8 { return None; }            // 8비트 인덱스만 지원
    if width == 0 || height == 0 { return None; }
    if colormap_count == 0 || colormap_count % 3 != 0 { return None; }
    if strip_offsets.is_empty() || strip_offsets.len() != strip_byte_counts.len() { return None; }

    // ── ColorMap 읽기 ──────────────────────────────────────────────────────
    // ColorMap은 SHORT(u16) 배열: R[0..n], G[0..n], B[0..n]
    let palette_size = colormap_count / 3;
    let mut colormap = vec![0u16; colormap_count];
    for i in 0..colormap_count {
        colormap[i] = read_u16(colormap_offset + i * 2)?;
    }

    // ── 각 Strip 디코딩 ────────────────────────────────────────────────────
    let expected_pixels = (width * height) as usize;
    let mut indices: Vec<u8> = Vec::with_capacity(expected_pixels);

    for (strip_idx, (&offset, &byte_count)) in strip_offsets.iter().zip(&strip_byte_counts).enumerate() {
        let start = offset as usize;
        let end   = start + byte_count as usize;
        let compressed = bytes.get(start..end)?;

        match compression {
            1 => {
                // No compression — そのままコピー
                indices.extend_from_slice(compressed);
            }
            5 => {
                // TIFF LZW: with_tiff_size_switch 필수 (early change 방식)
                use weezl::{BitOrder, decode::Decoder as LzwDecoder};
                let decoded = LzwDecoder::with_tiff_size_switch(BitOrder::Msb, 8)
                    .decode(compressed)
                    .ok()?;
                indices.extend_from_slice(&decoded);
            }
            32773 => {
                // PackBits
                let mut i = 0usize;
                while i < compressed.len() {
                    let n = compressed[i] as i8;
                    i += 1;
                    if n >= 0 {
                        let run = (n as usize) + 1;
                        let src = compressed.get(i..i+run)?;
                        indices.extend_from_slice(src);
                        i += run;
                    } else if n != -128 {
                        let repeat = (-n as usize) + 1;
                        let byte = *compressed.get(i)?;
                        i += 1;
                        indices.extend(std::iter::repeat(byte).take(repeat));
                    }
                }
            }
            _ => return None,
        }
    }

    // ── Predictor=2 (Horizontal differencing) 역변환 ──────────────────────
    // LZW 압축 전에 수평 차분이 적용된 경우, 각 행의 누적합으로 원래 인덱스 복원
    if predictor == 2 {
        let w = width as usize;
        for row in indices.chunks_mut(w) {
            for x in 1..row.len() {
                row[x] = row[x].wrapping_add(row[x - 1]);
            }
        }
    }

    // ── 인덱스 → RGB 변환 ─────────────────────────────────────────────────
    indices.truncate(expected_pixels);
    if indices.len() < expected_pixels { return None; }

    let r_base = 0;
    let g_base = palette_size;
    let b_base = palette_size * 2;

    // 팔레트를 8비트 RGB LUT로 미리 변환 (colormap[i]>>8)
    let lut: Vec<[u8; 3]> = (0..palette_size).map(|p| [
        (colormap[r_base + p] >> 8) as u8,
        (colormap[g_base + p] >> 8) as u8,
        (colormap[b_base + p] >> 8) as u8,
    ]).collect();

    // max_size가 지정된 경우 인덱스 버퍼를 먼저 다운샘플 후 변환 (풀 RGB 버퍼 생략)
    if let Some(max_px) = max_size {
        let scale = (max_px as f32 / width.max(height) as f32).min(1.0);
        if scale < 1.0 {
            let out_w = ((width as f32 * scale).round() as u32).max(1);
            let out_h = ((height as f32 * scale).round() as u32).max(1);
            let mut rgb = vec![0u8; (out_w * out_h) as usize * 3];
            for dy in 0..out_h {
                let sy = ((dy as f32 + 0.5) / out_h as f32 * height as f32) as usize;
                let sy = sy.min(height as usize - 1);
                for dx in 0..out_w {
                    let sx = ((dx as f32 + 0.5) / out_w as f32 * width as f32) as usize;
                    let sx = sx.min(width as usize - 1);
                    let idx = indices[sy * width as usize + sx] as usize;
                    let dst = (dy * out_w + dx) as usize * 3;
                    let c = if idx < palette_size { lut[idx] } else { [0,0,0] };
                    rgb[dst] = c[0]; rgb[dst+1] = c[1]; rgb[dst+2] = c[2];
                }
            }
            return image::RgbImage::from_raw(out_w, out_h, rgb)
                .map(image::DynamicImage::ImageRgb8);
        }
    }

    // 풀 해상도 변환
    let mut rgb = vec![0u8; expected_pixels * 3];
    for (i, &idx) in indices.iter().enumerate() {
        let p = idx as usize;
        let dst = i * 3;
        let c = if p < palette_size { lut[p] } else { [0,0,0] };
        rgb[dst] = c[0]; rgb[dst+1] = c[1]; rgb[dst+2] = c[2];
    }

    image::RgbImage::from_raw(width, height, rgb)
        .map(image::DynamicImage::ImageRgb8)
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

    if let Ok(img) = image::open(path) {
        return Ok(img);
    }

    // TIFF 전용 폴백: FillOrder=2(LSB-first) 파일 → 비트 반전 후 재시도
    if let Some(fixed) = try_fix_tiff_lsb_fill_order(&bytes) {
        if let Ok(img) = image::load_from_memory(&fixed) {
            return Ok(img);
        }
    }

    // TIFF Palette/Indexed color (PhotometricInterpretation=3) 전용 폴백
    if let Some(img) = try_decode_palette_tiff(&bytes) {
        return Ok(img);
    }

    Err(format!("이미지 디코드 실패: {}", path.display()))
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

    let adapter_info = adapter.get_info();
    let adapter_name = format!("{} ({:?})", adapter_info.name, adapter_info.device_type);

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
        adapter_name,
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

fn guess_mime_from_real_format(format: RealImageFormat) -> &'static str {
    match format {
        RealImageFormat::Jpeg => "image/jpeg",
        RealImageFormat::Png => "image/png",
        RealImageFormat::WebP => "image/webp",
        RealImageFormat::Gif => "image/gif",
        RealImageFormat::Bmp => "image/bmp",
        RealImageFormat::Tiff => "image/tiff",
        RealImageFormat::Avif => "image/avif",
        RealImageFormat::HeicLike => "image/heic",
        RealImageFormat::Jxl => "image/jxl",
        RealImageFormat::Qoi => "image/qoi",
    }
}

fn display_format_from_real_format(format: RealImageFormat) -> &'static str {
    match format {
        RealImageFormat::Jpeg => "JPEG",
        RealImageFormat::Png => "PNG",
        RealImageFormat::WebP => "WebP",
        RealImageFormat::Avif => "AVIF",
        RealImageFormat::HeicLike => "HEIC",
        RealImageFormat::Jxl => "JXL",
        RealImageFormat::Qoi => "QOI",
        RealImageFormat::Tiff => "TIFF",
        RealImageFormat::Bmp => "BMP",
        RealImageFormat::Gif => "GIF",
    }
}

fn build_display_data_url(path: &Path) -> Option<String> {
    let bytes = fs::read(path).ok()?;
    let detected_format = detect_real_image_format(path)?;

    // WebView2(Chromium) DOES NOT support TIFF/BMP natively.
    // Also handle extension-mismatch cases by checking real file signature.
    if matches!(detected_format, RealImageFormat::Tiff | RealImageFormat::Bmp) {
        let is_palette = matches!(detected_format, RealImageFormat::Tiff) && is_palette_tiff(&bytes);

        // Palette TIFF는 image crate가 지원 안 하므로 바로 전용 디코더로
        // max_size=1200: 인덱스 단계에서 다운샘플 → 풀 RGB 버퍼 생성 불필요
        let img = if is_palette {
            try_decode_palette_tiff_inner(&bytes, Some(1200))
        } else {
            // 1차 시도: 표준 디코드
            image::load_from_memory(&bytes).ok().or_else(|| {
                // 2차 시도: FillOrder=2(LSB-first) TIFF 전용 패치 후 재시도
                if matches!(detected_format, RealImageFormat::Tiff) {
                    try_fix_tiff_lsb_fill_order(&bytes)
                        .and_then(|fixed| image::load_from_memory(&fixed).ok())
                } else {
                    None
                }
            }).or_else(|| {
                // 3차 시도: Palette/Indexed color TIFF 폴백 (비Palette인데 표준 실패 시 최후 수단)
                if matches!(detected_format, RealImageFormat::Tiff) {
                    try_decode_palette_tiff(&bytes)
                } else {
                    None
                }
            })
        };
        if let Some(img) = img {
            // Thumbnail for preview performance
            let preview = img.thumbnail(1200, 1200);
            let mut buffer = Cursor::new(Vec::new());
            if preview.write_to(&mut buffer, image::ImageFormat::Png).is_ok() {
                let encoded = BASE64_STANDARD.encode(buffer.into_inner());
                return Some(format!("data:image/png;base64,{}", encoded));
            }
        }
        // TIFF/BMP를 어떻게도 디코드할 수 없으면 None 반환 (브라우저가 렌더 불가)
        return None;
    }

    let encoded = BASE64_STANDARD.encode(bytes);
    Some(format!(
        "data:{};base64,{}",
        guess_mime_from_real_format(detected_format),
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
    let format = detect_real_image_format(&file_path)
        .map(display_format_from_real_format)
        .map(|v| v.to_string());

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
                format,
                color: None,
                error: Some(e),
            });
        }
    };

    let (width, height) = match read_image_dimensions(&file_path) {
        (Some(w), Some(h)) => (w, h),
        _ => img.dimensions(),
    };
    let color = read_color_from_file_header(&file_path)
        .or_else(|| Some(format!("{:?}", img.color())));

    Ok(ImageDetails {
        display_data_url: build_display_data_url(&file_path),
        size_bytes,
        width: Some(width),
        height: Some(height),
        dpi_x,
        dpi_y,
        format,
        color,
        error: None,
    })
}

/// Sample ~1% of pixels in a grid pattern to determine actual color mode.
/// Returns the *lowest* mode that accurately describes the image content.
fn analyze_color_mode(data: &[u8], width: u32, height: u32) -> ColorMode {
    if width == 0 || height == 0 || data.len() < 4 {
        return ColorMode::Rgba;
    }

    let total_pixels = (width as u64) * (height as u64);
    // Sample stride: ~1% of pixels, at least 1, at most a reasonable cap
    let stride = ((total_pixels / 100).max(1) as u32).min(width.max(height));

    let mut has_alpha = false;
    let mut is_gray = true;
    let mut unique_colors: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut check_mono = true;

    let mut y = 0u32;
    while y < height {
        let mut x = 0u32;
        while x < width {
            let idx = ((y as u64 * width as u64 + x as u64) * 4) as usize;
            if idx + 3 >= data.len() {
                x += stride;
                continue;
            }
            let r = data[idx];
            let g = data[idx + 1];
            let b = data[idx + 2];
            let a = data[idx + 3];

            if a < 255 {
                has_alpha = true;
            }

            if is_gray {
                let dr = (r as i16 - g as i16).unsigned_abs();
                let dg = (g as i16 - b as i16).unsigned_abs();
                if dr > 4 || dg > 4 {
                    is_gray = false;
                    check_mono = false;
                }
            }

            if check_mono && is_gray {
                let packed = ((r as u32) << 16) | ((g as u32) << 8) | (b as u32);
                unique_colors.insert(packed);
                if unique_colors.len() > 2 {
                    check_mono = false;
                }
            }

            x += stride;
        }
        y += stride;
    }

    if has_alpha {
        return ColorMode::Rgba;
    }
    if !is_gray {
        return ColorMode::Rgb;
    }
    // All samples are gray — check if monochrome (pure black/white only)
    if check_mono && unique_colors.len() <= 2 {
        let all_bw = unique_colors.iter().all(|&c| c == 0x000000 || c == 0xFFFFFF);
        if all_bw {
            return ColorMode::Monochrome;
        }
    }
    ColorMode::Grayscale
}

fn compute_optimization_plan(file: &MigrationTaskFile, criteria: OptimizationCriteria, image_data: Option<&[u8]>) -> Option<OptimizationPlan> {
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

    // Max resolution is orientation-aware:
    // landscape image -> use configured (max_width, max_height)
    // portrait image  -> swap configured values
    let (effective_max_width, effective_max_height) = match (file.width, file.height) {
        (Some(w), Some(h)) if w < h => (criteria.max_height, criteria.max_width),
        _ => (criteria.max_width, criteria.max_height),
    };

    if criteria.use_max_width && criteria.use_max_height {
        // Combined max-resolution mode: use short-side threshold only.
        // If source short side is already <= target short side, skip resize.
        if let (Some(width), Some(height)) = (file.width, file.height) {
            let source_short = width.min(height);
            let target_short = effective_max_width.min(effective_max_height);
            if source_short > target_short {
                required_scales.push(target_short as f32 / source_short as f32);
            }
        }
    } else {
        // Backward-compatible path when only one criterion is enabled.
        if criteria.use_max_width {
            if let Some(width) = file.width {
                if width > effective_max_width {
                    required_scales.push(effective_max_width as f32 / width as f32);
                }
            }
        }

        if criteria.use_max_height {
            if let Some(height) = file.height {
                if height > effective_max_height {
                    required_scales.push(effective_max_height as f32 / height as f32);
                }
            }
        }
    }

    // Determine color transform
    let (color_transform, pending_color) = if criteria.color.enabled {
        let target = criteria.color.target_mode;
        if let (Some(data), Some(w), Some(h)) = (image_data, file.width, file.height) {
            // Pixel data available — resolve now
            let actual = analyze_color_mode(data, w, h);
            let resolved = if actual == ColorMode::Monochrome && target == ColorMode::Grayscale {
                None
            } else if actual <= target {
                Some(actual.min(target))
            } else {
                None
            };
            (resolved, None)
        } else {
            // Defer analysis — will be done after decoding inside the resize function
            (None, Some(target))
        }
    } else {
        (None, None)
    };

    let has_scale = !required_scales.is_empty();
    let scale = if has_scale {
        required_scales
            .into_iter()
            .fold(1.0_f32, |acc, value| acc.min(value))
            .clamp(0.01, 1.0)
    } else {
        1.0
    };

    let needs_resize = has_scale && scale < 1.0;
    let needs_color = color_transform.is_some() || pending_color.is_some();

    if !needs_resize && !needs_color {
        return None;
    }

    Some(OptimizationPlan {
        scale: if needs_resize { scale } else { 1.0 },
        apply_target_dpi,
        color_transform,
        pending_color,
    })
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

    // DPI 기준 미지정이어도, 원본 헤더(JFIF/pHYs/TIFF IFD/BMP 등)에서 DPI를 읽을 수 있으면 보존한다.
    let inferred_source_dpi = if set_target_dpi.is_none() {
        let (sx, sy) = read_dpi_from_file_header(source_path);
        match (sx, sy) {
            (Some(x), Some(y)) => Some(((x + y) * 0.5).round().clamp(1.0, 1200.0) as u32),
            (Some(x), None) => Some(x.round().clamp(1.0, 1200.0) as u32),
            (None, Some(y)) => Some(y.round().clamp(1.0, 1200.0) as u32),
            _ => None,
        }
    } else {
        None
    };
    let effective_target_dpi = set_target_dpi.or(inferred_source_dpi);

    // 원본 EXIF 읽기: 없으면 빈 메타데이터로 시작 (에러 아님)
    let exif_read_result = LittleExifMetadata::new_from_path(source_path);
    let no_source_exif = exif_read_result.is_err();
    let mut metadata = exif_read_result.unwrap_or_else(|_| LittleExifMetadata::new());

    // 본문 픽셀은 방향 정규화해서 저장하므로 EXIF Orientation도 1로 맞춘다.
    if !no_source_exif {
        metadata.set_tag(LittleExifTag::Orientation(vec![1]));
    }

    // DPI 기준 최적화 활성 시, 대상 DPI를 명시적으로 고정 저장한다.
    if let Some(target_dpi) = effective_target_dpi {
        let r = LittleExifUR64::from(target_dpi);
        metadata.set_tag(LittleExifTag::XResolution(vec![r.clone()]));
        metadata.set_tag(LittleExifTag::YResolution(vec![r]));
        metadata.set_tag(LittleExifTag::ResolutionUnit(vec![2]));
    }

    // EXIF가 없었고 DPI도 쓸 게 없으면 메타데이터 복원 자체가 필요 없음
    if no_source_exif && effective_target_dpi.is_none() {
        return "메타데이터 복원 생략(원본 EXIF 없음)".to_string();
    }

    let metadata_write_ok = metadata.write_to_file(dest_path).is_ok();
    // PNG는 EXIF 기록 성공 여부와 별개로 pHYs(DPI) 보정이 필요하므로 계속 진행한다.
    if metadata_write_ok && !matches!(detected, Some(RealImageFormat::Png)) {
        return if no_source_exif {
            "메타데이터 복원 완료(DPI 신규 기록)".to_string()
        } else {
            "메타데이터 복원 완료".to_string()
        };
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
            // PNG DPI는 EXIF보다 pHYs 청크가 우선적으로 사용되는 경우가 많다.
            // 1) target_dpi 지정 시: pHYs를 지정 DPI로 강제
            // 2) 지정이 없으면: 원본 pHYs를 대상에 복원
            let desired_phys = if let Some(dpi) = effective_target_dpi {
                let ppm = ((dpi as f32) / 0.0254).round().max(1.0) as u32;
                Some((ppm, ppm, 1u8))
            } else {
                read_png_phys_chunk(&source_bytes).filter(|(_, _, unit)| *unit == 1)
            };

            if let Some((xppu, yppu, unit)) = desired_phys {
                match upsert_png_phys_chunk(&dest_bytes, xppu, yppu, unit) {
                    Ok(updated) => fs::write(dest_path, updated).map_err(|e| e.to_string()),
                    Err(e) => Err(e),
                }
            } else {
                // pHYs가 없으면 기존 EXIF 복사 폴백 유지
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
        Ok(_) if metadata_write_ok => "메타데이터 복원 완료".to_string(),
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

/// Apply a color transform to a resized SlimgImageData and save to dest_path.
/// `is_jpeg_source`: if true, use mozjpeg grayscale JPEG output; otherwise use PNG for gray/mono.
/// Returns an error string on failure, or a note string describing the color transform applied.
/// For Rgb/Rgba modes this function does NOT save the file — caller must use normal convert path.
fn apply_color_transform_and_save(
    image: &SlimgImageData,
    color_mode: ColorMode,
    dest_path: &Path,
    is_jpeg_source: bool,
    encode_quality: u8,
) -> Result<String, String> {
    let w = image.width;
    let h = image.height;
    let rgba = &image.data;

    match color_mode {
        ColorMode::Monochrome => {
            // Threshold to 1-bit B&W, encode as grayscale PNG (Luma8)
            let luma: Vec<u8> = rgba
                .chunks_exact(4)
                .map(|px| {
                    let gray = (px[0] as u32 * 299 + px[1] as u32 * 587 + px[2] as u32 * 114) / 1000;
                    if gray >= 128 { 255u8 } else { 0u8 }
                })
                .collect();
            let gray_img = image::GrayImage::from_raw(w, h, luma)
                .ok_or_else(|| "흑백 이미지 버퍼 생성 실패".to_string())?;
            gray_img.save_with_format(dest_path, image::ImageFormat::Png)
                .map_err(|e| format!("흑백 PNG 저장 실패: {}", e))?;
            Ok("색상변환: Monochrome(→ PNG)".to_string())
        }
        ColorMode::Grayscale => {
            let luma: Vec<u8> = rgba
                .chunks_exact(4)
                .map(|px| {
                    ((px[0] as u32 * 299 + px[1] as u32 * 587 + px[2] as u32 * 114) / 1000) as u8
                })
                .collect();
            if is_jpeg_source {
                // mozjpeg grayscale JPEG
                let result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<Vec<u8>, String> {
                    let mut compress = mozjpeg::Compress::new(mozjpeg::ColorSpace::JCS_GRAYSCALE);
                    compress.set_size(w as usize, h as usize);
                    compress.set_quality(encode_quality as f32);
                    compress.set_progressive_mode();
                    let mut compressor = compress
                        .start_compress(Vec::new())
                        .map_err(|e| format!("mozjpeg grayscale 시작 실패: {}", e))?;
                    compressor
                        .write_scanlines(&luma)
                        .map_err(|e| format!("mozjpeg grayscale 쓰기 실패: {}", e))?;
                    compressor.finish().map_err(|e| format!("mozjpeg grayscale 완료 실패: {}", e))
                }));
                let bytes = match result {
                    Ok(Ok(b)) => b,
                    Ok(Err(e)) => return Err(e),
                    Err(_) => return Err("mozjpeg grayscale 인코딩 패닉".to_string()),
                };
                fs::write(dest_path, &bytes)
                    .map_err(|e| format!("Grayscale JPEG 저장 실패: {}", e))?;
                Ok("색상변환: Grayscale(→ JPEG)".to_string())
            } else {
                let gray_img = image::GrayImage::from_raw(w, h, luma)
                    .ok_or_else(|| "Grayscale 이미지 버퍼 생성 실패".to_string())?;
                gray_img.save_with_format(dest_path, image::ImageFormat::Png)
                    .map_err(|e| format!("Grayscale PNG 저장 실패: {}", e))?;
                Ok("색상변환: Grayscale(→ PNG)".to_string())
            }
        }
        ColorMode::Rgb | ColorMode::Rgba => {
            // No special handling — caller uses normal convert path
            Ok(String::new())
        }
    }
}

fn run_slimg_resize_with_cancel(
    file: &MigrationTaskFile,
    plan: OptimizationPlan,
    cancel_flag: &AtomicBool,
    criteria: OptimizationCriteria,
    migration_options: MigrationOptions,
) -> ProcessOutcome {
    run_slimg_resize_with_cancel_with_decoded(file, plan, cancel_flag, criteria, migration_options, None)
}

fn run_slimg_resize_with_cancel_with_decoded(
    file: &MigrationTaskFile,
    plan: OptimizationPlan,
    cancel_flag: &AtomicBool,
    criteria: OptimizationCriteria,
    migration_options: MigrationOptions,
    pre_decoded: Option<SlimgImageData>,
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
        let (new_w, new_h) = if plan.scale < 1.0 {
            (
                ((oriented.width() as f32) * plan.scale).round().max(1.0) as u32,
                ((oriented.height() as f32) * plan.scale).round().max(1.0) as u32,
            )
        } else {
            (oriented.width(), oriented.height())
        };
        let resized = if plan.scale < 1.0 {
            oriented.resize_exact(new_w, new_h, FilterType::Lanczos3)
        } else {
            oriented
        };

        let is_tiff = matches!(real_format, Some(RealImageFormat::Tiff));
        let output_fmt = if is_tiff { image::ImageFormat::Tiff } else { image::ImageFormat::Bmp };
        let fmt_label = if is_tiff { "TIFF" } else { "BMP" };

        // Resolve color transform (lazy if pending)
        let effective_color_transform = plan.color_transform.or_else(|| {
            plan.pending_color.and_then(|target| {
                let rgba_data = resized.to_rgba8();
                let actual = analyze_color_mode(rgba_data.as_raw(), rgba_data.width(), rgba_data.height());
                if actual == ColorMode::Monochrome && target == ColorMode::Grayscale {
                    None
                } else if actual <= target {
                    Some(actual.min(target))
                } else {
                    None
                }
            })
        });

        // For Mono/Gray: convert to Luma8 and save in original format (TIFF→TIFF, BMP→BMP)
        if matches!(effective_color_transform, Some(ColorMode::Monochrome) | Some(ColorMode::Grayscale)) {
            let cm = effective_color_transform.unwrap();
            let rgba_data = resized.to_rgba8();
            let w = rgba_data.width();
            let h = rgba_data.height();
            let luma: Vec<u8> = rgba_data.pixels().map(|px| {
                let v = (px[0] as u32 * 299 + px[1] as u32 * 587 + px[2] as u32 * 114) / 1000;
                if cm == ColorMode::Monochrome { if v >= 128 { 255 } else { 0 } } else { v as u8 }
            }).collect();
            let gray_dyn = match image::GrayImage::from_raw(w, h, luma) {
                Some(img) => image::DynamicImage::ImageLuma8(img),
                None => return ProcessOutcome::Failed(format!("Grayscale 버퍼 생성 실패: {}", file.relative_path)),
            };
            if let Err(e) = gray_dyn.save_with_format(&file.dest_path, output_fmt) {
                return ProcessOutcome::Failed(format!("색상 변환 저장 실패: {} ({})", file.relative_path, e));
            }
            let color_note = if cm == ColorMode::Monochrome {
                format!("색상변환: Monochrome(→ {})", fmt_label)
            } else {
                format!("색상변환: Grayscale(→ {})", fmt_label)
            };
            let optimized_size_bytes = fs::metadata(&file.dest_path).map(|m| m.len()).unwrap_or(0);
            let metadata_message = if migration_options.restore_metadata {
                copy_metadata_best_effort(&file.source_path, &file.dest_path, if plan.apply_target_dpi { Some(criteria.target_dpi as u32) } else { None })
            } else { "메타데이터 복원 비활성화".to_string() };
            if optimized_size_bytes > source_size_bytes {
                let _ = fs::copy(&file.source_path, &file.dest_path);
                let restored = fs::metadata(&file.dest_path).map(|m| m.len()).unwrap_or(source_size_bytes);
                return ProcessOutcome::Success(ProcessSuccess {
                    action: "skipped", source_size_bytes, dest_size_bytes: restored,
                    message: "최적화 결과가 원본보다 커서 원본으로 복원 후 스킵 처리".to_string(),
                    fallback_code: None,
                });
            }
            return ProcessOutcome::Success(ProcessSuccess {
                action: "optimized", source_size_bytes, dest_size_bytes: optimized_size_bytes,
                message: format!("{} 최적화 완료(방향 정규화 적용) / {} / {}", fmt_label, metadata_message, color_note),
                fallback_code: None,
            });
        }

        // No color transform (or RGB/RGBA): save in original format
        let color_note = match effective_color_transform {
            Some(ColorMode::Rgb) => " / 색상변환: RGB(알파 제거)".to_string(),
            _ => String::new(),
        };
        let final_image = resized;

        if let Err(e) = final_image.save_with_format(&file.dest_path, output_fmt) {
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
        let optimized_size_bytes = fs::metadata(&file.dest_path).map(|m| m.len()).unwrap_or(0);
        if optimized_size_bytes > source_size_bytes {
            if let Err(e) = fs::copy(&file.source_path, &file.dest_path) {
                return ProcessOutcome::Failed(format!(
                    "원본 복원 실패(최적화 결과가 더 큼): {} -> {} ({})",
                    file.source_path.display(),
                    file.dest_path.display(),
                    e
                ));
            }
            let restored_size_bytes = fs::metadata(&file.dest_path).map(|m| m.len()).unwrap_or(source_size_bytes);
            return ProcessOutcome::Success(ProcessSuccess {
                action: "skipped",
                source_size_bytes,
                dest_size_bytes: restored_size_bytes,
                message: format!("{} 최적화 결과가 원본보다 커서 원본으로 복원 후 스킵 처리", fmt_label),
                fallback_code: None,
            });
        }

        return ProcessOutcome::Success(ProcessSuccess {
            action: "optimized",
            source_size_bytes,
            dest_size_bytes: optimized_size_bytes,
            message: format!("{} 최적화 완료(방향 정규화 적용) / {}{}", fmt_label, metadata_message, color_note),
            fallback_code: None,
        });
    }

    let source_size_bytes = fs::metadata(&file.source_path)
        .map(|m| m.len())
        .unwrap_or(0);
    let (decoded, decoded_format) = if let Some(pre) = pre_decoded {
        // Use pre-decoded data; we need the format from the file extension
        let fmt = match file.real_format {
            RealImageFormat::Jpeg => slimg_core::Format::Jpeg,
            RealImageFormat::Png => slimg_core::Format::Png,
            RealImageFormat::WebP => slimg_core::Format::WebP,
            RealImageFormat::Avif => slimg_core::Format::Avif,
            RealImageFormat::Qoi => slimg_core::Format::Qoi,
            _ => slimg_core::Format::Png,
        };
        (pre, fmt)
    } else {
        match decode_file(&file.source_path) {
            Ok(value) => value,
            Err(e) => {
                return ProcessOutcome::Failed(format!(
                    "slimg-core 디코드 실패: {} ({})",
                    file.relative_path, e
                ));
            }
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
        quality: migration_options.encode_quality,
        resize: None,
        crop: None,
        extend: None,
        fill_color: None,
    };

    let (preprocessed, accel_note, accel_fallback_code) = if plan.scale < 1.0 {
        let new_w = ((normalized.width as f32) * plan.scale).round().max(1.0) as u32;
        let new_h = ((normalized.height as f32) * plan.scale).round().max(1.0) as u32;
        // Determine effective mode and reason for Auto
        let (effective_mode, auto_cpu_reason) = match migration_options.acceleration_mode {
            AccelerationMode::Cpu => (AccelerationMode::Cpu, None),
            AccelerationMode::GpuPreferred => (AccelerationMode::GpuPreferred, None),
            AccelerationMode::Auto => {
                match get_gpu_resize_context() {
                    Err(e) => (AccelerationMode::Cpu, Some(format!("GPU 초기화 실패: {}", e))),
                    Ok(_) => {
                        let src_px = (normalized.width as u64).saturating_mul(normalized.height as u64);
                        let dst_px = (new_w as u64).saturating_mul(new_h as u64);
                        let max_px = src_px.max(dst_px);
                        let scale_ratio = if src_px > 0 { ((dst_px as f64) / (src_px as f64)).sqrt() as f32 } else { 1.0 };
                        let delta = (1.0 - scale_ratio).abs();
                        if should_use_gpu_in_auto_mode(normalized.width, normalized.height, new_w, new_h) {
                            (AccelerationMode::GpuPreferred, None)
                        } else if max_px < 2_000_000 {
                            (AccelerationMode::Cpu, Some(format!("픽셀 수 미달 ({}MP < 2MP)", max_px / 1_000_000)))
                        } else {
                            (AccelerationMode::Cpu, Some(format!("scale delta 미달 ({:.1}% < 8%)", delta * 100.0)))
                        }
                    }
                }
            }
        };

        match effective_mode {
            AccelerationMode::Cpu => {
                match slimg_resize(&normalized, &ResizeMode::Scale(plan.scale as f64)) {
                    Ok(img) => {
                        let note = if matches!(migration_options.acceleration_mode, AccelerationMode::Auto) {
                            match &auto_cpu_reason {
                                Some(reason) => format!("가속: Auto->CPU ({})", reason),
                                None => "가속: Auto->CPU".to_string(),
                            }
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
                Ok(Ok(rgba)) => {
                    let adapter_name = get_gpu_resize_context().ok().map(|c| c.adapter_name.clone()).unwrap_or_default();
                    (
                        SlimgImageData::new(new_w, new_h, rgba),
                        if matches!(migration_options.acceleration_mode, AccelerationMode::Auto) {
                            format!("가속: Auto->GPU(wgpu) [{}]", adapter_name)
                        } else {
                            format!("가속: GPU(wgpu) [{}]", adapter_name)
                        },
                        None,
                    )
                },
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
        }
    } else {
        // No resize needed — use normalized as-is
        (normalized, "가속: 리사이즈 없음".to_string(), None)
    };

    // Resolve color transform — either pre-resolved or pending (lazy analysis)
    let effective_color_transform = plan.color_transform.or_else(|| {
        plan.pending_color.and_then(|target| {
            let actual = analyze_color_mode(&preprocessed.data, preprocessed.width, preprocessed.height);
            if actual == ColorMode::Monochrome && target == ColorMode::Grayscale {
                None
            } else if actual <= target {
                Some(actual.min(target))
            } else {
                None
            }
        })
    });

    // Apply color transform if needed
    if let Some(cm) = effective_color_transform {
        if cm == ColorMode::Monochrome || cm == ColorMode::Grayscale {
            let is_jpeg = decoded_format == slimg_core::Format::Jpeg;
            match apply_color_transform_and_save(&preprocessed, cm, &file.dest_path, is_jpeg, migration_options.encode_quality) {
                Ok(color_note) => {
                    if cancel_flag.load(Ordering::Relaxed) {
                        return ProcessOutcome::Cancelled;
                    }
                    let optimized_size_bytes = fs::metadata(&file.dest_path).map(|m| m.len()).unwrap_or(0);
                    if optimized_size_bytes > source_size_bytes {
                        let _ = fs::copy(&file.source_path, &file.dest_path);
                        let restored = fs::metadata(&file.dest_path).map(|m| m.len()).unwrap_or(source_size_bytes);
                        return ProcessOutcome::Success(ProcessSuccess {
                            action: "skipped", source_size_bytes, dest_size_bytes: restored,
                            message: "최적화 결과가 원본보다 커서 원본으로 복원 후 스킵 처리".to_string(),
                            fallback_code: None,
                        });
                    }
                    let metadata_message = if migration_options.restore_metadata {
                        copy_metadata_best_effort(&file.source_path, &file.dest_path, if plan.apply_target_dpi { Some(criteria.target_dpi as u32) } else { None })
                    } else { "메타데이터 복원 비활성화".to_string() };
                    return ProcessOutcome::Success(ProcessSuccess {
                        action: "optimized", source_size_bytes, dest_size_bytes: optimized_size_bytes,
                        message: format!("최적화 완료(방향 정규화 적용) / {} / {} / {}", metadata_message, accel_note, color_note),
                        fallback_code: accel_fallback_code,
                    });
                }
                Err(e) => return ProcessOutcome::Failed(format!("색상 변환 실패: {} ({})", file.relative_path, e)),
            }
        }
        // For Rgb/Rgba: fall through to normal convert path (slimg-core strips alpha for JPEG automatically)
    }

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

    let color_suffix = if effective_color_transform.map_or(false, |c| c == ColorMode::Rgb || c == ColorMode::Rgba) {
        " / 색상변환: RGB(알파 제거)"
    } else { "" };
    ProcessOutcome::Success(ProcessSuccess {
        action: "optimized",
        source_size_bytes,
        dest_size_bytes: optimized_size_bytes,
        message: format!("최적화 완료(방향 정규화 적용) / {} / {}{}", metadata_message, accel_note, color_suffix),
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

    // Policy: palette 1-bit sources are excluded from color transform.
    // Resize/quality optimization still applies via normal pipeline.
    let mut effective_criteria = criteria;
    if criteria.color.enabled && is_palette_1bit_source(&file.source_path) {
        effective_criteria.color.enabled = false;
    }

    // Quick plan check (without pixel data) to see if scale is needed.
    // For color criteria, we need decoded pixels, so we decode lazily below.
    let quick_plan = compute_optimization_plan(file, effective_criteria, None);

    // If color criteria enabled, we need to decode to get actual color mode.
    // Decode here if we haven't already (i.e., no scale was needed but color might be).
    if effective_criteria.color.enabled {
        // For BMP/TIFF use dynamic image path; for others use slimg-core.
        let is_bmp_tiff = matches!(file.real_format, RealImageFormat::Bmp | RealImageFormat::Tiff);
        if !is_bmp_tiff {
            let decode_result = decode_file(&file.source_path);
            match decode_result {
                Ok((decoded, _)) => {
                    // Re-compute plan with real pixel data
                    let refined_plan = compute_optimization_plan(file, effective_criteria, Some(&decoded.data));
                    if let Some(plan) = refined_plan {
                        return run_slimg_resize_with_cancel_with_decoded(
                            file, plan, cancel_flag, effective_criteria, options, Some(decoded),
                        );
                    }
                    // No optimization needed even with color info
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
                    return ProcessOutcome::Success(ProcessSuccess {
                        action: "skipped",
                        source_size_bytes,
                        dest_size_bytes,
                        message: "기준 미초과로 복사 처리".to_string(),
                        fallback_code: None,
                    });
                }
                Err(_) => {
                    // Decode failed; fall through to quick_plan path below
                }
            }
        }
    }

    if let Some(plan) = quick_plan {
        return run_slimg_resize_with_cancel(file, plan, cancel_flag, effective_criteria, options);
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
    // Windows에서 슬래시/백슬래시 혼용 방지: 슬래시를 백슬래시로 정규화
    let dest_base = PathBuf::from(dest_path.replace('/', std::path::MAIN_SEPARATOR_STR));

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
    use_color: Option<bool>,
    target_color_mode: Option<String>,
    encode_quality: Option<u8>,
) -> Result<(), String> {
    let color_mode = match target_color_mode.as_deref().unwrap_or("rgba") {
        "monochrome" => ColorMode::Monochrome,
        "grayscale" => ColorMode::Grayscale,
        "rgb" => ColorMode::Rgb,
        _ => ColorMode::Rgba,
    };
    let criteria = OptimizationCriteria {
        use_dpi: use_dpi.unwrap_or(true),
        target_dpi: target_dpi.unwrap_or(300).clamp(72, 1200) as f32,
        use_max_width: use_max_width.unwrap_or(false),
        max_width: max_width.unwrap_or(4000).clamp(64, 20000),
        use_max_height: use_max_height.unwrap_or(false),
        max_height: max_height.unwrap_or(4000).clamp(64, 20000),
        color: ColorCriteria {
            enabled: use_color.unwrap_or(false),
            target_mode: color_mode,
        },
    };

    if !criteria.use_dpi && !criteria.use_max_width && !criteria.use_max_height && !criteria.color.enabled {
        return Err("최적화 기준을 하나 이상 선택하세요.".to_string());
    }

    let options = MigrationOptions {
        restore_metadata: restore_metadata.unwrap_or(true),
        acceleration_mode: acceleration_mode
            .map(AccelerationMode::from)
            .unwrap_or(AccelerationMode::Auto),
        encode_quality: encode_quality.unwrap_or(100).clamp(1, 100),
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
