# slimg 윈도우 빌드 가이드 (Windows Build Guide)

이 문서는 Windows 환경에서 `slimg`를 소스에서 빌드하고 설치하기 위한 통합 가이드를 제공합니다. 수많은 시행착오 끝에 확인된 가장 안정적인 셋팅 방법입니다.

## 1. 필수 도구 설치

관리자 권한의 커맨드 라인(PowerShell 또는 CMD)에서 다음 명령어를 순서대로 실행하여 필수 도구를 설치합니다.

### Rust 설치
[rustup.rs](https://rustup.rs/)에서 설치하거나 `winget`을 사용합니다.
```powershell
winget install Rust.Rustup
```

### C++ 빌드 도구 (MSVC) 설치
Visual Studio 2022 기반의 C++ 빌드 도구가 필요합니다.
```powershell
winget install Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

### 기타 종속성 도구 설치
```powershell
# NASM (MozJPEG / rav1e 최적화용)
winget install NASM.NASM

# LLVM (libclang - libjxl-sys 빌드용)
winget install LLVM.LLVM

# Meson & Ninja (dav1d 빌드용)
pip install meson ninja
```

## 2. 환경 변수 설정

설치된 도구들이 `cargo` 빌드 과정에서 인식될 수 있도록 시스템 환경 변수(PATH)에 다음 경로들을 추가해야 합니다. (사용자 환경에 따라 경로가 다를 수 있으니 확인 필요)

- **NASM**: `C:\Users\<사용자명>\AppData\Local\bin\NASM`
- **Meson/Ninja**: `C:\Users\<사용자명>\AppData\Roaming\Python\Python313\Scripts` (Python 버전에 따라 다름)
- **LLVM (libclang)**: `C:\Program Files\LLVM\bin`

### 필수 환경 변수 추가
`LIBCLANG_PATH` 변수를 별도로 생성하여 설정해야 `bindgen` 오류가 발생하지 않습니다.
- 변수명: `LIBCLANG_PATH`
- 값: `C:\Program Files\LLVM\bin`

## 3. 빌드 환경 실행 (중요)

일반적인 PowerShell 보다는 **"x64 Native Tools Command Prompt for VS 2022"**를 사용하거나, 터미널에서 `vcvars64.bat`을 호출한 세션에서 빌드해야 합니다.

```powershell
# 예시: vcvars64.bat 호출 (경로는 설치 위치에 따라 다름)
& "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" x64
```

## 4. 빌드 이슈 해결 (AVIF 우회)

Windows 환경에서는 `pkg-config` 부재로 인해 `dav1d-sys` 빌드가 실패하는 경우가 많습니다. 현재 가장 확실한 우회 방법은 빌드 시 AVIF 기능을 비활성화하는 것입니다.

### Cargo.toml 수정 (`crates/slimg-core/Cargo.toml`)
`image` 크레이트와 `ravif` 의존성을 다음과 같이 수정합니다.

```toml
[dependencies]
# avif-native 제거 및 default-features 비활성화
image = { version = "0.25", default-features = false, features = ["jpeg", "png", "webp"] }
# ravif 주석 처리 (dav1d-sys 전파 방지)
# ravif = "0.13"
```

### 소스 코드 수정 (`crates/slimg-core/src/codec/avif.rs`)
AVIF 코덱 구현부를 에러를 반환하는 스텁(Stub)으로 대체합니다.

```rust
use crate::error::{Error, Result};
use crate::format::Format;
use super::{Codec, EncodeOptions, ImageData};

pub struct AvifCodec;
impl Codec for AvifCodec {
    fn format(&self) -> Format { Format::Avif }
    fn decode(&self, _data: &[u8]) -> Result<ImageData> {
        Err(Error::Decode("AVIF 지원이 비활성화된 빌드입니다.".to_string()))
    }
    fn encode(&self, _image: &ImageData, _options: &EncodeOptions) -> Result<Vec<u8>> {
        Err(Error::Encode("AVIF 지원이 비활성화된 빌드입니다.".to_string()))
    }
}
```

## 5. 최종 빌드 및 설치

모든 설정이 완료되었다면 다음 명령어로 빌드합니다.

```powershell
# CLI 도구만 빌드
cargo build --release --manifest-path cli/Cargo.toml

# 또는 전체 설치 (PATH에 등록된 경우)
cargo install --path cli
```

빌드가 완료된 실행 파일은 `target/release/slimg.exe`에서 확인할 수 있습니다.
