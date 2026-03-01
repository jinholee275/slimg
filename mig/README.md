# Image Optimization Tool

이미지 폴더를 대상으로 마이그레이션(최적화/복사) 작업을 수행하는 데스크톱 앱입니다.
Tauri(v2) + Vue 3 + TypeScript + Rust 백엔드로 구성되어 있습니다.

- 앱 이름: `Image Optimization Tool`
- 프론트엔드: Vite + Vue 3 (`src/`)
- 백엔드: Tauri Rust (`src-tauri/`)
- 코어 처리: `slimg-core` (path dependency)

## 주요 기능

- 원본/대상 폴더 지정 후 이미지 배치 마이그레이션
- 기준 조건(DPI, 최대 가로/세로) 기반 최적화 대상 판별
- 스킵/성공/실패 상태 관리 및 필터링
- 진행률/상태 표시, 취소 기능
- 이미지 비교 뷰어 및 메타데이터 확인
- GPU 가속 모드(`Auto`, `CPU`, `GPU 우선`) + CPU 폴백

## 프로젝트 구조

```text
mig/
  src/                 # Vue UI
  src-tauri/           # Tauri(Rust) 백엔드
  dist/                # 프론트 빌드 결과
```

주의: `src-tauri/Cargo.toml`은 아래 경로 의존성을 사용합니다.

```toml
slimg-core = { path = "../../crates/slimg-core" }
```

즉, 현재 `mig` 폴더가 `slimg` 저장소 내부(`slimg/mig`)에 위치한다고 가정합니다.
폴더 구조가 다르면 위 경로를 맞게 수정해야 합니다.

## 개발환경 요구사항

- Rust (stable, 권장 최신)
- Bun 1.x
- Node.js 18+
- Tauri v2 빌드 필수 도구

macOS 기준 추가 권장:
- Xcode Command Line Tools

플랫폼별 Tauri 필수 패키지는 공식 가이드를 참고하세요.

## 설치

프로젝트 루트(`mig`)에서:

```bash
bun install
```

Rust 의존성은 첫 `cargo` 빌드 시 자동으로 내려받습니다.

## 개발 실행

### 1) 프론트엔드만 실행

```bash
bun run dev
```

### 2) Tauri 앱 실행(권장)

```bash
bun run tauri dev
```

`tauri.conf.json`에서 개발 전 프론트 명령으로 `bun run dev`를 사용하도록 설정되어 있습니다.

## 빌드

### 프론트엔드 빌드

```bash
bun run build
```

### 데스크톱 앱 빌드

```bash
bun run tauri build
```

출력물은 플랫폼별로 `src-tauri/target/` 아래 생성됩니다.

## 품질 확인

### 프론트 타입/빌드 확인

```bash
bun run build
```

### 백엔드 컴파일 확인

```bash
cargo check -p image-optimization-tool --manifest-path src-tauri/Cargo.toml
```

## 성능 참고

환경에 따라 다르지만, **맥북프로 M1 + SSD 기준** 최근 측정치로 약 `분당 120건` 처리 사례가 있었습니다.
실제 처리량은 이미지 포맷/해상도, 저장장치, CPU/GPU 모드에 크게 영향을 받습니다.

## 트러블슈팅

### 1) `slimg-core` 경로 에러

증상:
- `failed to load manifest for dependency slimg-core`

확인:
- `src-tauri/Cargo.toml`의 `slimg-core` path가 현재 폴더 구조와 맞는지 확인

### 2) GPU 관련 오류 발생 시

- 가속 모드를 `CPU`로 변경해 우선 동작 확인
- `Auto`/`GPU 우선`은 환경에 따라 CPU 폴백이 발생할 수 있음

### 3) Cargo 캐시/락 문제

- 빌드 중 파일 잠금 메시지가 반복되면 잠시 후 재시도
- 백그라운드 `cargo` 프로세스가 오래 점유 중인지 확인

## 라이선스

내부 프로젝트 정책을 따릅니다.
