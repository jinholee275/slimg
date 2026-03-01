## Image Optimization Tool 설계서

### 1. 시스템 아키텍처 (System Architecture)

본 시스템은 **Tauri** 프레임워크를 기반으로 구축됩니다. 이를 통해 단일 코드베이스로 Windows, macOS, Linux에서 동작하는 네이티브 데스크톱 애플리케이션을 개발합니다.

- **Frontend (Vue.js)**: 사용자 인터페이스(UI)와 사용자 경험(UX)을 담당합니다. HTML, CSS, TypeScript로 작성되며, 모든 시각적 요소의 렌더링, 사용자 입력 처리, 상태 관리를 수행합니다.
- **Backend (Rust)**: 핵심 비즈니스 로직과 시스템 연동을 담당합니다. 파일 시스템 접근(폴더 스캔, 파일 읽기/쓰기), 이미지 메타데이터 분석, 외부 CLI(`slimg`) 실행 및 병렬 처리를 수행합니다.
- **통신 (Tauri API)**: 프론트엔드와 백엔드는 Tauri의 `invoke` API를 통해 비동기적으로 통신합니다. 프론트엔드는 Rust 함수를 호출하여 결과를 받고, 백엔드는 프론트엔드로 이벤트를 보내 작업 진행 상태 등을 알릴 수 있습니다.

```
+------------------------------------------------+
|              Tauri Native Window               |
| +-------------------------------------------+  |
| |            Frontend (Vue.js)              |  |
| | +-----------+ +-----------+ +-----------+ |  |
| | | File View | | Image View| | Metadata  | |  |
| | +-----------+ +-----------+ +-----------+ |  |
| |       ^             ^             ^       |  |
| +-------|-------------|-------------|-------+  |
|         |             |             |          |
|         +----(Tauri invoke/event)----+          |
|                       v                        |
| +-------------------------------------------+  |
| |              Backend (Rust)               |  |
| | +----------------+ +--------------------+ |  |
| | | File System    | | Image Processor    | |  |
| | | (Scan, R/W)    | | (Metadata, slimg)  | |  |
| | +----------------+ +--------------------+ |  |
| +-------------------------------------------+  |
+------------------------------------------------+
```

### 2. 컴포넌트 설계 (Component Design)

#### 2.1. Frontend (Vue.js Components)

- `App.vue`: 애플리케이션의 최상위 컴포넌트. 좌/우 스플릿 레이아웃을 구성하고, 주요 컴포넌트들을 배치하며, 전역 상태를 관리합니다.
- `FolderSelector.vue`: '원본/대상 폴더 선택' 버튼과 선택된 경로를 표시하는 컴포넌트. 버튼 클릭 시 백엔드의 파일 다이얼로그 함수를 호출합니다.
- `FileListView.vue`: 파일 목록을 표시하는 재사용 가능한 컴포넌트. 파일 객체 배열을 `props`로 받아 리스트를 렌더링하고, 파일 선택 시 `emit`으로 이벤트를 발생시킵니다. (좌/우측 뷰에서 각각 사용)
- `ImageViewer.vue`: 선택된 이미지 파일을 미리 보여주는 컴포넌트.
- `MetadataDisplay.vue`: 선택된 이미지의 메타데이터(경로, 크기, DPI 등)를 표시하는 컴포넌트.
- `ActionControls.vue`: '마이그레이션 시작' 버튼 및 진행 상태(프로그레스 바, 처리된 파일 수)를 표시하는 컴포넌트.

#### 2.2. Backend (Rust Modules & Structs)

- `main.rs`: Tauri 애플리케이션을 초기화하고, 프론트엔드에서 호출할 Rust 함수(Tauri Command)들을 등록합니다.
- `commands.rs`: 프론트엔드에 노출될 핵심 기능들을 정의합니다.
    - `select_folder()` -> `String`: 네이티브 폴더 선택 다이얼로그를 엽니다.
    - `scan_folder(path: String)` -> `Vec<ImageInfo>`: 지정된 경로를 재귀적으로 스캔하여 지원하는 이미지 파일 목록과 기본 정보를 반환합니다.
    - `get_image_metadata(path: String)` -> `ImageInfo`: 단일 이미지 파일의 상세 메타데이터를 추출합니다.
    - `start_migration(source_path: String, dest_path: String)`: 마이그레이션 프로세스를 시작합니다. (비동기 함수)
- `image_handler.rs`: 이미지 처리 관련 로직을 담당합니다.
    - `ImageInfo` (Struct): 이미지 정보를 담는 구조체. 프론트엔드로 전달될 수 있도록 `serde::{Serialize, Deserialize}`를 구현합니다.
        ```rust
        #[derive(Serialize, Deserialize)]
        struct ImageInfo {
            path: String,
            size: u64,
            dpi: Option<(u32, u32)>,
            resolution: (u32, u32),
            color_mode: String,
        }
        ```
    - 이미지 DPI, 해상도 등 메타데이터를 읽는 함수. (예: `kamadak-exif`, `png` 크레이트 사용)
    - `slimg` CLI를 호출하여 이미지를 변환하는 함수.
- `utils.rs`: 병렬 처리를 위한 헬퍼 함수들을 포함. (예: `rayon` 크레이트 사용)

### 3. 데이터 흐름 및 핵심 로직

#### 3.1. 이미지 스캔 및 목록 표시

1.  **사용자**가 '원본 폴더 선택' 버튼을 클릭합니다.
2.  **프론트엔드**(`FolderSelector.vue`)는 `invoke('select_folder')`를 호출합니다.
3.  **백엔드**(Rust)는 폴더 경로를 반환하고, 프론트엔드는 이 경로를 상태에 저장합니다.
4.  경로가 유효하면, 프론트엔드는 `invoke('scan_folder', { path: "..." })`를 호출합니다.
5.  **백엔드**는 해당 폴더의 모든 이미지 파일을 스캔하여 `Vec<ImageInfo>`를 프론트엔드로 반환합니다.
6.  **프론트엔드**(`App.vue`)는 받은 데이터를 상태에 저장하고, `FileListView.vue`에 `props`로 전달하여 좌측 목록을 갱신합니다.

#### 3.2. 마이그레이션 프로세스

1.  **사용자**가 '마이그레이션 시작' 버튼을 클릭합니다.
2.  **프론트엔드**(`ActionControls.vue`)는 `invoke('start_migration', { source_path: "...", dest_path: "..." })`를 호출하며 UI를 '처리 중' 상태로 변경합니다.
3.  **백엔드**(`start_migration` 함수)가 다음 로직을 수행합니다.
    a. 원본 폴더 내 모든 이미지 파일 목록을 다시 수집합니다.
    b. **`rayon`** 크레이트의 `par_iter()`를 사용하여 파일 목록을 병렬로 처리합니다.
    c. 각 스레드에서 다음을 수행합니다:
        i. 이미지의 DPI 메타데이터를 읽습니다.
        ii. 대상 폴더에 원본과 동일한 하위 경로를 생성합니다 (`std::fs::create_dir_all`).
        iii. **조건 분기:**
            - **DPI > 300**: `std::process::Command`를 사용해 `slimg -d 300 <원본경로> <대상경로>` 명령을 실행합니다.
            - **DPI <= 300**: `std::fs::copy`를 사용해 파일을 그대로 복사합니다.
        iv. 처리 완료 시 프론트엔드로 `migration-progress` 이벤트를 발생시켜 진행률을 알립니다.
4.  모든 파일 처리가 완료되면 백엔드는 `migration-complete` 이벤트를 발생시킵니다.
5.  **프론트엔드**는 완료 이벤트를 수신하면, '대상 폴더'를 기준으로 `scan_folder`를 다시 호출하여 우측 파일 목록을 갱신합니다.

### 4. 상태 관리 (Frontend State)

Vue의 `ref`와 `reactive`를 사용하여 `App.vue`에서 다음과 같은 주요 상태를 관리합니다.

```typescript
const sourcePath = ref<string>('');
const destPath = ref<string>('');

const sourceFiles = ref<ImageInfo[]>([]);
const destFiles = ref<ImageInfo[]>([]);

const selectedSourceFile = ref<ImageInfo | null>(null);
const selectedDestFile = ref<ImageInfo | null>(null);

const migrationStatus = ref<'idle' | 'processing' | 'done'>('idle');
const migrationProgress = ref({ current: 0, total: 0 });
```

### 5. 외부 도구 연동 (`slimg`)

- Rust의 `std::process::Command` 모듈을 사용하여 `slimg`를 자식 프로세스로 실행합니다.
- `Command::new("slimg").args(["-d", "300", source_file, dest_file]).output()` 또는 `.status()`를 호출하여 실행 결과를 확인합니다.
- 자식 프로세스의 `stdout`, `stderr` 및 종료 코드를 로깅하여 오류 발생 시 원인을 추적할 수 있도록 합니다.
- `slimg`가 시스템 PATH에 없을 경우를 대비하여, 애플리케이션 번들 내에 `slimg` 실행 파일을 포함하거나 사용자에게 설치 경로를 설정하게 하는 옵션을 고려합니다. (초기 버전에서는 PATH에 의존)
