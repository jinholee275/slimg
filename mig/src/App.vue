<script setup lang="ts">
import { ref, computed, onBeforeUnmount, onMounted, nextTick } from 'vue';
import { open } from '@tauri-apps/plugin-dialog';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

type FileItem = {
  relativePath: string;
  absolutePath: string;
  sizeBytes: number;
};

type ImageDetails = {
  displayDataUrl: string | null;
  sizeBytes: number | null;
  width: number | null;
  height: number | null;
  dpiX: number | null;
  dpiY: number | null;
  format: string | null;
  color: string | null;
  error: string | null;
};

type ImageDetailsState = ImageDetails & { loading?: boolean };

type MigrationProgressEvent = {
  total: number;
  processed: number;
  succeeded: number;
  failed: number;
  message: string;
  currentRelativePath: string | null;
  currentAction: string | null;
  currentSourceSizeBytes: number | null;
  currentDestSizeBytes: number | null;
  done: boolean;
  canceled: boolean;
};

type MigrationProgressItemUpdate = {
  relativePath: string;
  action: string;
  sourceSizeBytes: number | null;
  destSizeBytes: number | null;
  message: string;
  fallbackCode: string | null;
};

type MigrationProgressBatchEvent = {
  total: number;
  processed: number;
  succeeded: number;
  failed: number;
  message: string;
  updates: MigrationProgressItemUpdate[];
  done: boolean;
  canceled: boolean;
};

type MigrationItemResult = {
  status: 'optimized' | 'skipped' | 'failed';
  sourceSizeBytes: number | null;
  destSizeBytes: number | null;
  message: string;
  fallbackCode: string | null;
};

type FolderSummary = {
  fileCount: number;
  totalSizeBytes: number;
};

type ConcurrencyProfile = {
  cpuCores: number;
  min: number;
  max: number;
  defaultValue: number;
};

type AccelerationMode = 'auto' | 'cpu' | 'gpu';

type ImageVerificationResult = {
  similarity: number;
  verdict: 'pass' | 'warn' | 'fail' | string;
  bestTransform: string;
  orientationIssue: boolean;
  aspectIssue: boolean;
  aspectRatioDelta: number;
  scaleRatioDelta: number;
  sourceWidth: number;
  sourceHeight: number;
  destWidth: number;
  destHeight: number;
  message: string;
};

const sourcePath = ref<string | null>(null);
const destPath = ref<string | null>(null);

const sourceFiles = ref<FileItem[]>([]);
const sourceScanError = ref<string | null>(null);
const scanningSourceFolder = ref(false);
const detailsMap = ref<Record<string, ImageDetailsState>>({});
const destDetailsMap = ref<Record<string, ImageDetailsState | null>>({});
const migrationRunning = ref(false);
const migrationProgress = ref<MigrationProgressEvent>({
  total: 0,
  processed: 0,
  succeeded: 0,
  failed: 0,
  message: '준비',
  currentRelativePath: null,
  currentAction: null,
  currentSourceSizeBytes: null,
  currentDestSizeBytes: null,
  done: false,
  canceled: false,
});
const failedRelativePaths = ref<string[]>([]);
const failedRelativePathSet = ref<Set<string>>(new Set());
const failedCursor = ref<number>(-1);
const migrationResults = ref<Record<string, MigrationItemResult>>({});
const sourceSummary = ref<FolderSummary | null>(null);
const destSummary = ref<FolderSummary | null>(null);
const copiedFeedbackKey = ref<string | null>(null);
const copiedFeedbackVersion = ref(0);
const copiedToastVisible = ref(false);
const viewerVisible = ref(false);
const viewerImageUrl = ref<string | null>(null);
const viewerSourceImageUrl = ref<string | null>(null);
const viewerDestImageUrl = ref<string | null>(null);
const viewerTitle = ref('');
const viewerZoom = ref(1);
const viewerSplit = ref(50);
const viewerSplitDragging = ref(false);
const viewerPanning = ref(false);
const viewerPanX = ref(0);
const viewerPanY = ref(0);
const viewerCanvasRef = ref<HTMLElement | null>(null);
const itemVerificationMap = ref<
  Record<
    string,
    {
      loading: boolean;
      result: ImageVerificationResult | null;
      error: string | null;
    }
  >
>({});
const progressDialogVisible = ref(false);
const migrationStartedAt = ref<number | null>(null);
const migrationEndedAt = ref<number | null>(null);
const timeNow = ref(Date.now());

const cardHeight = ref(600);
const splitRatioInCard = ref(50);
const useDpiCriteria = ref(true);
const targetDpi = ref(300);
const useMaxResolutionCriteria = ref(false);
const maxWidthPx = ref(4000);
const maxHeightPx = ref(4000);
const restoreMetadata = ref(true);
const accelerationMode = ref<AccelerationMode>('auto');
const useColorCriteria = ref(false);
const targetColorMode = ref<'monochrome' | 'grayscale' | 'rgb' | 'rgba'>('grayscale');
const encodeQuality = ref(100);
const concurrencyProfile = ref<ConcurrencyProfile>({
  cpuCores: 4,
  min: 1,
  max: 4,
  defaultValue: 2,
});
const selectedConcurrency = ref(2);
const statusFilters = ref({
  pending: true,
  skipped: true,
  optimized: true,
  failed: true,
});
const fallbackFilter = ref<'all' | 'none' | 'LIMIT' | 'INIT_FAIL' | 'RUNTIME_FAIL' | 'PANIC'>('all');

const listScrollRef = ref<HTMLElement | null>(null);
const virtualContentRef = ref<HTMLElement | null>(null);
const scrollTop = ref(0);
const contentScrollTop = ref(0);
const viewportHeight = ref(0);

const VISIBLE_BUFFER = 4;
const CARD_GAP = 10;
const DETAIL_CONCURRENCY = 2;

let loadVersion = 0;
let requestEpoch = 0;
let activeLoads = 0;
let detailQueue: FileItem[] = [];
const queuedKeys = new Set<string>();
const loadingKeys = new Set<string>();
let queueRafId: number | null = null;
let queueNeedsCancel = false;
let destLoadVersion = 0;
let destRequestEpoch = 0;
let destActiveLoads = 0;
let destDetailQueue: FileItem[] = [];
const destQueuedKeys = new Set<string>();
const destLoadingKeys = new Set<string>();
let destQueueRafId: number | null = null;
let destQueueNeedsCancel = false;
let unlistenProgress: (() => void) | null = null;
let unlistenProgressBatch: (() => void) | null = null;
let unlistenDone: (() => void) | null = null;
let copiedFeedbackTimer: number | null = null;
let progressFlushTimer: number | null = null;
let latestProgressPayload: MigrationProgressEvent | null = null;
const PROGRESS_FLUSH_MS = 80;
let resultFlushTimer: number | null = null;
const pendingResultUpdates: Record<string, MigrationItemResult> = {};
const pendingFailedPaths: string[] = [];
const RESULT_FLUSH_MS = 100;
let migrationClockTimer: number | null = null;
let verificationEpoch = 0;
let verificationActiveLoads = 0;
let verificationQueue: FileItem[] = [];
const verificationQueuedKeys = new Set<string>();
const verificationLoadingKeys = new Set<string>();
const VERIFY_CONCURRENCY = 1;
let scrollIdleTimer: number | null = null;
const SCROLL_IDLE_MS = 90;
let viewerPanLastX = 0;
let viewerPanLastY = 0;
const VIEWER_SPLIT_MIN = 5;
const VIEWER_SPLIT_MAX = 95;
let viewerPanRafId: number | null = null;
let viewerPendingPanDx = 0;
let viewerPendingPanDy = 0;
let viewerZoomRafId: number | null = null;
let viewerPendingZoomFactor = 1;

const cardOuterHeight = computed(() => cardHeight.value + CARD_GAP);
const cardStyle = computed(() => ({
  height: `${cardHeight.value}px`,
  minHeight: `${cardHeight.value}px`,
  gridTemplateColumns: `${splitRatioInCard.value}fr 120px ${100 - splitRatioInCard.value}fr`,
}));

const filteredSourceFiles = computed(() => {
  return sourceFiles.value.filter((file) => {
    const result = migrationResults.value[file.relativePath];
    const status = result ? result.status : 'pending';
    if (!statusFilters.value[status]) return false;
    if (fallbackFilter.value === 'all') return true;
    const code = result?.fallbackCode ?? null;
    if (fallbackFilter.value === 'none') return code == null;
    return code === fallbackFilter.value;
  });
});

function getVisibleRange(total: number, currentScrollTop: number, currentViewportHeight: number) {
  if (total <= 0 || currentViewportHeight <= 0) {
    return { start: 0, end: -1 };
  }

  const start = Math.max(0, Math.floor(currentScrollTop / cardOuterHeight.value) - VISIBLE_BUFFER);
  const end = Math.min(
    total - 1,
    Math.ceil((currentScrollTop + currentViewportHeight) / cardOuterHeight.value) + VISIBLE_BUFFER,
  );

  return { start, end };
}

const virtualRange = computed(() =>
  getVisibleRange(filteredSourceFiles.value.length, contentScrollTop.value, viewportHeight.value),
);

const totalVirtualHeight = computed(() => filteredSourceFiles.value.length * cardOuterHeight.value);
const visibleItems = computed(() => {
  const { start, end } = virtualRange.value;
  if (end < start) return [] as Array<{ file: FileItem; absoluteIndex: number }>;
  const items: Array<{ file: FileItem; absoluteIndex: number }> = [];
  for (let idx = start; idx <= end; idx += 1) {
    const file = filteredSourceFiles.value[idx];
    if (file) {
      items.push({ file, absoluteIndex: idx });
    }
  }
  return items;
});

function emptyDetails(loading = false): ImageDetailsState {
  return {
    displayDataUrl: null,
    sizeBytes: null,
    width: null,
    height: null,
    dpiX: null,
    dpiY: null,
    format: null,
    color: null,
    error: null,
    loading,
  };
}

function summarizeFiles(files: FileItem[]): FolderSummary {
  return {
    fileCount: files.length,
    totalSizeBytes: files.reduce((acc, file) => acc + file.sizeBytes, 0),
  };
}

function resetDetailLoadingState() {
  detailsMap.value = {};
  detailQueue = [];
  activeLoads = 0;
  queuedKeys.clear();
  loadingKeys.clear();
  loadVersion += 1;
  requestEpoch += 1;
}

function resetDestDetailLoadingState() {
  destDetailsMap.value = {};
  destDetailQueue = [];
  destActiveLoads = 0;
  destQueuedKeys.clear();
  destLoadingKeys.clear();
  destLoadVersion += 1;
  destRequestEpoch += 1;
}

function resetItemVerificationState() {
  itemVerificationMap.value = {};
  verificationQueue = [];
  verificationQueuedKeys.clear();
  verificationLoadingKeys.clear();
  verificationActiveLoads = 0;
  verificationEpoch += 1;
}

function enqueueFileDetails(file: FileItem) {
  const key = file.relativePath;
  if (detailsMap.value[key] || queuedKeys.has(key) || loadingKeys.has(key)) return;

  queuedKeys.add(key);
  detailQueue.push(file);
  detailsMap.value[key] = emptyDetails(true);
}

function pumpDetailsQueue(version: number, epoch: number) {
  while (activeLoads < DETAIL_CONCURRENCY && detailQueue.length > 0) {
    const file = detailQueue.shift()!;
    const key = file.relativePath;

    queuedKeys.delete(key);
    loadingKeys.add(key);
    activeLoads += 1;

    void (async () => {
      try {
        const details = await invoke<ImageDetails>('get_image_details', { path: file.absolutePath });
        if (version === loadVersion && epoch === requestEpoch) {
          detailsMap.value[key] = { ...details, loading: false };
        }
      } catch (error) {
        if (version === loadVersion && epoch === requestEpoch) {
          detailsMap.value[key] = {
            ...emptyDetails(false),
            error: String(error),
          };
        }
      } finally {
        loadingKeys.delete(key);
        activeLoads -= 1;
        if (version === loadVersion && epoch === requestEpoch) {
          pumpDetailsQueue(version, epoch);
        }
      }
    })();
  }
}

function enqueueDestFileDetails(file: FileItem) {
  const key = file.relativePath;
  if (destDetailsMap.value[key] !== undefined || destQueuedKeys.has(key) || destLoadingKeys.has(key)) return;

  destQueuedKeys.add(key);
  destDetailQueue.push(file);
  destDetailsMap.value[key] = { ...emptyDetails(true) };
}

function pumpDestDetailsQueue(version: number, epoch: number) {
  while (destActiveLoads < DETAIL_CONCURRENCY && destDetailQueue.length > 0) {
    const file = destDetailQueue.shift()!;
    const key = file.relativePath;

    destQueuedKeys.delete(key);
    destLoadingKeys.add(key);
    destActiveLoads += 1;

    void (async () => {
      try {
        if (!destPath.value) {
          if (version === destLoadVersion && epoch === destRequestEpoch) {
            destDetailsMap.value[key] = null;
          }
          return;
        }

        const details = await invoke<ImageDetails | null>('get_destination_image_details', {
          destBasePath: destPath.value,
          relativePath: file.relativePath,
        });

        if (version === destLoadVersion && epoch === destRequestEpoch) {
          destDetailsMap.value[key] = details ? { ...details, loading: false } : null;
        }
      } catch (error) {
        if (version === destLoadVersion && epoch === destRequestEpoch) {
          destDetailsMap.value[key] = {
            ...emptyDetails(false),
            error: String(error),
          };
        }
      } finally {
        destLoadingKeys.delete(key);
        destActiveLoads -= 1;
        if (version === destLoadVersion && epoch === destRequestEpoch) {
          pumpDestDetailsQueue(version, epoch);
        }
      }
    })();
  }
}

function rebuildQueueByVisibleRange() {
  detailQueue = [];
  queuedKeys.clear();

  if (filteredSourceFiles.value.length === 0) return;
  const { start, end } = virtualRange.value;
  if (end < start) return;

  const center = (start + end) / 2;
  const indexes: number[] = [];
  for (let idx = start; idx <= end; idx += 1) indexes.push(idx);
  indexes.sort((a, b) => Math.abs(a - center) - Math.abs(b - center));

  for (const idx of indexes) {
    const file = filteredSourceFiles.value[idx];
    if (!file) continue;
    const key = file.relativePath;
    const current = detailsMap.value[key];

    const staleLoading =
      !!current &&
      !!current.loading &&
      !current.displayDataUrl &&
      current.width == null &&
      current.height == null &&
      !current.error &&
      !loadingKeys.has(key) &&
      !queuedKeys.has(key);

    if (staleLoading) delete detailsMap.value[key];

    if (detailsMap.value[key] || loadingKeys.has(key)) continue;
    enqueueFileDetails(file);
  }
}

function rebuildDestQueueByVisibleRange() {
  destDetailQueue = [];
  destQueuedKeys.clear();

  if (filteredSourceFiles.value.length === 0 || !destPath.value) return;
  const { start, end } = virtualRange.value;
  if (end < start) return;

  const center = (start + end) / 2;
  const indexes: number[] = [];
  for (let idx = start; idx <= end; idx += 1) indexes.push(idx);
  indexes.sort((a, b) => Math.abs(a - center) - Math.abs(b - center));

  for (const idx of indexes) {
    const file = filteredSourceFiles.value[idx];
    if (!file) continue;
    const key = file.relativePath;
    const current = destDetailsMap.value[key];

    const staleLoading =
      !!current &&
      !!current.loading &&
      !current.displayDataUrl &&
      current.width == null &&
      current.height == null &&
      !current.error &&
      !destLoadingKeys.has(key) &&
      !destQueuedKeys.has(key);

    if (staleLoading) delete destDetailsMap.value[key];

    if (destDetailsMap.value[key] !== undefined || destLoadingKeys.has(key)) continue;
    enqueueDestFileDetails(file);
  }
}

function queueVisibleDetails(cancelPending: boolean) {
  if (cancelPending) {
    requestEpoch += 1;
    detailQueue = [];
    queuedKeys.clear();
    loadingKeys.clear();

    for (const [key, details] of Object.entries(detailsMap.value)) {
      if (details.loading && !details.displayDataUrl && details.width == null && details.height == null && !details.error) {
        delete detailsMap.value[key];
      }
    }
  }

  rebuildQueueByVisibleRange();
  pumpDetailsQueue(loadVersion, requestEpoch);
}

function queueVisibleDestDetails(cancelPending: boolean) {
  if (cancelPending) {
    destRequestEpoch += 1;
    destDetailQueue = [];
    destQueuedKeys.clear();
    destLoadingKeys.clear();

    for (const [key, details] of Object.entries(destDetailsMap.value)) {
      if (details && details.loading && !details.displayDataUrl && details.width == null && details.height == null && !details.error) {
        delete destDetailsMap.value[key];
      }
    }
  }

  rebuildDestQueueByVisibleRange();
  pumpDestDetailsQueue(destLoadVersion, destRequestEpoch);
}

function scheduleVisibleQueue(cancelPending: boolean) {
  queueNeedsCancel = queueNeedsCancel || cancelPending;
  if (queueRafId != null) return;

  queueRafId = requestAnimationFrame(() => {
    const shouldCancel = queueNeedsCancel;
    queueNeedsCancel = false;
    queueRafId = null;
    queueVisibleDetails(shouldCancel);
  });
}

function scheduleDestVisibleQueue(cancelPending: boolean) {
  destQueueNeedsCancel = destQueueNeedsCancel || cancelPending;
  if (destQueueRafId != null) return;

  destQueueRafId = requestAnimationFrame(() => {
    const shouldCancel = destQueueNeedsCancel;
    destQueueNeedsCancel = false;
    destQueueRafId = null;
    queueVisibleDestDetails(shouldCancel);
  });
}

async function refreshSourceFiles(path: string) {
  sourceScanError.value = null;
  resetFailedNavigation();
  resetMigrationResults();
  resetItemVerificationState();
  scanningSourceFolder.value = true;
  try {
    const files = await invoke<FileItem[]>('scan_folder', { path });
    sourceFiles.value = files;
    sourceSummary.value = summarizeFiles(files);
    resetDetailLoadingState();
    resetDestDetailLoadingState();
    await nextTick();
    updateViewportMetrics();
  } catch (error) {
    sourceFiles.value = [];
    sourceSummary.value = null;
    resetDetailLoadingState();
    resetDestDetailLoadingState();
    resetItemVerificationState();
    sourceScanError.value = String(error);
  } finally {
    scanningSourceFolder.value = false;
  }
}

async function refreshDestSummary(path: string) {
  try {
    const files = await invoke<FileItem[]>('scan_folder', { path });
    destSummary.value = summarizeFiles(files);
  } catch {
    destSummary.value = null;
  }
}

async function selectSourceFolder() {
  const result = await open({
    directory: true,
    multiple: false,
    title: '원본 폴더 선택',
  });

  if (typeof result === 'string') {
    sourcePath.value = result;
    await refreshSourceFiles(result);
  }
}

async function reloadSourceFolder() {
  if (!sourcePath.value) return;
  await refreshSourceFiles(sourcePath.value);
}

async function selectDestFolder() {
  const result = await open({
    directory: true,
    multiple: false,
    title: '대상 폴더 선택',
  });

  if (typeof result === 'string') {
    resetFailedNavigation();
    resetMigrationResults();
    resetItemVerificationState();
    destPath.value = result;
    resetDestDetailLoadingState();
    updateViewportMetrics();
    await refreshDestSummary(result);
  }
}

function reloadDestFolder() {
  if (!destPath.value) return;
  resetFailedNavigation();
  resetMigrationResults();
  resetItemVerificationState();
  resetDestDetailLoadingState();
  updateViewportMetrics();
  void refreshDestSummary(destPath.value);
}

function clearSourceFolder() {
  resetFailedNavigation();
  resetMigrationResults();
  sourcePath.value = null;
  sourceFiles.value = [];
  sourceSummary.value = null;
  resetDetailLoadingState();
  resetDestDetailLoadingState();
  resetItemVerificationState();
  sourceScanError.value = null;
}

function clearDestFolder() {
  resetFailedNavigation();
  resetMigrationResults();
  destPath.value = null;
  destSummary.value = null;
  resetDestDetailLoadingState();
  resetItemVerificationState();
}

function onListScroll() {
  const el = listScrollRef.value;
  if (!el) return;

  scrollTop.value = el.scrollTop;
  const contentTop = virtualContentRef.value?.offsetTop ?? 0;
  contentScrollTop.value = Math.max(0, el.scrollTop - contentTop);
  viewportHeight.value = el.clientHeight;
  scheduleQueuesOnScrollIdle();
}

function scheduleQueuesOnScrollIdle() {
  if (scrollIdleTimer != null) {
    window.clearTimeout(scrollIdleTimer);
  }
  scrollIdleTimer = window.setTimeout(() => {
    scrollIdleTimer = null;
    scheduleVisibleQueue(false);
    scheduleDestVisibleQueue(false);
  }, SCROLL_IDLE_MS);
}

function updateViewportMetrics() {
  const el = listScrollRef.value;
  if (!el) return;

  viewportHeight.value = el.clientHeight;
  scrollTop.value = el.scrollTop;
  const contentTop = virtualContentRef.value?.offsetTop ?? 0;
  contentScrollTop.value = Math.max(0, el.scrollTop - contentTop);
  scheduleVisibleQueue(true);
  scheduleDestVisibleQueue(true);
}

function resetFailedNavigation() {
  failedRelativePaths.value = [];
  failedRelativePathSet.value = new Set();
  failedCursor.value = -1;
}

function resetMigrationResults() {
  migrationResults.value = {};
  for (const key of Object.keys(pendingResultUpdates)) {
    delete pendingResultUpdates[key];
  }
  pendingFailedPaths.length = 0;
  if (resultFlushTimer != null) {
    window.clearTimeout(resultFlushTimer);
    resultFlushTimer = null;
  }
}

function toggleStatusFilter(status: 'pending' | 'skipped' | 'optimized' | 'failed') {
  statusFilters.value[status] = !statusFilters.value[status];
  updateViewportMetrics();
}

function enqueueItemVerification(file: FileItem) {
  const key = file.relativePath;
  if (migrationResultFor(key)?.status !== 'optimized') return;
  if (itemVerificationMap.value[key] || verificationQueuedKeys.has(key) || verificationLoadingKeys.has(key)) return;
  verificationQueuedKeys.add(key);
  verificationQueue.push(file);
  itemVerificationMap.value[key] = {
    loading: true,
    result: null,
    error: null,
  };
}

function pumpItemVerificationQueue(epoch: number) {
  while (verificationActiveLoads < VERIFY_CONCURRENCY && verificationQueue.length > 0) {
    const file = verificationQueue.shift()!;
    const key = file.relativePath;
    if (migrationResultFor(key)?.status !== 'optimized') {
      delete itemVerificationMap.value[key];
      verificationQueuedKeys.delete(key);
      continue;
    }
    const destAbsolutePath = toDestAbsolutePath(file);

    verificationQueuedKeys.delete(key);
    verificationLoadingKeys.add(key);
    verificationActiveLoads += 1;

    void (async () => {
      try {
        if (!destAbsolutePath) {
          if (epoch === verificationEpoch) {
            delete itemVerificationMap.value[key];
          }
          return;
        }

        const result = await invoke<ImageVerificationResult>('verify_image_pair', {
          sourcePath: file.absolutePath,
          destPath: destAbsolutePath,
        });

        if (epoch === verificationEpoch) {
          itemVerificationMap.value[key] = {
            loading: false,
            result,
            error: null,
          };
        }
      } catch (error) {
        if (epoch === verificationEpoch) {
          itemVerificationMap.value[key] = {
            loading: false,
            result: null,
            error: String(error),
          };
        }
      } finally {
        verificationLoadingKeys.delete(key);
        verificationActiveLoads -= 1;
        if (epoch === verificationEpoch) {
          pumpItemVerificationQueue(epoch);
        }
      }
    })();
  }
}

function startBatchVerificationForOptimizedItems() {
  if (!destPath.value || sourceFiles.value.length === 0) return;
  verificationEpoch += 1;
  verificationQueue = [];
  verificationQueuedKeys.clear();
  verificationLoadingKeys.clear();
  verificationActiveLoads = 0;
  itemVerificationMap.value = {};

  for (const file of sourceFiles.value) {
    const key = file.relativePath;
    if (migrationResultFor(key)?.status !== 'optimized') continue;
    enqueueItemVerification(file);
  }
  pumpItemVerificationQueue(verificationEpoch);
}

function itemVerificationVerdictClass(relativePath: string): string {
  if (migrationResultFor(relativePath)?.status !== 'optimized') return 'pending';
  const verdict = itemVerificationMap.value[relativePath]?.result?.verdict;
  if (verdict === 'pass') return 'pass';
  if (verdict === 'warn') return 'warn';
  if (verdict === 'fail') return 'fail';
  return 'pending';
}

function itemVerificationVerdictLabel(relativePath: string): string {
  if (migrationResultFor(relativePath)?.status !== 'optimized') return '';
  const state = itemVerificationMap.value[relativePath];
  if (!destPath.value) return '검증 대기';
  if (!state) return '검증 대기';
  if (state.loading) return '검증 중...';
  if (state.error) return '검증 오류';
  const verdict = state.result?.verdict;
  if (verdict === 'pass') return '검증 통과';
  if (verdict === 'warn') return '주의';
  if (verdict === 'fail') return '검증 실패';
  return verdict ? `결과: ${verdict}` : '검증 대기';
}

function shouldShowVerification(relativePath: string): boolean {
  return migrationResultFor(relativePath)?.status === 'optimized';
}

function migrationResultFor(relativePath: string): MigrationItemResult | null {
  return migrationResults.value[relativePath] ?? null;
}

function migrationFallbackCodeLabel(relativePath: string): string | null {
  const code = migrationResultFor(relativePath)?.fallbackCode;
  if (!code) return null;
  if (code === 'LIMIT') return '폴백: LIMIT';
  if (code === 'INIT_FAIL') return '폴백: INIT_FAIL';
  if (code === 'RUNTIME_FAIL') return '폴백: RUNTIME_FAIL';
  if (code === 'PANIC') return '폴백: PANIC';
  return `폴백: ${code}`;
}

function migrationStatusClass(relativePath: string): string {
  const result = migrationResultFor(relativePath);
  if (!result) return 'pending';
  return result.status;
}

function migrationStatusLabel(relativePath: string): string {
  const result = migrationResultFor(relativePath);
  if (!result) return '대기';
  if (result.status === 'optimized') return '최적화 완료';
  if (result.status === 'skipped') return '스킵(복사)';
  return '실패';
}

onMounted(() => {
  window.addEventListener('resize', updateViewportMetrics);
  window.addEventListener('keydown', onGlobalKeydown);
  window.addEventListener('mousemove', onGlobalMouseMove);
  window.addEventListener('mouseup', stopViewerSplitDrag);
  void nextTick().then(() => updateViewportMetrics());

  void (async () => {
    try {
      const profile = await invoke<ConcurrencyProfile>('get_concurrency_profile');
      concurrencyProfile.value = profile;
      selectedConcurrency.value = profile.defaultValue;
    } catch {
      selectedConcurrency.value = concurrencyProfile.value.defaultValue;
    }

    migrationRunning.value = await invoke<boolean>('migration_running');
    if (migrationRunning.value) {
      progressDialogVisible.value = true;
      migrationStartedAt.value = Date.now();
      migrationEndedAt.value = null;
      startMigrationClock();
    }
    unlistenProgress = await listen<MigrationProgressEvent>('migration-progress', (event) => {
      latestProgressPayload = event.payload;
      scheduleProgressFlush(false);
    });
    unlistenProgressBatch = await listen<MigrationProgressBatchEvent>('migration-progress-batch', (event) => {
      const payload = event.payload;
      const last = payload.updates.length > 0 ? payload.updates[payload.updates.length - 1] : null;
      latestProgressPayload = {
        total: payload.total,
        processed: payload.processed,
        succeeded: payload.succeeded,
        failed: payload.failed,
        message: payload.message,
        currentRelativePath: last?.relativePath ?? null,
        currentAction: last?.action ?? null,
        currentSourceSizeBytes: last?.sourceSizeBytes ?? null,
        currentDestSizeBytes: last?.destSizeBytes ?? null,
        done: payload.done,
        canceled: payload.canceled,
      };
      scheduleProgressFlush(false);

      for (const item of payload.updates) {
        if (item.action !== 'optimized' && item.action !== 'skipped' && item.action !== 'failed') continue;
        pendingResultUpdates[item.relativePath] = {
          status: item.action,
          sourceSizeBytes: item.sourceSizeBytes ?? null,
          destSizeBytes: item.destSizeBytes ?? null,
          message: item.message,
          fallbackCode: item.fallbackCode ?? null,
        };
        if (item.action === 'failed') {
          pendingFailedPaths.push(item.relativePath);
        }
      }
      if (payload.updates.length > 0) {
        scheduleResultFlush(false);
      }
    });
    unlistenDone = await listen<MigrationProgressEvent>('migration-done', (event) => {
      latestProgressPayload = event.payload;
      scheduleProgressFlush(true);
      scheduleResultFlush(true);
      migrationEndedAt.value = Date.now();
      stopMigrationClock();
      if (destPath.value) {
        resetItemVerificationState();
        resetDestDetailLoadingState();
        updateViewportMetrics();
        void refreshDestSummary(destPath.value);
        startBatchVerificationForOptimizedItems();
      }
    });
  })();
});

onBeforeUnmount(() => {
  window.removeEventListener('resize', updateViewportMetrics);
  window.removeEventListener('keydown', onGlobalKeydown);
  window.removeEventListener('mousemove', onGlobalMouseMove);
  window.removeEventListener('mouseup', stopViewerSplitDrag);
  if (queueRafId != null) cancelAnimationFrame(queueRafId);
  if (destQueueRafId != null) cancelAnimationFrame(destQueueRafId);
  if (viewerPanRafId != null) cancelAnimationFrame(viewerPanRafId);
  if (viewerZoomRafId != null) cancelAnimationFrame(viewerZoomRafId);
  if (scrollIdleTimer != null) window.clearTimeout(scrollIdleTimer);
  if (copiedFeedbackTimer != null) window.clearTimeout(copiedFeedbackTimer);
  if (progressFlushTimer != null) window.clearTimeout(progressFlushTimer);
  if (resultFlushTimer != null) window.clearTimeout(resultFlushTimer);
  stopMigrationClock();
  if (unlistenProgress) unlistenProgress();
  if (unlistenProgressBatch) unlistenProgressBatch();
  if (unlistenDone) unlistenDone();
});

const isMigrationDisabled = computed(() => !sourcePath.value || !destPath.value || migrationRunning.value);

async function startMigration() {
  if (isMigrationDisabled.value) return;

  migrationProgress.value = {
    total: 0,
    processed: 0,
    succeeded: 0,
    failed: 0,
    message: '마이그레이션 시작 요청 중...',
    currentRelativePath: null,
    currentAction: null,
    currentSourceSizeBytes: null,
    currentDestSizeBytes: null,
    done: false,
    canceled: false,
  };
  latestProgressPayload = migrationProgress.value;
  resetFailedNavigation();
  resetMigrationResults();
  resetItemVerificationState();
  progressDialogVisible.value = true;
  migrationStartedAt.value = Date.now();
  migrationEndedAt.value = null;
  startMigrationClock();

  try {
    await invoke('start_migration', {
      sourcePath: sourcePath.value,
      destPath: destPath.value,
      concurrencyLimit: selectedConcurrency.value,
      useDpi: useDpiCriteria.value,
      targetDpi: targetDpi.value,
      useMaxWidth: useMaxResolutionCriteria.value,
      maxWidth: maxWidthPx.value,
      useMaxHeight: useMaxResolutionCriteria.value,
      maxHeight: maxHeightPx.value,
      restoreMetadata: restoreMetadata.value,
      accelerationMode: accelerationMode.value,
      useColor: useColorCriteria.value,
      targetColorMode: targetColorMode.value,
      encodeQuality: encodeQuality.value,
    });
    migrationRunning.value = true;
  } catch (error) {
    migrationRunning.value = false;
    progressDialogVisible.value = false;
    migrationStartedAt.value = null;
    migrationEndedAt.value = null;
    stopMigrationClock();
    migrationProgress.value = {
      total: 0,
      processed: 0,
      succeeded: 0,
      failed: 0,
      message: `시작 실패: ${String(error)}`,
      currentRelativePath: null,
      currentAction: null,
      currentSourceSizeBytes: null,
      currentDestSizeBytes: null,
      done: true,
      canceled: false,
    };
  }
}

async function cancelMigration() {
  if (!migrationRunning.value) return;
  try {
    await invoke('cancel_migration');
  } catch (error) {
    migrationProgress.value = {
      ...migrationProgress.value,
      message: `취소 요청 실패: ${String(error)}`,
    };
  }
}

function formatBytes(sizeBytes: number): string {
  if (sizeBytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const exponent = Math.min(Math.floor(Math.log(sizeBytes) / Math.log(1024)), units.length - 1);
  const value = sizeBytes / 1024 ** exponent;
  return `${value.toFixed(value >= 100 || exponent === 0 ? 0 : 1)} ${units[exponent]}`;
}

function formatDpi(dpiX: number | null, dpiY: number | null): string {
  if (dpiX == null || dpiY == null) return '-';
  return `${dpiX.toFixed(1)} x ${dpiY.toFixed(1)}`;
}

function formatResolution(width: number | null, height: number | null): string {
  if (width == null || height == null) return '-';
  return `${width} x ${height}`;
}

async function copyToClipboard(text: string | null) {
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();
    document.execCommand('copy');
    document.body.removeChild(textarea);
  }
}

async function copyWithFeedback(text: string | null, key: string) {
  if (!text) return;
  await copyToClipboard(text);
  copiedFeedbackVersion.value += 1;
  copiedFeedbackKey.value = key;
  copiedToastVisible.value = true;

  if (copiedFeedbackTimer != null) window.clearTimeout(copiedFeedbackTimer);
  copiedFeedbackTimer = window.setTimeout(() => {
    copiedFeedbackKey.value = null;
    copiedToastVisible.value = false;
    copiedFeedbackTimer = null;
  }, 900);
}

function toDestAbsolutePath(file: FileItem): string | null {
  if (!destPath.value) return null;
  const base = destPath.value.replace(/[\\/]+$/, '');
  const relative = file.relativePath.replace(/^[\\/]+/, '');
  return `${base}/${relative}`;
}

function clampZoom(value: number): number {
  return Math.min(6, Math.max(0.2, value));
}

function openImageViewer(
  sourceImageUrl: string | null | undefined,
  destImageUrl: string | null | undefined,
  title: string,
) {
  const source = sourceImageUrl ?? null;
  const dest = destImageUrl ?? null;
  const single = source || dest;
  if (!single) return;

  viewerSourceImageUrl.value = source;
  viewerDestImageUrl.value = dest;
  viewerImageUrl.value = source && dest ? null : single;
  viewerTitle.value = title;
  viewerZoom.value = 1;
  viewerSplit.value = 50;
  viewerPanX.value = 0;
  viewerPanY.value = 0;
  viewerPanning.value = false;
  viewerVisible.value = true;
}

function closeImageViewer() {
  viewerVisible.value = false;
  viewerImageUrl.value = null;
  viewerSourceImageUrl.value = null;
  viewerDestImageUrl.value = null;
  viewerTitle.value = '';
  viewerZoom.value = 1;
  viewerSplitDragging.value = false;
  viewerPanning.value = false;
  viewerPanX.value = 0;
  viewerPanY.value = 0;
  if (viewerPanRafId != null) {
    cancelAnimationFrame(viewerPanRafId);
    viewerPanRafId = null;
  }
  if (viewerZoomRafId != null) {
    cancelAnimationFrame(viewerZoomRafId);
    viewerZoomRafId = null;
  }
  viewerPendingPanDx = 0;
  viewerPendingPanDy = 0;
  viewerPendingZoomFactor = 1;
}

function zoomInViewer() {
  viewerZoom.value = clampZoom(viewerZoom.value * 1.2);
}

function zoomOutViewer() {
  viewerZoom.value = clampZoom(viewerZoom.value / 1.2);
}

function resetViewerZoom() {
  viewerZoom.value = 1;
  viewerPanX.value = 0;
  viewerPanY.value = 0;
}

function onViewerWheel(event: WheelEvent) {
  event.preventDefault();
  const delta = event.deltaY > 0 ? 0.92 : 1.08;
  viewerPendingZoomFactor *= delta;
  if (viewerZoomRafId != null) return;
  viewerZoomRafId = requestAnimationFrame(() => {
    viewerZoomRafId = null;
    viewerZoom.value = clampZoom(viewerZoom.value * viewerPendingZoomFactor);
    viewerPendingZoomFactor = 1;
  });
}

const isCompareViewer = computed(() => !!viewerSourceImageUrl.value && !!viewerDestImageUrl.value);
function updateViewerSplitByClientX(clientX: number) {
  const el = viewerCanvasRef.value;
  if (!el) return;
  const rect = el.getBoundingClientRect();
  if (rect.width <= 0) return;
  const ratio = ((clientX - rect.left) / rect.width) * 100;
  viewerSplit.value = Math.max(VIEWER_SPLIT_MIN, Math.min(VIEWER_SPLIT_MAX, ratio));
}

function startViewerSplitDrag(event: MouseEvent) {
  viewerSplitDragging.value = true;
  updateViewerSplitByClientX(event.clientX);
}

function startViewerPan(event: MouseEvent) {
  if (event.button !== 0) return;
  const target = event.target as HTMLElement | null;
  if (target?.closest('.compare-divider')) return;
  event.preventDefault();
  viewerPanning.value = true;
  viewerPanLastX = event.clientX;
  viewerPanLastY = event.clientY;
}

function preventImageDrag(event: DragEvent) {
  event.preventDefault();
}

function onGlobalMouseMove(event: MouseEvent) {
  if (viewerSplitDragging.value) {
    updateViewerSplitByClientX(event.clientX);
  }
  if (viewerPanning.value) {
    const dx = event.clientX - viewerPanLastX;
    const dy = event.clientY - viewerPanLastY;
    viewerPendingPanDx += dx;
    viewerPendingPanDy += dy;
    viewerPanLastX = event.clientX;
    viewerPanLastY = event.clientY;
    if (viewerPanRafId == null) {
      viewerPanRafId = requestAnimationFrame(() => {
        viewerPanRafId = null;
        viewerPanX.value += viewerPendingPanDx;
        viewerPanY.value += viewerPendingPanDy;
        viewerPendingPanDx = 0;
        viewerPendingPanDy = 0;
      });
    }
  }
}

function stopViewerSplitDrag() {
  viewerSplitDragging.value = false;
  viewerPanning.value = false;
}

function onGlobalKeydown(event: KeyboardEvent) {
  if (!viewerVisible.value) return;

  if (event.key === 'Escape') {
    closeImageViewer();
    return;
  }
  if (event.key === '+' || event.key === '=') {
    zoomInViewer();
    return;
  }
  if (event.key === '-') {
    zoomOutViewer();
    return;
  }
  if (event.key === '0') {
    resetViewerZoom();
  }
}

function flushProgressNow() {
  if (!latestProgressPayload) return;
  migrationProgress.value = latestProgressPayload;
  migrationRunning.value = !latestProgressPayload.done;
}

function scheduleProgressFlush(force = false) {
  if (force) {
    if (progressFlushTimer != null) {
      window.clearTimeout(progressFlushTimer);
      progressFlushTimer = null;
    }
    flushProgressNow();
    return;
  }

  if (progressFlushTimer != null) return;
  progressFlushTimer = window.setTimeout(() => {
    progressFlushTimer = null;
    flushProgressNow();
  }, PROGRESS_FLUSH_MS);
}

function closeProgressDialog() {
  if (!migrationProgress.value.done) return;
  progressDialogVisible.value = false;
}

function flushResultUpdatesNow() {
  const resultEntries = Object.entries(pendingResultUpdates);
  if (resultEntries.length > 0) {
    for (const [path, result] of resultEntries) {
      migrationResults.value[path] = result;
      delete pendingResultUpdates[path];
    }
  }

  if (pendingFailedPaths.length > 0) {
    const toAppend: string[] = [];
    for (const path of pendingFailedPaths) {
      if (!failedRelativePathSet.value.has(path)) {
        failedRelativePathSet.value.add(path);
        toAppend.push(path);
      }
    }
    pendingFailedPaths.length = 0;
    if (toAppend.length > 0) {
      failedRelativePaths.value.push(...toAppend);
      if (failedCursor.value < 0) {
        failedCursor.value = 0;
      }
    }
  }

}

function scheduleResultFlush(force = false) {
  if (force) {
    if (resultFlushTimer != null) {
      window.clearTimeout(resultFlushTimer);
      resultFlushTimer = null;
    }
    flushResultUpdatesNow();
    return;
  }

  if (resultFlushTimer != null) return;
  resultFlushTimer = window.setTimeout(() => {
    resultFlushTimer = null;
    flushResultUpdatesNow();
  }, RESULT_FLUSH_MS);
}

function startMigrationClock() {
  if (migrationClockTimer != null) return;
  timeNow.value = Date.now();
  migrationClockTimer = window.setInterval(() => {
    timeNow.value = Date.now();
  }, 1000);
}

function stopMigrationClock() {
  if (migrationClockTimer != null) {
    window.clearInterval(migrationClockTimer);
    migrationClockTimer = null;
  }
}

function applyResolutionPreset(width: number, height: number) {
  useMaxResolutionCriteria.value = true;
  maxWidthPx.value = width;
  maxHeightPx.value = height;
}

function clampRangeValue(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, Math.round(value)));
}

function onTargetDpiInputChange() {
  targetDpi.value = clampRangeValue(targetDpi.value, 72, 1200);
}

const maxResolutionText = computed(() => `${maxWidthPx.value}x${maxHeightPx.value}`);

function formatDuration(totalSeconds: number): string {
  const sec = Math.max(0, Math.round(totalSeconds));
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  if (h > 0) return `${h}시간 ${m}분 ${s}초`;
  if (m > 0) return `${m}분 ${s}초`;
  return `${s}초`;
}

const migrationProgressPercent = computed(() => {
  const total = migrationProgress.value.total;
  if (total <= 0) return 0;
  return Math.min(100, Math.round((migrationProgress.value.processed / total) * 100));
});

const statusLineText = computed(() => {
  return `상태: ${migrationProgress.value.message}`;
});

const sourceListSummaryText = computed(() => {
  if (!sourceSummary.value) return '원본: -';
  return `원본 ${sourceSummary.value.fileCount}개 / ${formatBytes(sourceSummary.value.totalSizeBytes)}`;
});

const destListSummaryText = computed(() => {
  if (!destSummary.value) return '대상: -';
  return `대상 ${destSummary.value.fileCount}개 / ${formatBytes(destSummary.value.totalSizeBytes)}`;
});

const optimizedCount = computed(() =>
  Object.values(migrationResults.value).filter((item) => item.status === 'optimized').length,
);
const skippedCount = computed(() =>
  Object.values(migrationResults.value).filter((item) => item.status === 'skipped').length,
);
const failedCount = computed(() =>
  Object.values(migrationResults.value).filter((item) => item.status === 'failed').length,
);
const pendingCount = computed(() => {
  const total = migrationProgress.value.total;
  if (total <= 0) return 0;
  return Math.max(0, total - (optimizedCount.value + skippedCount.value + failedCount.value));
});

const filterCounts = computed(() => {
  const counts = {
    pending: 0,
    skipped: 0,
    optimized: 0,
    failed: 0,
  };

  for (const file of sourceFiles.value) {
    const result = migrationResults.value[file.relativePath];
    const status = result ? result.status : 'pending';
    counts[status] += 1;
  }

  return counts;
});

const fallbackFilterCounts = computed(() => {
  const counts = {
    all: sourceFiles.value.length,
    none: 0,
    LIMIT: 0,
    INIT_FAIL: 0,
    RUNTIME_FAIL: 0,
    PANIC: 0,
  };

  for (const file of sourceFiles.value) {
    const code = migrationResults.value[file.relativePath]?.fallbackCode ?? null;
    if (!code) {
      counts.none += 1;
      continue;
    }
    if (code === 'LIMIT' || code === 'INIT_FAIL' || code === 'RUNTIME_FAIL' || code === 'PANIC') {
      counts[code] += 1;
    }
  }

  return counts;
});

const optimizedReducedBytes = computed(() => {
  let reduced = 0;
  for (const item of Object.values(migrationResults.value)) {
    if (item.status !== 'optimized') continue;
    if (item.sourceSizeBytes == null || item.destSizeBytes == null) continue;
    const delta = item.sourceSizeBytes - item.destSizeBytes;
    if (delta > 0) reduced += delta;
  }
  return reduced;
});

const reductionInfo = computed(() => {
  if (!sourceSummary.value) {
    return {
      label: '용량 비교를 위해 원본/대상 폴더를 지정하세요.',
      positive: false,
    };
  }

  const reduced = optimizedReducedBytes.value;
  const percent = sourceSummary.value.totalSizeBytes > 0
    ? (reduced / sourceSummary.value.totalSizeBytes) * 100
    : 0;

  return {
    label: `${formatBytes(reduced)} 감소 (${percent.toFixed(1)}%)`,
    positive: reduced > 0,
  };
});

const elapsedSeconds = computed(() => {
  if (!migrationStartedAt.value) return 0;
  const end = migrationEndedAt.value ?? timeNow.value;
  return Math.max(0, (end - migrationStartedAt.value) / 1000);
});

const elapsedText = computed(() => formatDuration(elapsedSeconds.value));

const etaSeconds = computed(() => {
  const processed = migrationProgress.value.processed;
  const total = migrationProgress.value.total;
  if (!migrationRunning.value || processed <= 0 || total <= processed) return null;
  const secPerItem = elapsedSeconds.value / processed;
  return secPerItem * (total - processed);
});

const etaText = computed(() => (etaSeconds.value == null ? '-' : formatDuration(etaSeconds.value)));
const avgPerFileSeconds = computed(() => {
  const processed = migrationProgress.value.processed;
  if (processed <= 0) return null;
  return elapsedSeconds.value / processed;
});
const avgPerFileText = computed(() => (avgPerFileSeconds.value == null ? '-' : formatDuration(avgPerFileSeconds.value)));
</script>

<template>
  <div class="app-container">
    <header class="top-control-area">
      <section class="folder-row">
        <div class="folder-panel">
          <button @click="selectSourceFolder">원본 폴더 선택...</button>
          <div class="folder-path-group">
            <button class="icon-tool-button icon-left" @click="reloadSourceFolder" :disabled="!sourcePath" title="원본 폴더 새로고침">
              ↻
            </button>
            <div class="image-path path-copy-text folder-path-text" :title="sourcePath ?? ''" @click="copyWithFeedback(sourcePath, 'folder-source')">
              {{ sourcePath ?? '' }}
            </div>
            <button class="icon-tool-button icon-right" @click="clearSourceFolder" title="원본 폴더 초기화">
              ✕
            </button>
          </div>
        </div>
        <div class="folder-panel">
          <button @click="selectDestFolder">대상 폴더 선택...</button>
          <div class="folder-path-group">
            <button class="icon-tool-button icon-left" @click="reloadDestFolder" :disabled="!destPath" title="대상 폴더 새로고침">
              ↻
            </button>
            <div class="image-path path-copy-text folder-path-text" :title="destPath ?? ''" @click="copyWithFeedback(destPath, 'folder-dest')">
              {{ destPath ?? '' }}
            </div>
            <button class="icon-tool-button icon-right" @click="clearDestFolder" title="대상 폴더 초기화">
              ✕
            </button>
          </div>
        </div>
        <div class="toolbar-actions">
          <button class="start-migration-button" @click="startMigration" :disabled="isMigrationDisabled">
            {{ migrationRunning ? '처리 중...' : '마이그레이션 시작' }}
          </button>
        </div>
      </section>
    </header>

    <main class="main-content">
      <section class="second-toolbar-row">
        <div class="toolbar-sliders three-lines">
          <div class="toolbar-line">
            <div class="toolbar-slider">
              <label for="card-height-range">카드 높이</label>
              <input
                id="card-height-range"
                v-model.number="cardHeight"
                type="range"
                min="180"
                max="2200"
                step="10"
                @input="updateViewportMetrics"
              />
              <span>{{ cardHeight }}px</span>
            </div>
            <div class="toolbar-slider">
              <label for="concurrency-range">병렬 처리</label>
              <input
                id="concurrency-range"
                v-model.number="selectedConcurrency"
                type="range"
                :min="concurrencyProfile.min"
                :max="concurrencyProfile.max"
                step="1"
              />
              <span>{{ selectedConcurrency }} (코어 {{ concurrencyProfile.cpuCores }})</span>
            </div>
            <label class="criteria-toggle">
              <input v-model="restoreMetadata" type="checkbox" />
              <span>메타데이터 복원</span>
            </label>
            <div class="toolbar-slider acceleration-mode">
              <label for="acceleration-mode">가속 모드</label>
              <select id="acceleration-mode" v-model="accelerationMode">
                <option value="auto">Auto</option>
                <option value="cpu">CPU</option>
                <option value="gpu">GPU 우선</option>
              </select>
            </div>
          </div>
          <div class="toolbar-line">
            <div class="toolbar-slider">
              <label for="encode-quality-range">인코딩 품질</label>
              <input
                id="encode-quality-range"
                v-model.number="encodeQuality"
                type="range"
                min="1"
                max="100"
                step="1"
              />
              <div class="number-input-wrap">
                <input
                  v-model.number="encodeQuality"
                  class="range-number-input"
                  type="number"
                  min="1"
                  max="100"
                  step="1"
                  @change="encodeQuality = Math.min(100, Math.max(1, encodeQuality))"
                />
                <span class="number-input-unit">%</span>
              </div>
            </div>
            <div class="toolbar-slider">
              <label class="criteria-toggle">
                <input v-model="useDpiCriteria" type="checkbox" />
                <span>기준 DPI</span>
              </label>
              <input
                id="target-dpi-range"
                v-model.number="targetDpi"
                type="range"
                min="72"
                max="1200"
                step="1"
                :disabled="!useDpiCriteria"
              />
              <div class="number-input-wrap">
                <input
                  v-model.number="targetDpi"
                  class="range-number-input"
                  type="number"
                  min="72"
                  max="1200"
                  step="1"
                  :disabled="!useDpiCriteria"
                  @change="onTargetDpiInputChange"
                />
                <span class="number-input-unit">DPI</span>
              </div>
            </div>
          </div>
          <div class="toolbar-line">
            <div class="toolbar-slider">
              <label class="criteria-toggle">
                <input v-model="useMaxResolutionCriteria" type="checkbox" />
                <span>최대 해상도</span>
              </label>
              <input
                class="range-number-input max-resolution-text"
                type="text"
                :value="maxResolutionText"
                readonly
                :disabled="!useMaxResolutionCriteria"
              />
              <div class="preset-buttons">
                <button class="preset-button" @click="applyResolutionPreset(640, 480)">VGA</button>
                <button class="preset-button" @click="applyResolutionPreset(1024, 768)">XGA</button>
                <button class="preset-button" @click="applyResolutionPreset(1280, 960)">SXGA</button>
                <button class="preset-button" @click="applyResolutionPreset(1600, 1200)">UXGA</button>
                <button class="preset-button" @click="applyResolutionPreset(2048, 1536)">QXGA</button>
                <button class="preset-button" @click="applyResolutionPreset(3200, 2400)">QUXGA</button>
                <button class="preset-button" @click="applyResolutionPreset(4096, 3072)">4K 4:3</button>
                <button class="preset-button" @click="applyResolutionPreset(1280, 720)">HD</button>
                <button class="preset-button" @click="applyResolutionPreset(1920, 1080)">FHD</button>
                <button class="preset-button" @click="applyResolutionPreset(2560, 1440)">QHD</button>
                <button class="preset-button" @click="applyResolutionPreset(3840, 2160)">4K</button>
              </div>
            </div>
          </div>
          <div class="toolbar-line">
            <div class="toolbar-slider color-criteria-row">
              <label class="criteria-toggle">
                <input v-model="useColorCriteria" type="checkbox" />
                <span>색상 기준</span>
              </label>
              <div class="color-mode-buttons" :class="{ disabled: !useColorCriteria }">
                <button
                  v-for="mode in (['monochrome', 'grayscale', 'rgb', 'rgba'] as const)"
                  :key="mode"
                  class="color-mode-btn"
                  :class="{ active: targetColorMode === mode }"
                  :disabled="!useColorCriteria"
                  @click="targetColorMode = mode"
                >
                  {{ mode === 'monochrome' ? '흑백' : mode === 'grayscale' ? '회색' : mode.toUpperCase() }}
                </button>
              </div>
              <span class="color-mode-hint" v-if="useColorCriteria">
                {{ targetColorMode === 'monochrome' ? '흑백 이미지만 1bit 변환'
                  : targetColorMode === 'grayscale' ? '회색조 이미지 Grayscale 변환'
                  : targetColorMode === 'rgb' ? '불투명 이미지 알파 제거'
                  : 'RGBA 유지 (변환 없음)' }}
              </span>
            </div>
          </div>
        </div>
        <div class="toolbar-filters">
          <button class="filter-button" :class="{ active: statusFilters.pending }" @click="toggleStatusFilter('pending')">
            대기 ({{ filterCounts.pending }})
          </button>
          <button class="filter-button" :class="{ active: statusFilters.skipped }" @click="toggleStatusFilter('skipped')">
            스킵 ({{ filterCounts.skipped }})
          </button>
          <button class="filter-button" :class="{ active: statusFilters.optimized }" @click="toggleStatusFilter('optimized')">
            성공 ({{ filterCounts.optimized }})
          </button>
          <button class="filter-button" :class="{ active: statusFilters.failed }" @click="toggleStatusFilter('failed')">
            실패 ({{ filterCounts.failed }})
          </button>
          <div class="fallback-filter">
            <label for="fallback-filter-select">GPU 폴백</label>
            <select id="fallback-filter-select" v-model="fallbackFilter" @change="updateViewportMetrics">
              <option value="all">전체 ({{ fallbackFilterCounts.all }})</option>
              <option value="none">없음 ({{ fallbackFilterCounts.none }})</option>
              <option value="LIMIT">LIMIT ({{ fallbackFilterCounts.LIMIT }})</option>
              <option value="INIT_FAIL">INIT_FAIL ({{ fallbackFilterCounts.INIT_FAIL }})</option>
              <option value="RUNTIME_FAIL">RUNTIME_FAIL ({{ fallbackFilterCounts.RUNTIME_FAIL }})</option>
              <option value="PANIC">PANIC ({{ fallbackFilterCounts.PANIC }})</option>
            </select>
          </div>
        </div>
      </section>

      <div class="section-title-row">
        <div class="section-side-summary left">{{ sourceListSummaryText }}</div>
        <div class="section-title-wrap">
          <div class="section-title">이미지 목록</div>
          <div v-if="migrationProgress.done" class="section-title-reduction" :class="{ positive: reductionInfo.positive }">
            {{ reductionInfo.label }}
          </div>
        </div>
        <div class="section-side-summary right">{{ destListSummaryText }}</div>
      </div>

      <section ref="listScrollRef" class="list-panel" @scroll="onListScroll">
        <div v-if="sourceScanError" class="panel-message">{{ sourceScanError }}</div>
        <div v-else-if="!sourcePath" class="panel-message">원본 폴더를 먼저 선택하세요.</div>
        <div v-else-if="sourceFiles.length === 0" class="panel-message">표시할 원본 파일이 없습니다.</div>
        <div v-else-if="filteredSourceFiles.length === 0" class="panel-message">선택된 필터에 해당하는 항목이 없습니다.</div>
        <div v-else ref="virtualContentRef" class="virtual-content">
          <div class="virtual-viewport" :style="{ height: `${totalVirtualHeight}px` }">
            <div
              v-for="{ file, absoluteIndex } in visibleItems"
              :key="file.relativePath"
              class="image-card-wrap"
              :style="{ transform: `translateY(${absoluteIndex * cardOuterHeight}px)` }"
            >
              <div
                class="image-card"
                :class="{
                  'failed-item': failedRelativePathSet.has(file.relativePath),
                  'failed-active': failedCursor >= 0 && failedRelativePaths[failedCursor] === file.relativePath,
                }"
                :style="cardStyle"
              >
              <div class="card-pane left-pane">
                <div class="path-row source-path-row">
                  <div class="image-path path-copy-text path-main" :title="file.absolutePath" @click="copyWithFeedback(file.absolutePath, `src-${file.relativePath}`)">
                    {{ file.relativePath }}
                  </div>
                  <div class="path-size">{{ formatBytes(file.sizeBytes) }}</div>
                </div>
                <div class="image-preview-wrap">
                  <img
                    v-if="detailsMap[file.relativePath]?.displayDataUrl"
                    class="image-preview clickable-preview"
                    :src="detailsMap[file.relativePath]?.displayDataUrl ?? ''"
                    :alt="file.relativePath"
                    loading="lazy"
                    @click="
                      openImageViewer(
                        detailsMap[file.relativePath]?.displayDataUrl,
                        destDetailsMap[file.relativePath]?.displayDataUrl ?? null,
                        file.relativePath,
                      )
                    "
                  />
                  <div v-else class="image-preview-placeholder">
                    {{ detailsMap[file.relativePath]?.loading ? '원본 로딩 중...' : '이미지를 표시할 수 없습니다.' }}
                  </div>
                  <div class="image-meta-float">
                    <div>포맷: {{ detailsMap[file.relativePath]?.format ?? '-' }}</div>
                    <div>
                      해상도:
                      {{
                        formatResolution(
                          detailsMap[file.relativePath]?.width ?? null,
                          detailsMap[file.relativePath]?.height ?? null,
                        )
                      }}
                    </div>
                    <div>
                      DPI:
                      {{
                        formatDpi(
                          detailsMap[file.relativePath]?.dpiX ?? null,
                          detailsMap[file.relativePath]?.dpiY ?? null,
                        )
                      }}
                    </div>
                    <div>컬러: {{ detailsMap[file.relativePath]?.color ?? '-' }}</div>
                  </div>
                </div>
                <div v-if="detailsMap[file.relativePath]?.error" class="card-error">
                  {{ detailsMap[file.relativePath].error }}
                </div>
              </div>

              <div class="card-middle">
                <div class="result-chip" :class="migrationStatusClass(file.relativePath)">
                  {{ migrationStatusLabel(file.relativePath) }}
                </div>
                <div
                  v-if="migrationFallbackCodeLabel(file.relativePath)"
                  class="fallback-chip"
                >
                  {{ migrationFallbackCodeLabel(file.relativePath) }}
                </div>
                <div
                  v-if="shouldShowVerification(file.relativePath)"
                  class="verify-chip"
                  :class="itemVerificationVerdictClass(file.relativePath)"
                >
                  {{ itemVerificationVerdictLabel(file.relativePath) }}
                </div>
                <div
                  v-if="shouldShowVerification(file.relativePath) && itemVerificationMap[file.relativePath]?.result"
                  class="verify-score"
                >
                  {{ (((itemVerificationMap[file.relativePath]?.result?.similarity ?? 0) * 100)).toFixed(1) }}%
                </div>
                <div
                  v-if="shouldShowVerification(file.relativePath) && (itemVerificationMap[file.relativePath]?.result?.orientationIssue || itemVerificationMap[file.relativePath]?.result?.aspectIssue)"
                  class="verify-warning"
                >
                  {{ itemVerificationMap[file.relativePath]?.result?.orientationIssue ? '회전/반전' : '비율/스케일' }}
                </div>
              </div>

              <div class="card-pane right-pane">
                <div class="path-row dest-path-row">
                  <div class="path-size">
                    {{
                      destDetailsMap[file.relativePath]?.sizeBytes != null
                        ? formatBytes(destDetailsMap[file.relativePath]?.sizeBytes ?? 0)
                        : '-'
                    }}
                  </div>
                  <div class="image-path path-copy-text path-main" :title="toDestAbsolutePath(file) ?? ''" @click="copyWithFeedback(toDestAbsolutePath(file), `dest-${file.relativePath}`)">
                    {{ file.relativePath }}
                  </div>
                </div>
                <div class="image-preview-wrap">
                  <img
                    v-if="destPath && destDetailsMap[file.relativePath]?.displayDataUrl"
                    class="image-preview clickable-preview"
                    :src="destDetailsMap[file.relativePath]?.displayDataUrl ?? ''"
                    :alt="file.relativePath"
                    loading="lazy"
                    @click="
                      openImageViewer(
                        detailsMap[file.relativePath]?.displayDataUrl,
                        destDetailsMap[file.relativePath]?.displayDataUrl,
                        file.relativePath,
                      )
                    "
                  />
                  <div v-else-if="!destPath" class="image-preview-placeholder">
                    대상 폴더 미선택
                  </div>
                  <div v-else-if="destDetailsMap[file.relativePath]?.loading" class="image-preview-placeholder">
                    대상 확인 중...
                  </div>
                  <div v-else class="image-preview-placeholder">
                    동일 경로/이름 대상 파일 없음
                  </div>
                  <div class="image-meta-float">
                    <template v-if="destDetailsMap[file.relativePath]">
                      <div>포맷: {{ destDetailsMap[file.relativePath]?.format ?? '-' }}</div>
                      <div>
                        DPI:
                        {{
                          formatDpi(
                            destDetailsMap[file.relativePath]?.dpiX ?? null,
                            destDetailsMap[file.relativePath]?.dpiY ?? null,
                          )
                        }}
                      </div>
                      <div>
                        해상도:
                        {{
                          formatResolution(
                            destDetailsMap[file.relativePath]?.width ?? null,
                            destDetailsMap[file.relativePath]?.height ?? null,
                          )
                        }}
                      </div>
                      <div>컬러: {{ destDetailsMap[file.relativePath]?.color ?? '-' }}</div>
                    </template>
                    <template v-else>
                      <div>상태: 미생성</div>
                    </template>
                  </div>
                </div>
                <div v-if="destDetailsMap[file.relativePath]?.error" class="card-error">
                  {{ destDetailsMap[file.relativePath]?.error }}
                </div>
              </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </main>

    <div v-if="copiedToastVisible" :key="`copy-toast-${copiedFeedbackVersion}`" class="copy-toast">
      클립보드에 복사되었습니다
    </div>

    <div v-if="viewerVisible" class="image-viewer-overlay" @click.self="closeImageViewer" @wheel="onViewerWheel">
      <div class="image-viewer-panel">
        <div class="image-viewer-topbar">
          <div class="image-viewer-title" :title="viewerTitle">{{ viewerTitle }}</div>
          <div class="image-viewer-actions">
            <template v-if="isCompareViewer">
              <label class="viewer-split-label" for="viewer-split-range">스플릿</label>
              <input
                id="viewer-split-range"
                v-model.number="viewerSplit"
                class="viewer-split-range"
                type="range"
                :min="VIEWER_SPLIT_MIN"
                :max="VIEWER_SPLIT_MAX"
                step="1"
              />
            </template>
            <button @click="zoomOutViewer">-</button>
            <button @click="resetViewerZoom">{{ Math.round(viewerZoom * 100) }}%</button>
            <button @click="zoomInViewer">+</button>
            <button @click="closeImageViewer">닫기</button>
          </div>
        </div>
        <div
          ref="viewerCanvasRef"
          class="image-viewer-canvas"
          :class="{ panning: viewerPanning }"
          @mousedown="startViewerPan"
        >
          <template v-if="isCompareViewer">
            <div class="compare-layer compare-layer-base">
              <img
                class="image-viewer-image"
                :src="viewerDestImageUrl ?? ''"
                :style="{ transform: `translate(${viewerPanX}px, ${viewerPanY}px) scale(${viewerZoom})` }"
                alt="최적화 이미지"
                draggable="false"
                @dragstart="preventImageDrag"
                @dblclick="resetViewerZoom"
              />
            </div>
            <div
              class="compare-layer compare-layer-overlay"
              :style="{ clipPath: `inset(0 ${100 - viewerSplit}% 0 0)` }"
            >
              <img
                class="image-viewer-image"
                :src="viewerSourceImageUrl ?? ''"
                :style="{ transform: `translate(${viewerPanX}px, ${viewerPanY}px) scale(${viewerZoom})` }"
                alt="원본 이미지"
                draggable="false"
                @dragstart="preventImageDrag"
                @dblclick="resetViewerZoom"
              />
            </div>
            <div class="compare-divider" :style="{ left: `${viewerSplit}%` }" @mousedown.prevent="startViewerSplitDrag">
              <div class="compare-divider-handle">↔</div>
            </div>
            <div class="compare-label compare-label-left">원본</div>
            <div class="compare-label compare-label-right">최적화</div>
          </template>
          <img
            v-else-if="viewerImageUrl"
            class="image-viewer-image"
            :src="viewerImageUrl"
            :style="{ transform: `translate(${viewerPanX}px, ${viewerPanY}px) scale(${viewerZoom})` }"
            alt="이미지 단독 보기"
            draggable="false"
            @dragstart="preventImageDrag"
            @dblclick="resetViewerZoom"
          />
        </div>
      </div>
    </div>

    <footer class="status-bar">
      <div class="status-row">
        <span>{{ statusLineText }}</span>
      </div>
    </footer>

    <div v-if="scanningSourceFolder" class="progress-dialog-overlay">
      <div class="progress-dialog" style="width: 480px;">
        <div class="progress-dialog-body" style="text-align: center; padding: 40px 20px;">
          <div style="margin-bottom: 20px; display: flex; justify-content: center;">
            <div style="width: 40px; height: 40px; border: 4px solid #4a4a4a; border-top-color: #6cb2ff; border-radius: 50%; animation: scan-spin 1s infinite linear;"></div>
          </div>
          <div style="font-size: 18px; font-weight: 800; color: #eef8ff; margin-bottom: 12px;">
            디렉토리 스캔 중...
          </div>
          <div style="font-size: 14px; color: #a3b8cc; line-height: 1.6;">
            선택하신 원본 경로에서 파일 목록을 분석하고 있습니다.<br />
            매우 많은 파일들이 포진되어 있다면 시간이 다소 걸릴 수 있습니다.<br />
            잠시만 대기해주세요.
          </div>
        </div>
      </div>
    </div>

    <div v-if="progressDialogVisible" class="progress-dialog-overlay">
      <div class="progress-dialog">
        <div class="progress-dialog-header">
          <div class="progress-dialog-title">마이그레이션 진행 현황</div>
          <button @click="closeProgressDialog" :disabled="!migrationProgress.done">닫기</button>
        </div>

        <div class="progress-dialog-body">
          <div class="progress-line">
            <span class="progress-line-label">진행률</span>
            <strong class="progress-line-percent">{{ migrationProgressPercent }}%</strong>
          </div>
          <div class="progress-track">
            <div class="progress-fill" :style="{ width: `${migrationProgressPercent}%` }"></div>
          </div>
          <div class="progress-line-sub">
            <span class="progress-line-status">{{ statusLineText }}</span>
            <span class="progress-line-count">{{ migrationProgress.processed }} / {{ migrationProgress.total }}</span>
          </div>

          <div class="progress-stats">
            <div>대기: {{ pendingCount }}건</div>
            <div>스킵: {{ skippedCount }}건</div>
            <div>성공: {{ optimizedCount }}건</div>
            <div>실패: {{ failedCount }}건</div>
            <div>진행 시간: {{ elapsedText }}</div>
            <div>예상 남은 시간: {{ etaText }}</div>
            <div>파일당 평균: {{ avgPerFileText }}</div>
          </div>

          <div class="reduction-panel" :class="{ positive: reductionInfo.positive }">
            <div class="reduction-label">성공(최적화) 처리 기준 절감량 / 원본 전체 대비</div>
            <div class="reduction-value">{{ reductionInfo.label }}</div>
          </div>
        </div>

        <div class="progress-dialog-footer">
          <button @click="cancelMigration" :disabled="!migrationRunning">마이그레이션 취소</button>
        </div>
      </div>
    </div>
  </div>
</template>

<style>
html,
body,
#app {
  height: 100%;
  margin: 0;
  padding: 0;
  overflow: hidden;
  background-color: #2f2f2f;
  color: #f6f6f6;
  font-family: Inter, Avenir, Helvetica, Arial, sans-serif;
  font-size: 14px;
}

.app-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

.top-control-area {
  padding: 8px 12px;
  background-color: #3a3a3a;
  border-bottom: 1px solid #4a4a4a;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 8px;
}

.toolbar-slider {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.toolbar-sliders {
  display: flex;
  align-items: center;
  gap: 16px;
  min-width: 0;
  flex-wrap: wrap;
}

.toolbar-sliders.two-lines,
.toolbar-sliders.three-lines {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 6px;
}

.toolbar-line {
  display: flex;
  align-items: center;
  gap: 16px;
  min-width: 0;
  flex-wrap: wrap;
}

.preset-buttons {
  display: flex;
  align-items: center;
  gap: 6px;
}

.preset-button {
  padding: 3px 10px;
  font-size: 12px;
  background-color: #3a3a3a;
  border-color: #5a5a5a;
}

.max-resolution-text {
  width: 110px;
  text-align: center;
  letter-spacing: 0.4px;
}

.color-criteria-row {
  flex-wrap: wrap;
}

.color-mode-buttons {
  display: flex;
  gap: 4px;
}

.color-mode-buttons.disabled {
  opacity: 0.4;
}

.color-mode-btn {
  padding: 2px 10px;
  font-size: 12px;
  background-color: #3a3a3a;
  border: 1px solid #5a5a5a;
  border-radius: 4px;
  color: #cfcfcf;
  cursor: pointer;
}

.color-mode-btn.active {
  background-color: #4a90d9;
  border-color: #6aaeff;
  color: #fff;
}

.color-mode-btn:disabled {
  cursor: default;
}

.color-mode-hint {
  color: #888;
  font-size: 11px;
  white-space: nowrap;
}

.criteria-toggle {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: #cfcfcf;
  font-size: 12px;
  white-space: nowrap;
}

.toolbar-slider label,
.toolbar-slider span {
  color: #cfcfcf;
  font-size: 12px;
  white-space: nowrap;
}

.toolbar-slider input[type='range'] {
  width: 180px;
}

.toolbar-slider select {
  padding: 2px 6px;
  font-size: 12px;
  color: #ddd;
  background: #2a2a2a;
  border: 1px solid #505050;
  border-radius: 4px;
}

.number-input-wrap {
  position: relative;
  display: inline-flex;
  align-items: center;
}

.number-input-unit {
  position: absolute;
  right: 6px;
  font-size: 11px;
  color: #888;
  pointer-events: none;
  user-select: none;
}

.range-number-input {
  width: 62px;
  padding: 2px 22px 2px 6px;
  font-size: 12px;
  color: #ddd;
  background: #2a2a2a;
  border: 1px solid #505050;
  border-radius: 4px;
  text-align: right;
}

.toolbar-actions {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
}

.toolbar-filters {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}

.fallback-filter {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-left: 8px;
}

.fallback-filter label {
  font-size: 12px;
  color: #cfcfcf;
}

.fallback-filter select {
  padding: 2px 6px;
  font-size: 12px;
  color: #ddd;
  background: #2a2a2a;
  border: 1px solid #505050;
  border-radius: 4px;
}

.filter-button {
  background-color: #303030;
  border-color: #555;
  color: #a8a8a8;
}

.filter-button.active {
  background-color: #486a89;
  border-color: #6fa7d3;
  color: #f4f9ff;
}

.start-migration-button {
  background: linear-gradient(180deg, #31b46f, #249456);
  border-color: #3dc47f;
  color: #f7fff9;
  font-weight: 800;
}

.start-migration-button:hover {
  background: linear-gradient(180deg, #39c079, #279f5e);
}

.start-migration-button:disabled {
  background: #3f5d4d;
  border-color: #557364;
  color: #9fb5a8;
}

.main-content {
  flex: 1;
  min-height: 0;
  padding: 10px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.second-toolbar-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 2px 0;
}

.folder-row {
  display: grid;
  grid-template-columns: 1fr 1fr auto;
  gap: 10px;
  align-items: center;
}

.folder-panel {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 8px;
  align-items: center;
  min-width: 0;
}

.folder-path-group {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  align-items: stretch;
  min-width: 0;
}

.icon-tool-button {
  min-width: 30px;
  padding: 0 8px;
  border-radius: 0;
  font-size: 14px;
  line-height: 1;
}

.icon-tool-button.icon-left {
  border-top-left-radius: 4px;
  border-bottom-left-radius: 4px;
}

.icon-tool-button.icon-right {
  border-top-right-radius: 4px;
  border-bottom-right-radius: 4px;
}

.list-panel {
  min-height: 0;
  overflow-y: auto;
  overflow-x: hidden;
  overflow-anchor: none;
  border: 1px dashed #555;
  border-radius: 6px;
  padding: 10px;
  background-color: #282828;
}

.virtual-content {
  overflow-anchor: none;
}

.section-title-row {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
}

.section-title {
  color: #bbb;
  font-size: 16px;
  font-weight: 800;
  text-align: center;
  white-space: nowrap;
}

.section-title-wrap {
  display: inline-flex;
  align-items: baseline;
  justify-content: center;
  gap: 10px;
  min-width: 0;
}

.section-title-reduction {
  font-size: 16px;
  font-weight: 800;
  color: #f0d28a;
  letter-spacing: 0.1px;
  white-space: nowrap;
}

.section-title-reduction.positive {
  color: #86f5c7;
}

.section-side-summary {
  color: #9da6b0;
  font-size: 16px;
  font-weight: 700;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.section-side-summary.right {
  text-align: right;
}

.panel-message {
  color: #8e8e8e;
  line-height: 1.5;
}

.image-card {
  display: grid;
  border: 1px solid #454545;
  border-radius: 6px;
  background-color: #222;
  box-sizing: border-box;
  overflow: hidden;
  gap: 0;
}

.image-card-wrap {
  position: absolute;
  left: 0;
  right: 0;
}

.image-card.failed-item {
  border-color: #8a4747;
}

.image-card.failed-active {
  border-color: #ff6868;
  box-shadow: 0 0 0 1px rgba(255, 104, 104, 0.6) inset;
}

.card-pane {
  padding: 8px;
  display: flex;
  flex-direction: column;
  min-width: 0;
  min-height: 0;
}

.left-pane {
  border-right: 1px solid #3b3b3b;
}

.card-middle {
  border-right: 1px solid #3b3b3b;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  gap: 8px;
  padding: 8px;
  min-width: 0;
  background: #202020;
}

.result-chip {
  font-size: 12px;
  font-weight: 700;
  border-radius: 999px;
  padding: 4px 10px;
  border: 1px solid #555;
  color: #ddd;
  background: #2f2f2f;
  white-space: nowrap;
}

.result-chip.pending {
  border-color: #555;
  color: #bfbfbf;
}

.result-chip.optimized {
  border-color: #2d8f66;
  color: #7ff0c0;
  background: rgba(45, 143, 102, 0.2);
}

.result-chip.skipped {
  border-color: #78622e;
  color: #f0d28a;
  background: rgba(120, 98, 46, 0.2);
}

.result-chip.failed {
  border-color: #a14a4a;
  color: #ff9b9b;
  background: rgba(161, 74, 74, 0.2);
}

.fallback-chip {
  font-size: 10px;
  font-weight: 700;
  border-radius: 999px;
  padding: 3px 8px;
  border: 1px solid #5f5f5f;
  color: #d9d9d9;
  background: rgba(110, 110, 110, 0.2);
  white-space: nowrap;
}

.verify-chip {
  margin-top: 6px;
  font-size: 11px;
  font-weight: 700;
  border-radius: 999px;
  padding: 3px 8px;
  border: 1px solid #555;
  color: #c9d2dc;
  background: #2a2a2a;
  white-space: nowrap;
}

.verify-chip.pass {
  border-color: #2d8f66;
  color: #86f5c7;
  background: rgba(45, 143, 102, 0.2);
}

.verify-chip.warn {
  border-color: #8a7336;
  color: #f0d28a;
  background: rgba(138, 115, 54, 0.2);
}

.verify-chip.fail {
  border-color: #a14a4a;
  color: #ff9b9b;
  background: rgba(161, 74, 74, 0.2);
}

.verify-score {
  margin-top: 6px;
  font-size: 12px;
  font-weight: 800;
  color: #dfe8f2;
  font-variant-numeric: tabular-nums;
}

.verify-warning {
  margin-top: 4px;
  font-size: 10px;
  color: #ffb3b3;
  text-align: center;
}

.right-pane .pane-title,
.right-pane .image-path {
  width: 100%;
  text-align: right;
}

.pane-title {
  color: #d5d5d5;
  font-size: 12px;
  font-weight: 600;
  margin-bottom: 6px;
}

.image-path {
  color: #c9c9c9;
  font-size: 12px;
  margin-bottom: 8px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.path-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
  min-width: 0;
}

.path-main {
  margin-bottom: 0;
  min-width: 0;
  flex: 1;
}

.path-size {
  flex-shrink: 0;
  color: #b7b7b7;
  font-size: 11px;
  font-variant-numeric: tabular-nums;
}

.dest-path-row .path-main {
  text-align: right;
}

.path-copy-text {
  cursor: pointer;
}

.folder-path-text {
  margin-bottom: 0;
  padding: 2px 8px;
  min-height: 18px;
  display: flex;
  align-items: center;
  background-color: #2a2a2a;
  border: 1px solid #4a4a4a;
  border-left: none;
  border-right: none;
  border-radius: 0;
}

.copy-toast {
  position: fixed;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  z-index: 9999;
  font-size: 14px;
  font-weight: 600;
  color: #eef8ff;
  background: rgba(18, 58, 84, 0.94);
  border: 1px solid rgba(121, 195, 255, 0.75);
  border-radius: 999px;
  padding: 10px 16px;
  pointer-events: none;
  box-shadow: 0 10px 24px rgba(0, 0, 0, 0.35);
  animation: copy-toast-fade 0.9s ease-out forwards;
}

@keyframes copy-toast-fade {
  0% {
    opacity: 0;
    transform: translate(-50%, calc(-50% + 10px)) scale(0.96);
  }
  15% {
    opacity: 1;
    transform: translate(-50%, -50%) scale(1);
  }
  80% {
    opacity: 1;
    transform: translate(-50%, -50%) scale(1);
  }
  100% {
    opacity: 0;
    transform: translate(-50%, calc(-50% - 8px)) scale(1);
  }
}

@keyframes scan-spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.image-preview-wrap {
  position: relative;
  background-color: #1a1a1a;
  border: 1px solid #333;
  border-radius: 6px;
  overflow: hidden;
  flex: 1;
  min-height: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}

.image-preview,
.image-preview-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
}

.image-preview {
  object-fit: contain;
  object-position: center;
}

.clickable-preview {
  cursor: zoom-in;
}

.image-preview-placeholder {
  color: #858585;
  font-size: 12px;
  text-align: center;
  padding: 0 8px;
}

.image-meta-float {
  position: absolute;
  right: 8px;
  bottom: 8px;
  background: rgba(0, 0, 0, 0.68);
  color: #f1f1f1;
  border: 1px solid rgba(255, 255, 255, 0.14);
  border-radius: 6px;
  font-size: 11px;
  line-height: 1.35;
  padding: 6px 8px;
  max-width: calc(100% - 16px);
  backdrop-filter: blur(2px);
}

.card-error {
  margin-top: 6px;
  color: #f08c8c;
  font-size: 11px;
}

.image-viewer-overlay {
  position: fixed;
  inset: 0;
  z-index: 10000;
  background: rgba(0, 0, 0, 0.76);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
}

.image-viewer-panel {
  width: min(96vw, 1600px);
  height: min(92vh, 1100px);
  background: #171717;
  border: 1px solid #4d4d4d;
  border-radius: 10px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.image-viewer-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 8px 10px;
  border-bottom: 1px solid #3d3d3d;
  background: #252525;
}

.image-viewer-title {
  font-size: 12px;
  color: #d3d3d3;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.image-viewer-actions {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-shrink: 0;
}

.viewer-split-label {
  font-size: 12px;
  color: #c9d4df;
}

.viewer-split-range {
  width: 140px;
}

.image-viewer-canvas {
  flex: 1;
  min-height: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: auto;
  background: #101010;
  position: relative;
  cursor: grab;
  user-select: none;
}

.image-viewer-canvas.panning {
  cursor: grabbing;
}

.image-viewer-image {
  width: 100%;
  height: 100%;
  max-width: none;
  max-height: none;
  object-fit: contain;
  transform-origin: center center;
  transition: none;
  will-change: transform;
  -webkit-user-drag: none;
  user-select: none;
}

.compare-layer {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.compare-layer-overlay {
  pointer-events: none;
}

.compare-divider {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 2px;
  background: rgba(255, 255, 255, 0.92);
  box-shadow:
    -1px 0 0 rgba(0, 0, 0, 0.85),
     1px 0 0 rgba(0, 0, 0, 0.85);
  transform: translateX(-1px);
  cursor: ew-resize;
  z-index: 2;
}

.compare-divider-handle {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background: rgba(20, 20, 20, 0.9);
  color: #fff;
  border: 1px solid rgba(255, 255, 255, 0.55);
  border-radius: 999px;
  font-size: 12px;
  padding: 4px 8px;
  user-select: none;
}

.compare-label {
  position: absolute;
  top: 10px;
  z-index: 2;
  font-size: 12px;
  font-weight: 700;
  color: #eef3f8;
  background: rgba(0, 0, 0, 0.55);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 999px;
  padding: 3px 8px;
  pointer-events: none;
}

.compare-label-left {
  left: 10px;
}

.compare-label-right {
  right: 10px;
}

.virtual-spacer {
  width: 100%;
  pointer-events: none;
}

.virtual-viewport {
  position: relative;
  width: 100%;
}

.status-bar {
  position: relative;
  overflow: hidden;
  padding: 4px 12px;
  background-color: #3a3a3a;
  border-top: 1px solid #4a4a4a;
  flex-shrink: 0;
  font-size: 0.8em;
}

.status-row {
  position: relative;
  z-index: 1;
  display: flex;
  justify-content: center;
  gap: 16px;
  color: #d0d0d0;
  text-align: center;
}

.progress-dialog-overlay {
  position: fixed;
  inset: 0;
  z-index: 10020;
  background: rgba(0, 0, 0, 0.62);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 16px;
}

.progress-dialog {
  width: min(92vw, 560px);
  background: #222;
  border: 1px solid #555;
  border-radius: 10px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.45);
  overflow: hidden;
}

.progress-dialog-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  padding: 10px 12px;
  background: #313131;
  border-bottom: 1px solid #4a4a4a;
}

.progress-dialog-title {
  font-weight: 700;
  font-size: 14px;
}

.progress-dialog-body {
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.progress-line {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.progress-line-label {
  font-size: 16px;
  font-weight: 700;
}

.progress-line-percent {
  font-size: 28px;
  font-weight: 900;
  color: #8ec2ff;
  line-height: 1;
}

.progress-line-sub {
  margin-top: -4px;
  font-size: 13px;
  color: #c7d0da;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.progress-line-status {
  text-align: left;
  color: #bfc9d4;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.progress-line-count {
  text-align: right;
  white-space: nowrap;
}

.progress-stats {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px 12px;
  color: #d9d9d9;
  font-size: 15px;
}

.progress-dialog .progress-track {
  height: 16px;
  border-radius: 999px;
  background: #1d2731;
  border: 1px solid #4b5f75;
  overflow: hidden;
}

.progress-dialog .progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #2f7cf6, #6cb2ff);
  transition: width 0.18s ease-out;
}

.reduction-panel {
  background: #2b2b2b;
  border: 1px solid #4a4a4a;
  border-radius: 8px;
  padding: 10px;
}

.reduction-panel.positive {
  background: rgba(45, 143, 102, 0.16);
  border-color: #2d8f66;
}

.reduction-label {
  font-size: 12px;
  color: #bfc5ce;
}

.reduction-value {
  margin-top: 4px;
  font-size: 20px;
  font-weight: 800;
  color: #f4f8ff;
}

.reduction-panel.positive .reduction-value {
  color: #86f5c7;
}

.progress-dialog-footer {
  padding: 10px 12px 12px;
  border-top: 1px solid #4a4a4a;
  display: flex;
  justify-content: flex-end;
}

button {
  background-color: #4f4f4f;
  border: 1px solid #666;
  color: #fff;
  padding: 4px 12px;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s;
  white-space: nowrap;
  flex-shrink: 0;
}

button:hover {
  background-color: #5a5a5a;
}

button:disabled {
  background-color: #444;
  color: #888;
  cursor: not-allowed;
  border-color: #555;
}
</style>
