<script setup lang="ts">
type ImageMetaDetails = {
  format?: string | null;
  sizeBytes?: number | null;
  width?: number | null;
  height?: number | null;
  dpiX?: number | null;
  dpiY?: number | null;
  color?: string | null;
};

const props = withDefaults(defineProps<{
  details?: ImageMetaDetails | null;
  showMissing?: boolean;
  missingLabel?: string;
  formatBytes: (size: number) => string;
  formatResolution: (width: number | null, height: number | null) => string;
  formatDpi: (dpiX: number | null, dpiY: number | null) => string;
}>(), {
  details: null,
  showMissing: false,
  missingLabel: '상태: 미생성',
});
</script>

<template>
  <template v-if="props.details">
    <div>포맷: {{ props.details.format ?? '-' }}</div>
    <div>
      파일크기:
      {{ props.details.sizeBytes != null ? props.formatBytes(props.details.sizeBytes) : '-' }}
    </div>
    <div>
      해상도:
      {{ props.formatResolution(props.details.width ?? null, props.details.height ?? null) }}
    </div>
    <div>
      DPI:
      {{ props.formatDpi(props.details.dpiX ?? null, props.details.dpiY ?? null) }}
    </div>
    <div>컬러: {{ props.details.color ?? '-' }}</div>
  </template>
  <template v-else-if="props.showMissing">
    <div>{{ props.missingLabel }}</div>
  </template>
</template>
