<script setup lang="ts">
import { computed, ref } from 'vue'
import { ArrowRight, Tools, Loading, RefreshRight } from '@element-plus/icons-vue'
import type { ToolStep } from '../stores/chat'
import AgentBadge from './AgentBadge.vue'

const props = defineProps<{ step: ToolStep }>()
const open = ref(false)

const isRetry = computed(() => (props.step.retry_attempt ?? 0) > 0)
const isInterrupted = computed(() => Boolean(props.step.interrupted))
</script>

<template>
  <div :class="['tool', { retry: isRetry, interrupted: isInterrupted }]">
    <div class="head" @click="open = !open">
      <el-icon class="tag-icon"><Tools /></el-icon>
      <span class="name">{{ step.name }}</span>
      <AgentBadge v-if="step.agent_name" :agent_name="step.agent_name" />
      <span v-if="isRetry" class="retry-tag" :title="`第 ${step.retry_attempt} 次重试`">
        <el-icon class="retry-icon"><RefreshRight /></el-icon>
        重试 {{ step.retry_attempt }}
      </span>
      <span class="status">
        <el-icon v-if="!step.output" class="spin"><Loading /></el-icon>
        {{ step.output ? '已返回' : '调用中…' }}
      </span>
      <el-icon class="arrow" :class="{ open }"><ArrowRight /></el-icon>
    </div>
    <el-collapse-transition>
      <div v-if="open" class="body">
        <div class="label">参数</div>
        <pre class="block">{{ JSON.stringify(step.args, null, 2) }}</pre>
        <template v-if="step.output">
          <div class="label">返回</div>
          <pre class="block">{{ step.output }}</pre>
        </template>
      </div>
    </el-collapse-transition>
  </div>
</template>

<style scoped>
.tool {
  margin: 0 0 8px 31px;
  overflow: hidden;
  border-left: 3px solid var(--tool-border);
  background: var(--tool-bg);
  border-radius: var(--radius-sm);
  font-size: 13px;
}
.tool.retry {
  border-color: var(--warning);
  background: var(--warning-soft);
}
.tool.interrupted {
  border-color: var(--muted);
}
.tool.interrupted .status {
  font-size: 0;
}
.tool.interrupted .status .spin {
  display: none;
}
.tool.interrupted .status::after {
  content: '执行中断';
  font-size: 11px;
}
.head {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 38px;
  padding: 7px 10px;
  cursor: pointer;
  user-select: none;
}
.tag-icon {
  color: var(--primary);
  font-size: 14px;
}
.name {
  color: var(--text);
  font-weight: 600;
}
.status {
  display: flex;
  align-items: center;
  gap: 4px;
  margin-left: auto;
  color: var(--muted);
  font-size: 11px;
}
.retry-tag {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  font-size: 10px;
  color: var(--warning);
  white-space: nowrap;
}
.retry-icon {
  font-size: 11px;
}
.spin {
  animation: spin 1s linear infinite;
}
.arrow {
  color: var(--muted);
  font-size: 12px;
  transition: transform 0.2s;
}
.arrow.open {
  transform: rotate(90deg);
}
.body {
  padding: 10px;
  border-top: 1px solid var(--tool-border);
}
.label {
  margin: 4px 0;
  color: var(--muted);
  font-size: 11px;
  font-weight: 600;
}
.block {
  max-height: 240px;
  margin: 0 0 8px;
  overflow: auto;
  border: 1px solid var(--tool-border);
  background: var(--surface);
  font-size: 12px;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

@media (max-width: 720px) {
  .tool {
    margin-left: 0;
  }
  .head {
    gap: 6px;
  }
  .status {
    font-size: 0;
  }
  .status .el-icon {
    font-size: 12px;
  }
}
</style>
