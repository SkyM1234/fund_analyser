<script setup lang="ts">
import { computed, ref } from 'vue'
import { ArrowRight, Tools, Loading, RefreshRight } from '@element-plus/icons-vue'
import type { ToolStep } from '../stores/chat'
import AgentBadge from './AgentBadge.vue'

const props = defineProps<{ step: ToolStep }>()
const open = ref(false)

const isRetry = computed(() => (props.step.retry_attempt ?? 0) > 0)
</script>

<template>
  <div :class="['tool', { retry: isRetry }]">
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
  border: 1px solid var(--tool-border);
  background: var(--tool-bg);
  border-radius: var(--radius-sm);
  margin-bottom: 8px;
  font-size: 13px;
  overflow: hidden;
}
.tool.retry {
  border-style: dashed;
  border-color: #fbbf24;
  background: #fffbeb;
}
.head {
  display: flex; align-items: center; gap: 8px;
  padding: 7px 10px;
  cursor: pointer; user-select: none;
}
.tag-icon {
  color: var(--primary);
  font-size: 14px;
}
.name { font-weight: 600; color: var(--primary); }
.status { color: var(--muted); margin-left: auto; display: flex; align-items: center; gap: 4px; }
.retry-tag {
  display: inline-flex;
  align-items: center;
  gap: 2px;
  padding: 0 5px;
  border-radius: 999px;
  font-size: 10px;
  color: #92400e;
  background: #fef3c7;
  white-space: nowrap;
}
.retry-icon { font-size: 11px; }
.spin { animation: spin 1s linear infinite; }
.arrow { color: var(--muted); transition: transform 0.2s; font-size: 12px; }
.arrow.open { transform: rotate(90deg); }
.body { padding: 8px 10px 10px; border-top: 1px solid var(--tool-border); }
.label { font-size: 12px; color: var(--muted); margin: 4px 0 2px; }
.block {
  background: #fff; border: 1px solid var(--tool-border);
  margin: 0; max-height: 240px; overflow: auto;
  font-size: 12px;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
</style>
