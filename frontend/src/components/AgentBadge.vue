<script setup lang="ts">
import { computed } from 'vue'
import { Check, Close, Connection, DataLine, Document, Loading, Operation } from '@element-plus/icons-vue'

const props = defineProps<{
  agent_name: string
  status?: 'running' | 'completed' | 'failed'
}>()

const agentLabel = computed(() => {
  const map: Record<string, string> = {
    analysis_agent: '结果分析',
    rag_agent: '年报检索',
    market_agent: '实时数据',
    arbiter_agent: '数据仲裁',
    fund_scope_agent: '范围确认',
  }
  return map[props.agent_name] || props.agent_name
})

const agentIcon = computed(() => {
  const map: Record<string, typeof Document> = {
    analysis_agent: DataLine,
    rag_agent: Document,
    market_agent: DataLine,
    arbiter_agent: Connection,
    fund_scope_agent: Operation,
  }
  return map[props.agent_name] || Operation
})
</script>

<template>
  <span :class="['agent-badge', status || '']" :title="agent_name">
    <el-icon class="agent-icon"><component :is="agentIcon" /></el-icon>
    <span class="agent-label">{{ agentLabel }}</span>
    <el-icon v-if="status === 'running'" class="agent-status is-loading"><Loading /></el-icon>
    <el-icon v-else-if="status === 'completed'" class="agent-status completed"><Check /></el-icon>
    <el-icon v-else-if="status === 'failed'" class="agent-status failed"><Close /></el-icon>
  </span>
</template>

<style scoped>
.agent-badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  min-height: 24px;
  padding: 2px 7px;
  border-radius: 4px;
  font-size: 11px;
  user-select: none;
  white-space: nowrap;
  border: 1px solid var(--border);
  background: var(--surface-subtle);
  color: var(--text-secondary);
}
.agent-badge.running {
  border-color: #b9d1e3;
  background: var(--primary-soft);
  color: var(--primary);
}
.agent-badge.completed {
  border-color: #b9ddca;
  background: var(--success-soft);
  color: var(--success);
}
.agent-badge.failed {
  border-color: #edc4c4;
  background: var(--danger-soft);
  color: var(--danger);
}
.agent-icon {
  font-size: 13px;
}
.agent-label {
  font-weight: 500;
}
.agent-status {
  font-size: 11px;
}
.completed {
  color: var(--success);
}
.failed {
  color: var(--danger);
}
</style>
