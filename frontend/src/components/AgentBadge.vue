<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  agent_name: string
  status?: 'running' | 'completed' | 'failed'
}>()

const agentLabel = computed(() => {
  const map: Record<string, string> = {
    rag_agent: '年报检索',
    market_agent: '实时数据',
    arbiter_agent: '数据仲裁',
  }
  return map[props.agent_name] || props.agent_name
})

const agentIcon = computed(() => {
  const map: Record<string, string> = {
    rag_agent: '📄',
    market_agent: '📊',
    arbiter_agent: '⚖️',
  }
  return map[props.agent_name] || '🤖'
})
</script>

<template>
  <span :class="['agent-badge', status || '']" :title="agent_name">
    <span class="agent-icon">{{ agentIcon }}</span>
    <span class="agent-label">{{ agentLabel }}</span>
    <span v-if="status === 'running'" class="agent-status running-dot" />
    <span v-else-if="status === 'completed'" class="agent-status completed">✓</span>
    <span v-else-if="status === 'failed'" class="agent-status failed">✗</span>
  </span>
</template>

<style scoped>
.agent-badge {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  padding: 1px 7px;
  border-radius: 999px;
  font-size: 11px;
  line-height: 1.6;
  user-select: none;
  white-space: nowrap;
  border: 1px solid var(--border);
  background: var(--panel);
  color: var(--muted);
}
.agent-badge.running {
  border-color: #93c5fd;
  background: #eff6ff;
  color: #2563eb;
}
.agent-badge.completed {
  border-color: #86efac;
  background: #f0fdf4;
  color: #16a34a;
}
.agent-badge.failed {
  border-color: #fca5a5;
  background: #fef2f2;
  color: #dc2626;
}
.agent-icon {
  font-size: 12px;
  line-height: 1;
}
.agent-label {
  font-weight: 500;
}
.agent-status {
  font-size: 10px;
  line-height: 1;
}
.running-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #2563eb;
  animation: agent-pulse 1.2s ease-in-out infinite;
}
.completed {
  color: #16a34a;
  font-weight: 700;
}
.failed {
  color: #dc2626;
  font-weight: 700;
}

@keyframes agent-pulse {
  0%, 100% { opacity: 0.3; }
  50% { opacity: 1; }
}
</style>
