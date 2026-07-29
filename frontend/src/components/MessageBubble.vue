<script setup lang="ts">
import MarkdownIt from 'markdown-it'
import hljs from 'highlight.js'
import 'highlight.js/styles/github.css'
import { computed, ref } from 'vue'
import { DocumentCopy, Check, EditPen } from '@element-plus/icons-vue'
import ToolCallCard from './ToolCallCard.vue'
import AgentBadge from './AgentBadge.vue'
import type { Message } from '../stores/chat'

const props = defineProps<{
  msg: Message
  canEdit?: boolean
}>()

const emit = defineEmits<{
  edit: [id: string, newContent: string]
}>()

const md = new MarkdownIt({
  linkify: true,
  breaks: true,
  highlight(code, lang) {
    if (lang && hljs.getLanguage(lang)) {
      try {
        return hljs.highlight(code, { language: lang }).value
      } catch {}
    }
    return ''
  },
})

const rendered = computed(() => md.render(props.msg.content || ''))

// 编辑状态
const isEditing = ref(false)
const editContent = ref('')
const copySuccess = ref(false)

const startEdit = () => {
  editContent.value = props.msg.content
  isEditing.value = true
}

const cancelEdit = () => {
  isEditing.value = false
  editContent.value = ''
}

const confirmEdit = () => {
  const newContent = editContent.value.trim()
  if (newContent && newContent !== props.msg.content) {
    emit('edit', props.msg.id, newContent)
  }
  isEditing.value = false
}

const copyContent = async () => {
  try {
    await navigator.clipboard.writeText(props.msg.content)
    copySuccess.value = true
    setTimeout(() => {
      copySuccess.value = false
    }, 2000)
  } catch (err) {
    console.error('复制失败:', err)
  }
}
</script>

<template>
  <div :class="['bubble', msg.role]">
    <div class="role">{{ msg.role === 'user' ? '我' : '助手' }}</div>

    <!-- Agent 执行摘要：显示本条消息关联的子Agent及其状态 -->
    <div v-if="msg.agents.length" class="agent-summary">
      <AgentBadge
        v-for="a in msg.agents"
        :key="a.task_id || a.agent_name"
        :agent_name="a.agent_name"
        :status="a.status"
      />
    </div>

    <!-- Supervisor 规划摘要：简洁展示任务分配 -->
    <div v-if="msg.plan && msg.plan.length" class="plan-summary">
      <div class="plan-title">📋 执行计划</div>
      <div v-for="t in msg.plan" :key="t.task_id" class="plan-task">
        <AgentBadge :agent_name="t.assigned_agent" />
        <span class="plan-desc">{{ t.description }}</span>
        <span v-if="t.fund_codes.length" class="plan-funds">{{ t.fund_codes.join(', ') }}</span>
      </div>
    </div>

    <ToolCallCard v-for="(t, i) in msg.tools" :key="i" :step="t" />

    <!-- 用户消息：普通状态 -->
    <div v-if="msg.role === 'user' && !isEditing" class="user-message-wrapper">
      <div class="content user-content">{{ msg.content }}</div>

      <!-- 操作按钮：鼠标悬停时显示 -->
      <div v-if="canEdit" class="action-buttons">
        <el-tooltip :content="copySuccess ? '已复制' : '复制'" placement="bottom">
          <el-button class="action-btn" text :icon="copySuccess ? Check : DocumentCopy" @click="copyContent" />
        </el-tooltip>
        <el-tooltip content="编辑并回溯" placement="bottom">
          <el-button class="action-btn" text :icon="EditPen" @click="startEdit" />
        </el-tooltip>
      </div>
    </div>

    <!-- 用户消息：编辑状态 -->
    <div v-if="msg.role === 'user' && isEditing" class="edit-area">
      <el-input
        v-model="editContent"
        type="textarea"
        :rows="3"
        class="edit-input"
        autofocus
      />
      <div class="edit-actions">
        <el-button size="small" @click="cancelEdit">取消</el-button>
        <el-button size="small" type="primary" @click="confirmEdit">确定</el-button>
      </div>
    </div>

    <!-- 助手消息 -->
    <div v-else-if="msg.role === 'assistant' && (msg.content || msg.error)" class="content md" v-html="rendered" />

    <div v-if="msg.pending && !msg.retryNotice" class="pending">生成中…</div>
    <div v-if="msg.retryNotice" class="retry-notice">内容未通过合规审查，正在重新生成…</div>
    <div v-if="msg.error" class="err">出错：{{ msg.error }}</div>
  </div>
</template>

<style scoped>
.bubble { margin-bottom: 16px; animation: fade-in 0.25s ease; }
.role { font-size: 12px; color: var(--muted); margin-bottom: 4px; }
.content { padding: 10px 14px; border-radius: var(--radius-md); line-height: 1.7; }
.user-content {
  background: var(--user-bg); color: var(--user-fg);
  display: inline-block;
  max-width: 600px;
  text-align: left;
  word-wrap: break-word;
  overflow-wrap: break-word;
  box-shadow: var(--shadow-sm);
}
.bubble.user { text-align: right; }
.bubble.user .role { text-align: right; }
.md { background: var(--panel); border: 1px solid var(--border); box-shadow: var(--shadow-sm); }
.md :deep(p) { margin: 6px 0; }
.md :deep(table) { border-collapse: collapse; }
.md :deep(th), .md :deep(td) { border: 1px solid var(--border); padding: 4px 8px; }
.pending { font-size: 12px; color: var(--muted); margin-top: 4px; }
.pending::after { content: ''; display: inline-block; width: 6px; height: 6px; margin-left: 6px; border-radius: 50%; background: var(--primary); animation: pulse 1.2s ease-in-out infinite; }
.retry-notice { font-size: 12px; color: #d97706; margin-top: 4px; }
.err { font-size: 12px; color: #c0392b; margin-top: 4px; }

/* Agent 执行摘要 */
.agent-summary {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-bottom: 8px;
}

/* Supervisor 规划摘要 */
.plan-summary {
  margin-bottom: 10px;
  padding: 8px 10px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--panel);
  font-size: 12px;
}
.plan-title {
  font-weight: 600;
  color: var(--text);
  margin-bottom: 6px;
}
.plan-task {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 4px;
  padding: 2px 0;
}
.plan-desc {
  color: var(--text);
  flex: 1;
}
.plan-funds {
  color: var(--primary);
  font-size: 11px;
  font-weight: 500;
  font-family: monospace;
}

@keyframes pulse {
  0%, 100% { opacity: 0.3; }
  50% { opacity: 1; }
}

/* 用户消息包装器 */
.user-message-wrapper {
  display: inline-block;
  position: relative;
  max-width: 100%;
}

/* 操作按钮容器 */
.action-buttons {
  display: flex;
  gap: 2px;
  justify-content: flex-end;
  margin-top: 4px;
  opacity: 0;
  transition: opacity 0.2s;
}

.user-message-wrapper:hover .action-buttons {
  opacity: 1;
}

.action-btn {
  color: var(--muted);
  padding: 4px;
}

.action-btn:hover {
  color: var(--primary);
}

/* 编辑区域 */
.edit-area {
  display: inline-block;
  max-width: 80%;
  min-width: 320px;
  text-align: left;
}
.edit-actions {
  display: flex;
  gap: 8px;
  margin-top: 8px;
  justify-content: flex-end;
}
</style>
