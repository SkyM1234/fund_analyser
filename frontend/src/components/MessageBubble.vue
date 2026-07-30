<script setup lang="ts">
import MarkdownIt from 'markdown-it'
import hljs from 'highlight.js'
import 'highlight.js/styles/github.css'
import { computed, ref } from 'vue'
import { Check, DataAnalysis, DocumentCopy, EditPen, List, User } from '@element-plus/icons-vue'
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
    <div class="role">
      <span class="role-icon">
        <el-icon><User v-if="msg.role === 'user'" /><DataAnalysis v-else /></el-icon>
      </span>
      <span>{{ msg.role === 'user' ? '我' : '基金助手' }}</span>
    </div>

    <div v-if="msg.agents.length" class="agent-summary">
      <AgentBadge
        v-for="a in msg.agents"
        :key="a.task_id || a.agent_name"
        :agent_name="a.agent_name"
        :status="a.status"
      />
    </div>

    <div v-if="msg.plan && msg.plan.length" class="plan-summary">
      <div class="plan-title">
        <el-icon><List /></el-icon>
        <span>执行计划</span>
      </div>
      <div v-for="t in msg.plan" :key="t.task_id" class="plan-task">
        <AgentBadge :agent_name="t.assigned_agent" />
        <span class="plan-desc">{{ t.description }}</span>
        <span v-if="t.fund_codes.length" class="plan-funds">{{ t.fund_codes.join(', ') }}</span>
      </div>
    </div>

    <ToolCallCard v-for="(t, i) in msg.tools" :key="i" :step="t" />

    <div v-if="msg.role === 'user' && !isEditing" class="user-message-wrapper">
      <div class="content user-content">{{ msg.content }}</div>

      <div v-if="canEdit" class="action-buttons">
        <el-tooltip :content="copySuccess ? '已复制' : '复制'" placement="bottom">
          <el-button class="action-btn" text :icon="copySuccess ? Check : DocumentCopy" @click="copyContent" />
        </el-tooltip>
        <el-tooltip content="编辑并回溯" placement="bottom">
          <el-button class="action-btn" text :icon="EditPen" @click="startEdit" />
        </el-tooltip>
      </div>
    </div>

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

    <div v-else-if="msg.role === 'assistant' && (msg.content || msg.error)" class="content md" v-html="rendered" />

    <div v-if="msg.pending && !msg.retryNotice" class="pending">正在生成</div>
    <div v-if="msg.retryNotice" class="retry-notice">内容未通过合规审查，正在重新生成</div>
    <div v-if="msg.error" class="err">出错：{{ msg.error }}</div>
  </div>
</template>

<style scoped>
.bubble {
  margin-bottom: 26px;
  animation: fade-in 0.25s ease;
}
.role {
  display: flex;
  align-items: center;
  gap: 7px;
  margin-bottom: 8px;
  color: var(--text-secondary);
  font-size: 12px;
  font-weight: 600;
}
.role-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border: 1px solid var(--border);
  border-radius: 50%;
  background: var(--surface-subtle);
  color: var(--text-secondary);
  font-size: 13px;
}
.content {
  border-radius: var(--radius-md);
  line-height: 1.75;
}
.user-content {
  display: inline-block;
  max-width: min(680px, 88%);
  padding: 10px 14px;
  background: var(--user-bg);
  color: var(--user-fg);
  text-align: left;
  overflow-wrap: break-word;
}
.bubble.user {
  text-align: right;
}
.bubble.user .role {
  flex-direction: row-reverse;
}
.bubble.user .role-icon {
  border-color: #c8dce9;
  background: var(--primary-soft);
  color: var(--primary);
}
.md {
  padding: 0;
  color: var(--text);
  font-size: 14px;
}
.md :deep(p) {
  margin: 0 0 12px;
}
.md :deep(p:last-child) {
  margin-bottom: 0;
}
.md :deep(h1),
.md :deep(h2),
.md :deep(h3),
.md :deep(h4) {
  margin: 22px 0 10px;
  color: var(--text);
  line-height: 1.4;
}
.md :deep(h1) {
  font-size: 20px;
}
.md :deep(h2) {
  font-size: 18px;
}
.md :deep(h3),
.md :deep(h4) {
  font-size: 15px;
}
.md :deep(ul),
.md :deep(ol) {
  margin: 8px 0 14px;
  padding-left: 22px;
}
.md :deep(li) {
  margin: 5px 0;
}
.md :deep(blockquote) {
  margin: 12px 0;
  padding: 3px 14px;
  border-left: 3px solid var(--primary);
  background: var(--surface-subtle);
  color: var(--text-secondary);
}
.md :deep(table) {
  display: block;
  max-width: 100%;
  margin: 14px 0;
  overflow-x: auto;
  border-collapse: collapse;
}
.md :deep(th),
.md :deep(td) {
  min-width: 90px;
  padding: 8px 10px;
  border: 1px solid var(--border);
  text-align: left;
}
.md :deep(th) {
  background: var(--surface-subtle);
  color: var(--text-secondary);
  font-weight: 600;
}
.md :deep(a) {
  color: var(--primary);
  text-decoration: none;
}
.md :deep(a:hover) {
  text-decoration: underline;
}
.pending,
.retry-notice,
.err {
  margin-top: 7px;
  font-size: 12px;
}
.pending {
  color: var(--muted);
}
.pending::after {
  display: inline-block;
  width: 6px;
  height: 6px;
  margin-left: 7px;
  border-radius: 50%;
  background: var(--primary);
  content: "";
  animation: pulse 1.2s ease-in-out infinite;
}
.retry-notice {
  color: var(--warning);
}
.err {
  color: var(--danger);
}

.agent-summary {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin: 0 0 10px 31px;
}

.plan-summary {
  margin: 0 0 12px 31px;
  padding: 9px 11px;
  border-left: 3px solid var(--primary);
  border-radius: var(--radius-sm);
  background: var(--surface-subtle);
  font-size: 12px;
}
.plan-title {
  display: flex;
  align-items: center;
  gap: 6px;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 8px;
}
.plan-task {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 0;
}
.plan-desc {
  flex: 1;
  color: var(--text-secondary);
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

.user-message-wrapper {
  display: inline-block;
  position: relative;
  max-width: 100%;
}

.action-buttons {
  display: flex;
  justify-content: flex-end;
  margin-top: 4px;
  opacity: 0;
  transition: opacity 0.2s;
}

.user-message-wrapper:hover .action-buttons {
  opacity: 1;
}

.action-btn {
  width: 28px;
  height: 28px;
  color: var(--muted);
  padding: 0;
}

.action-btn:hover {
  color: var(--primary);
  background: var(--primary-soft);
}

.edit-area {
  display: inline-block;
  width: min(680px, 88%);
  text-align: left;
}
.edit-actions {
  display: flex;
  gap: 8px;
  margin-top: 8px;
  justify-content: flex-end;
}

@media (hover: none) {
  .action-buttons {
    opacity: 1;
  }
}

@media (max-width: 720px) {
  .bubble {
    margin-bottom: 22px;
  }
  .agent-summary,
  .plan-summary {
    margin-left: 0;
  }
  .user-content,
  .edit-area {
    max-width: 92%;
    width: auto;
  }
}
</style>
