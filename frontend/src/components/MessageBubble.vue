<script setup lang="ts">
import MarkdownIt from 'markdown-it'
import hljs from 'highlight.js'
import 'highlight.js/styles/github.css'
import { computed, ref } from 'vue'
import { ArrowRight, Check, DataAnalysis, DocumentCopy, EditPen, List, User } from '@element-plus/icons-vue'
import ToolCallCard from './ToolCallCard.vue'
import AgentBadge from './AgentBadge.vue'
import type { Message, TraceEvent } from '../stores/chat'

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

interface ExecutionGroup {
  task_id: string
  agent_name: string
  description: string
  status?: 'running' | 'completed' | 'failed'
  tools: Message['tools']
  thoughts: Message['thoughts']
  events: TraceEvent[]
}

const executionGroups = computed<ExecutionGroup[]>(() => {
  const groups: ExecutionGroup[] = []
  const byTaskId = new Map<string, ExecutionGroup>()

  const addGroup = (
    taskId: string,
    agentName: string,
    description: string,
    status?: 'running' | 'completed' | 'failed',
  ) => {
    const existing = byTaskId.get(taskId)
    if (existing) {
      existing.agent_name = agentName || existing.agent_name
      existing.description = description || existing.description
      existing.status = status || existing.status
      return existing
    }

    const group: ExecutionGroup = {
      task_id: taskId,
      agent_name: agentName,
      description,
      status,
      tools: [],
      thoughts: [],
      events: [],
    }
    groups.push(group)
    byTaskId.set(taskId, group)
    return group
  }

  for (const task of props.msg.plan || []) {
    addGroup(task.task_id, task.assigned_agent, task.description)
  }
  for (const agent of props.msg.agents) {
    addGroup(agent.task_id, agent.agent_name, agent.description, agent.status)
  }

  for (const tool of props.msg.tools) {
    let group = tool.task_id ? byTaskId.get(tool.task_id) : undefined
    if (!group && tool.agent_name) {
      group = groups.find((item) => item.agent_name === tool.agent_name)
    }
    if (!group) {
      group = addGroup(
        tool.task_id || `tool-${groups.length + 1}`,
        tool.agent_name || 'unknown_agent',
        '未关联任务的工具调用',
      )
    }
    group.tools.push(tool)
  }

  for (const thought of props.msg.thoughts) {
    let group = thought.task_id ? byTaskId.get(thought.task_id) : undefined
    if (!group && thought.agent_name) {
      group = groups.find((item) => item.agent_name === thought.agent_name)
    }
    if (!group) {
      group = addGroup(
        thought.task_id || `thought-${groups.length + 1}`,
        thought.agent_name || 'unknown_agent',
        'LLM thought',
      )
    }
    group.thoughts.push(thought)
  }

  if (props.msg.trace_events.length) {
    const eventGroups = new Map<string, ExecutionGroup>()
    for (const group of groups) eventGroups.set(group.task_id, group)

    const toolEvents = new Map<string, TraceEvent>()
    for (const event of [...props.msg.trace_events].sort((a, b) => a.sequence - b.sequence)) {
      const taskId = event.task_id || `trace-${groups.length + 1}`
      let group = eventGroups.get(taskId)
      if (!group) {
        group = addGroup(taskId, event.agent_name || 'unknown_agent', '执行任务')
        eventGroups.set(taskId, group)
      }

      if (event.type === 'tool_result' && event.tool_call_id) {
        const toolEvent = toolEvents.get(event.tool_call_id)
        if (toolEvent) {
          toolEvent.output = event.output
          continue
        }
      }
      if (event.type === 'tool_call' && event.tool_call_id) {
        toolEvents.set(event.tool_call_id, event)
      }
      group.events.push(event)
    }
  } else {
    for (const group of groups) {
      group.events = [
        ...group.thoughts.map((thought) => ({
          type: 'agent_thought' as const,
          event_id: thought.thought_id,
          sequence: 0,
          thought_id: thought.thought_id,
          content: thought.content,
          agent_name: thought.agent_name,
          task_id: thought.task_id,
        })),
        ...group.tools.map((tool, index) => ({
          type: 'tool_call' as const,
          event_id: tool.tool_call_id || `${group.task_id}-${index}`,
          sequence: index + 1,
          name: tool.name,
          args: tool.args,
          output: tool.output,
          tool_call_id: tool.tool_call_id,
          agent_name: tool.agent_name,
          task_id: tool.task_id,
        })),
      ]
    }
  }

  return groups
})

const executionToolCount = computed(() =>
  executionGroups.value.reduce(
    (total, group) =>
      total + group.events.filter((event) => event.type === 'tool_call').length,
    0,
  ),
)
const executionThoughtCount = computed(() =>
  executionGroups.value.reduce(
    (total, group) => total + group.events.filter((event) => event.type === 'agent_thought').length,
    0,
  ),
)

const executionOpen = ref(true)
const openThoughts = ref(new Set<string>())
const collapsedTasks = ref(new Set<string>())
const isEditing = ref(false)
const editContent = ref('')
const copySuccess = ref(false)

const isTaskOpen = (taskId: string) => !collapsedTasks.value.has(taskId)

const toggleTask = (taskId: string) => {
  const next = new Set(collapsedTasks.value)
  if (next.has(taskId)) {
    next.delete(taskId)
  } else {
    next.add(taskId)
  }
  collapsedTasks.value = next
}

const toggleThought = (thoughtId: string) => {
  const next = new Set(openThoughts.value)
  if (next.has(thoughtId)) {
    next.delete(thoughtId)
  } else {
    next.add(thoughtId)
  }
  openThoughts.value = next
}

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

    <div v-if="msg.trace_events.length > 0 || msg.tools.length > 0 || msg.thoughts.length > 0" class="execution-trace">
      <div
        class="trace-title"
        role="button"
        tabindex="0"
        :aria-expanded="executionOpen"
        @click="executionOpen = !executionOpen"
        @keydown.enter.prevent="executionOpen = !executionOpen"
        @keydown.space.prevent="executionOpen = !executionOpen"
      >
        <el-icon class="trace-arrow" :class="{ open: executionOpen }"><ArrowRight /></el-icon>
        <el-icon><List /></el-icon>
        <span>执行轨迹</span>
        <span class="trace-meta">
          {{ executionGroups.length }} 个任务 · {{ executionToolCount }} 次工具调用
        </span>
      </div>
      <el-collapse-transition>
        <div v-if="executionOpen">
          <div
            v-for="(group, groupIndex) in executionGroups"
            :key="group.task_id"
            class="execution-task"
          >
            <div
              class="task-head"
              role="button"
              tabindex="0"
              :aria-expanded="isTaskOpen(group.task_id)"
              @click="toggleTask(group.task_id)"
              @keydown.enter.prevent="toggleTask(group.task_id)"
              @keydown.space.prevent="toggleTask(group.task_id)"
            >
              <span class="task-index">{{ groupIndex + 1 }}</span>
              <div class="task-main">
                <div class="task-agent">
                  <AgentBadge
                    v-if="group.agent_name && group.agent_name !== 'unknown_agent'"
                    :agent_name="group.agent_name"
                    :status="group.status"
                  />
                  <span class="task-id">{{ group.task_id }}</span>
                </div>
                <div class="task-description">{{ group.description }}</div>
              </div>
              <span class="task-count">
                {{ group.events.filter((event) => event.type === 'tool_call').length }} 个工具
              </span>
              <el-icon class="task-arrow" :class="{ open: isTaskOpen(group.task_id) }">
                <ArrowRight />
              </el-icon>
            </div>
            <el-collapse-transition>
              <div v-if="isTaskOpen(group.task_id)">
                <div v-if="group.events.length" class="task-events">
                  <div
                    v-for="event in group.events"
                    :key="event.event_id"
                  >
                    <ToolCallCard
                      v-if="event.type === 'tool_call'"
                      :step="{
                        name: event.name || '',
                        args: event.args,
                        output: event.output,
                        tool_call_id: event.tool_call_id,
                        agent_name: event.agent_name,
                        task_id: event.task_id,
                      }"
                    />
                    <div v-else-if="event.type === 'agent_thought'" class="thought">
                      <div
                        class="thought-head"
                        role="button"
                        tabindex="0"
                        :aria-expanded="openThoughts.has(event.event_id)"
                        @click="toggleThought(event.event_id)"
                        @keydown.enter.prevent="toggleThought(event.event_id)"
                        @keydown.space.prevent="toggleThought(event.event_id)"
                      >
                        <el-icon
                          class="thought-arrow"
                          :class="{ open: openThoughts.has(event.event_id) }"
                        >
                          <ArrowRight />
                        </el-icon>
                        <span class="thought-label">思考过程</span>
                      </div>
                      <el-collapse-transition>
                        <div v-if="openThoughts.has(event.event_id)" class="thought-content">
                          {{ event.content }}
                        </div>
                      </el-collapse-transition>
                    </div>
                    <div v-else-if="event.type === 'tool_retry'" class="trace-retry">
                      <el-icon><ArrowRight /></el-icon>
                      <span>工具重试{{ event.attempt ? ` · 第 ${event.attempt} 次` : '' }}</span>
                      <span v-if="event.reason">{{ event.reason }}</span>
                    </div>
                  </div>
                </div>
              </div>
            </el-collapse-transition>
          </div>
        </div>
      </el-collapse-transition>
    </div>

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
    <div v-if="msg.retryNotice" class="retry-notice">内容未通过合规审核，正在重新生成</div>
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

.execution-trace {
  margin: 0 0 12px 31px;
  overflow: hidden;
  border-left: 3px solid var(--primary);
  border-radius: var(--radius-sm);
  background: var(--surface-subtle);
  font-size: 12px;
}
.trace-title {
  display: flex;
  align-items: center;
  gap: 6px;
  min-height: 34px;
  padding: 7px 11px;
  border-bottom: 1px solid var(--border);
  color: var(--text);
  font-weight: 600;
  cursor: pointer;
  user-select: none;
}
.trace-title:hover {
  background: var(--surface-hover);
}
.trace-title:focus-visible {
  outline: 2px solid var(--primary);
  outline-offset: -2px;
}
.trace-arrow {
  color: var(--muted);
  font-size: 12px;
  transition: transform 0.2s ease;
}
.trace-arrow.open {
  transform: rotate(90deg);
}
.trace-meta {
  margin-left: auto;
  color: var(--muted);
  font-size: 11px;
  font-weight: 400;
}
.execution-task {
  padding: 9px 11px 7px;
  border-bottom: 1px solid var(--border);
}
.execution-task:last-child {
  border-bottom: 0;
}
.task-head {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
  cursor: pointer;
  user-select: none;
}
.task-head:hover {
  background: var(--surface-hover);
}
.task-head:focus-visible {
  outline: 2px solid var(--primary);
  outline-offset: 2px;
}
.task-index {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  flex: 0 0 20px;
  border: 1px solid var(--border-strong);
  border-radius: 50%;
  background: var(--surface);
  color: var(--primary);
  font-size: 11px;
  font-weight: 700;
}
.task-main {
  min-width: 0;
  flex: 1;
}
.task-agent {
  display: flex;
  align-items: center;
  gap: 7px;
  min-width: 0;
}
.task-id {
  overflow: hidden;
  color: var(--muted);
  font-family: monospace;
  font-size: 10px;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.task-description {
  margin-top: 4px;
  color: var(--text-secondary);
  line-height: 1.45;
  overflow-wrap: anywhere;
}
.task-count {
  flex: 0 0 auto;
  color: var(--muted);
  font-size: 11px;
}
.task-arrow {
  flex: 0 0 auto;
  color: var(--muted);
  font-size: 12px;
  transition: transform 0.2s ease;
}
.task-arrow.open {
  transform: rotate(90deg);
}
.task-events {
  margin-top: 7px;
}
.thought {
  margin: 0 0 8px 31px;
  padding: 8px 10px;
  border-left: 3px solid var(--primary);
  border-radius: var(--radius-sm);
  background: var(--surface);
}
.thought:last-child {
  margin-bottom: 0;
}
.thought-head {
  display: flex;
  align-items: center;
  gap: 5px;
  min-height: 18px;
  cursor: pointer;
  user-select: none;
}
.thought-head:hover .thought-label {
  color: var(--text);
}
.thought-head:focus-visible {
  outline: 2px solid var(--primary);
  outline-offset: 2px;
}
.thought-arrow {
  color: var(--muted);
  font-size: 11px;
  transition: transform 0.2s ease;
}
.thought-arrow.open {
  transform: rotate(90deg);
}
.thought-label {
  color: var(--primary);
  font-size: 11px;
  font-weight: 600;
}
.thought-content {
  margin: 4px 0 0 16px;
  color: var(--text-secondary);
  line-height: 1.55;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
.trace-retry {
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 0 0 8px 31px;
  color: var(--warning);
  font-size: 12px;
  line-height: 1.4;
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
  padding: 0;
  color: var(--muted);
}
.action-btn:hover {
  background: var(--primary-soft);
  color: var(--primary);
}
.edit-area {
  display: inline-block;
  width: min(680px, 88%);
  text-align: left;
}
.edit-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 8px;
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
  .execution-trace {
    margin-left: 0;
  }
  .trace-meta {
    display: none;
  }
  .task-count {
    font-size: 10px;
  }
  .thought,
  .trace-retry {
    margin-left: 0;
  }
  .user-content,
  .edit-area {
    width: auto;
    max-width: 92%;
  }
}
</style>
