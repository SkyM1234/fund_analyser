<script setup lang="ts">
import { computed, ref, onMounted, watch } from 'vue'
import { Plus, Delete, ChatDotRound, Close, Loading } from '@element-plus/icons-vue'
import { listSessions, deleteSession, type SessionItem } from '../api/session'
import { useChatStore } from '../stores/chat'

const props = defineProps<{
  visible: boolean
}>()

const emit = defineEmits<{
  close: []
}>()

const store = useChatStore()
const sessions = ref<SessionItem[]>([])
const loading = ref(false)
let loadSeq = 0

const visibleSessions = computed<SessionItem[]>(() => {
  const merged = [...sessions.value]
  const knownIds = new Set(merged.map((session) => session.thread_id))

  for (const running of store.runningSessions) {
    if (knownIds.has(running.thread_id)) continue
    merged.unshift({
      ...running,
      last_checkpoint: '',
      created_at: null,
    })
  }

  return merged
})

async function load() {
  const seq = ++loadSeq
  loading.value = true
  try {
    const result = await listSessions()
    if (seq !== loadSeq) return
    sessions.value = result
  } catch (e) {
    console.error('加载会话失败', e)
  } finally {
    if (seq === loadSeq) loading.value = false
  }
}

async function onDelete(threadId: string) {
  if (store.isSessionStreaming(threadId)) return
  try {
    await deleteSession(threadId)
    sessions.value = sessions.value.filter((s) => s.thread_id !== threadId)
    // 如果删除的是当前正在显示的会话，重置到新建会话界面
    if (threadId === store.sessionId) {
      store.clear()
    }
  } catch (e) {
    console.error('删除会话失败', e)
  }
}

async function onSelect(threadId: string) {
  await store.loadSession(threadId)
}

function onNew() {
  store.clear()
}

watch(
  () => props.visible,
  (visible) => {
    if (visible) load()
  },
)

watch(
  () => store.sessionsVersion,
  () => {
    if (props.visible) load()
  },
)

onMounted(() => {
  if (props.visible) load()
})
</script>

<template>
  <div class="sidebar">
    <div class="header">
      <div>
        <h2>会话</h2>
        <span class="session-count">{{ visibleSessions.length }} 个历史记录</span>
      </div>
      <el-tooltip content="收起侧栏" placement="right">
        <el-button
          class="close-btn"
          text
          :icon="Close"
          aria-label="收起侧栏"
          @click="emit('close')"
        />
      </el-tooltip>
    </div>
    <div class="new-btn-wrap">
      <el-button type="primary" :icon="Plus" class="new-btn" @click="onNew">
        新建会话
      </el-button>
    </div>
    <div v-if="loading" class="loading">
      <el-skeleton :rows="4" animated />
    </div>
    <el-empty v-else-if="!visibleSessions.length" description="暂无会话" :image-size="72" class="empty" />
    <div v-else class="list">
      <div
        v-for="s in visibleSessions"
        :key="s.thread_id"
        :class="[
          'item',
          {
            active: s.thread_id === store.sessionId,
            running: store.isSessionStreaming(s.thread_id),
          },
        ]"
        @click="onSelect(s.thread_id)"
      >
        <el-icon v-if="store.isSessionStreaming(s.thread_id)" class="item-icon is-loading">
          <Loading />
        </el-icon>
        <el-icon v-else class="item-icon"><ChatDotRound /></el-icon>
        <div class="item-body">
          <div class="title">{{ s.first_message || '新会话' }}</div>
          <div class="meta">
            {{ store.isSessionStreaming(s.thread_id) ? '生成中...' : `${s.checkpoint_count} 条消息` }}
          </div>
        </div>
        <el-popconfirm title="确定删除该会话？" @confirm="onDelete(s.thread_id)">
          <template #reference>
            <el-button
              class="del"
              text
              :icon="Delete"
              :disabled="store.isSessionStreaming(s.thread_id)"
              aria-label="删除会话"
              title="删除会话"
              @click.stop
            />
          </template>
        </el-popconfirm>
      </div>
    </div>
  </div>
</template>

<style scoped>
.sidebar {
  width: 272px;
  height: 100%;
  background: var(--surface);
  display: flex;
  flex-direction: column;
}
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-height: 66px;
  padding: 14px 12px 8px 18px;
}
.header h2 {
  margin: 0;
  font-size: 14px;
  font-weight: 650;
  color: var(--text);
}
.session-count {
  display: block;
  margin-top: 3px;
  color: var(--muted);
  font-size: 11px;
}
.close-btn {
  width: 32px;
  height: 32px;
  padding: 0;
  color: var(--muted);
  font-size: 15px;
}
.close-btn:hover {
  color: var(--text);
  background: var(--surface-hover);
}
.new-btn-wrap {
  padding: 7px 14px 14px;
}
.new-btn {
  width: 100%;
  height: 38px;
  justify-content: flex-start;
  padding-left: 15px;
  box-shadow: 0 2px 6px rgba(23, 105, 170, 0.16);
}
.loading {
  padding: 12px 16px;
}
.empty {
  flex: 1;
}
.list {
  flex: 1;
  overflow-y: auto;
  padding: 2px 8px 12px;
}
.item {
  position: relative;
  display: flex;
  align-items: center;
  gap: 9px;
  min-height: 56px;
  margin-bottom: 3px;
  padding: 8px 8px 8px 11px;
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: background 0.15s ease, color 0.15s ease;
}
.item:hover {
  background: var(--surface-hover);
}
.item:hover .del {
  opacity: 1;
}
.item.active {
  background: var(--primary-soft);
}
.item.active::before {
  position: absolute;
  inset: 10px auto 10px 0;
  width: 3px;
  border-radius: 2px;
  background: var(--primary);
  content: "";
}
.item.running:not(.active) {
  background: var(--success-soft);
}
.item-icon {
  color: var(--muted);
  font-size: 16px;
  flex-shrink: 0;
}
.item.active .item-icon {
  color: var(--primary);
}
.item.running .item-icon {
  color: var(--success);
}
.item-body {
  flex: 1;
  min-width: 0;
}
.title {
  margin-bottom: 3px;
  overflow: hidden;
  color: var(--text);
  font-size: 13px;
  font-weight: 500;
  line-height: 1.35;
  white-space: nowrap;
  text-overflow: ellipsis;
}
.meta {
  color: var(--muted);
  font-size: 11px;
}
.item.running .meta {
  color: var(--success);
}
.del {
  width: 28px;
  height: 28px;
  padding: 0;
  opacity: 0;
  color: var(--muted);
  flex-shrink: 0;
  transition: opacity 0.15s;
}
.del:hover {
  color: var(--danger);
  background: var(--danger-soft);
}

@media (hover: none) {
  .del {
    opacity: 1;
  }
}
</style>
