<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import { Plus, Delete, ChatDotRound, Close } from '@element-plus/icons-vue'
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
      <h3>会话历史</h3>
      <el-button class="close-btn" text :icon="Close" @click="emit('close')" />
    </div>
    <div class="new-btn-wrap">
      <el-button type="primary" :icon="Plus" class="new-btn" @click="onNew">新建会话</el-button>
    </div>
    <div v-if="loading" class="loading">
      <el-skeleton :rows="4" animated />
    </div>
    <el-empty v-else-if="!sessions.length" description="暂无会话" :image-size="72" class="empty" />
    <div v-else class="list">
      <div
        v-for="s in sessions"
        :key="s.thread_id"
        :class="['item', { active: s.thread_id === store.sessionId }]"
        @click="onSelect(s.thread_id)"
      >
        <el-icon class="item-icon"><ChatDotRound /></el-icon>
        <div class="item-body">
          <div class="title">{{ s.first_message || '新会话' }}</div>
          <div class="meta">{{ s.checkpoint_count }} 条消息</div>
        </div>
        <el-popconfirm title="确定删除该会话？" @confirm="onDelete(s.thread_id)">
          <template #reference>
            <el-button class="del" text :icon="Delete" @click.stop />
          </template>
        </el-popconfirm>
      </div>
    </div>
  </div>
</template>

<style scoped>
.sidebar {
  width: 280px;
  height: 100%;
  background: var(--panel);
  display: flex;
  flex-direction: column;
}
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 12px 8px 16px;
}
.header h3 {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
  color: var(--text);
}
.close-btn {
  color: var(--muted);
  font-size: 16px;
}
.new-btn-wrap {
  padding: 8px 16px 12px;
}
.new-btn {
  width: 100%;
}
.loading { padding: 16px; }
.empty { flex: 1; }
.list {
  flex: 1;
  overflow-y: auto;
  padding: 4px 8px 8px;
}
.item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 10px;
  border-radius: var(--radius-sm);
  margin-bottom: 2px;
  cursor: pointer;
  transition: background 0.15s;
}
.item:hover {
  background: var(--bg);
}
.item:hover .del {
  opacity: 1;
}
.item.active {
  background: var(--tool-bg);
}
.item-icon {
  color: var(--muted);
  font-size: 16px;
  flex-shrink: 0;
}
.item.active .item-icon {
  color: var(--primary);
}
.item-body {
  flex: 1;
  min-width: 0;
}
.title {
  font-size: 13px;
  color: var(--text);
  margin-bottom: 2px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.meta {
  font-size: 11px;
  color: var(--muted);
}
.del {
  opacity: 0;
  transition: opacity 0.15s;
  color: var(--muted);
  flex-shrink: 0;
}
.del:hover {
  color: #c0392b;
}
</style>
