<script setup lang="ts">
import { nextTick, ref, watch } from 'vue'
import { Promotion, CircleClose } from '@element-plus/icons-vue'
import { useChatStore } from '../stores/chat'
import MessageBubble from './MessageBubble.vue'

const store = useChatStore()
const input = ref('')
const scroller = ref<HTMLDivElement | null>(null)

const onSubmit = async () => {
  const text = input.value.trim()
  if (!text || store.streaming) return
  input.value = ''
  await store.send(text)
}

const onKeydown = (e: KeyboardEvent) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    onSubmit()
  }
}

const onEdit = async (msgId: string, newContent: string) => {
  await store.rewindAndResend(msgId, newContent)
}

watch(
  () => store.messages.map((m) => m.content + m.tools.length).join('|'),
  async () => {
    await nextTick()
    scroller.value?.scrollTo({ top: scroller.value.scrollHeight, behavior: 'smooth' })
  },
)
</script>

<template>
  <div class="chat">
    <div class="msgs" ref="scroller">
      <div v-if="!store.messages.length" class="hint">
        <div class="hint-title">试试问我：</div>
        <div class="hint-examples">
          <button class="example" @click="input = '金融科技ETF汇添富的基金经理是谁？'">金融科技ETF汇添富的基金经理是谁？</button>
          <button class="example" @click="input = '159103 和 159299 哪个规模更大？'">159103 和 159299 哪个规模更大？</button>
        </div>
        <div class="tip">💡 点击左上角菜单按钮查看历史会话</div>
      </div>
      <MessageBubble
        v-for="(m, idx) in store.messages"
        :key="m.id"
        :msg="m"
        :canEdit="m.role === 'user' && !store.streaming"
        @edit="onEdit"
      />
    </div>
    <div class="composer">
      <el-input
        v-model="input"
        type="textarea"
        :rows="2"
        resize="none"
        placeholder="输入问题，Enter 发送 / Shift+Enter 换行"
        :disabled="store.streaming"
        @keydown="onKeydown"
      />
      <div class="actions">
        <el-button v-if="store.streaming" :icon="CircleClose" @click="store.abort">中断</el-button>
        <el-button
          type="primary"
          :icon="Promotion"
          :loading="store.streaming"
          :disabled="store.streaming || !input.trim()"
          @click="onSubmit"
        >
          {{ store.streaming ? '生成中…' : '发送' }}
        </el-button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.chat { display: flex; flex-direction: column; height: 100%; flex: 1; max-width: 920px; margin: 0 auto; width: 100%; }
.msgs { flex: 1; overflow-y: auto; padding: 24px; }
.hint {
  color: var(--muted);
  font-size: 13px;
  line-height: 1.8;
  animation: fade-in 0.3s ease;
}
.hint-title { font-size: 14px; color: var(--text); margin-bottom: 10px; font-weight: 500; }
.hint-examples { display: flex; flex-direction: column; gap: 8px; margin-bottom: 14px; }
.example {
  align-self: flex-start;
  border: 1px solid var(--border);
  background: var(--panel);
  color: var(--text);
  padding: 8px 14px;
  border-radius: 999px;
  font-size: 13px;
  transition: all 0.15s;
}
.example:hover {
  border-color: var(--primary);
  color: var(--primary);
  box-shadow: var(--shadow-sm);
}
.tip { color: var(--primary); font-size: 12px; }
.composer {
  border-top: 1px solid var(--border);
  background: var(--panel);
  padding: 12px 24px;
}
.actions { display: flex; gap: 8px; justify-content: flex-end; margin-top: 8px; }

@media (max-width: 640px) {
  .msgs { padding: 16px; }
  .composer { padding: 10px 14px; }
}
</style>
