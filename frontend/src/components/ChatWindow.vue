<script setup lang="ts">
import { nextTick, ref, watch } from 'vue'
import { ArrowRight, CircleClose, DataLine, Promotion } from '@element-plus/icons-vue'
import { useChatStore } from '../stores/chat'
import MessageBubble from './MessageBubble.vue'

const store = useChatStore()
const input = ref('')
const scroller = ref<HTMLDivElement | null>(null)

const onSubmit = async () => {
  const text = input.value.trim()
  if (!text || store.streaming || store.loadingSession) return
  input.value = ''
  await store.send(text)
}

const onKeydown = (e: KeyboardEvent) => {
  if (e.isComposing || e.keyCode === 229) return
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
      <div class="message-list">
      <div v-if="!store.messages.length" class="hint">
        <div class="hint-icon"><el-icon><DataLine /></el-icon></div>
        <h2>开始分析基金</h2>
        <div class="hint-examples" aria-label="示例问题">
          <button class="example" @click="input = '金融科技ETF汇添富的基金经理是谁？'">
            <span>金融科技ETF汇添富的基金经理是谁？</span>
            <el-icon><ArrowRight /></el-icon>
          </button>
          <button class="example" @click="input = '159103 和 159299 哪个规模更大？'">
            <span>159103 和 159299 哪个规模更大？</span>
            <el-icon><ArrowRight /></el-icon>
          </button>
        </div>
      </div>
      <div v-if="store.loadingSession" class="session-loading">正在载入会话...</div>
      <MessageBubble
        v-for="m in store.messages"
        :key="m.id"
        :msg="m"
        :canEdit="m.role === 'user' && !store.streaming && !store.loadingSession"
        @edit="onEdit"
      />
      </div>
    </div>
    <div class="composer">
      <div class="composer-shell">
        <el-input
          v-model="input"
          class="question-input"
          type="textarea"
          :autosize="{ minRows: 2, maxRows: 6 }"
          resize="none"
          placeholder="输入基金问题..."
          aria-label="基金问题"
          :disabled="store.streaming || store.loadingSession"
          @keydown="onKeydown"
        />
        <div class="composer-footer">
          <div class="actions">
            <el-button
              v-if="store.streaming"
              :icon="CircleClose"
              :loading="store.cancelling"
              :disabled="store.cancelling"
              @click="store.abort"
            >
              中断
            </el-button>
            <el-tooltip :content="store.streaming ? '生成中' : '发送'" placement="top">
              <span class="send-control">
                <el-button
                  class="send-btn"
                  type="primary"
                  :icon="Promotion"
                  :loading="store.streaming"
                  :disabled="store.streaming || store.loadingSession || !input.trim()"
                  :aria-label="store.streaming ? '生成中' : '发送'"
                  @click="onSubmit"
                />
              </span>
            </el-tooltip>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.chat {
  display: flex;
  flex: 1;
  flex-direction: column;
  width: 100%;
  min-width: 0;
  height: 100%;
  background: var(--surface);
}
.msgs {
  flex: 1;
  min-height: 0;
  width: 100%;
  overflow-y: auto;
  scrollbar-gutter: stable;
}
.message-list {
  width: 100%;
  max-width: 960px;
  margin: 0 auto;
  padding: 32px 40px 8px;
}
.hint {
  display: flex;
  flex-direction: column;
  align-items: center;
  max-width: 620px;
  margin: min(14vh, 120px) auto 0;
  text-align: center;
  color: var(--muted);
  font-size: 13px;
  line-height: 1.6;
  animation: fade-in 0.3s ease;
}
.hint-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 48px;
  height: 48px;
  margin-bottom: 18px;
  border: 1px solid #c8dce9;
  border-radius: var(--radius-md);
  background: var(--primary-soft);
  color: var(--primary);
  font-size: 24px;
}
.hint h2 {
  margin: 0;
  color: var(--text);
  font-size: 20px;
  font-weight: 650;
}
.hint-examples {
  display: grid;
  width: 100%;
  margin-top: 26px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}
.example {
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-height: 58px;
  gap: 12px;
  padding: 10px 13px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text);
  font-size: 13px;
  line-height: 1.4;
  text-align: left;
  transition: border-color 0.15s ease, background 0.15s ease, color 0.15s ease;
}
.example:hover {
  border-color: var(--primary);
  background: var(--primary-soft);
  color: var(--primary);
}
.example .el-icon {
  flex: 0 0 auto;
}
.session-loading {
  padding: 20px 0;
  color: var(--muted);
  font-size: 13px;
  text-align: center;
}
.composer {
  flex-shrink: 0;
  width: 100%;
  padding: 12px 40px max(20px, env(safe-area-inset-bottom));
  background: var(--surface);
}
.composer-shell {
  width: 100%;
  max-width: 880px;
  margin: 0 auto;
  overflow: hidden;
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-md);
  background: var(--surface);
  box-shadow: 0 3px 14px rgba(23, 33, 43, 0.05);
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.composer-shell:focus-within {
  border-color: var(--primary);
  box-shadow: 0 4px 18px rgba(23, 105, 170, 0.12);
}
.question-input :deep(.el-textarea__inner) {
  min-height: 66px !important;
  padding: 14px 15px 8px;
  border: 0;
  border-radius: 0;
  background: transparent;
  box-shadow: none;
  color: var(--text);
  font-size: 15px;
  line-height: 1.55;
}
.question-input :deep(.el-textarea__inner:hover),
.question-input :deep(.el-textarea__inner:focus) {
  box-shadow: none;
}
.composer-footer {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 12px;
  min-height: 46px;
  padding: 6px 8px 8px 14px;
}
.actions {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
  align-items: center;
}
.send-control {
  display: inline-flex;
}
.send-btn {
  width: 34px;
  height: 34px;
  margin: 0;
  padding: 0;
  font-size: 17px;
}

@media (max-width: 720px) {
  .message-list {
    padding: 22px 16px 16px;
  }
  .hint {
    margin-top: 8vh;
  }
  .hint-examples {
    grid-template-columns: 1fr;
  }
  .composer {
    padding: 8px 12px max(12px, env(safe-area-inset-bottom));
  }
}
</style>
