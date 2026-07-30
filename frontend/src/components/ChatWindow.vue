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
        <div class="hint-icon"><el-icon><DataLine /></el-icon></div>
        <h2>开始分析基金</h2>
        <p>输入基金名称、基金代码或比较需求，获取基于数据的回答。</p>
        <div class="hint-examples" aria-label="示例问题">
          <button class="example" @click="input = '金融科技ETF汇添富的基金经理是谁？'">
            <span>查询基金经理与基本信息</span>
            <el-icon><ArrowRight /></el-icon>
          </button>
          <button class="example" @click="input = '159103 和 159299 哪个规模更大？'">
            <span>比较 159103 与 159299 的基金规模</span>
            <el-icon><ArrowRight /></el-icon>
          </button>
        </div>
      </div>
      <div v-if="store.loadingSession" class="session-loading">正在载入会话...</div>
      <MessageBubble
        v-for="(m, idx) in store.messages"
        :key="m.id"
        :msg="m"
        :canEdit="m.role === 'user' && !store.streaming && !store.loadingSession"
        @edit="onEdit"
      />
    </div>
    <div class="composer">
      <div class="composer-shell">
        <el-input
          v-model="input"
          class="question-input"
          type="textarea"
          :rows="2"
          resize="none"
          placeholder="输入基金问题..."
          :disabled="store.streaming || store.loadingSession"
          @keydown="onKeydown"
        />
        <div class="composer-footer">
          <span class="keyboard-hint">Enter 发送 · Shift + Enter 换行</span>
          <div class="actions">
            <el-button v-if="store.streaming" :icon="CircleClose" @click="store.abort">
              中断
            </el-button>
            <el-button
              type="primary"
              :icon="Promotion"
              :loading="store.streaming"
              :disabled="store.streaming || store.loadingSession || !input.trim()"
              @click="onSubmit"
            >
              {{ store.streaming ? '生成中' : '发送' }}
            </el-button>
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
  width: 100%;
  max-width: 940px;
  margin: 0 auto;
  overflow-y: auto;
  padding: 32px 34px 24px;
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
.hint p {
  margin: 8px 0 26px;
  color: var(--text-secondary);
  font-size: 13px;
}
.hint-examples {
  display: grid;
  width: 100%;
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
  width: 100%;
  padding: 12px 24px 18px;
  background: linear-gradient(to bottom, rgba(255, 255, 255, 0), var(--surface) 16px);
}
.composer-shell {
  width: 100%;
  max-width: 872px;
  margin: 0 auto;
  overflow: hidden;
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-md);
  background: var(--surface);
  box-shadow: 0 4px 18px rgba(23, 33, 43, 0.08);
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
  line-height: 1.55;
}
.question-input :deep(.el-textarea__inner:hover),
.question-input :deep(.el-textarea__inner:focus) {
  box-shadow: none;
}
.composer-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  min-height: 46px;
  padding: 6px 8px 8px 14px;
}
.keyboard-hint {
  color: var(--muted);
  font-size: 11px;
}
.actions {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
}

@media (max-width: 720px) {
  .msgs {
    padding: 22px 16px 16px;
  }
  .hint {
    margin-top: 8vh;
  }
  .hint-examples {
    grid-template-columns: 1fr;
  }
  .composer {
    padding: 8px 10px 10px;
  }
  .keyboard-hint {
    display: none;
  }
  .composer-footer {
    justify-content: flex-end;
  }
}
</style>
