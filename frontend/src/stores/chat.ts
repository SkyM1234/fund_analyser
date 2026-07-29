import { defineStore } from 'pinia'
import { ref } from 'vue'
import { sendChatStream, type ChatHistoryItem } from '../api/chat'

export interface ToolStep {
  name: string
  args: unknown
  output?: string
  agent_name?: string
  retry_attempt?: number
}

export interface AgentStep {
  agent_name: string
  task_id: string
  description: string
  status: 'running' | 'completed' | 'failed'
}

export interface PlanTask {
  task_id: string
  task_type: string
  description: string
  assigned_agent: string
  fund_codes: string[]
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  tools: ToolStep[]
  agents: AgentStep[]
  plan?: PlanTask[]
  pending?: boolean
  error?: string
  retryNotice?: string
}

const uid = () => {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID()
  }
  // Fallback: crypto.randomUUID 仅在 localhost / HTTPS 下可用
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0
    return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16)
  })
}

export const useChatStore = defineStore('chat', () => {
  const messages = ref<Message[]>([])
  const streaming = ref(false)
  const sessionId = ref<string>(uid())
  const sessionsVersion = ref(0)
  let aborter: AbortController | null = null

  const buildHistory = (): ChatHistoryItem[] =>
    messages.value
      .filter((m) => !m.pending && !m.error)
      .map((m) => ({ role: m.role, content: m.content }))

  // 待处理的 retry 标记：tool_retry 事件到达后暂存，由下一个 onToolCall 消费
  let _pendingRetry: { agent_name: string; attempt: number } | null = null

  async function send(text: string) {
    if (!text.trim() || streaming.value) return

    _pendingRetry = null
    messages.value.push({ id: uid(), role: 'user', content: text, tools: [], agents: [] })
    messages.value.push({
      id: uid(),
      role: 'assistant',
      content: '',
      tools: [],
      agents: [],
      pending: true,
    })
    // 通过 messages.value 访问，拿到的是响应式 Proxy；直接用本地引用改属性不会触发更新
    const assistant = () => messages.value[messages.value.length - 1]
    const historySnapshot = buildHistory().slice(0, -1)

    streaming.value = true
    aborter = new AbortController()

    await sendChatStream(
      {
        message: text,
        session_id: sessionId.value,
        history: historySnapshot,
        signal: aborter.signal,
      },
      {
        onMessageStart: () => {
          // 新的一条助手消息开始：可能是首次生成，也可能是合规重试后的重新生成。
          // 清空已渲染的正文，避免重试前后的两段内容拼接在一起
          const a = assistant()
          a.content = ''
          a.retryNotice = undefined
        },
        onToken: (delta) => {
          assistant().content += delta
        },
        onRetryNotice: (reason) => {
          // 合规重试即将重新生成：立即清空当前已渲染的正文，
          // 避免重试前的完整错误答案在下一次 message_start 到来前一直停留在屏幕上
          const a = assistant()
          a.content = ''
          a.retryNotice = reason
        },
        onToolCall: (name, args, agentName) => {
          const step: ToolStep = { name, args }
          if (agentName) step.agent_name = agentName
          if (_pendingRetry) {
            step.retry_attempt = _pendingRetry.attempt
            _pendingRetry = null
          }
          assistant().tools.push(step)
        },
        onToolResult: (name, output) => {
          const tools = assistant().tools
          const step = [...tools].reverse().find((t) => t.name === name && !t.output)
          if (step) step.output = output
          else tools.push({ name, args: null, output })
        },
        onAgentStart: (agent_name, task_id, description) => {
          assistant().agents.push({
            agent_name,
            task_id,
            description,
            status: 'running',
          })
        },
        onAgentEnd: (agent_name, task_id, status) => {
          const agents = assistant().agents
          // 找到最后一个同名且 running 的 agent，更新其状态
          const agent = [...agents].reverse().find(
            (a) => a.agent_name === agent_name && a.status === 'running'
          )
          if (agent) {
            agent.status = status as 'completed' | 'failed'
            if (task_id) agent.task_id = task_id
          }
        },
        onPlanCreated: (plan, _reasoning) => {
          assistant().plan = plan
        },
        onToolRetry: (agent_name, _task_id, attempt, _reason) => {
          // 暂存：由紧接着的 onToolCall 消费并写到 ToolStep.retry_attempt
          _pendingRetry = { agent_name: agent_name, attempt }
        },
        onDone: () => {
          const a = assistant()
          a.pending = false
          a.retryNotice = undefined
          streaming.value = false
          sessionsVersion.value++
        },
        onError: (msg) => {
          const a = assistant()
          a.error = msg
          a.pending = false
          streaming.value = false
        },
      },
    )
  }

  function abort() {
    aborter?.abort()
    streaming.value = false
    const last = messages.value[messages.value.length - 1]
    if (last?.pending) last.pending = false
  }

  function clear() {
    if (streaming.value) abort()
    messages.value = []
    sessionId.value = uid()
  }

  async function loadSession(threadId: string) {
    if (streaming.value) return
    const { getSession } = await import('../api/session')
    const detail = await getSession(threadId)
    sessionId.value = threadId
    messages.value = []
    for (const m of detail.messages) {
      messages.value.push({
        id: uid(),
        role: m.role as 'user' | 'assistant',
        content: m.content,
        tools: (m.tools || []).map((t) => ({
          name: t.name,
          args: t.args,
          output: t.output,
          agent_name: t.agent_name,
          retry_attempt: t.retry_attempt,
        })),
        agents: (m.agents || []).map((a) => ({
          ...a,
          status: a.status as 'running' | 'completed' | 'failed',
        })),
      })
    }
  }

  async function rewindAndResend(msgId: string, newContent: string) {
    if (streaming.value) return

    // 找到要编辑的消息索引
    const msgIndex = messages.value.findIndex((m) => m.id === msgId)
    if (msgIndex === -1) return

    // 调用后端 API 回溯到该消息之前
    const { rewindSession } = await import('../api/session')
    await rewindSession(sessionId.value, msgIndex)

    // 截断前端消息列表（删除该消息及之后的所有消息）
    messages.value = messages.value.slice(0, msgIndex)

    // 发送新的消息
    await send(newContent)
  }

  return { messages, streaming, sessionId, sessionsVersion, send, abort, clear, loadSession, rewindAndResend }
})
