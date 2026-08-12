import { defineStore } from 'pinia'
import { computed, reactive, ref } from 'vue'
import { sendChatStream, type ChatHistoryItem } from '../api/chat'

export interface ToolStep {
  name: string
  args: unknown
  tool_call_id?: string
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

export interface RunningSession {
  thread_id: string
  first_message: string | null
  checkpoint_count: number
}

interface ConversationState {
  messages: Message[]
  streaming: boolean
  loading: boolean
  loaded: boolean
}

const uid = () => {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID()
  }
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (char) => {
    const random = Math.random() * 16 | 0
    return (char === 'x' ? random : (random & 0x3 | 0x8)).toString(16)
  })
}

export const useChatStore = defineStore('chat', () => {
  const initialSessionId = uid()
  const sessionId = ref(initialSessionId)
  const sessionsVersion = ref(0)
  const conversations = reactive(new Map<string, ConversationState>())
  const aborters = new Map<string, AbortController>()

  function createConversation(loaded = true): ConversationState {
    return reactive({
      messages: [],
      streaming: false,
      loading: false,
      loaded,
    })
  }

  function getConversation(threadId: string, loaded = true) {
    let conversation = conversations.get(threadId)
    if (!conversation) {
      conversation = createConversation(loaded)
      conversations.set(threadId, conversation)
    }
    return conversation
  }

  conversations.set(initialSessionId, createConversation())

  const currentConversation = computed(() => getConversation(sessionId.value))
  const messages = computed(() => currentConversation.value.messages)
  const streaming = computed(() => currentConversation.value.streaming)
  const loadingSession = computed(() => currentConversation.value.loading)
  const runningSessions = computed<RunningSession[]>(() =>
    Array.from(conversations.entries())
      .filter(([, conversation]) => conversation.streaming)
      .map(([threadId, conversation]) => ({
        thread_id: threadId,
        first_message:
          conversation.messages.find((message) => message.role === 'user')?.content ?? null,
        checkpoint_count: conversation.messages.length,
      })),
  )

  const buildHistory = (conversationMessages: Message[]): ChatHistoryItem[] =>
    conversationMessages
      .filter((message) => !message.pending && !message.error)
      .map((message) => ({ role: message.role, content: message.content }))

  async function send(text: string) {
    const trimmedText = text.trim()
    const targetSessionId = sessionId.value
    const conversation = getConversation(targetSessionId)
    if (!trimmedText || conversation.streaming || conversation.loading) return

    const historySnapshot = buildHistory(conversation.messages)
    const assistantMessage: Message = {
      id: uid(),
      role: 'assistant',
      content: '',
      tools: [],
      agents: [],
      pending: true,
    }

    conversation.messages.push({
      id: uid(),
      role: 'user',
      content: trimmedText,
      tools: [],
      agents: [],
    })
    conversation.messages.push(assistantMessage)
    const assistant = conversation.messages[conversation.messages.length - 1]
    conversation.loaded = true
    conversation.streaming = true

    const aborter = new AbortController()
    aborters.set(targetSessionId, aborter)
    let pendingRetry: { agent_name: string; attempt: number } | null = null

    try {
      await sendChatStream(
        {
          message: trimmedText,
          session_id: targetSessionId,
          history: historySnapshot,
          signal: aborter.signal,
        },
        {
          onMessageStart: () => {
            assistant.content = ''
            assistant.retryNotice = undefined
          },
          onToken: (delta) => {
            assistant.content += delta
          },
          onRetryNotice: (reason) => {
            assistant.content = ''
            assistant.retryNotice = reason
          },
          onToolCall: (name, args, agentName, toolCallId) => {
            const step: ToolStep = { name, args }
            if (agentName) step.agent_name = agentName
            if (toolCallId) step.tool_call_id = toolCallId
            if (pendingRetry) {
              step.retry_attempt = pendingRetry.attempt
              pendingRetry = null
            }
            assistant.tools.push(step)
          },
          onToolResult: (name, output, toolCallId) => {
            const step = toolCallId
              ? assistant.tools.find((tool) => tool.tool_call_id === toolCallId)
              : [...assistant.tools]
                .reverse()
                .find((tool) => tool.name === name && !tool.output)
            if (step) step.output = output
            else assistant.tools.push({ name, args: null, output, tool_call_id: toolCallId })
          },
          onAgentStart: (agent_name, task_id, description) => {
            assistant.agents.push({
              agent_name,
              task_id,
              description,
              status: 'running',
            })
          },
          onAgentEnd: (agent_name, task_id, status) => {
            const agent = [...assistant.agents].reverse().find(
              (item) => item.agent_name === agent_name && item.status === 'running',
            )
            if (agent) {
              agent.status = status as 'completed' | 'failed'
              if (task_id) agent.task_id = task_id
            }
          },
          onPlanCreated: (plan, _reasoning) => {
            assistant.plan = plan
          },
          onToolRetry: (agent_name, _task_id, attempt, _reason) => {
            pendingRetry = { agent_name, attempt }
          },
          onDone: () => {
            assistant.pending = false
            assistant.retryNotice = undefined
            sessionsVersion.value++
          },
          onError: (message) => {
            assistant.error = message
            assistant.pending = false
            sessionsVersion.value++
          },
        },
      )
    } catch (error: any) {
      if (error?.name !== 'AbortError') {
        assistant.error = String(error?.message ?? error)
        sessionsVersion.value++
      }
    } finally {
      if (aborters.get(targetSessionId) === aborter) {
        aborters.delete(targetSessionId)
        conversation.streaming = false
      }
      assistant.pending = false
    }
  }

  function abort() {
    const conversation = getConversation(sessionId.value)
    aborters.get(sessionId.value)?.abort()
    aborters.delete(sessionId.value)
    conversation.streaming = false
    const last = conversation.messages[conversation.messages.length - 1]
    if (last?.pending) last.pending = false
  }

  function clear() {
    const newSessionId = uid()
    conversations.set(newSessionId, createConversation())
    sessionId.value = newSessionId
    sessionsVersion.value++
  }

  function reset() {
    for (const aborter of aborters.values()) aborter.abort()
    aborters.clear()
    conversations.clear()
    const newSessionId = uid()
    conversations.set(newSessionId, createConversation())
    sessionId.value = newSessionId
  }

  async function loadSession(threadId: string) {
    let conversation = conversations.get(threadId)
    if (!conversation) {
      conversation = createConversation(false)
      conversations.set(threadId, conversation)
    }

    sessionId.value = threadId
    if (conversation.loaded || conversation.loading) return

    conversation.loading = true
    const { getSession } = await import('../api/session')
    try {
      const detail = await getSession(threadId)
      conversation.messages.splice(
        0,
        conversation.messages.length,
        ...detail.messages.map((message) => ({
          id: uid(),
          role: message.role as 'user' | 'assistant',
          content: message.content,
          tools: (message.tools || []).map((tool) => ({
            name: tool.name,
            args: tool.args,
            tool_call_id: tool.tool_call_id,
            output: tool.output,
            agent_name: tool.agent_name,
            retry_attempt: tool.retry_attempt,
          })),
          agents: (message.agents || []).map((agent) => ({
            ...agent,
            status: agent.status as 'running' | 'completed' | 'failed',
          })),
        })),
      )
      conversation.loaded = true
    } finally {
      conversation.loading = false
    }
  }

  async function rewindAndResend(msgId: string, newContent: string) {
    const conversation = getConversation(sessionId.value)
    if (conversation.streaming || conversation.loading) return

    const msgIndex = conversation.messages.findIndex((message) => message.id === msgId)
    if (msgIndex === -1) return

    const { rewindSession } = await import('../api/session')
    await rewindSession(sessionId.value, msgIndex)
    conversation.messages.splice(msgIndex)
    await send(newContent)
  }

  const isSessionStreaming = (threadId: string) =>
    conversations.get(threadId)?.streaming ?? false

  return {
    messages,
    streaming,
    loadingSession,
    runningSessions,
    sessionId,
    sessionsVersion,
    send,
    abort,
    clear,
    reset,
    loadSession,
    rewindAndResend,
    isSessionStreaming,
  }
})
