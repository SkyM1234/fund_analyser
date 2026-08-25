import { defineStore } from 'pinia'
import { computed, reactive, ref } from 'vue'
import {
  cancelChatTask,
  resumeChatStream,
  sendChatStream,
  type ChatHistoryItem,
} from '../api/chat'

export interface ToolStep {
  name: string
  args: unknown
  tool_call_id?: string
  output?: string
  agent_name?: string
  task_id?: string
  retry_attempt?: number
  interrupted?: boolean
}

export interface ThoughtStep {
  thought_id: string
  content: string
  agent_name?: string
  task_id?: string
}

export type TraceEventType =
  | 'agent_thought'
  | 'tool_call'
  | 'tool_result'
  | 'tool_retry'

export interface TraceEvent {
  type: TraceEventType
  event_id: string
  sequence: number
  decision_id?: string
  related_tool_call_ids?: string[]
  thought_id?: string
  content?: string
  name?: string
  args?: unknown
  output?: string
  tool_call_id?: string
  agent_name?: string
  task_id?: string
  attempt?: number
  reason?: string
  interrupted?: boolean
}

export interface AgentStep {
  agent_name: string
  task_id: string
  description: string
  status: 'running' | 'completed' | 'failed'
  sequence?: number
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
  thoughts: ThoughtStep[]
  trace_events: TraceEvent[]
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

function mergeTraceEvents(current: TraceEvent[], incoming: any[]): TraceEvent[] {
  const byId = new Map<string, TraceEvent>()
  for (const event of [...current, ...incoming]) {
    if (!event || typeof event !== 'object') continue
    const normalized: TraceEvent = {
      ...event,
      type: event.type as TraceEventType,
      event_id: String(event.event_id ?? ''),
      sequence: Number(event.sequence ?? 0),
    }
    const key = normalized.event_id || `${normalized.type}:${normalized.sequence}`
    byId.set(key, normalized)
  }
  return [...byId.values()].sort((a, b) => a.sequence - b.sequence)
}

function markInterruptedTraceTools(events: TraceEvent[]) {
  const completedCallIds = new Set(
    events
      .filter((event) => event.type === 'tool_result' && event.tool_call_id)
      .map((event) => event.tool_call_id),
  )
  for (const event of events) {
    if (
      event.type === 'tool_call' &&
      event.tool_call_id &&
      !completedCallIds.has(event.tool_call_id)
    ) {
      event.interrupted = true
    }
  }
}

const CURRENT_SESSION_KEY = 'fund_analyser_current_session_id'

export const useChatStore = defineStore('chat', () => {
  const initialSessionId = localStorage.getItem(CURRENT_SESSION_KEY) || uid()
  const sessionId = ref(initialSessionId)
  const sessionsVersion = ref(0)
  const conversations = reactive(new Map<string, ConversationState>())
  const aborters = new Map<string, AbortController>()
  const taskRunIds = new Map<string, string>()
  const cancellingSessions = reactive(new Set<string>())

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

  conversations.set(initialSessionId, createConversation(false))

  const currentConversation = computed(() => getConversation(sessionId.value))
  const messages = computed(() => currentConversation.value.messages)
  const streaming = computed(() => currentConversation.value.streaming)
  const cancelling = computed(() => cancellingSessions.has(sessionId.value))
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
    localStorage.setItem(CURRENT_SESSION_KEY, targetSessionId)
    const conversation = getConversation(targetSessionId)
    if (!trimmedText || conversation.streaming || conversation.loading) return

    const historySnapshot = buildHistory(conversation.messages)
    const assistantMessage: Message = {
      id: uid(),
      role: 'assistant',
      content: '',
      tools: [],
      thoughts: [],
      trace_events: [],
      agents: [],
      pending: true,
    }

    conversation.messages.push({
      id: uid(),
      role: 'user',
      content: trimmedText,
      tools: [],
      thoughts: [],
      trace_events: [],
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
          onStarted: (runId) => {
            if (runId) taskRunIds.set(targetSessionId, runId)
          },
          onAttemptStart: (_attempt, workerRecovery, checkpointTrace) => {
            if (!workerRecovery) return
            assistant.trace_events = mergeTraceEvents(
              assistant.trace_events,
              checkpointTrace || [],
            )
            markInterruptedTraceTools(assistant.trace_events)
            assistant.retryNotice = undefined
          },
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
          onToolCall: (name, args, agentName, taskId, toolCallId) => {
            const step: ToolStep = { name, args }
            if (agentName) step.agent_name = agentName
            if (taskId) step.task_id = taskId
            if (toolCallId) step.tool_call_id = toolCallId
            if (pendingRetry) {
              step.retry_attempt = pendingRetry.attempt
              pendingRetry = null
            }
            assistant.tools.push(step)
          },
          onToolResult: (name, output, taskId, toolCallId) => {
            const step = toolCallId
              ? assistant.tools.find((tool) => tool.tool_call_id === toolCallId)
              : [...assistant.tools]
                .reverse()
                .find(
                  (tool) =>
                    tool.name === name &&
                    !tool.output &&
                    (!taskId || tool.task_id === taskId),
                )
            if (step) {
              step.output = output
              if (taskId) step.task_id = taskId
            } else {
              assistant.tools.push({
                name,
                args: null,
                output,
                task_id: taskId,
                tool_call_id: toolCallId,
              })
            }
          },
          onAgentStart: (agent_name, task_id, description, sequence) => {
            assistant.agents.push({
              agent_name,
              task_id,
              description,
              status: 'running',
              sequence: Number.isFinite(Number(sequence)) ? Number(sequence) : undefined,
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
          onAgentThought: (thought_id, content, agent_name, task_id) => {
            assistant.thoughts.push({
              thought_id,
              content,
              agent_name,
              task_id,
            })
          },
          onTraceEvent: (event) => {
            assistant.trace_events = mergeTraceEvents(assistant.trace_events, [event])
          },
          onDone: () => {
            assistant.pending = false
            assistant.retryNotice = undefined
            sessionsVersion.value++
          },
          onCancelled: (message) => {
            assistant.pending = false
            assistant.retryNotice = undefined
            assistant.error = undefined
            conversation.streaming = false
            taskRunIds.delete(targetSessionId)
            sessionsVersion.value++
            if (message) {
              assistant.content = assistant.content || message
            }
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
      taskRunIds.delete(targetSessionId)
      cancellingSessions.delete(targetSessionId)
      assistant.pending = false
    }
  }

  async function abort() {
    const targetSessionId = sessionId.value
    const conversation = getConversation(targetSessionId)
    const aborter = aborters.get(targetSessionId)
    if (!aborter || cancellingSessions.has(targetSessionId)) return

    const runId = taskRunIds.get(targetSessionId)
    const last = conversation.messages[conversation.messages.length - 1]
    cancellingSessions.add(targetSessionId)
    try {
      if (runId) await cancelChatTask(runId)
    } catch (error: any) {
      if (last?.pending) {
        last.error = `取消请求失败：${String(error?.message ?? error)}`
      }
    } finally {
      aborter.abort()
      aborters.delete(targetSessionId)
      taskRunIds.delete(targetSessionId)
      cancellingSessions.delete(targetSessionId)
      conversation.streaming = false
      if (last?.pending) last.pending = false
      sessionsVersion.value++
    }
  }

  function clear() {
    const newSessionId = uid()
    conversations.set(newSessionId, createConversation())
    sessionId.value = newSessionId
    localStorage.setItem(CURRENT_SESSION_KEY, newSessionId)
    sessionsVersion.value++
  }

  function reset() {
    for (const aborter of aborters.values()) aborter.abort()
    aborters.clear()
    taskRunIds.clear()
    cancellingSessions.clear()
    conversations.clear()
    const newSessionId = uid()
    conversations.set(newSessionId, createConversation())
    sessionId.value = newSessionId
    localStorage.setItem(CURRENT_SESSION_KEY, newSessionId)
  }

  async function loadSession(threadId: string, force = false) {
    let conversation = conversations.get(threadId)
    if (!conversation) {
      conversation = createConversation(false)
      conversations.set(threadId, conversation)
    }

    sessionId.value = threadId
    localStorage.setItem(CURRENT_SESSION_KEY, threadId)
    if ((conversation.loaded && !force) || conversation.loading) return

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
            task_id: tool.task_id,
            retry_attempt: tool.retry_attempt,
          })),
          thoughts: (message.thoughts || []).map((thought) => ({
            thought_id: thought.thought_id,
            content: thought.content,
            agent_name: thought.agent_name,
            task_id: thought.task_id,
          })),
          trace_events: (message.trace_events || []).map((event) => ({
            ...event,
            type: event.type as TraceEventType,
            sequence: Number(event.sequence ?? 0),
          })).filter((event) =>
            ['agent_thought', 'tool_call', 'tool_result', 'tool_retry'].includes(event.type),
          ),
          agents: (message.agents || []).map((agent) => ({
            ...agent,
            status: agent.status as 'running' | 'completed' | 'failed',
            sequence: Number.isFinite(Number(agent.sequence))
              ? Number(agent.sequence)
              : undefined,
          })),
          plan: message.plan,
          pending: Boolean(message.pending),
        })),
      )
      conversation.loaded = true
      return detail
    } finally {
      conversation.loading = false
    }
  }

  async function resumeActiveTask(
    threadId: string,
    runId: string,
    assistant: Message,
  ) {
    const conversation = getConversation(threadId)
    const aborter = new AbortController()
    aborters.set(threadId, aborter)
    taskRunIds.set(threadId, runId)
    conversation.streaming = true

    try {
      await resumeChatStream(runId, {
        onAttemptStart: (_attempt, workerRecovery, checkpointTrace) => {
          if (!workerRecovery) return
          assistant.trace_events = mergeTraceEvents(
            assistant.trace_events,
            checkpointTrace || [],
          )
          markInterruptedTraceTools(assistant.trace_events)
          assistant.retryNotice = undefined
        },
        onMessageStart: () => {
          assistant.content = ''
          assistant.retryNotice = undefined
        },
        onToken: (delta) => { assistant.content += delta },
        onRetryNotice: (reason) => {
          assistant.content = ''
          assistant.retryNotice = reason
        },
        onToolCall: (name, args, agentName, taskId, toolCallId) => {
          if (toolCallId && assistant.tools.some((tool) => tool.tool_call_id === toolCallId)) {
            return
          }
          assistant.tools.push({
            name,
            args,
            agent_name: agentName,
            task_id: taskId,
            tool_call_id: toolCallId,
          })
        },
        onToolResult: (name, output, taskId, toolCallId) => {
          const step = toolCallId
            ? assistant.tools.find((tool) => tool.tool_call_id === toolCallId)
            : [...assistant.tools].reverse().find(
                (tool) => tool.name === name && !tool.output,
              )
          if (step) {
            step.output = output
            step.task_id = taskId
          }
        },
        onAgentStart: (agent_name, task_id, description, sequence) => {
          if (assistant.agents.some(
            (agent) => agent.agent_name === agent_name && agent.task_id === task_id,
          )) return
          assistant.agents.push({
            agent_name,
            task_id,
            description,
            status: 'running',
            sequence: Number(sequence ?? 0),
          })
        },
        onAgentEnd: (agent_name, task_id, status) => {
          const agent = [...assistant.agents].reverse().find(
            (item) => item.agent_name === agent_name && item.status === 'running',
          )
          if (agent) {
            agent.status = status as 'completed' | 'failed'
            agent.task_id = task_id
          }
        },
        onPlanCreated: (plan) => { assistant.plan = plan },
        onAgentThought: (thought_id, content, agent_name, task_id) => {
          if (assistant.thoughts.some((thought) => thought.thought_id === thought_id)) return
          assistant.thoughts.push({ thought_id, content, agent_name, task_id })
        },
        onTraceEvent: (event) => {
          assistant.trace_events = mergeTraceEvents(assistant.trace_events, [event])
        },
        onDone: () => {
          assistant.pending = false
          assistant.retryNotice = undefined
          conversation.streaming = false
          sessionsVersion.value++
        },
        onCancelled: () => {
          assistant.pending = false
          assistant.retryNotice = undefined
          conversation.streaming = false
        },
        onError: (message) => {
          assistant.error = message
          assistant.pending = false
          conversation.streaming = false
        },
      }, aborter.signal)
    } finally {
      if (aborters.get(threadId) === aborter) aborters.delete(threadId)
      taskRunIds.delete(threadId)
      conversation.streaming = false
      assistant.pending = false
    }
  }

  async function restoreCurrentSession() {
    const threadId = sessionId.value
    const conversation = getConversation(threadId, false)
    if (conversation.loaded || conversation.loading) return

    try {
      const { listSessions } = await import('../api/session')
      const sessions = await listSessions()
      const currentExists = sessions.some((session) => session.thread_id === threadId)
      const target = currentExists ? threadId : sessions[0]?.thread_id

      if (target) {
        const detail = await loadSession(target)
        const activeTask = detail?.active_task
        const targetConversation = getConversation(target)
        let assistant = targetConversation.messages[targetConversation.messages.length - 1]
        if (activeTask?.run_id && assistant?.role !== 'assistant') {
          // A checkpoint may contain only the user message while the worker
          // is between its first state write and the first AI/trace write.
          // Keep an event sink so the reconnect cannot discard the stream.
          assistant = {
            id: uid(),
            role: 'assistant',
            content: '',
            tools: [],
            thoughts: [],
            trace_events: [],
            agents: [],
            pending: true,
          }
          targetConversation.messages.push(assistant)
        }
        if (activeTask?.run_id && assistant?.role === 'assistant') {
          assistant.pending = true
          await resumeActiveTask(target, activeTask.run_id, assistant)
        }
      } else {
        conversation.loaded = true
      }
    } catch (error: any) {
      console.error('恢复会话失败', error)
    }
  }

  async function rewindAndResend(msgId: string, newContent: string) {
    const conversation = getConversation(sessionId.value)
    if (conversation.streaming || conversation.loading) return

    const msgIndex = conversation.messages.findIndex((message) => message.id === msgId)
    if (msgIndex === -1) return

    const messageIndex = conversation.messages
      .slice(0, msgIndex + 1)
      .filter((message) => message.role === 'user')
      .length - 1
    const { rewindSession } = await import('../api/session')
    await rewindSession(sessionId.value, messageIndex)
    await loadSession(sessionId.value, true)
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
    cancelling,
    clear,
    reset,
    loadSession,
    restoreCurrentSession,
    rewindAndResend,
    isSessionStreaming,
  }
})
