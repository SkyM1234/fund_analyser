/**
 * SSE 流式聊天客户端：兼容 POST 的手动 SSE 解析。
 * 后端事件：message_start / token / retry_notice / tool_call / tool_result / done / error
 */

export interface ChatHistoryItem {
  role: 'user' | 'assistant'
  content: string
}

import { authFetch } from './http'

export interface StreamHandlers {
  onStarted?: (runId: string, taskId: string) => void
  onAttemptStart?: (
    attempt: number,
    workerRecovery: boolean,
    checkpointTrace?: any[],
  ) => void
  onMessageStart?: () => void
  onToken?: (delta: string) => void
  onRetryNotice?: (reason: string) => void
  onToolCall?: (
    name: string,
    args: unknown,
    agent_name?: string,
    task_id?: string,
    tool_call_id?: string,
  ) => void
  onToolResult?: (
    name: string,
    output: string,
    task_id?: string,
    tool_call_id?: string,
  ) => void
  onAgentStart?: (
    agent_name: string,
    task_id: string,
    description: string,
    sequence?: number,
  ) => void
  onAgentEnd?: (agent_name: string, task_id: string, status: string) => void
  onPlanCreated?: (plan: Array<{ task_id: string; task_type: string; description: string; assigned_agent: string; fund_codes: string[] }>, reasoning: string) => void
  onToolRetry?: (agent_name: string, task_id: string, attempt: number, reason: string) => void
  onAgentThought?: (
    thought_id: string,
    content: string,
    agent_name: string,
    task_id: string,
  ) => void
  onTraceEvent?: (event: any) => void
  onDone?: () => void
  onCancelled?: (message: string) => void
  onError?: (msg: string) => void
}

export interface SendOptions {
  message: string
  session_id: string  // 必需，会话 ID
  history?: ChatHistoryItem[]
  signal?: AbortSignal
  resume_run_id?: string
}

export async function sendChatStream(opts: SendOptions, handlers: StreamHandlers) {
  const resuming = Boolean(opts.resume_run_id)
  const resp = await authFetch(
    resuming
      ? `/api/chat/tasks/${encodeURIComponent(opts.resume_run_id!)}/stream`
      : '/api/chat/stream',
    resuming
      ? { method: 'GET', headers: { Accept: 'text/event-stream' }, signal: opts.signal }
      : {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
          body: JSON.stringify({
            message: opts.message,
            session_id: opts.session_id,
            history: opts.history ?? [],
          }),
          signal: opts.signal,
        },
  )

  if (!resp.ok || !resp.body) {
    handlers.onError?.(`HTTP ${resp.status}`)
    return
  }

  const reader = resp.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  // SSE 帧以空行 (\n\n) 分隔；每帧若干 "field: value\n"
  const dispatch = (frame: string) => {
    let event = 'message'
    const dataLines: string[] = []
    for (const line of frame.split('\n')) {
      if (!line) continue
      if (line.startsWith(':')) continue // 注释/心跳
      const idx = line.indexOf(':')
      const field = idx === -1 ? line : line.slice(0, idx)
      const value = idx === -1 ? '' : line.slice(idx + 1).replace(/^ /, '')
      if (field === 'event') event = value
      else if (field === 'data') dataLines.push(value)
    }
    if (!dataLines.length) return
    const raw = dataLines.join('\n')
    let payload: any
    try {
      payload = JSON.parse(raw)
    } catch {
      payload = raw
    }
    switch (event) {
      case 'started':
        handlers.onStarted?.(payload.run_id ?? '', payload.task_id ?? '')
        break
      case 'attempt_start':
        handlers.onAttemptStart?.(
          Number(payload.attempt ?? 1),
          Boolean(payload.worker_recovery),
          Array.isArray(payload.checkpoint_trace) ? payload.checkpoint_trace : undefined,
        )
        break
      case 'message_start':
        handlers.onMessageStart?.()
        break
      case 'token':
        handlers.onToken?.(payload.delta ?? '')
        break
      case 'retry_notice':
        handlers.onRetryNotice?.(payload.reason ?? '')
        break
      case 'tool_call':
        handlers.onToolCall?.(
          payload.name,
          payload.args,
          payload.agent_name,
          payload.task_id,
          payload.tool_call_id,
        )
        break
      case 'tool_result':
        handlers.onToolResult?.(
          payload.name,
          payload.output,
          payload.task_id,
          payload.tool_call_id,
        )
        break
      case 'agent_start':
        handlers.onAgentStart?.(
          payload.agent_name ?? '',
          payload.task_id ?? '',
          payload.description ?? '',
          payload.sequence,
        )
        break
      case 'agent_end':
        handlers.onAgentEnd?.(payload.agent_name ?? '', payload.task_id ?? '', payload.status ?? 'completed')
        break
      case 'plan_created':
        handlers.onPlanCreated?.(payload.plan ?? [], payload.reasoning ?? '')
        break
      case 'tool_retry':
        handlers.onToolRetry?.(payload.agent_name ?? '', payload.task_id ?? '', payload.attempt ?? 1, payload.reason ?? '')
        break
      case 'agent_thought':
        handlers.onAgentThought?.(
          payload.thought_id ?? '',
          payload.content ?? '',
          payload.agent_name ?? '',
          payload.task_id ?? '',
        )
        break
      case 'done':
        handlers.onDone?.()
        break
      case 'cancelled':
        handlers.onCancelled?.(payload.message ?? 'Task cancelled by user')
        break
      case 'error':
        handlers.onError?.(payload.message ?? 'unknown error')
        break
    }
    if (
      ['agent_thought', 'tool_call', 'tool_result', 'tool_retry'].includes(event)
    ) {
      handlers.onTraceEvent?.({ type: event, ...payload })
    }
  }

  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      // 统一行尾，避免 CRLF 与 LF 混用导致切不出帧
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n')

      let sep
      while ((sep = buffer.indexOf('\n\n')) !== -1) {
        const frame = buffer.slice(0, sep)
        buffer = buffer.slice(sep + 2)
        dispatch(frame)
      }
    }
    if (buffer.trim()) dispatch(buffer)
  } catch (e: any) {
    if (e?.name !== 'AbortError') handlers.onError?.(String(e?.message ?? e))
  }
}

export function resumeChatStream(
  runId: string,
  handlers: StreamHandlers,
  signal?: AbortSignal,
) {
  return sendChatStream(
    {
      message: '',
      session_id: '',
      resume_run_id: runId,
      signal,
    },
    handlers,
  )
}

export async function cancelChatTask(runId: string) {
  const resp = await authFetch(`/api/chat/tasks/${encodeURIComponent(runId)}/cancel`, {
    method: 'POST',
    headers: { Accept: 'application/json' },
  })

  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`
    try {
      const payload = await resp.json()
      detail = payload.detail ?? detail
    } catch {
      // 错误响应不是 JSON 时保留 HTTP 状态码。
    }
    throw new Error(detail)
  }

  return resp.json() as Promise<{
    run_id: string
    status: string
    cancel_requested: boolean
  }>
}
