/**
 * 会话管理 API
 */
import { authFetch } from './http'

export interface SessionItem {
  thread_id: string
  first_message: string | null
  last_checkpoint: string
  checkpoint_count: number
  created_at: string | null
}

export interface SessionDetail {
  thread_id: string
  messages: Array<{
    role: string
    content: string
    tools?: Array<{
      name: string
      args: any
      tool_call_id?: string
      output?: string
      agent_name?: string
      task_id?: string
      retry_attempt?: number
    }>
    thoughts?: Array<{
      thought_id: string
      content: string
      agent_name?: string
      task_id?: string
    }>
    trace_events?: Array<{
      type: string
      event_id: string
      sequence: number
      [key: string]: any
    }>
    agents?: Array<{
      agent_name: string
      task_id: string
      description: string
      status: string
      sequence?: number
    }>
    plan?: Array<{
      task_id: string
      task_type: string
      description: string
      assigned_agent: string
      fund_codes: string[]
    }>
    pending?: boolean
  }>
  checkpoint_count: number
  active_task?: {
    run_id: string
    task_id: string | null
    status: string
    attempt: number
  } | null
}

export async function listSessions(): Promise<SessionItem[]> {
  const resp = await authFetch('/api/sessions')
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
  return resp.json()
}

export async function getSession(threadId: string): Promise<SessionDetail> {
  const resp = await authFetch(`/api/sessions/${threadId}`)
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
  return resp.json()
}

export async function deleteSession(threadId: string): Promise<void> {
  const resp = await authFetch(`/api/sessions/${threadId}`, { method: 'DELETE' })
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
}

export async function rewindSession(threadId: string, messageIndex: number): Promise<void> {
  const resp = await authFetch(`/api/sessions/${threadId}/rewind`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message_index: messageIndex })
  })
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
}
