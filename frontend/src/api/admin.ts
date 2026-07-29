/**
 * 管理员 API
 */
import { authFetch } from './http'

export interface AdminUserItem {
  id: number
  username: string
  email: string | null
  role: string
  is_active: boolean
  created_at: string
  session_count: number
}

export interface AdminSessionItem {
  thread_id: string
  user_id: number
  username: string
  title: string | null
  created_at: string
  updated_at: string
}

export interface AdminTokenItem {
  id: number
  expires_at: string
  created_at: string
  revoked: boolean
}

async function checkOk(resp: Response): Promise<Response> {
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}))
    throw new Error(body.detail || `HTTP ${resp.status}`)
  }
  return resp
}

export async function listUsers(): Promise<AdminUserItem[]> {
  return (await checkOk(await authFetch('/api/admin/users'))).json()
}

export async function setUserActive(userId: number, isActive: boolean): Promise<AdminUserItem> {
  return (await checkOk(await authFetch(`/api/admin/users/${userId}/active`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ is_active: isActive }),
  }))).json()
}

export async function deleteUser(userId: number): Promise<void> {
  await checkOk(await authFetch(`/api/admin/users/${userId}`, { method: 'DELETE' }))
}

export async function resetPassword(userId: number, newPassword: string): Promise<void> {
  await checkOk(await authFetch(`/api/admin/users/${userId}/reset-password`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ new_password: newPassword }),
  }))
}

export async function listUserTokens(userId: number): Promise<AdminTokenItem[]> {
  return (await checkOk(await authFetch(`/api/admin/users/${userId}/tokens`))).json()
}

export async function revokeToken(tokenId: number): Promise<void> {
  await checkOk(await authFetch(`/api/admin/tokens/${tokenId}`, { method: 'DELETE' }))
}

export async function listAllSessions(): Promise<AdminSessionItem[]> {
  return (await checkOk(await authFetch('/api/admin/sessions'))).json()
}

export async function getAnySession(threadId: string) {
  return (await checkOk(await authFetch(`/api/admin/sessions/${threadId}`))).json()
}

export async function deleteAnySession(threadId: string): Promise<void> {
  await checkOk(await authFetch(`/api/admin/sessions/${threadId}`, { method: 'DELETE' }))
}
