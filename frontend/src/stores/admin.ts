import { defineStore } from 'pinia'
import { ref } from 'vue'
import {
  deleteAnySession,
  deleteUser,
  listAllSessions,
  listUsers,
  resetPassword,
  setUserActive,
  type AdminSessionItem,
  type AdminUserItem,
} from '../api/admin'

export const useAdminStore = defineStore('admin', () => {
  const users = ref<AdminUserItem[]>([])
  const sessions = ref<AdminSessionItem[]>([])
  const loadingUsers = ref(false)
  const loadingSessions = ref(false)

  async function loadUsers() {
    loadingUsers.value = true
    try {
      users.value = await listUsers()
    } finally {
      loadingUsers.value = false
    }
  }

  async function loadSessions() {
    loadingSessions.value = true
    try {
      sessions.value = await listAllSessions()
    } finally {
      loadingSessions.value = false
    }
  }

  async function toggleActive(userId: number, isActive: boolean) {
    const updated = await setUserActive(userId, isActive)
    const idx = users.value.findIndex((u) => u.id === userId)
    if (idx !== -1) users.value[idx] = updated
  }

  async function resetUserPassword(userId: number, newPassword: string) {
    await resetPassword(userId, newPassword)
  }

  async function removeSession(threadId: string) {
    await deleteAnySession(threadId)
    sessions.value = sessions.value.filter((s) => s.thread_id !== threadId)
  }

  async function removeUser(userId: number) {
    await deleteUser(userId)
    users.value = users.value.filter((u) => u.id !== userId)
  }

  return {
    users,
    sessions,
    loadingUsers,
    loadingSessions,
    loadUsers,
    loadSessions,
    toggleActive,
    resetUserPassword,
    removeSession,
    removeUser,
  }
})
