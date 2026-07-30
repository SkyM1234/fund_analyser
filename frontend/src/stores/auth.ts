import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

interface TokenResponse {
  access_token: string
  refresh_token: string
  token_type: string
}

interface UserInfo {
  id: number
  username: string
  email: string | null
  role: string
}

const ACCESS_KEY = 'fund_analyser_access_token'
const REFRESH_KEY = 'fund_analyser_refresh_token'

export const useAuthStore = defineStore('auth', () => {
  const accessToken = ref<string | null>(localStorage.getItem(ACCESS_KEY))
  const refreshToken = ref<string | null>(localStorage.getItem(REFRESH_KEY))
  const user = ref<UserInfo | null>(null)
  const isAuthenticated = computed(() => !!accessToken.value)

  // 并发请求同时触发 401 时，共享同一次刷新，避免刷新令牌被重复使用（旋转机制下第二次会被拒绝）
  let refreshingPromise: Promise<boolean> | null = null

  function setTokens(tokens: TokenResponse) {
    accessToken.value = tokens.access_token
    refreshToken.value = tokens.refresh_token
    localStorage.setItem(ACCESS_KEY, tokens.access_token)
    localStorage.setItem(REFRESH_KEY, tokens.refresh_token)
  }

  function clearTokens() {
    accessToken.value = null
    refreshToken.value = null
    user.value = null
    localStorage.removeItem(ACCESS_KEY)
    localStorage.removeItem(REFRESH_KEY)
  }

  async function fetchMe() {
    const requestMe = (token: string) =>
      fetch('/api/auth/me', {
        headers: { Authorization: `Bearer ${token}` },
      })

    let requestToken = accessToken.value
    if (!requestToken) return

    let resp = await requestMe(requestToken)
    if (resp.status === 401 && refreshToken.value) {
      if (accessToken.value === requestToken) {
        const refreshed = await refreshAccessToken()
        if (!refreshed) return
      }

      requestToken = accessToken.value
      if (!requestToken) return
      resp = await requestMe(requestToken)
    }

    if (resp.ok) user.value = await resp.json()
  }

  async function login(username: string, password: string) {
    const resp = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    })
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}))
      throw new Error(body.detail || `登录失败 (HTTP ${resp.status})`)
    }
    setTokens(await resp.json())
    await fetchMe()
  }

  async function register(username: string, password: string, email?: string) {
    const resp = await fetch('/api/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password, email: email || null }),
    })
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}))
      // FastAPI 422 验证错误 detail 是数组，提取第一条消息；其他错误 detail 是字符串
      const detail = Array.isArray(body.detail)
        ? body.detail.map((e: any) => e.msg).join('；')
        : body.detail
      throw new Error(detail || `注册失败 (HTTP ${resp.status})`)
    }
    // 注册成功不自动登录，由调用方决定后续行为
  }

  async function refreshAccessToken(): Promise<boolean> {
    if (!refreshToken.value) return false
    if (refreshingPromise) return refreshingPromise

    refreshingPromise = (async () => {
      try {
        const resp = await fetch('/api/auth/refresh', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ refresh_token: refreshToken.value }),
        })
        if (!resp.ok) {
          clearTokens()
          return false
        }
        setTokens(await resp.json())
        return true
      } catch {
        return false
      } finally {
        refreshingPromise = null
      }
    })()

    return refreshingPromise
  }

  async function logout() {
    const rt = refreshToken.value
    clearTokens()
    if (rt) {
      try {
        await fetch('/api/auth/logout', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ refresh_token: rt }),
        })
      } catch {
        // 登出请求失败也无妨，本地 token 已清除
      }
    }
  }

  if (accessToken.value) fetchMe()

  return {
    accessToken,
    refreshToken,
    user,
    isAuthenticated,
    login,
    register,
    logout,
    refreshAccessToken,
  }
})
