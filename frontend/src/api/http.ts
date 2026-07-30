/**
 * 带鉴权的 fetch 封装：自动附加 Authorization 头；遇到 401 时尝试用 refresh_token 换取新
 * access_token 并重试一次；仍失败则清空本地 token，交给上层（路由/App）跳转登录页。
 */
import { useAuthStore } from '../stores/auth'

export async function authFetch(url: string, init: RequestInit = {}): Promise<Response> {
  const auth = useAuthStore()

  const withToken = (token: string | null): RequestInit => ({
    ...init,
    headers: {
      ...(init.headers || {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  })

  const requestToken = auth.accessToken
  let resp = await fetch(url, withToken(requestToken))

  if (resp.status === 401 && auth.refreshToken) {
    if (auth.accessToken !== requestToken) {
      resp = await fetch(url, withToken(auth.accessToken))
    } else {
      const ok = await auth.refreshAccessToken()
      if (ok) {
        resp = await fetch(url, withToken(auth.accessToken))
      }
    }
  }

  return resp
}
