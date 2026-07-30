<script setup lang="ts">
import { ref } from 'vue'
import { DataAnalysis, Lock, Message, User } from '@element-plus/icons-vue'
import { useAuthStore } from '../stores/auth'

const auth = useAuthStore()
const mode = ref<'login' | 'register'>('login')
const username = ref('')
const password = ref('')
const email = ref('')
const error = ref('')
const success = ref('')
const loading = ref(false)

async function onSubmit() {
  error.value = ''
  success.value = ''

  if (!username.value.trim()) {
    error.value = '请输入用户名'
    return
  }
  if (username.value.trim().length < 3) {
    error.value = '用户名至少需要 3 个字符'
    return
  }
  if (!password.value) {
    error.value = '请输入密码'
    return
  }
  if (password.value.length < 6) {
    error.value = '密码至少需要 6 个字符'
    return
  }

  loading.value = true
  try {
    if (mode.value === 'login') {
      await auth.login(username.value.trim(), password.value)
    } else {
      await auth.register(username.value.trim(), password.value, email.value.trim())
      success.value = '注册成功，请登录'
      mode.value = 'login'
      password.value = ''
    }
  } catch (e: any) {
    error.value = e?.message || '操作失败，请重试'
  } finally {
    loading.value = false
  }
}

function toggleMode() {
  mode.value = mode.value === 'login' ? 'register' : 'login'
  error.value = ''
  success.value = ''
}
</script>

<template>
  <div class="login-page">
    <main class="login-panel">
      <div class="brand">
        <span class="brand-mark"><el-icon><DataAnalysis /></el-icon></span>
        <div>
          <h1>基金问答助手</h1>
          <p>专业基金数据分析</p>
        </div>
      </div>

      <div class="heading">
        <h2>{{ mode === 'login' ? '欢迎回来' : '创建账号' }}</h2>
        <p>{{ mode === 'login' ? '登录后继续查看基金分析与历史会话' : '填写信息以开始使用' }}</p>
      </div>

      <el-form class="form" @submit.prevent="onSubmit">
        <label class="field-label" for="username">用户名</label>
        <el-input
          id="username"
          v-model="username"
          :prefix-icon="User"
          placeholder="请输入用户名"
          size="large"
          @keydown.enter="onSubmit"
        />

        <label class="field-label" for="password">密码</label>
        <el-input
          id="password"
          v-model="password"
          :prefix-icon="Lock"
          type="password"
          placeholder="请输入密码"
          size="large"
          show-password
          @keydown.enter="onSubmit"
        />

        <template v-if="mode === 'register'">
          <label class="field-label" for="email">邮箱 <span>选填</span></label>
          <el-input
            id="email"
            v-model="email"
            :prefix-icon="Message"
            placeholder="请输入邮箱"
            size="large"
            @keydown.enter="onSubmit"
          />
        </template>

        <el-alert v-if="success" :title="success" type="success" show-icon :closable="false" />
        <el-alert v-if="error" :title="error" type="error" show-icon :closable="false" />

        <el-button class="submit-btn" type="primary" size="large" :loading="loading" @click="onSubmit">
          {{ mode === 'login' ? '登录' : '注册' }}
        </el-button>
      </el-form>

      <div class="switch">
        <span>{{ mode === 'login' ? '还没有账号？' : '已有账号？' }}</span>
        <button type="button" @click="toggleMode">
          {{ mode === 'login' ? '立即注册' : '直接登录' }}
        </button>
      </div>
    </main>
    <footer>Fund Intelligence Workspace</footer>
  </div>
</template>

<style scoped>
.login-page {
  position: relative;
  display: grid;
  min-height: 100%;
  padding: 32px 20px 58px;
  background:
    linear-gradient(var(--border) 1px, transparent 1px),
    linear-gradient(90deg, var(--border) 1px, transparent 1px),
    var(--bg);
  background-size: 48px 48px;
  place-items: center;
}
.login-page::before {
  position: absolute;
  inset: 0;
  background: rgba(244, 246, 248, 0.82);
  content: "";
}
.login-panel {
  position: relative;
  width: min(100%, 410px);
  padding: 30px 32px 28px;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: var(--surface);
  box-shadow: var(--shadow-md);
}
.brand {
  display: flex;
  align-items: center;
  gap: 11px;
  padding-bottom: 22px;
  border-bottom: 1px solid var(--border);
}
.brand-mark {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 38px;
  height: 38px;
  border-radius: var(--radius-sm);
  background: var(--primary);
  color: #fff;
  font-size: 21px;
}
.brand h1 {
  margin: 0;
  color: var(--text);
  font-size: 15px;
  font-weight: 650;
}
.brand p {
  margin: 3px 0 0;
  color: var(--muted);
  font-size: 11px;
}
.heading {
  margin: 24px 0 20px;
}
.heading h2 {
  margin: 0;
  color: var(--text);
  font-size: 22px;
  font-weight: 650;
}
.heading p {
  margin: 7px 0 0;
  color: var(--text-secondary);
  font-size: 13px;
  line-height: 1.5;
}
.form {
  display: flex;
  flex-direction: column;
}
.field-label {
  margin: 0 0 6px;
  color: var(--text-secondary);
  font-size: 12px;
  font-weight: 600;
}
.field-label:not(:first-child) {
  margin-top: 14px;
}
.field-label span {
  color: var(--muted);
  font-weight: 400;
}
.form :deep(.el-alert) {
  margin-top: 14px;
}
.submit-btn {
  width: 100%;
  margin-top: 20px;
}
.switch {
  display: flex;
  justify-content: center;
  gap: 5px;
  margin-top: 18px;
  color: var(--muted);
  font-size: 13px;
}
.switch button {
  padding: 0;
  border: 0;
  background: transparent;
  color: var(--primary);
  font-size: 13px;
  font-weight: 600;
}
.switch button:hover {
  color: var(--primary-hover);
  text-decoration: underline;
}
footer {
  position: absolute;
  bottom: 22px;
  color: var(--muted);
  font-size: 10px;
  letter-spacing: 0;
}

@media (max-width: 520px) {
  .login-page {
    padding: 0;
    background: var(--surface);
  }
  .login-page::before {
    display: none;
  }
  .login-panel {
    width: 100%;
    padding: 24px;
    border: 0;
    box-shadow: none;
  }
  footer {
    bottom: 14px;
  }
}
</style>
