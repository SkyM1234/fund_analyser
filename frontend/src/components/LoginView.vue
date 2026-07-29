<script setup lang="ts">
import { ref } from 'vue'
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

  // 客户端校验
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
    <div class="card">
      <h1>基金问答助手</h1>
      <p class="sub">{{ mode === 'login' ? '登录以继续' : '创建新账号' }}</p>

      <el-form @submit.prevent="onSubmit" class="form">
        <el-input v-model="username" placeholder="用户名" size="large" @keydown.enter="onSubmit" />
        <el-input
          v-model="password"
          type="password"
          placeholder="密码"
          size="large"
          show-password
          @keydown.enter="onSubmit"
        />
        <el-input
          v-if="mode === 'register'"
          v-model="email"
          placeholder="邮箱（可选）"
          size="large"
          @keydown.enter="onSubmit"
        />

        <el-alert v-if="success" :title="success" type="success" show-icon :closable="false" />
        <el-alert v-if="error" :title="error" type="error" show-icon :closable="false" />

        <el-button type="primary" size="large" :loading="loading" @click="onSubmit">
          {{ mode === 'login' ? '登录' : '注册' }}
        </el-button>
      </el-form>

      <div class="switch">
        <span v-if="mode === 'login'">
          还没有账号？<a @click="toggleMode">立即注册</a>
        </span>
        <span v-else>
          已有账号？<a @click="toggleMode">直接登录</a>
        </span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.login-page {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--bg);
}
.card {
  width: 340px;
  background: var(--panel);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-md);
  padding: 32px 28px;
  text-align: center;
}
.card h1 { margin: 0 0 4px; font-size: 18px; font-weight: 600; color: var(--text); }
.sub { margin: 0 0 20px; font-size: 13px; color: var(--muted); }
.form { display: flex; flex-direction: column; gap: 12px; }
.switch { margin-top: 16px; font-size: 13px; color: var(--muted); }
.switch a { color: var(--primary); cursor: pointer; margin-left: 4px; }
.switch a:hover { text-decoration: underline; }
</style>
