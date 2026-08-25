<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { DataAnalysis, Menu, Setting, SwitchButton } from '@element-plus/icons-vue'
import ChatWindow from './components/ChatWindow.vue'
import SessionSidebar from './components/SessionSidebar.vue'
import LoginView from './components/LoginView.vue'
import AdminView from './components/AdminView.vue'
import { useAuthStore } from './stores/auth'
import { useChatStore } from './stores/chat'

const showSidebar = ref(true)
const view = ref<'chat' | 'admin'>('chat')
const auth = useAuthStore()
const chat = useChatStore()
const userInitial = computed(() => auth.user?.username?.trim().charAt(0).toUpperCase() || 'U')

async function onLogout() {
  await auth.logout()
  chat.reset()
}

onMounted(async () => {
  if (window.matchMedia('(max-width: 720px)').matches) {
    showSidebar.value = false
  }
  await auth.initialize()
  if (auth.user) {
    await chat.restoreCurrentSession()
  }
})
</script>

<template>
  <LoginView v-if="!auth.isAuthenticated" />
  <AdminView
    v-else-if="view === 'admin'"
    @back="view = 'chat'"
    @logout="onLogout"
  />
  <div v-else class="layout">
    <header class="topbar">
      <el-tooltip content="会话列表" placement="bottom">
        <el-button
          class="icon-btn menu-btn"
          text
          :icon="Menu"
          aria-label="会话列表"
          @click="showSidebar = !showSidebar"
        />
      </el-tooltip>
      <div class="brand">
        <span class="brand-mark"><el-icon><DataAnalysis /></el-icon></span>
        <div class="brand-copy">
          <h1>基金问答助手</h1>
          <span class="sub">专业基金数据分析</span>
        </div>
      </div>
      <div class="spacer" />
      <div v-if="auth.user" class="user-block">
        <span class="avatar">{{ userInitial }}</span>
        <span class="username">{{ auth.user.username }}</span>
      </div>
      <el-tooltip v-if="auth.user?.role === 'admin'" content="管理后台" placement="bottom">
        <el-button
          class="icon-btn"
          text
          :icon="Setting"
          aria-label="管理后台"
          @click="view = 'admin'"
        />
      </el-tooltip>
      <el-tooltip content="退出登录" placement="bottom">
        <el-button
          class="icon-btn"
          text
          :icon="SwitchButton"
          aria-label="退出登录"
          @click="onLogout"
        />
      </el-tooltip>
    </header>
    <main class="main-content">
      <button
        v-if="showSidebar"
        class="sidebar-overlay"
        aria-label="关闭会话列表"
        @click="showSidebar = false"
      />
      <div class="sidebar-wrap" :class="{ collapsed: !showSidebar }">
        <SessionSidebar :visible="showSidebar" @close="showSidebar = false" />
      </div>
      <ChatWindow />
    </main>
  </div>
</template>

<style scoped>
.layout {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--bg);
}
.topbar {
  z-index: 5;
  display: flex;
  align-items: center;
  min-height: 58px;
  gap: 8px;
  padding: 8px 18px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  box-shadow: var(--shadow-sm);
}
.icon-btn {
  width: 36px;
  height: 36px;
  margin: 0;
  padding: 0;
  color: var(--text-secondary);
  font-size: 17px;
}
.icon-btn:hover {
  color: var(--primary);
  background: var(--primary-soft);
}
.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}
.brand-mark {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  flex: 0 0 32px;
  border-radius: var(--radius-sm);
  background: var(--primary);
  color: #fff;
  font-size: 18px;
}
.brand-copy {
  min-width: 0;
  line-height: 1.2;
}
.topbar h1 {
  margin: 0;
  color: var(--text);
  font-size: 15px;
  font-weight: 650;
}
.sub {
  display: block;
  margin-top: 2px;
  color: var(--muted);
  font-size: 11px;
}
.spacer {
  flex: 1;
}
.user-block {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
  margin-right: 4px;
}
.avatar {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  flex: 0 0 28px;
  border-radius: 50%;
  background: var(--success-soft);
  color: var(--success);
  font-size: 12px;
  font-weight: 700;
}
.username {
  max-width: 140px;
  overflow: hidden;
  color: var(--text-secondary);
  font-size: 13px;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.main-content {
  position: relative;
  display: flex;
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

.sidebar-wrap {
  z-index: 3;
  width: 272px;
  flex-shrink: 0;
  overflow: hidden;
  border-right: 1px solid var(--border);
  background: var(--surface);
  transition: width 0.18s ease, border-color 0.18s ease;
}
.sidebar-wrap.collapsed {
  width: 0;
  border-right-color: transparent;
}
.sidebar-overlay {
  display: none;
}

@media (max-width: 720px) {
  .topbar {
    min-height: 54px;
    padding: 7px 10px;
  }
  .brand-mark,
  .sub,
  .username {
    display: none;
  }
  .brand {
    gap: 0;
  }
  .topbar h1 {
    font-size: 14px;
  }
  .user-block {
    margin-right: 0;
  }
  .sidebar-wrap {
    position: absolute;
    inset: 0 auto 0 0;
    width: 100%;
    max-width: 292px;
    box-shadow: var(--shadow-md);
  }
  .sidebar-wrap.collapsed {
    width: 0;
    max-width: 0;
  }
  .sidebar-overlay {
    position: absolute;
    z-index: 2;
    inset: 0;
    display: block;
    width: 100%;
    padding: 0;
    border: 0;
    background: rgba(23, 33, 43, 0.34);
  }
}
</style>
