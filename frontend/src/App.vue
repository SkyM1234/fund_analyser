<script setup lang="ts">
import { ref } from 'vue'
import { Menu, Setting, SwitchButton } from '@element-plus/icons-vue'
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

async function onLogout() {
  await auth.logout()
  chat.reset()
}
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
      <el-button class="menu-btn" text :icon="Menu" @click="showSidebar = !showSidebar" />
      <h1>基金问答助手</h1>
      <span class="sub">RAG · ReAct · DeepSeek</span>
      <div class="spacer" />
      <span v-if="auth.user" class="username">{{ auth.user.username }}</span>
      <el-button
        v-if="auth.user?.role === 'admin'"
        class="admin-btn"
        text
        :icon="Setting"
        title="管理后台"
        @click="view = 'admin'"
      />
      <el-button class="logout-btn" text :icon="SwitchButton" @click="onLogout" title="登出">登出</el-button>
    </header>
    <main class="main-content">
      <div class="sidebar-wrap" :class="{ collapsed: !showSidebar }">
        <SessionSidebar :visible="showSidebar" @close="showSidebar = false" />
      </div>
      <ChatWindow />
    </main>
  </div>
</template>

<style scoped>
.layout { display: flex; flex-direction: column; height: 100%; }
.topbar {
  display: flex; align-items: center; gap: 12px;
  padding: 10px 20px;
  background: var(--panel);
  border-bottom: 1px solid var(--border);
  box-shadow: var(--shadow-sm);
  z-index: 1;
}
.menu-btn {
  font-size: 18px;
  color: var(--text);
}
.topbar h1 { margin: 0; font-size: 16px; font-weight: 600; }
.sub { font-size: 12px; color: var(--muted); }
.spacer { flex: 1; }
.username { font-size: 13px; color: var(--muted); }
.admin-btn { font-size: 18px; color: var(--muted); }
.logout-btn { font-size: 15px; color: var(--muted); }
.main-content { flex: 1; min-height: 0; display: flex; }

.sidebar-wrap {
  width: 280px;
  flex-shrink: 0;
  overflow: hidden;
  border-right: 1px solid var(--border);
  transition: width 0.2s ease, border-color 0.2s ease;
}
.sidebar-wrap.collapsed {
  width: 0;
  border-right-color: transparent;
}

@media (max-width: 640px) {
  .sub { display: none; }
  .topbar { padding: 10px 14px; }
  .sidebar-wrap {
    position: absolute;
    inset: 49px 0 0 0;
    z-index: 2;
    width: 100%;
    max-width: 280px;
    box-shadow: var(--shadow-md);
  }
  .sidebar-wrap.collapsed {
    width: 0;
    max-width: 0;
  }
}
</style>
