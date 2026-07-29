<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { ArrowLeft, ChatDotRound, SwitchButton, User } from '@element-plus/icons-vue'
import { useAdminStore } from '../stores/admin'
import { useAuthStore } from '../stores/auth'
import { getAnySession, listUserTokens, revokeToken, type AdminTokenItem } from '../api/admin'
import AdminUsersTable from './AdminUsersTable.vue'
import AdminSessionsTable from './AdminSessionsTable.vue'

const emit = defineEmits<{
  back: []
  logout: []
}>()

const admin = useAdminStore()
const auth = useAuthStore()

const activeMenu = ref<'users' | 'sessions'>('users')

// ---- 登录设备对话框 ----
const tokenDialogVisible = ref(false)
const tokens = ref<AdminTokenItem[]>([])
const tokensLoading = ref(false)

// ---- 会话详情对话框 ----
const sessionDialogVisible = ref(false)
const sessionDetail = ref<{ thread_id: string; messages: any[] } | null>(null)
const sessionDetailLoading = ref(false)

onMounted(() => {
  admin.loadUsers()
  admin.loadSessions()
})

// ---- 用户操作 ----
async function onToggleActive(userId: number, isActive: boolean) {
  try {
    await admin.toggleActive(userId, isActive)
    ElMessage.success(isActive ? '账号已启用' : '账号已停用')
  } catch (e: any) {
    ElMessage.error(e?.message || '操作失败')
    await admin.loadUsers()
  }
}

async function onResetPassword(userId: number, username: string) {
  try {
    const { value } = await ElMessageBox.prompt(
      `为用户 "${username}" 设置新密码（至少 6 位，将强制其重新登录）`,
      '重置密码',
      { inputPattern: /.{6,}/, inputErrorMessage: '密码至少 6 位' }
    )
    await admin.resetUserPassword(userId, value)
    ElMessage.success('密码已重置')
  } catch (e: any) {
    if (e === 'cancel' || e === 'close') return
    ElMessage.error(e?.message || '重置失败')
  }
}

async function onShowTokens(userId: number) {
  tokenDialogVisible.value = true
  tokensLoading.value = true
  try {
    tokens.value = await listUserTokens(userId)
  } catch (e: any) {
    ElMessage.error(e?.message || '加载失败')
  } finally {
    tokensLoading.value = false
  }
}

async function onRevokeToken(tokenId: number) {
  try {
    await revokeToken(tokenId)
    tokens.value = tokens.value.filter((t) => t.id !== tokenId)
    ElMessage.success('已下线该登录设备')
  } catch (e: any) {
    ElMessage.error(e?.message || '操作失败')
  }
}

// ---- 会话操作 ----
async function onViewSession(threadId: string) {
  sessionDialogVisible.value = true
  sessionDetailLoading.value = true
  sessionDetail.value = null
  try {
    sessionDetail.value = await getAnySession(threadId)
  } catch (e: any) {
    ElMessage.error(e?.message || '加载失败')
  } finally {
    sessionDetailLoading.value = false
  }
}

async function onDeleteSession(threadId: string) {
  try {
    await ElMessageBox.confirm('删除后无法恢复，确认删除该会话？', '确认删除', { type: 'warning' })
    await admin.removeSession(threadId)
    ElMessage.success('已删除')
  } catch (e: any) {
    if (e === 'cancel' || e === 'close') return
    ElMessage.error(e?.message || '删除失败')
  }
}

async function onDeleteUser(userId: number, username: string) {
  try {
    await ElMessageBox.confirm(
      `将删除用户 "${username}" 及其所有会话数据，此操作不可恢复！确认删除？`,
      '删除用户',
      { type: 'warning', confirmButtonText: '确认删除', cancelButtonText: '取消' }
    )
    await admin.removeUser(userId)
    ElMessage.success(`用户 "${username}" 已删除`)
  } catch (e: any) {
    if (e === 'cancel' || e === 'close') return
    ElMessage.error(e?.message || '删除失败')
  }
}
</script>

<template>
  <div class="admin-layout">
    <!-- 管理后台专用顶栏 -->
    <header class="admin-topbar">
      <el-button text :icon="ArrowLeft" @click="emit('back')">返回首页</el-button>
      <h1>管理后台</h1>
      <div class="spacer" />
      <span v-if="auth.user" class="username">{{ auth.user.username }}</span>
      <el-button text :icon="SwitchButton" @click="emit('logout')" title="登出">登出</el-button>
    </header>

    <!-- 主体：侧边栏 + 内容 -->
    <el-container class="admin-body">
      <el-aside class="admin-sidebar" width="220px">
        <el-menu
          :default-active="activeMenu"
          class="admin-menu"
          @select="(key: string) => activeMenu = key as 'users' | 'sessions'"
        >
          <el-menu-item index="users">
            <el-icon><User /></el-icon>
            <span>用户管理</span>
          </el-menu-item>
          <el-menu-item index="sessions">
            <el-icon><ChatDotRound /></el-icon>
            <span>会话管理</span>
          </el-menu-item>
        </el-menu>
      </el-aside>

      <el-main class="admin-content">
        <AdminUsersTable
          v-if="activeMenu === 'users'"
          :users="admin.users"
          :loading="admin.loadingUsers"
          @toggle-active="onToggleActive"
          @reset-password="onResetPassword"
          @show-tokens="onShowTokens"
          @delete-user="onDeleteUser"
        />
        <AdminSessionsTable
          v-else-if="activeMenu === 'sessions'"
          :sessions="admin.sessions"
          :loading="admin.loadingSessions"
          @view-session="onViewSession"
          @delete-session="onDeleteSession"
        />
      </el-main>
    </el-container>

    <!-- 登录设备对话框 -->
    <el-dialog v-model="tokenDialogVisible" title="登录设备（未过期的 refresh token）" width="500">
      <el-table :data="tokens" v-loading="tokensLoading" size="small">
        <el-table-column prop="id" label="ID" width="60" />
        <el-table-column prop="created_at" label="创建时间" />
        <el-table-column prop="expires_at" label="过期时间" />
        <el-table-column label="操作" width="100">
          <template #default="{ row }">
            <el-button size="small" type="danger" @click="onRevokeToken(row.id)">下线</el-button>
          </template>
        </el-table-column>
      </el-table>
      <el-empty v-if="!tokensLoading && tokens.length === 0" description="没有有效的登录设备" />
    </el-dialog>

    <!-- 会话详情对话框 -->
    <el-dialog v-model="sessionDialogVisible" title="会话详情" width="600">
      <div v-loading="sessionDetailLoading" class="session-messages">
        <div v-for="(m, i) in sessionDetail?.messages || []" :key="i" class="msg" :class="m.role">
          <strong>{{ m.role === 'user' ? '用户' : '助手' }}：</strong>
          <span>{{ m.content }}</span>
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<style scoped>
.admin-layout {
  height: 100%;
  display: flex;
  flex-direction: column;
  background: var(--bg);
}

/* ---- 顶栏 ---- */
.admin-topbar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 20px;
  background: var(--panel);
  border-bottom: 1px solid var(--border);
  box-shadow: var(--shadow-sm);
  flex-shrink: 0;
}
.admin-topbar h1 {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  color: var(--text);
}
.admin-topbar .el-button { font-size: 15px; color: var(--muted); }
.spacer { flex: 1; }
.username { font-size: 13px; color: var(--muted); }

/* ---- 主体 ---- */
.admin-body {
  flex: 1;
  min-height: 0;
}

/* ---- 侧边栏 ---- */
.admin-sidebar {
  background: var(--panel);
  border-right: 1px solid var(--border);
  overflow-y: auto;
}
.admin-menu {
  border-right: none;
  background: var(--panel);
}
.admin-menu :deep(.el-menu-item) {
  color: var(--text);
}
.admin-menu :deep(.el-menu-item.is-active) {
  color: var(--primary);
  background: var(--tool-bg);
}
.admin-menu :deep(.el-menu-item:hover) {
  background: var(--bg);
}

/* ---- 内容区 ---- */
.admin-content {
  background: var(--bg);
  padding: 20px;
  overflow: auto;
}

/* ---- 会话详情 ---- */
.session-messages {
  max-height: 400px;
  overflow: auto;
}
.msg {
  padding: 8px 0;
  border-bottom: 1px solid var(--border);
  font-size: 13px;
}
.msg.user strong { color: var(--primary); }
</style>
