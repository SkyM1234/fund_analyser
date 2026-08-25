<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { ArrowLeft, ChatDotRound, DataAnalysis, SwitchButton, User } from '@element-plus/icons-vue'
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
const userInitial = computed(() => auth.user?.username?.trim().charAt(0).toUpperCase() || 'A')
const pageTitle = computed(() => activeMenu.value === 'users' ? '用户管理' : '会话管理')
const pageDescription = computed(() =>
  activeMenu.value === 'users'
    ? `共 ${admin.users.length} 个用户账号`
    : `共 ${admin.sessions.length} 个会话记录`
)

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
      {
        customClass: 'admin-action-message-box',
        inputPattern: /.{6,}/,
        inputErrorMessage: '密码至少 6 位',
      }
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
    await ElMessageBox.confirm('删除后无法恢复，确认删除该会话？', '确认删除', {
      type: 'warning',
      customClass: 'admin-action-message-box',
    })
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
      {
        type: 'warning',
        customClass: 'admin-action-message-box',
        confirmButtonText: '确认删除',
        cancelButtonText: '取消',
      }
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
    <header class="admin-topbar">
      <el-button class="back-btn" text :icon="ArrowLeft" @click="emit('back')">返回</el-button>
      <div class="admin-brand">
        <span class="brand-mark"><el-icon><DataAnalysis /></el-icon></span>
        <div>
          <h1>基金问答助手</h1>
          <span>管理后台</span>
        </div>
      </div>
      <div class="spacer" />
      <div v-if="auth.user" class="user-block">
        <span class="avatar">{{ userInitial }}</span>
        <span class="username">{{ auth.user.username }}</span>
      </div>
      <el-tooltip content="退出登录" placement="bottom">
        <el-button
          class="logout-btn"
          text
          :icon="SwitchButton"
          aria-label="退出登录"
          @click="emit('logout')"
        />
      </el-tooltip>
    </header>

    <el-container class="admin-body">
      <el-aside class="admin-sidebar" width="204px">
        <div class="nav-label">管理功能</div>
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
        <div class="content-inner">
          <div class="page-heading">
            <h2>{{ pageTitle }}</h2>
            <p>{{ pageDescription }}</p>
          </div>
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
        </div>
      </el-main>
    </el-container>

    <el-dialog v-model="tokenDialogVisible" title="登录设备" width="500">
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

.admin-topbar {
  display: flex;
  align-items: center;
  min-height: 58px;
  gap: 10px;
  padding: 8px 18px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  box-shadow: var(--shadow-sm);
  flex-shrink: 0;
}
.back-btn {
  margin-right: 4px;
  color: var(--text-secondary);
}
.admin-brand {
  display: flex;
  align-items: center;
  gap: 9px;
}
.brand-mark {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 30px;
  height: 30px;
  border-radius: var(--radius-sm);
  background: var(--primary);
  color: #fff;
  font-size: 17px;
}
.admin-brand h1 {
  margin: 0;
  color: var(--text);
  font-size: 14px;
  font-weight: 650;
}
.admin-brand span {
  display: block;
  margin-top: 1px;
  color: var(--muted);
  font-size: 10px;
}
.spacer {
  flex: 1;
}
.user-block {
  display: flex;
  align-items: center;
  gap: 8px;
}
.avatar {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: var(--success-soft);
  color: var(--success);
  font-size: 12px;
  font-weight: 700;
}
.username {
  max-width: 130px;
  overflow: hidden;
  color: var(--text-secondary);
  font-size: 13px;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.logout-btn {
  width: 36px;
  height: 36px;
  margin: 0;
  padding: 0;
  color: var(--text-secondary);
  font-size: 17px;
}
.logout-btn:hover {
  color: var(--danger);
  background: var(--danger-soft);
}

.admin-body {
  flex: 1;
  min-height: 0;
}

.admin-sidebar {
  background: var(--surface);
  border-right: 1px solid var(--border);
  overflow-y: auto;
}
.nav-label {
  padding: 20px 18px 8px;
  color: var(--muted);
  font-size: 11px;
  font-weight: 600;
}
.admin-menu {
  border-right: none;
  background: var(--surface);
}
.admin-menu :deep(.el-menu-item) {
  height: 44px;
  margin: 3px 8px;
  border-radius: var(--radius-sm);
  color: var(--text);
  user-select: none;
}
.admin-menu :deep(.el-menu-item.is-active) {
  color: var(--primary);
  background: var(--primary-soft);
  font-weight: 600;
}
.admin-menu :deep(.el-menu-item:hover) {
  background: var(--surface-hover);
}

.admin-content {
  background: var(--bg);
  padding: 28px;
  overflow: auto;
}
.content-inner {
  width: 100%;
  max-width: 1180px;
  margin: 0 auto;
}
.page-heading {
  margin-bottom: 18px;
}
.page-heading h2 {
  margin: 0;
  color: var(--text);
  font-size: 20px;
  font-weight: 650;
}
.page-heading p {
  margin: 5px 0 0;
  color: var(--muted);
  font-size: 12px;
}

.session-messages {
  max-height: 400px;
  overflow: auto;
}
.msg {
  padding: 10px 0;
  border-bottom: 1px solid var(--border);
  font-size: 13px;
  line-height: 1.6;
}
.msg.user strong {
  color: var(--primary);
}

@media (max-width: 720px) {
  .admin-topbar {
    min-height: 54px;
    padding: 7px 10px;
  }
  .brand-mark,
  .admin-brand span,
  .username {
    display: none;
  }
  .admin-body {
    display: flex;
    flex-direction: column;
  }
  .admin-sidebar {
    width: 100% !important;
    overflow: visible;
    border-right: 0;
    border-bottom: 1px solid var(--border);
  }
  .nav-label {
    display: none;
  }
  .admin-menu {
    display: flex;
    padding: 4px 6px;
  }
  .admin-menu :deep(.el-menu-item) {
    flex: 1;
    justify-content: center;
    margin: 0 2px;
  }
  .admin-content {
    padding: 20px 12px;
  }
}
</style>
