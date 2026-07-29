<script setup lang="ts">
import type { AdminUserItem } from '../api/admin'

defineProps<{
  users: AdminUserItem[]
  loading: boolean
}>()

const emit = defineEmits<{
  'toggle-active': [userId: number, isActive: boolean]
  'reset-password': [userId: number, username: string]
  'show-tokens': [userId: number]
  'delete-user': [userId: number, username: string]
}>()
</script>

<template>
  <el-table :data="users" v-loading="loading" size="default">
    <el-table-column prop="id" label="ID" width="60" />
    <el-table-column prop="username" label="用户名" width="140" />
    <el-table-column prop="email" label="邮箱" />
    <el-table-column prop="role" label="角色" width="90">
      <template #default="{ row }">
        <el-tag :type="row.role === 'admin' ? 'danger' : 'info'">{{ row.role }}</el-tag>
      </template>
    </el-table-column>
    <el-table-column prop="session_count" label="会话数" width="90" />
    <el-table-column label="状态" width="100">
      <template #default="{ row }">
        <el-switch
          :model-value="row.is_active"
          @change="(v: boolean) => emit('toggle-active', row.id, v)"
        />
      </template>
    </el-table-column>
    <el-table-column label="操作" width="290">
      <template #default="{ row }">
        <el-button size="small" @click="emit('reset-password', row.id, row.username)">重置密码</el-button>
        <el-button size="small" @click="emit('show-tokens', row.id)">登录设备</el-button>
        <el-button size="small" type="danger" @click="emit('delete-user', row.id, row.username)">删除</el-button>
      </template>
    </el-table-column>
  </el-table>
</template>
