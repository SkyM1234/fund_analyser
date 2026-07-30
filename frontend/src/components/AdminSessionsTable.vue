<script setup lang="ts">
import { Delete, View } from '@element-plus/icons-vue'
import type { AdminSessionItem } from '../api/admin'

defineProps<{
  sessions: AdminSessionItem[]
  loading: boolean
}>()

const emit = defineEmits<{
  'view-session': [threadId: string]
  'delete-session': [threadId: string]
}>()
</script>

<template>
  <el-table :data="sessions" v-loading="loading" empty-text="暂无会话" size="default">
    <el-table-column prop="username" label="用户" width="140" />
    <el-table-column prop="title" label="标题">
      <template #default="{ row }">{{ row.title || '(未命名)' }}</template>
    </el-table-column>
    <el-table-column prop="updated_at" label="最后更新" width="180" />
    <el-table-column label="操作" width="180" fixed="right">
      <template #default="{ row }">
        <el-button :icon="View" size="small" @click="emit('view-session', row.thread_id)">
          查看
        </el-button>
        <el-button
          :icon="Delete"
          size="small"
          type="danger"
          plain
          @click="emit('delete-session', row.thread_id)"
        >
          删除
        </el-button>
      </template>
    </el-table-column>
  </el-table>
</template>
