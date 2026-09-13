const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const { createRequire } = require('node:module')
const frontend = path.resolve(__dirname, '..')
const requireFrontend = createRequire(path.join(frontend, 'package.json'))
const ts = requireFrontend('typescript')
const { createPinia, setActivePinia } = requireFrontend('pinia')
const storage = new Map()
global.localStorage = {
  getItem: (key) => storage.get(key) || null,
  setItem: (key, value) => storage.set(key, value),
  removeItem: (key) => storage.delete(key),
}

function load(relativePath, dependencies) {
  const source = fs.readFileSync(path.join(frontend, relativePath), 'utf8')
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  })
  const module = { exports: {} }
  new Function('require', 'exports', 'module', outputText)(
    (name) => dependencies[name] || requireFrontend(name), module.exports, module,
  )
  return module.exports
}

let controller
const api = load('src/api/chat.ts', {
  './http': {
    authFetch: async () => new Response(new ReadableStream({
      start(value) { controller = value },
    }), { headers: { 'Content-Type': 'text/event-stream' } }),
  },
})
const history = {
  listSessions: async () => [{ thread_id: 'restored' }],
  getSession: async () => ({
    messages: [{ role: 'user', content: 'query' }],
    active_task: { run_id: 'restored-run' },
  }),
}
const { useChatStore } = load('src/stores/chat.ts', {
  '../api/chat': api,
  '../api/session': history,
})
const tick = () => new Promise((resolve) => setImmediate(resolve))
async function emit(event, data = {}) {
  const frame = new TextEncoder().encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
  // Exercise frame buffering, including UTF-8 byte boundaries.
  for (let offset = 0; offset < frame.length; offset += 7) {
    controller.enqueue(frame.slice(offset, offset + 7))
  }
  await tick()
}

async function main() {
  setActivePinia(createPinia())
  const chat = useChatStore()
  const sending = chat.send('query')
  await tick()
  const message = chat.messages.at(-1)
  for (const phase of ['preparing', 'analyzing', 'synthesizing', 'reviewing']) {
    await emit('answer_progress', { phase })
    assert.equal(message.answerPhase, phase)
    assert.equal(message.content, '')
    assert.equal(message.pending, true)
  }
  await emit('retry_notice', { reason: 'Rejected' })
  assert.equal(message.answerPhase, 'recovering')
  await emit('answer_progress', { phase: 'revising' })
  assert.equal(message.retryNotice, undefined)
  await emit('answer_progress', { phase: 'reviewing' })
  await emit('answer_progress', { phase: 'unknown' })
  assert.equal(message.answerPhase, 'reviewing')
  await emit('message_start')
  await emit('token', { delta: 'Approved answer' })
  await emit('done')
  controller.close()
  await sending
  assert.equal(message.content, 'Approved answer')
  assert.equal(message.pending, false)
  assert.equal(message.answerPhase, undefined)

  setActivePinia(createPinia())
  const restored = useChatStore()
  const restoring = restored.restoreCurrentSession()
  await tick()
  const resumed = restored.messages.at(-1)
  await emit('answer_progress', { phase: 'reviewing' })
  assert.equal(resumed.answerPhase, 'reviewing')
  await emit('attempt_start', { attempt: 2, worker_recovery: true })
  assert.equal(resumed.answerPhase, 'recovering')
  await emit('answer_progress', { phase: 'revising' })
  assert.equal(resumed.answerPhase, 'revising')
  await emit('cancelled')
  controller.close()
  await restoring
  assert.equal(resumed.pending, false)
  assert.equal(resumed.answerPhase, undefined)

  const failing = restored.send('another query')
  await tick()
  const failed = restored.messages.at(-1)
  await emit('answer_progress', { phase: 'reviewing' })
  await emit('error', { message: 'Review unavailable' })
  controller.close()
  await failing
  assert.equal(failed.pending, false)
  assert.equal(failed.answerPhase, undefined)
  assert.equal(failed.content, '')
  assert.equal(failed.error, 'Review unavailable')
  console.log('PASS: SSE progress, revision, approval, resume, recovery, cancellation, error')
}
main().catch((error) => { console.error(error); process.exitCode = 1 })
