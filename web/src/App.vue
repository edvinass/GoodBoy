<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

const copied = ref(false)
const origin = ref('')
const rainCanvas = ref(null)
let rainCleanup = null

onMounted(() => {
  origin.value = window.location.origin
  rainCleanup = startMatrixRain(rainCanvas.value)
})

onBeforeUnmount(() => {
  if (rainCleanup) rainCleanup()
})

function startMatrixRain(canvas) {
  if (!canvas) return () => {}
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  const ctx = canvas.getContext('2d')
  if (!ctx) return () => {}

  const glyphs =
    'ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ0123456789<>/\\|=+*-_:;.'
  const fontSize = 16
  let columns = 0
  let drops = []
  let dpr = 1
  let rafId = 0
  let lastFrame = 0
  const frameInterval = 1000 / 24

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2)
    const { clientWidth, clientHeight } = canvas
    canvas.width = clientWidth * dpr
    canvas.height = clientHeight * dpr
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.font = `${fontSize}px "IBM Plex Mono", ui-monospace, monospace`
    columns = Math.max(1, Math.floor(clientWidth / fontSize))
    drops = new Array(columns)
      .fill(0)
      .map(() => Math.random() * (clientHeight / fontSize))
    ctx.fillStyle = 'rgba(5, 8, 5, 1)'
    ctx.fillRect(0, 0, clientWidth, clientHeight)
  }

  function draw(now) {
    rafId = requestAnimationFrame(draw)
    if (now - lastFrame < frameInterval) return
    lastFrame = now

    const w = canvas.clientWidth
    const h = canvas.clientHeight

    ctx.fillStyle = 'rgba(5, 8, 5, 0.18)'
    ctx.fillRect(0, 0, w, h)

    for (let i = 0; i < drops.length; i++) {
      const ch = glyphs.charAt(Math.floor(Math.random() * glyphs.length))
      const x = i * fontSize
      const y = drops[i] * fontSize

      ctx.fillStyle = 'rgba(180, 255, 200, 0.95)'
      ctx.fillText(ch, x, y)

      ctx.fillStyle = 'rgba(0, 255, 65, 0.85)'
      ctx.fillText(ch, x, y - fontSize)

      if (y > h && Math.random() > 0.975) {
        drops[i] = 0
      }
      drops[i] += 1
    }
  }

  resize()
  const ro = new ResizeObserver(resize)
  ro.observe(canvas)

  if (!reduceMotion) {
    rafId = requestAnimationFrame(draw)
  }

  return () => {
    cancelAnimationFrame(rafId)
    ro.disconnect()
  }
}

const installUrl = computed(() =>
  origin.value ? `${origin.value}/install.sh` : 'https://your-app.up.railway.app/install.sh',
)

const installCommand = computed(
  () => `curl -fsSL ${installUrl.value} | bash`,
)

async function copy(text) {
  try {
    await navigator.clipboard.writeText(text)
    copied.value = true
    setTimeout(() => {
      copied.value = false
    }, 2000)
  } catch {
    const el = document.createElement('textarea')
    el.value = text
    document.body.appendChild(el)
    el.select()
    document.execCommand('copy')
    document.body.removeChild(el)
    copied.value = true
    setTimeout(() => {
      copied.value = false
    }, 2000)
  }
}

const bannerLines = [
  '███╗   ██╗ ███████╗  ██████╗ ',
  '████╗  ██║ ██╔════╝ ██╔═══██╗',
  '██╔██╗ ██║ █████╗   ██║   ██║',
  '██║╚██╗██║ ██╔══╝   ██║   ██║',
  '██║ ╚████║ ███████╗ ╚██████╔╝',
  '╚═╝  ╚═══╝ ╚══════╝  ╚═════╝ ',
  '',
  'Neo v0.1.6',
  'Model gpt-5.4-nano',
  'Tip: Press Ctrl+C to interrupt safely',
]

const conversations = [
  {
    id: 'bugfix',
    label: 'Fix a bug',
    turns: [
      { role: 'user', text: 'pytest is failing in test_routing.py — can you take a look?' },
      {
        role: 'agent',
        subtitle: 'thought',
        text: 'I should run the failing test first to see the actual error before guessing what changed.',
      },
      {
        role: 'agent',
        subtitle: 'thought',
        text: 'The error says test_routes_to_local_model expected a tuple but got a RouteDecision. The router was refactored to return a dataclass — the test was never updated.',
      },
      {
        role: 'agent',
        text: 'Found it — test_routes_to_local_model still asserts on the old tuple shape. The router now returns a RouteDecision dataclass. Want me to update the test?',
      },
      { role: 'user', text: 'Yes, please.' },
      { type: 'activity', text: 'wrote test_routing.py' },
      {
        type: 'diff',
        path: 'test_routing.py',
        hunk: 'lines 8–12',
        lines: [
          { kind: 'del', n: 8, text: 'assert router.route(query) == ("openai", "gpt-4")' },
          { kind: 'add', n: 8, text: 'decision = router.route(query)' },
          { kind: 'add', n: 9, text: 'assert isinstance(decision, RouteDecision)' },
          { kind: 'add', n: 10, text: 'assert decision.provider == "openai"' },
          { kind: 'add', n: 11, text: 'assert decision.model == "gpt-4"' },
        ],
      },
      {
        role: 'agent',
        text: 'Done — 12/12 passing. Want me to run the full suite?',
      },
    ],
  },
  {
    id: 'refactor',
    label: 'Refactor',
    turns: [
      { role: 'user', text: 'The retry block in llm.py is duplicated three times. Pull it into a helper.' },
      { type: 'activity', text: 'read llm.py' },
      {
        role: 'agent',
        subtitle: 'thought',
        text: 'Three near-identical retry loops around the OpenAI, Anthropic, and local calls. I can extract a with_retries(fn, *, attempts, backoff) helper and reuse it in all three places while keeping the existing 0.5s / 1s / 2s timings.',
      },
      {
        role: 'agent',
        text: 'I see three near-identical retry loops. I\'ll extract with_retries(fn, *, attempts, backoff) and reuse it. Existing backoff timings stay (0.5s, 1s, 2s). Sound good?',
      },
      { role: 'user', text: 'Go for it.' },
      { type: 'activity', text: 'wrote llm.py' },
      {
        type: 'diff',
        path: 'llm.py',
        hunk: 'lines 42–51',
        lines: [
          { kind: 'del', n: 42, text: 'for attempt in range(3):' },
          { kind: 'del', n: 43, text: '    try:' },
          { kind: 'del', n: 44, text: '        return openai.responses.create(...)' },
          { kind: 'del', n: 45, text: '    except RetryableError:' },
          { kind: 'del', n: 46, text: '        time.sleep([0.5, 1.0, 2.0][attempt])' },
          { kind: 'add', n: 42, text: 'return with_retries(' },
          { kind: 'add', n: 43, text: '    lambda: openai.responses.create(...),' },
          { kind: 'add', n: 44, text: '    attempts=3,' },
          { kind: 'add', n: 45, text: '    backoff=(0.5, 1.0, 2.0),' },
          { kind: 'add', n: 46, text: ')' },
        ],
      },
      {
        role: 'agent',
        text: 'Done — net 24 fewer lines, all 9 LLM tests still pass. Timings unchanged.',
      },
    ],
  },
  {
    id: 'explain',
    label: 'Explain',
    turns: [
      { role: 'user', text: 'How does Neo decide which model to use for a given turn?' },
      { type: 'activity', text: 'read routing.py' },
      {
        role: 'agent',
        subtitle: 'thought',
        text: 'The user wants the high-level flow. I should describe the classifier, the per-bucket override lookup, and the local-preference shortcut.',
      },
      {
        role: 'agent',
        text:
          'Routing happens in routing.py. The flow is:\n' +
          '  1. Classify the user message into one of: chat, code_edit, long_context, tool_heavy.\n' +
          '  2. Look up the configured model for that bucket in settings.routes.\n' +
          '  3. Fall back to the default model if the bucket has no override.\n\n' +
          'Local models are preferred for chat when prefer_local=true, which keeps quick exchanges off the network.',
      },
    ],
  },
]

const activeId = ref(conversations[0].id)
const active = computed(
  () => conversations.find((c) => c.id === activeId.value) ?? conversations[0],
)

const subtitleMeta = {
  thought: { emoji: '💭', label: 'Thought' },
  plan: { emoji: '📋', label: 'Plan' },
  shell: { emoji: '⚡', label: 'Shell' },
}

function subLabel(sub) {
  return subtitleMeta[sub]?.label ?? sub
}

function subEmoji(sub) {
  return subtitleMeta[sub]?.emoji ?? ''
}

const features = [
  {
    title: 'Runs in your terminal',
    body:
      'Neo is a single CLI you launch from any project. It reads files, runs shell and Python, and edits code in place — no editor plugin, no daemon, no cloud sandbox.',
  },
  {
    title: 'Bring your own model',
    body:
      'Mix and match providers per session. Use a frontier model for hard refactors, a cheap nano model for routine edits, or a local GGUF model when you want to stay offline.',
  },
  {
    title: 'Plan mode & reasoning control',
    body:
      'Toggle between auto / off / always plan mode, and pick reasoning effort (low → xhigh) on the fly. Neo plans before it edits and verifies before it claims done.',
  },
  {
    title: 'Safe by default',
    body:
      'Risky shell commands ask for confirmation. Every session writes a structured log under ~/.neo/logs that you can replay in the built-in viewer.',
  },
]

const providers = [
  {
    name: 'OpenAI',
    badge: 'API',
    desc: 'Responses API with hosted web_search, file_search, code_interpreter, image_generation, and MCP tools.',
    models: [
      { id: 'gpt-5.5', note: 'frontier · premium' },
      { id: 'gpt-5.4', note: 'frontier · high' },
      { id: 'gpt-5.4-mini', note: 'frontier · medium' },
      { id: 'gpt-5.4-nano', note: 'frontier · low · default' },
      { id: 'gpt-5 · gpt-5-mini · gpt-5-nano', note: 'previous gen' },
      { id: 'gpt-4.1 · gpt-4.1-mini · gpt-4.1-nano', note: 'long context · general' },
    ],
  },
  {
    name: 'Anthropic',
    badge: 'API',
    desc: 'Claude with adaptive thinking and 1M-token context for long-horizon agentic work.',
    models: [
      { id: 'claude-opus-4-8', note: 'frontier · premium' },
      { id: 'claude-opus-4-7', note: 'frontier · premium' },
      { id: 'claude-sonnet-4-6', note: 'balanced default' },
      { id: 'claude-haiku-4-5', note: 'fast · cheap' },
    ],
  },
  {
    name: 'DeepSeek',
    badge: 'API',
    desc: 'Cost-efficient 1M-context models with optional thinking mode.',
    models: [
      { id: 'deepseek-v4-pro', note: 'reasoning · medium' },
      { id: 'deepseek-v4-flash', note: 'general · minimal' },
    ],
  },
  {
    name: 'Local (GGUF)',
    badge: 'on-device',
    desc: 'In-process inference via llama.cpp. No keys, no network, no daemon — models download on first use.',
    models: [
      { id: 'qwen2.5-coder-7b', note: 'coding · default · ~4.7 GB' },
      { id: 'qwen2.5-coder-14b', note: 'coding · 16 GB+ RAM' },
      { id: 'deepseek-coder-v2-lite', note: 'MoE · 16 GB+ RAM' },
      { id: 'llama-3.2-3b · phi-3.5-mini', note: 'small · fast' },
      { id: 'qwen2.5-7b · mistral-7b · gemma-2-9b · granite-3.1-8b', note: 'general' },
    ],
  },
]
</script>

<template>
  <div class="page">
    <header class="header">
      <canvas ref="rainCanvas" class="header-rain" aria-hidden="true"></canvas>
      <div class="header-content">
        <div class="header-text">
          <span class="eyebrow">Neo · v0.1.6</span>
          <h1>NEO - The terminal coding agent</h1>
          <p class="tagline">
            A local-first AI pair-programmer that lives in your shell. Bring your own
            OpenAI, Anthropic, or DeepSeek key — or run entirely offline on a local model.
          </p>
          <ul class="header-badges" aria-label="Supported providers">
            <li>OpenAI</li>
            <li>Anthropic</li>
            <li>DeepSeek</li>
            <li>Local · GGUF</li>
          </ul>
        </div>
      </div>
    </header>

    <main>
      <section class="install-card" aria-labelledby="install-heading">
        <h2 id="install-heading">Install</h2>
        <p class="lead">
          One command on macOS or Linux. Requires Python 3.10+.
        </p>

        <div class="command-block">
          <pre class="command"><code>{{ installCommand }}</code></pre>
          <button
            type="button"
            class="copy-btn"
            :aria-label="copied ? 'Copied' : 'Copy install command'"
            @click="copy(installCommand)"
          >
            {{ copied ? 'Copied' : 'Copy' }}
          </button>
        </div>
      </section>

      <section class="steps">
        <h3 class="section-eyebrow">After install</h3>
        <ol>
          <li><code>neo</code> — first run configures API key and default model</li>
          <li><code>cd your-project && neo</code> — run the agent in any repo</li>
        </ol>
      </section>

      <section class="features" aria-labelledby="features-heading">
        <h3 id="features-heading" class="section-eyebrow">Why Neo</h3>
        <div class="feature-grid">
          <article v-for="f in features" :key="f.title" class="feature-card">
            <h4>{{ f.title }}</h4>
            <p>{{ f.body }}</p>
          </article>
        </div>
      </section>

      <section class="providers" aria-labelledby="providers-heading">
        <h3 id="providers-heading" class="section-eyebrow">Supported APIs &amp; models</h3>
        <p class="section-lead">
          Pick any provider at <code>neo setup</code> or switch mid-session with
          <code>/model</code>. All models share the same tool surface — shell, Python,
          file edits, plan mode, and structured logging.
        </p>
        <div class="provider-grid">
          <article v-for="p in providers" :key="p.name" class="provider-card">
            <header class="provider-head">
              <h4>{{ p.name }}</h4>
              <span class="provider-badge">{{ p.badge }}</span>
            </header>
            <p class="provider-desc">{{ p.desc }}</p>
            <ul class="model-list">
              <li v-for="m in p.models" :key="m.id">
                <code>{{ m.id }}</code>
                <span class="model-note">{{ m.note }}</span>
              </li>
            </ul>
          </article>
        </div>
      </section>

      <section class="examples" aria-labelledby="examples-heading">
        <h3 id="examples-heading" class="section-eyebrow">See it in action</h3>

        <div class="example-tabs" role="tablist">
          <button
            v-for="c in conversations"
            :key="c.id"
            type="button"
            role="tab"
            :aria-selected="activeId === c.id"
            class="example-tab"
            :class="{ active: activeId === c.id }"
            @click="activeId = c.id"
          >
            {{ c.label }}
          </button>
        </div>

        <div class="term" role="tabpanel">
          <div class="term-chrome">
            <span class="term-dot term-dot-red" aria-hidden="true"></span>
            <span class="term-dot term-dot-yellow" aria-hidden="true"></span>
            <span class="term-dot term-dot-green" aria-hidden="true"></span>
            <span class="term-chrome-title">neo — ~/repos/your-project</span>
          </div>

          <div class="term-body">
            <div class="term-panel term-panel-banner">
              <pre class="term-banner-art">{{ bannerLines.join('\n') }}</pre>
            </div>

            <template v-for="(turn, i) in active.turns" :key="i">
              <div
                v-if="turn.role === 'user'"
                class="term-panel term-panel-user"
              >
                <span class="term-panel-title">
                  <span class="term-panel-emoji">👤</span> You
                </span>
                <div class="term-panel-body">
                  <p>{{ turn.text }}</p>
                </div>
              </div>

              <div
                v-else-if="turn.role === 'agent'"
                class="term-panel"
                :class="{ 'term-panel-thought': turn.subtitle === 'thought' }"
              >
                <span class="term-panel-title">
                  <span class="term-panel-emoji">◆</span> Neo<template
                    v-if="turn.subtitle"
                  >  <span
                    class="term-panel-sub"
                  ><span class="term-panel-sub-emoji">{{ subEmoji(turn.subtitle) }}</span> {{ subLabel(turn.subtitle) }}</span></template>
                </span>
                <div class="term-panel-body">
                  <p>{{ turn.text }}</p>
                </div>
              </div>

              <div v-else-if="turn.type === 'activity'" class="term-activity">
                <span class="term-activity-mark">◦</span>
                <span>{{ turn.text }}</span>
              </div>

              <div v-else-if="turn.type === 'diff'" class="term-panel term-panel-diff">
                <span class="term-panel-title">
                  <span class="term-panel-emoji">◆</span> Neo  <span
                    class="term-panel-sub"
                  >(changes · {{ turn.path }})</span>
                </span>
                <div class="term-panel-body">
                  <div class="term-diff-hunk">{{ turn.hunk }}</div>
                  <pre class="term-diff"><template
                    v-for="(line, idx) in turn.lines"
                    :key="idx"
                  ><span class="term-diff-row" :class="`term-diff-${line.kind}`"><span class="term-diff-num">{{ String(line.n).padStart(3, ' ') }}</span><span class="term-diff-mark">{{ line.kind === 'add' ? '+' : line.kind === 'del' ? '-' : ' ' }}</span><span class="term-diff-text">{{ line.text }}</span></span>{{ '\n' }}</template></pre>
                </div>
              </div>
            </template>

            <div class="term-input">
              <div class="term-input-frame">
                <div class="term-input-top">
                  <span class="term-input-corner">╭</span><span class="term-input-border"></span><span class="term-input-corner">╮</span>
                </div>
                <div class="term-input-row">
                  <span class="term-input-edge">│</span>
                  <span class="term-input-prompt">❯</span>
                  <span class="term-input-placeholder">Ask anything</span>
                  <span class="term-input-edge term-input-edge-right">│</span>
                </div>
                <div class="term-input-bottom">
                  <span class="term-input-corner">╰</span><span class="term-input-border"></span><span class="term-input-corner">╯</span>
                </div>
              </div>
              <div class="term-input-hint">
                @ files · / commands · ? help · Escape stop · Cmd+D clear
              </div>
            </div>
          </div>
        </div>
      </section>
    </main>

    <footer class="footer">
      <a href="https://github.com/edvinass/Neo" target="_blank" rel="noopener noreferrer">
        GitHub
      </a>
      <span class="sep">·</span>
      <a :href="installUrl" target="_blank" rel="noopener noreferrer">install.sh</a>
      <span class="sep">·</span>
      <a href="/logs.html">log viewer</a>
    </footer>
  </div>
</template>
