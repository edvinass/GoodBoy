<script setup>
import { computed, onMounted, ref } from 'vue'

const copied = ref(false)
const origin = ref('')

onMounted(() => {
  origin.value = window.location.origin
})

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
  '      / \\__',
  '     (    @\\___',
  '      /         O',
  '     /   (_____/',
  '    /_____/   U',
  '',
  'GoodBoy v0.1.0',
  'Model gpt-5',
  'Reasoning medium',
  'Autonomous coding agent',
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
      { type: 'activity', text: 'ran pytest goodboy/tests/test_routing.py -x' },
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
      { type: 'activity', text: 'ran pytest goodboy/tests/test_routing.py' },
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
      { type: 'activity', text: 'ran pytest goodboy/tests/test_llm.py' },
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
      { role: 'user', text: 'How does GoodBoy decide which model to use for a given turn?' },
      { type: 'activity', text: 'read goodboy/agent/routing.py' },
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
</script>

<template>
  <div class="page">
    <header class="header">
      <img class="logo" src="/logo.svg" alt="" aria-hidden="true" />
      <div>
        <h1>GoodBoy</h1>
        <p class="tagline">Local autonomous agent harness for your machine</p>
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
        <h3>After install</h3>
        <ol>
          <li><code>goodboy</code> — first run configures API key and default model</li>
          <li><code>cd your-project && goodboy</code> — run the agent in any repo</li>
        </ol>
      </section>

      <section class="examples" aria-labelledby="examples-heading">
        <h3 id="examples-heading" class="examples-head">See it in action</h3>

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
            <span class="term-chrome-title">goodboy — ~/repos/your-project</span>
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
                  <span class="term-panel-emoji">🐶</span> GoodBoy<template
                    v-if="turn.subtitle"
                  >  <span
                    class="term-panel-sub"
                  >({{ turn.subtitle }})</span></template>
                </span>
                <div class="term-panel-body">
                  <p>{{ turn.text }}</p>
                </div>
              </div>

              <div v-else-if="turn.type === 'activity'" class="term-activity">
                <span class="term-activity-mark">◦</span>
                <span>{{ turn.text }}</span>
              </div>
            </template>

            <div class="term-input">
              <div class="term-input-rule"></div>
              <div class="term-input-placeholder">Ask anything</div>
              <div class="term-input-rule"></div>
              <div class="term-input-hint">
                @ - files, / - commands, ? - help, Escape - stop, Cmd+D - clear
              </div>
            </div>
          </div>
        </div>
      </section>
    </main>

    <footer class="footer">
      <a href="https://github.com/edvinass/GoodBoy" target="_blank" rel="noopener noreferrer">
        GitHub
      </a>
      <span class="sep">·</span>
      <a :href="installUrl" target="_blank" rel="noopener noreferrer">install.sh</a>
      <span class="sep">·</span>
      <a href="/logs.html">log viewer</a>
    </footer>
  </div>
</template>
