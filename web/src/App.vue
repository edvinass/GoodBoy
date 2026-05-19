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

const localInstallCommand = computed(
  () => `curl -fsSL ${installUrl.value} | GOODBOY_LOCAL=1 bash`,
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
</script>

<template>
  <div class="page">
    <header class="header">
      <div class="logo" aria-hidden="true">🐕</div>
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

        <p class="hint">
          Then open a new terminal and run <code>goodboy setup</code> once.
        </p>
      </section>

      <section class="secondary">
        <h3>Optional: local GGUF models</h3>
        <div class="command-block compact">
          <pre class="command"><code>{{ localInstallCommand }}</code></pre>
          <button
            type="button"
            class="copy-btn"
            aria-label="Copy local install command"
            @click="copy(localInstallCommand)"
          >
            Copy
          </button>
        </div>
      </section>

      <section class="steps">
        <h3>After install</h3>
        <ol>
          <li><code>goodboy setup</code> — API key and default model</li>
          <li><code>cd your-project && goodboy</code> — run the agent in any repo</li>
        </ol>
      </section>
    </main>

    <footer class="footer">
      <a href="https://github.com/edvinass/GoodBoy" target="_blank" rel="noopener noreferrer">
        GitHub
      </a>
      <span class="sep">·</span>
      <a :href="installUrl" target="_blank" rel="noopener noreferrer">install.sh</a>
    </footer>
  </div>
</template>
