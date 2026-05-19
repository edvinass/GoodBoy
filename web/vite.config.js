import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import {
  injectInstallBase,
  readInstallShTemplate,
} from './scripts/inject-install-sh.js'

function installShDevPlugin() {
  const template = readInstallShTemplate()
  return {
    name: 'goodboy-install-sh',
    enforce: 'pre',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = req.url?.split('?')[0]
        if (url !== '/install.sh') {
          next()
          return
        }
        const host = req.headers.host || 'localhost:5173'
        const proto = req.socket?.encrypted ? 'https' : 'http'
        const body = injectInstallBase(template, `${proto}://${host}`)
        res.setHeader('Content-Type', 'text/plain; charset=utf-8')
        res.end(body)
      })
    },
  }
}

export default defineConfig({
  plugins: [installShDevPlugin(), vue()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})
