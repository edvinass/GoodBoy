import express from 'express'
import path from 'path'
import { fileURLToPath } from 'url'
import {
  injectInstallBase,
  readInstallShTemplate,
  requestInstallBase,
} from './scripts/inject-install-sh.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const dist = path.join(__dirname, 'dist')
const port = Number(process.env.PORT) || 3000

const installShTemplate = readInstallShTemplate()

const app = express()

app.get('/install.sh', (req, res) => {
  const base = requestInstallBase(req)
  res.type('text/plain; charset=utf-8')
  res.send(injectInstallBase(installShTemplate, base))
})

app.use(express.static(dist, { index: false }))

app.use((_req, res) => {
  res.sendFile(path.join(dist, 'index.html'))
})

app.listen(port, () => {
  console.log(`Neo site listening on port ${port}`)
})
