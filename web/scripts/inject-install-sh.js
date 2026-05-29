import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const MARKER = 'NEO_INSTALL_BASE_DEFAULT='

/** @param {string} installSh */
export function injectInstallBase(installSh, base) {
  if (!installSh.includes(MARKER)) {
    throw new Error(`install.sh missing ${MARKER} marker`)
  }
  const line = `${MARKER}"${base.replace(/"/g, '\\"')}"`
  return installSh.replace(new RegExp(`^${MARKER}.*$`, 'm'), line)
}

export function readInstallShTemplate() {
  const candidates = [
    path.join(__dirname, '..', 'install.sh'),
    path.join(__dirname, '..', '..', 'install.sh'),
  ]
  for (const file of candidates) {
    if (fs.existsSync(file)) {
      return fs.readFileSync(file, 'utf8')
    }
  }
  throw new Error(`install.sh not found (tried ${candidates.join(', ')})`)
}

export function requestInstallBase(req) {
  const proto = req.get('x-forwarded-proto') || req.protocol || 'http'
  const host = req.get('x-forwarded-host') || req.get('host')
  return `${proto}://${host}`
}
