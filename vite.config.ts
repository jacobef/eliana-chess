import type { IncomingMessage, ServerResponse } from 'node:http'
import { defineConfig, type Connect, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'

import { handleHumanStats } from './server/humanStatsProxy'
import { handleOpenAIResponses } from './server/openaiResponsesProxy'

export const crossOriginHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
} as const

function openaiResponsesProxy() {
  const handle = async (
    req: IncomingMessage,
    res: ServerResponse,
    next: Connect.NextFunction,
  ) => {
    const handled = await handleOpenAIResponses(req, res)
    if (!handled) {
      next()
    }
  }

  return {
    name: 'openai-responses-proxy',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        void handle(req, res, next)
      })
    },
    configurePreviewServer(server) {
      server.middlewares.use((req, res, next) => {
        void handle(req, res, next)
      })
    },
  } satisfies Plugin
}

function humanStatsProxy() {
  const handle = async (
    req: IncomingMessage,
    res: ServerResponse,
    next: Connect.NextFunction,
  ) => {
    const handled = await handleHumanStats(req, res)
    if (!handled) {
      next()
    }
  }

  return {
    name: 'human-stats-proxy',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        void handle(req, res, next)
      })
    },
    configurePreviewServer(server) {
      server.middlewares.use((req, res, next) => {
        void handle(req, res, next)
      })
    },
  } satisfies Plugin
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), humanStatsProxy(), openaiResponsesProxy()],
  optimizeDeps: {
    exclude: ['lila-stockfish-web'],
  },
  preview: {
    host: '127.0.0.1',
    headers: crossOriginHeaders,
  },
  server: {
    host: '127.0.0.1',
    headers: crossOriginHeaders,
  },
})
