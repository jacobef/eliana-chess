import type { IncomingMessage, ServerResponse } from 'node:http'

import { lichessHumanBook } from './lichessHumanBook'

export const HUMAN_STATS_PATH = '/api/human-stats'
export const HUMAN_STATS_STATUS_PATH = '/api/human-stats/status'

export async function handleHumanStats(
  req: IncomingMessage,
  res: ServerResponse,
) {
  const parsedUrl = new URL(req.url ?? HUMAN_STATS_PATH, 'http://127.0.0.1')

  if (parsedUrl.pathname === HUMAN_STATS_STATUS_PATH) {
    if (req.method !== 'GET') {
      res.statusCode = 405
      res.setHeader('Allow', 'GET')
      res.setHeader('Content-Type', 'application/json')
      res.end(JSON.stringify({ error: { message: 'Method not allowed.' } }))
      return true
    }

    if (parsedUrl.searchParams.get('warm') !== '0') {
      void lichessHumanBook.warm().catch(() => undefined)
    }

    res.statusCode = 200
    res.setHeader('Content-Type', 'application/json')
    res.setHeader('Cache-Control', 'no-store')
    res.end(JSON.stringify(lichessHumanBook.getState()))
    return true
  }

  if (parsedUrl.pathname !== HUMAN_STATS_PATH) {
    return false
  }

  if (req.method !== 'GET') {
    res.statusCode = 405
    res.setHeader('Allow', 'GET')
    res.setHeader('Content-Type', 'application/json')
    res.end(JSON.stringify({ error: { message: 'Method not allowed.' } }))
    return true
  }

  const fen = parsedUrl.searchParams.get('fen')?.trim()
  if (!fen) {
    res.statusCode = 400
    res.setHeader('Content-Type', 'application/json')
    res.end(JSON.stringify({ error: { message: 'fen is required.' } }))
    return true
  }

  try {
    const payload = await lichessHumanBook.lookup(fen)
    res.statusCode = 200
    res.setHeader('Content-Type', 'application/json')
    res.setHeader('Cache-Control', 'no-store')
    res.end(JSON.stringify(payload))
    return true
  } catch (error) {
    res.statusCode = 500
    res.setHeader('Content-Type', 'application/json')
    res.end(
      JSON.stringify({
        error: {
          message:
            error instanceof Error
              ? error.message
              : 'Human move statistics could not be loaded.',
        },
      }),
    )
    return true
  }
}
