const HUMAN_STATS_PATH = '/api/human-stats'
const HUMAN_STATS_STATUS_PATH = '/api/human-stats/status'

export type HumanBookLoadState = {
  status: 'idle' | 'loading' | 'ready' | 'error'
  detail: string
  progress?: number
  error?: string
}

export type HumanPositionStats = {
  positionKey: string
  totalGames: number
  moveCounts: Record<string, number>
  policy: Record<string, number>
  minSampleGames: number
  maxIndexPlies: number
}

const responseCache = new Map<string, HumanPositionStats | null>()
const promiseCache = new Map<string, Promise<HumanPositionStats | null>>()

function normalizePositionKey(fen: string) {
  return fen.trim().split(/\s+/).slice(0, 4).join(' ')
}

export async function fetchHumanPositionStats(fen: string) {
  const positionKey = normalizePositionKey(fen)
  const cached = responseCache.get(positionKey)
  if (cached !== undefined) {
    return cached
  }

  const pending = promiseCache.get(positionKey)
  if (pending) {
    return await pending
  }

  const promise = (async () => {
    try {
      const response = await fetch(
        `${HUMAN_STATS_PATH}?fen=${encodeURIComponent(positionKey)}`,
      )
      if (!response.ok) {
        return null
      }

      const payload = (await response.json()) as HumanPositionStats
      responseCache.set(positionKey, payload)
      return payload
    } catch {
      return null
    } finally {
      promiseCache.delete(positionKey)
    }
  })()

  promiseCache.set(positionKey, promise)
  return await promise
}

export async function fetchHumanBookLoadState(warm = true) {
  const response = await fetch(
    `${HUMAN_STATS_STATUS_PATH}${warm ? '?warm=1' : '?warm=0'}`,
  )
  if (!response.ok) {
    throw new Error(`Human book status request failed (${response.status}).`)
  }

  return (await response.json()) as HumanBookLoadState
}
