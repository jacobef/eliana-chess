import { spawn } from 'node:child_process'
import { createReadStream } from 'node:fs'
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises'
import path from 'node:path'
import readline from 'node:readline'
import { fileURLToPath } from 'node:url'

import { Chess } from 'chess.js'

const SOURCE_FILENAME = 'lichess_db_standard_rated_2013-01.pgn.zst'
const INDEX_CACHE_FILENAME = 'lichess-human-book-2013-01.json'
export const HUMAN_BOOK_ARTIFACT_RELATIVE_PATH =
  'server/generated/lichess-human-book-2013-01.json'
const MIN_SAMPLE_GAMES = 100
const MAX_INDEX_PLIES = 24
const SCAN_PROGRESS_CAP = 94
const moduleDirectory = path.dirname(fileURLToPath(import.meta.url))
const projectRoot = path.resolve(moduleDirectory, '..')

export type HumanBookLoadState = {
  status: 'idle' | 'loading' | 'ready' | 'error'
  detail: string
  progress?: number
  error?: string
}

type StoredPositionEntry = {
  totalGames: number
  moveCounts: Record<string, number>
}

type StoredIndex = {
  sourceFilename: string
  builtAt: string
  minSampleGames: number
  maxIndexPlies: number
  positions: Record<string, StoredPositionEntry>
}

export type HumanBookLookup = {
  positionKey: string
  totalGames: number
  moveCounts: Record<string, number>
  policy: Record<string, number>
  minSampleGames: number
  maxIndexPlies: number
}

class LichessHumanBook {
  private readonly listeners = new Set<() => void>()
  private index: Map<string, StoredPositionEntry> | null = null
  private indexPromise: Promise<Map<string, StoredPositionEntry>> | null = null
  private state: HumanBookLoadState = {
    status: 'idle',
    detail: 'Waiting for local game file…',
  }

  subscribe(listener: () => void) {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  getState() {
    return this.state
  }

  warm() {
    return this.ensureIndex()
  }

  async lookup(fen: string): Promise<HumanBookLookup> {
    const positionKey = normalizePositionKey(fen)
    const index = await this.ensureIndex()
    const entry = index.get(positionKey)

    return {
      positionKey,
      totalGames: entry?.totalGames ?? 0,
      moveCounts: entry?.moveCounts ?? {},
      policy: entry ? buildPolicy(entry.moveCounts, entry.totalGames) : {},
      minSampleGames: MIN_SAMPLE_GAMES,
      maxIndexPlies: MAX_INDEX_PLIES,
    }
  }

  private async ensureIndex() {
    if (this.index) {
      return this.index
    }

    if (!this.indexPromise) {
      this.indexPromise = this.loadOrBuildIndex()
        .then((index) => {
          this.index = index
          return index
        })
        .finally(() => {
          this.indexPromise = null
        })
    }

    return await this.indexPromise
  }

  private async loadOrBuildIndex() {
    this.setState({
      status: 'loading',
      detail: 'Checking local game data…',
      progress: 0,
    })

    const bundled = await this.readBundledIndex()
    if (bundled) {
      this.setState({
        status: 'ready',
        detail: 'Local game file is ready',
        progress: 100,
      })
      return bundled
    }

    const cached = await this.readLocalCacheIndex()
    if (cached) {
      this.setState({
        status: 'ready',
        detail: 'Local game file is ready',
        progress: 100,
      })
      return cached
    }

    try {
      if (!(await sourceFileExists())) {
        throw new Error(
          [
            'Human-book artifact not found in the deployed function bundle.',
            `Checked: ${getBundledIndexPathCandidates().join(', ')}`,
          ].join(' '),
        )
      }

      const builtIndex = await buildHumanBookIndex((state) => {
        this.setState(state)
      })
      this.setState({
        status: 'loading',
        detail: 'Saving local game index…',
        progress: 98,
      })
      await persistIndexFile(getLocalCachePath(), builtIndex)
      this.setState({
        status: 'ready',
        detail: 'Local game file is ready',
        progress: 100,
      })
      return new Map(Object.entries(builtIndex.positions))
    } catch (error) {
      this.setState({
        status: 'error',
        detail: 'Local game file failed to load',
        error:
          error instanceof Error
            ? error.message
            : 'The local game file could not be indexed.',
      })
      throw error
    }
  }

  private async readBundledIndex() {
    this.setState({
      status: 'loading',
      detail: 'Loading prebuilt game index…',
      progress: 10,
    })

    for (const candidate of getBundledIndexPathCandidates()) {
      const bundled = await readIndexFile(candidate)
      if (bundled) {
        return bundled
      }
    }

    return null
  }

  private async readLocalCacheIndex() {
    const sourcePath = getSourcePath()
    const cachePath = getLocalCachePath()

    try {
      const [sourceStat, cacheStat] = await Promise.all([stat(sourcePath), stat(cachePath)])
      if (cacheStat.mtimeMs < sourceStat.mtimeMs) {
        return null
      }

      return await readIndexFile(cachePath)
    } catch {
      return null
    }
  }

  private setState(nextState: HumanBookLoadState) {
    this.state = nextState
    for (const listener of this.listeners) {
      listener()
    }
  }
}

function getSourcePath() {
  return path.resolve(projectRoot, SOURCE_FILENAME)
}

function getBundledIndexPathCandidates() {
  return Array.from(
    new Set([
      path.resolve(projectRoot, HUMAN_BOOK_ARTIFACT_RELATIVE_PATH),
      path.resolve(moduleDirectory, 'generated', INDEX_CACHE_FILENAME),
      path.resolve(moduleDirectory, '..', HUMAN_BOOK_ARTIFACT_RELATIVE_PATH),
      path.resolve(process.cwd(), HUMAN_BOOK_ARTIFACT_RELATIVE_PATH),
    ]),
  )
}

function getLocalCachePath() {
  return path.resolve(
    projectRoot,
    'node_modules',
    '.cache',
    'eliana',
    INDEX_CACHE_FILENAME,
  )
}

async function sourceFileExists() {
  try {
    await stat(getSourcePath())
    return true
  } catch {
    return false
  }
}

async function readIndexFile(indexPath: string) {
  try {
    const raw = await readFile(indexPath, 'utf8')
    const parsed = JSON.parse(raw) as StoredIndex
    if (
      parsed.sourceFilename !== SOURCE_FILENAME ||
      parsed.minSampleGames !== MIN_SAMPLE_GAMES ||
      parsed.maxIndexPlies !== MAX_INDEX_PLIES
    ) {
      return null
    }

    return new Map(Object.entries(parsed.positions))
  } catch {
    return null
  }
}

function normalizePositionKey(fen: string) {
  return fen.trim().split(/\s+/).slice(0, 4).join(' ')
}

function extractStartFen(rawPgn: string) {
  const match = rawPgn.match(/\[FEN\s+"([^"]+)"\]/i)
  return match?.[1] ?? null
}

function extractSanMoves(rawPgn: string) {
  const normalized = rawPgn.replace(/\r/g, '')
  const separatorIndex = normalized.indexOf('\n\n')
  const moveSection =
    separatorIndex === -1 ? normalized : normalized.slice(separatorIndex + 2)
  const moves: string[] = []
  let token = ''
  let inBraceComment = false
  let inLineComment = false
  let variationDepth = 0

  const flushToken = () => {
    if (!token) {
      return
    }

    let cleaned = token

    while (/^\d+\.(\.\.)?/.test(cleaned)) {
      cleaned = cleaned.replace(/^\d+\.(\.\.)?/, '')
    }

    cleaned = cleaned.replace(/[!?]+/g, '')

    if (
      cleaned &&
      !/^\$\d+$/.test(cleaned) &&
      !/^(1-0|0-1|1\/2-1\/2|\*)$/.test(cleaned)
    ) {
      moves.push(cleaned)
    }

    token = ''
  }

  for (let index = 0; index < moveSection.length; index += 1) {
    if (moves.length > MAX_INDEX_PLIES) {
      break
    }

    const char = moveSection[index]

    if (inBraceComment) {
      if (char === '}') {
        inBraceComment = false
      }
      continue
    }

    if (inLineComment) {
      if (char === '\n') {
        inLineComment = false
      }
      continue
    }

    if (variationDepth > 0) {
      if (char === '(') {
        variationDepth += 1
      } else if (char === ')') {
        variationDepth -= 1
      }
      continue
    }

    if (char === '{') {
      flushToken()
      inBraceComment = true
      continue
    }

    if (char === ';') {
      flushToken()
      inLineComment = true
      continue
    }

    if (char === '(') {
      flushToken()
      variationDepth = 1
      continue
    }

    if (/\s/.test(char)) {
      flushToken()
      continue
    }

    token += char
  }

  flushToken()
  return moves
}

function buildPolicy(moveCounts: Record<string, number>, totalGames: number) {
  const entries = Object.entries(moveCounts)
    .filter(([, count]) => count > 0)
    .sort((left, right) => right[1] - left[1])

  return Object.fromEntries(
    entries.map(([move, count]) => [move, count / totalGames] as const),
  )
}

async function persistIndexFile(indexPath: string, index: StoredIndex) {
  await mkdir(path.dirname(indexPath), { recursive: true })
  await writeFile(indexPath, JSON.stringify(index), 'utf8')
}

async function buildHumanBookIndex(
  onStateChange: (state: HumanBookLoadState) => void,
): Promise<StoredIndex> {
  const rawCounts = new Map<string, StoredPositionEntry>()
  const sourcePath = getSourcePath()
  const sourceStat = await stat(sourcePath)
  const sourceStream = createReadStream(sourcePath)
  let compressedBytesRead = 0
  let indexedGames = 0
  let lastReportedProgress = -1
  const reportScanProgress = () => {
    const progress = Math.min(
      SCAN_PROGRESS_CAP,
      Math.round((compressedBytesRead / Math.max(1, sourceStat.size)) * SCAN_PROGRESS_CAP),
    )

    if (progress === lastReportedProgress && progress !== 0) {
      return
    }

    lastReportedProgress = progress
    onStateChange({
      status: 'loading',
      detail:
        indexedGames > 0
          ? `Scanning local game file… ${indexedGames.toLocaleString()} games`
          : 'Scanning local game file…',
      progress,
    })
  }

  sourceStream.on('data', (chunk) => {
    compressedBytesRead += chunk.length
    reportScanProgress()
  })

  const decompressor = spawn('zstd', ['-dc', '--stdout'], {
    cwd: process.cwd(),
    stdio: ['pipe', 'pipe', 'pipe'],
  })
  let stderr = ''

  sourceStream.pipe(decompressor.stdin)

  decompressor.stderr.setEncoding('utf8')
  decompressor.stderr.on('data', (chunk: string) => {
    stderr += chunk
  })

  const lines = readline.createInterface({
    input: decompressor.stdout,
    crlfDelay: Infinity,
  })

  let currentGameLines: string[] = []

  const flushCurrentGame = () => {
    const rawPgn = currentGameLines.join('\n').trim()
    currentGameLines = []
    if (!rawPgn) {
      return
    }

    if (indexGame(rawPgn, rawCounts)) {
      indexedGames += 1
      if (indexedGames % 500 === 0) {
        reportScanProgress()
      }
    }
  }

  for await (const line of lines) {
    if (line.startsWith('[Event ') && currentGameLines.length > 0) {
      flushCurrentGame()
    }

    currentGameLines.push(line)
  }

  flushCurrentGame()
  onStateChange({
    status: 'loading',
    detail: 'Finishing local game index…',
    progress: 96,
  })

  const exitCode = await new Promise<number>((resolve, reject) => {
    sourceStream.on('error', reject)
    decompressor.on('error', reject)
    decompressor.on('close', resolve)
  })

  if (exitCode !== 0) {
    throw new Error(stderr.trim() || `zstd exited with code ${exitCode}.`)
  }

  const positions: Record<string, StoredPositionEntry> = {}
  for (const [positionKey, entry] of rawCounts) {
    if (entry.totalGames >= MIN_SAMPLE_GAMES) {
      positions[positionKey] = entry
    }
  }

  return {
    sourceFilename: SOURCE_FILENAME,
    builtAt: new Date().toISOString(),
    minSampleGames: MIN_SAMPLE_GAMES,
    maxIndexPlies: MAX_INDEX_PLIES,
    positions,
  }
}

function indexGame(rawPgn: string, rawCounts: Map<string, StoredPositionEntry>) {
  const startFen = extractStartFen(rawPgn) ?? new Chess().fen()
  const replay = new Chess(startFen)
  let ply = 0
  let recordedMove = false

  for (const san of extractSanMoves(rawPgn)) {
    if (ply > MAX_INDEX_PLIES) {
      break
    }

    const positionKey = normalizePositionKey(replay.fen())
    let move: ReturnType<Chess['move']>

    try {
      move = replay.move(san)
    } catch {
      break
    }

    if (!move) {
      break
    }

    const moveUci = `${move.from}${move.to}${move.promotion ?? ''}`
    const entry = rawCounts.get(positionKey) ?? {
      totalGames: 0,
      moveCounts: {},
    }

    entry.totalGames += 1
    entry.moveCounts[moveUci] = (entry.moveCounts[moveUci] ?? 0) + 1
    rawCounts.set(positionKey, entry)
    ply += 1
    recordedMove = true
  }

  return recordedMove
}

export const lichessHumanBook = new LichessHumanBook()

export async function buildAndPersistHumanBookArtifact(
  onStateChange: (state: HumanBookLoadState) => void = () => undefined,
) {
  const index = await buildHumanBookIndex(onStateChange)
  const artifactPath = path.resolve(projectRoot, HUMAN_BOOK_ARTIFACT_RELATIVE_PATH)
  await persistIndexFile(artifactPath, index)
  return {
    path: artifactPath,
    positionCount: Object.keys(index.positions).length,
  }
}
