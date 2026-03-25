import type StockfishWeb from 'lila-stockfish-web'
import createStockfish from 'lila-stockfish-web/sf171-79.js'
import stockfishWasmUrl from 'lila-stockfish-web/sf171-79.wasm?url'

export type StockfishLine = {
  move: string
  depth: number
  cp: number
  mate?: number
  pv: string[]
  multipv: number
}

type Listener = () => void

type PendingSearch = {
  resolve: (lines: StockfishLine[]) => void
  reject: (error: Error) => void
  lines: Map<number, StockfishLine>
}

class StockfishEngine {
  private readonly listeners = new Set<Listener>()
  private state: {
    status: 'idle' | 'loading' | 'ready' | 'error'
    detail: string
    error?: string
  } = {
    status: 'idle',
    detail: 'Waiting for Stockfish…',
  }
  private engine: StockfishWeb | null = null
  private enginePromise: Promise<StockfishWeb> | null = null
  private pending: PendingSearch | null = null
  private readyResolvers: Array<() => void> = []

  subscribe(listener: Listener) {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  getState() {
    return this.state
  }

  async ensureReady() {
    if (this.engine) {
      return this.engine
    }

    if (this.enginePromise) {
      return await this.enginePromise
    }

    this.enginePromise = this.initialize()
    return await this.enginePromise
  }

  shutdown() {
    if (!this.engine && !this.enginePromise) {
      return
    }

    this.stop()

    if (this.engine) {
      try {
        this.engine.listen = () => undefined
        this.engine.onError = () => undefined
        this.engine.uci('quit')
      } catch {
        // Ignore teardown failures from an already-exiting engine.
      }
    }

    this.engine = null
    this.enginePromise = null
    this.readyResolvers = []
    this.setState('idle', 'Stockfish suspended')
  }

  stop() {
    if (this.engine) {
      this.engine.uci('stop')
    }

    if (this.pending) {
      this.pending.reject(new Error('Stockfish search cancelled.'))
      this.pending = null
    }
  }

  async analyze(fen: string, depth: number, multiPv: number) {
    return await this.runSearch({
      fen,
      depth,
      multiPv,
    })
  }

  async analyzeMove(fen: string, move: string, depth: number) {
    const lines = await this.runSearch({
      fen,
      depth,
      multiPv: 1,
      searchMoves: [move],
    })

    return lines[0] ?? null
  }

  private async initialize() {
    try {
      this.setState('loading', 'Booting Stockfish WASM…')

      const engine = await createStockfish({
        wasmMemory: sharedWasmMemory(2560),
        locateFile: () => stockfishWasmUrl,
      })

      engine.listen = (chunk) => {
        this.onMessage(chunk)
      }
      engine.onError = (message) => {
        this.setState('error', 'Stockfish failed', message)
      }

      this.setState('loading', 'Loading NNUE networks…')
      const [bigNetwork, smallNetwork] = await Promise.all([
        fetchLocalBinary(`/stockfish/${engine.getRecommendedNnue(0)}`),
        fetchLocalBinary(`/stockfish/${engine.getRecommendedNnue(1)}`),
      ])

      engine.setNnueBuffer(new Uint8Array(bigNetwork), 0)
      engine.setNnueBuffer(new Uint8Array(smallNetwork), 1)
      engine.uci('uci')
      engine.uci('isready')

      this.engine = engine
      await this.waitForReady()

      this.setState('ready', 'Stockfish 17.1 ready')
      return engine
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : 'Stockfish could not be initialized.'
      this.setState('error', 'Stockfish failed', message)
      this.enginePromise = null
      throw error
    }
  }

  private async runSearch({
    fen,
    depth,
    multiPv,
    searchMoves,
  }: {
    fen: string
    depth: number
    multiPv: number
    searchMoves?: string[]
  }) {
    const engine = await this.ensureReady()
    await this.flush()

    return await new Promise<StockfishLine[]>((resolve, reject) => {
      this.pending = {
        resolve,
        reject,
        lines: new Map(),
      }

      engine.uci(`setoption name MultiPV value ${Math.max(1, multiPv)}`)
      engine.uci('ucinewgame')
      engine.uci(`position fen ${fen}`)
      engine.uci(
        searchMoves && searchMoves.length > 0
          ? `go depth ${depth} searchmoves ${searchMoves.join(' ')}`
          : `go depth ${depth}`,
      )
    })
  }

  private async flush() {
    if (!this.engine) {
      return
    }

    if (this.pending) {
      this.pending.reject(new Error('Superseded by a newer Stockfish search.'))
      this.pending = null
      this.engine.uci('stop')
    }

    this.engine.uci('isready')
    await this.waitForReady()
  }

  private waitForReady() {
    return new Promise<void>((resolve) => {
      this.readyResolvers.push(resolve)
    })
  }

  private onMessage(chunk: string) {
    for (const rawLine of chunk.split('\n')) {
      const line = rawLine.trim()
      if (!line) {
        continue
      }

      if (line === 'readyok') {
        const resolve = this.readyResolvers.shift()
        if (resolve) {
          resolve()
        }
        continue
      }

      if (!this.pending) {
        continue
      }

      if (line.startsWith('info ')) {
        const parsed = parseInfoLine(line)
        if (!parsed) {
          continue
        }

        const existing = this.pending.lines.get(parsed.multipv)
        if (!existing || parsed.depth >= existing.depth) {
          this.pending.lines.set(parsed.multipv, parsed)
        }
        continue
      }

      if (line.startsWith('bestmove ')) {
        const resolvedLines = [...this.pending.lines.values()].sort(
          (left, right) => left.multipv - right.multipv,
        )
        this.pending.resolve(resolvedLines)
        this.pending = null
      }
    }
  }

  private setState(
    status: 'idle' | 'loading' | 'ready' | 'error',
    detail: string,
    error?: string,
  ) {
    this.state = {
      status,
      detail,
      ...(error ? { error } : {}),
    }

    for (const listener of this.listeners) {
      listener()
    }
  }
}

function parseInfoLine(line: string): StockfishLine | null {
  const tokens = line.split(/\s+/)
  const depthIndex = tokens.indexOf('depth')
  const scoreIndex = tokens.indexOf('score')
  const pvIndex = tokens.indexOf('pv')

  if (depthIndex === -1 || scoreIndex === -1 || pvIndex === -1 || pvIndex + 1 >= tokens.length) {
    return null
  }

  const depth = Number.parseInt(tokens[depthIndex + 1], 10)
  const multipvIndex = tokens.indexOf('multipv')
  const multipv =
    multipvIndex === -1 ? 1 : Number.parseInt(tokens[multipvIndex + 1], 10)
  const scoreType = tokens[scoreIndex + 1]
  const scoreValue = Number.parseInt(tokens[scoreIndex + 2], 10)
  const pv = tokens
    .slice(pvIndex + 1)
    .filter((token) => /^[a-h][1-8][a-h][1-8][qrbn]?$/.test(token))

  if (!Number.isFinite(depth) || !Number.isFinite(multipv) || pv.length === 0) {
    return null
  }

  if (scoreType === 'cp') {
    return {
      move: pv[0],
      depth,
      cp: scoreValue,
      pv,
      multipv,
    }
  }

  if (scoreType === 'mate') {
    return {
      move: pv[0],
      depth,
      cp: scoreValue > 0 ? 100000 - scoreValue : -100000 - scoreValue,
      mate: scoreValue,
      pv,
      multipv,
    }
  }

  return null
}

function sharedWasmMemory(initialPages: number, maximumPages = 32767) {
  let shrinkFactor = 4
  let max = maximumPages

  while (true) {
    try {
      return new WebAssembly.Memory({
        shared: true,
        initial: initialPages,
        maximum: max,
      })
    } catch (error) {
      if (!(error instanceof RangeError) || max <= initialPages) {
        throw error
      }

      max = Math.max(initialPages, Math.ceil(max - max / shrinkFactor))
      shrinkFactor = shrinkFactor === 4 ? 3 : 4
    }
  }
}

async function fetchLocalBinary(path: string) {
  const response = await fetch(path)
  if (!response.ok) {
    throw new Error(`Failed to fetch ${path} (${response.status}).`)
  }

  return await response.arrayBuffer()
}

export const stockfishEngine = new StockfishEngine()
