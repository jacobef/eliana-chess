import { InferenceSession, Tensor, env } from 'onnxruntime-web'

import { BinaryAssetCache } from './storage'
import { allPossibleMovesReversed, mirrorMove, preprocess } from './tensor'

const MODEL_URL =
  'https://raw.githubusercontent.com/CSSLab/maia-platform-frontend/e23a50e/public/maia2/maia_rapid.onnx'
const MODEL_VERSION = '1'
const MODEL_CACHE_KEY = 'maia-rapid'

env.logLevel = 'warning'
env.wasm.numThreads = 1
env.wasm.proxy = false

export type EngineLoadState = {
  status: 'idle' | 'loading' | 'ready' | 'error'
  detail: string
  progress?: number
  error?: string
}

export type MaiaEvaluation = {
  policy: Record<string, number>
  value: number
  policySource: 'maia' | 'database'
  sampleSize: number | null
}

type Listener = () => void

class MaiaEngine {
  private readonly cache = new BinaryAssetCache('ElianaEngineAssets', 'maia-models')
  private readonly listeners = new Set<Listener>()
  private state: EngineLoadState = {
    status: 'idle',
    detail: 'Waiting for Maia model…',
  }
  private session: InferenceSession | null = null
  private sessionPromise: Promise<InferenceSession> | null = null

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
    if (this.session) {
      return this.session
    }

    if (this.sessionPromise) {
      return await this.sessionPromise
    }

    this.sessionPromise = this.loadSession()
    return await this.sessionPromise
  }

  async evaluate(fen: string, eloSelf: number, eloOppo: number) {
    const session = await this.ensureReady()
    const { boardInput, legalMoves, eloSelfCategory, eloOppoCategory } =
      preprocess(fen, eloSelf, eloOppo)

    const feeds: Record<string, Tensor> = {
      boards: new Tensor('float32', boardInput, [1, 18, 8, 8]),
      elo_self: new Tensor(
        'int64',
        BigInt64Array.from([BigInt(eloSelfCategory)]),
      ),
      elo_oppo: new Tensor(
        'int64',
        BigInt64Array.from([BigInt(eloOppoCategory)]),
      ),
    }

    const { logits_maia, logits_value } = await session.run(feeds)
    return processOutputs(fen, logits_maia, logits_value, legalMoves)
  }

  private async loadSession() {
    this.setState({
      status: 'loading',
      detail: 'Checking Maia cache…',
      progress: 0,
    })

    await this.cache.requestPersistentStorage()
    const cachedModel = await this.cache.get(MODEL_CACHE_KEY, MODEL_VERSION)
    const modelBuffer = cachedModel ?? (await this.downloadModel())

    try {
      this.setState({
        status: 'loading',
        detail: 'Loading Maia into ONNX Runtime…',
      })
      this.session = await InferenceSession.create(modelBuffer)
      this.setState({
        status: 'ready',
        detail: 'Maia human-move model loaded',
      })
      return this.session
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Maia could not be initialized.'

      this.setState({
        status: 'error',
        detail: 'Maia load failed',
        error: message,
      })
      this.sessionPromise = null
      throw error
    }
  }

  private async downloadModel() {
    this.setState({
      status: 'loading',
      detail: 'Downloading Maia model…',
      progress: 0,
    })

    const response = await fetch(MODEL_URL)
    if (!response.ok || !response.body) {
      throw new Error(`Failed to fetch Maia model (${response.status}).`)
    }

    const reader = response.body.getReader()
    const contentLength = Number.parseInt(response.headers.get('Content-Length') ?? '0', 10)
    const chunks: Uint8Array[] = []
    let receivedLength = 0

    while (true) {
      const { done, value } = await reader.read()
      if (done) {
        break
      }

      chunks.push(value)
      receivedLength += value.length

      if (contentLength > 0) {
        this.setState({
          status: 'loading',
          detail: 'Downloading Maia model…',
          progress: Math.max(
            1,
            Math.min(99, Math.round((receivedLength / contentLength) * 100)),
          ),
        })
      }
    }

    const merged = new Uint8Array(receivedLength)
    let offset = 0

    for (const chunk of chunks) {
      merged.set(chunk, offset)
      offset += chunk.length
    }

    await this.cache.set(MODEL_CACHE_KEY, MODEL_VERSION, merged.buffer)

    return merged.buffer
  }

  private setState(nextState: EngineLoadState) {
    this.state = nextState
    for (const listener of this.listeners) {
      listener()
    }
  }
}

function processOutputs(
  fen: string,
  logitsMaia: Tensor,
  logitsValue: Tensor,
  legalMoves: Float32Array,
): MaiaEvaluation {
  const logits = logitsMaia.data as Float32Array
  const value = logitsValue.data as Float32Array

  let winProbability = Math.min(Math.max(value[0] / 2 + 0.5, 0), 1)
  const isBlackTurn = fen.split(' ')[1] === 'b'

  if (isBlackTurn) {
    winProbability = 1 - winProbability
  }

  const legalMoveIndices: number[] = []
  for (let index = 0; index < legalMoves.length; index += 1) {
    if (legalMoves[index] > 0) {
      legalMoveIndices.push(index)
    }
  }

  const legalMoveUcis = legalMoveIndices.map((moveIndex) => {
    const move = allPossibleMovesReversed[String(moveIndex)]
    return isBlackTurn ? mirrorMove(move) : move
  })
  const legalLogits = legalMoveIndices.map((index) => logits[index])
  const maxLogit = Math.max(...legalLogits)
  const expLogits = legalLogits.map((logit) => Math.exp(logit - maxLogit))
  const sum = expLogits.reduce((total, value_) => total + value_, 0)

  const policyEntries = legalMoveUcis.map((move, index) => [
    move,
    expLogits[index] / sum,
  ] as const)
  policyEntries.sort(([, left], [, right]) => right - left)

  return {
    policy: Object.fromEntries(policyEntries),
    value: Math.round(winProbability * 10000) / 10000,
    policySource: 'maia',
    sampleSize: null,
  }
}

export const maiaEngine = new MaiaEngine()
