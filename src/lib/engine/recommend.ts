import { Chess } from 'chess.js'

import type { StockfishLine } from './stockfish'
import type { stockfishEngine } from './stockfish'

const MIN_HUMAN_PROBABILITY = 0.1
const FALLBACK_HUMAN_PROBABILITY = 0.06
const PRIMARY_CP_LOSS_LIMIT = 60
const RELAXED_CP_LOSS_LIMIT = 110
const MAX_MAIA_CANDIDATES = 5
const MAX_MAIA_CUMULATIVE = 0.75

export type DecoratedLine = StockfishLine & {
  san: string
  maiaProbability: number
  cpLoss: number
  explanation: string
}

export type HumanAwareRecommendation = {
  suggestion: DecoratedLine | null
  suggestionSource: 'engine-best' | 'human-aware'
  engineBest: DecoratedLine | null
  humanCandidates: DecoratedLine[]
  stockfishCandidates: DecoratedLine[]
  minHumanProbability: number
  primaryCpLossLimit: number
}

export async function buildHumanAwareRecommendation({
  fen,
  maiaPolicy,
  stockfishLines,
  stockfishEngine: engine,
  depth,
}: {
  fen: string
  maiaPolicy: Record<string, number>
  stockfishLines: StockfishLine[]
  stockfishEngine: typeof stockfishEngine
  depth: number
}): Promise<HumanAwareRecommendation> {
  const lineByMove = new Map(stockfishLines.map((line) => [line.move, line]))
  const maiaCandidateMoves = getTopMaiaMoves(maiaPolicy)

  for (const move of maiaCandidateMoves) {
    if (lineByMove.has(move)) {
      continue
    }

    const line = await engine.analyzeMove(fen, move, Math.max(10, depth - 2))
    if (line) {
      lineByMove.set(move, line)
    }
  }

  const engineCandidates = stockfishLines
    .map((line) => decorateLine(fen, line, maiaPolicy, stockfishLines[0]))
    .filter((line): line is DecoratedLine => line !== null)
  const humanCandidates = [...new Set(maiaCandidateMoves)]
    .map((move) => lineByMove.get(move))
    .filter((line): line is StockfishLine => Boolean(line))
    .map((line) => decorateLine(fen, line, maiaPolicy, stockfishLines[0]))
    .filter((line): line is DecoratedLine => line !== null)
    .sort((left, right) => {
      if (right.maiaProbability !== left.maiaProbability) {
        return right.maiaProbability - left.maiaProbability
      }
      return scoreValue(right) - scoreValue(left)
    })

  const engineBest = engineCandidates[0] ?? null

  if (!engineBest) {
    return {
      suggestion: null,
      suggestionSource: 'human-aware',
      engineBest: null,
      humanCandidates: [],
      stockfishCandidates: [],
      minHumanProbability: MIN_HUMAN_PROBABILITY,
      primaryCpLossLimit: PRIMARY_CP_LOSS_LIMIT,
    }
  }

  if (engineBest.maiaProbability >= MIN_HUMAN_PROBABILITY) {
    return {
      suggestion: {
        ...engineBest,
        explanation:
          "Stockfish's top move also looks realistic for a human at these ratings.",
      },
      suggestionSource: 'engine-best',
      engineBest,
      humanCandidates,
      stockfishCandidates: engineCandidates,
      minHumanProbability: MIN_HUMAN_PROBABILITY,
      primaryCpLossLimit: PRIMARY_CP_LOSS_LIMIT,
    }
  }

  const primaryCandidate =
    humanCandidates.find(
      (line) =>
        line.maiaProbability >= MIN_HUMAN_PROBABILITY &&
        line.cpLoss <= PRIMARY_CP_LOSS_LIMIT,
    ) ??
    humanCandidates.find(
      (line) =>
        line.maiaProbability >= FALLBACK_HUMAN_PROBABILITY &&
        line.cpLoss <= RELAXED_CP_LOSS_LIMIT,
    ) ??
    humanCandidates[0] ??
    engineBest

  const explanation =
    primaryCandidate.move === engineBest.move
      ? 'No stronger practical alternative cleared the filter, so the engine best remains the fallback.'
      : `Stockfish prefers ${engineBest.san}, but it looks much less likely to be played at these ratings, at about ${Math.round(
          engineBest.maiaProbability * 100,
        )}%.`

  return {
    suggestion: {
      ...primaryCandidate,
      explanation,
    },
    suggestionSource:
      primaryCandidate.move === engineBest.move ? 'engine-best' : 'human-aware',
    engineBest,
    humanCandidates,
    stockfishCandidates: engineCandidates,
    minHumanProbability: MIN_HUMAN_PROBABILITY,
    primaryCpLossLimit: PRIMARY_CP_LOSS_LIMIT,
  }
}

function getTopMaiaMoves(maiaPolicy: Record<string, number>) {
  const moves: string[] = []
  let cumulative = 0

  for (const [move, probability] of Object.entries(maiaPolicy)) {
    moves.push(move)
    cumulative += probability

    if (moves.length >= MAX_MAIA_CANDIDATES || cumulative >= MAX_MAIA_CUMULATIVE) {
      break
    }
  }

  return moves
}

function decorateLine(
  fen: string,
  line: StockfishLine | undefined,
  maiaPolicy: Record<string, number>,
  bestLine: StockfishLine | undefined,
): DecoratedLine | null {
  if (!line || !bestLine) {
    return null
  }

  const san = moveToSan(fen, line.move)
  if (!san) {
    return null
  }

  return {
    ...line,
    san,
    maiaProbability: maiaPolicy[line.move] ?? 0,
    cpLoss: Math.max(0, scoreValue(bestLine) - scoreValue(line)),
    explanation: '',
  }
}

function moveToSan(fen: string, move: string) {
  const chess = new Chess(fen)
  const result = chess.move({
    from: move.slice(0, 2),
    to: move.slice(2, 4),
    promotion: move[4] as 'q' | 'r' | 'b' | 'n' | undefined,
  })

  return result?.san ?? null
}

function scoreValue(line: StockfishLine) {
  if (line.mate !== undefined) {
    return line.mate > 0 ? 100000 - line.mate : -100000 - line.mate
  }

  return line.cp
}
