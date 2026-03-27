import { startTransition, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { Chess } from 'chess.js'
import type { Square } from 'chess.js'
import { Chessboard } from 'react-chessboard'

import './App.css'
import {
  maiaEngine,
  type EngineLoadState,
  type MaiaEvaluation,
} from './lib/engine/maia'
import {
  stockfishEngine,
  type StockfishLine,
} from './lib/engine/stockfish'
import {
  buildHumanAwareRecommendation,
  type DecoratedLine,
  type HumanAwareRecommendation,
} from './lib/engine/recommend'
import {
  fetchHumanBookLoadState,
  fetchHumanPositionStats,
} from './lib/engine/humanStats'
import {
  runMoveAnalysisWithLlm,
  type MoveChatMessage,
  type MoveAnalysisContext,
} from './lib/llm/moveAnalysis'

const DEFAULT_START_FEN = new Chess().fen()
const RATING_OPTIONS = [
  { chessCom: 800, maia: 1100 },
  { chessCom: 900, maia: 1200 },
  { chessCom: 1000, maia: 1300 },
  { chessCom: 1100, maia: 1400 },
  { chessCom: 1200, maia: 1500 },
  { chessCom: 1300, maia: 1600 },
  { chessCom: 1400, maia: 1700 },
  { chessCom: 1500, maia: 1800 },
  { chessCom: 1600, maia: 1900 },
  { chessCom: 1700, maia: 2000 },
  { chessCom: 1900, maia: 2200 },
] as const
const DEFAULT_ENGINE_DEPTH = 14
const ANALYSIS_MULTI_PV = 6
const ANALYSIS_CACHE_LIMIT = 240
const CURRENT_ANALYSIS_PRIORITY = 100
const PREFETCH_PRIORITY_START = 72
const PREFETCH_DELAY_MS = 90
const STOCKFISH_IDLE_SHUTDOWN_MS = 2500

type GameNode = {
  id: string
  fen: string
  parentId: string | null
  childIds: string[]
  moveUci?: string
  san?: string
  side?: 'w' | 'b'
}

type PlayerRatings = Partial<Record<'w' | 'b', number>>

type GameTree = {
  rootId: string
  nodes: Record<string, GameNode>
  selectedChildIds: Record<string, string>
  playerRatings: PlayerRatings
}

type PromotionChoice = {
  from: string
  to: string
  color: 'w' | 'b'
}

type AnalysisState =
  | { status: 'idle' | 'loading' }
  | { status: 'game-over' }
  | { status: 'error'; error: string }
  | {
      status: 'ready'
      maia: MaiaEvaluation
      recommendation: HumanAwareRecommendation
    }

type MoveCell = {
  nodeId: string
  san: string
  annotation: MoveAnnotation
}

type MoveRow = {
  moveNumber: number
  white?: MoveCell
  black?: MoveCell
}

type AnalysisRequest = {
  key: string
  fen: string
  whiteRating: number
  blackRating: number
}

type AnalysisQueueItem = AnalysisRequest & {
  priority: number
  kind: 'current' | 'background'
  interrupted: false | 'requeue' | 'drop'
  resolve: (state: AnalysisState) => void
}

type ScoreLike = Pick<StockfishLine, 'cp' | 'mate'>

type PlayedMoveCommentary = {
  san: string
  annotation: MoveAnnotation
  moveUci: string
  maiaProbability: number
  cpLoss: number
  line: ScoreLike
  explanation: string
  bestMove: string | null
  bestMoveUci: string | null
  bestRealisticMove: string | null
  bestRealisticMoveUci: string | null
}

type MoveAnnotation = '' | '!!' | '!' | '!?' | '?!' | '?' | '??'

type LlmThreadMessage = MoveChatMessage & {
  id: string
  model?: string
}

type LlmAnalysisState = {
  status: 'idle' | 'loading' | 'ready' | 'error'
  messages: LlmThreadMessage[]
  draft: string
  error: string | null
  retryPrompt: string | null
}

type StoredLlmThread = LlmAnalysisState & {
  nodeId: string
  moveLabel: string
}

type LlmNotification = {
  id: string
  threadKey: string
  nodeId: string
  moveLabel: string
  rating: number
  preview: string
}

type ChessComApiGame = {
  url?: string
  pgn?: string
  end_time?: number
  time_class?: string
  rated?: boolean
  white: {
    username?: string
  }
  black: {
    username?: string
  }
}

type ChessComRecentGame = {
  id: string
  pgn: string
  url: string | null
  summary: string
  playerColor: 'white' | 'black'
}

function createNodeId() {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID()
  }

  return `node-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`
}

function createThreadMessage(
  role: 'user' | 'assistant',
  text: string,
  model?: string,
): LlmThreadMessage {
  return {
    id: createNodeId(),
    role,
    text,
    ...(model ? { model } : {}),
  }
}

function createEmptyLlmAnalysisState(): LlmAnalysisState {
  return {
    status: 'idle',
    messages: [],
    draft: '',
    error: null,
    retryPrompt: null,
  }
}

function toVisibleLlmAnalysisState(
  thread: StoredLlmThread | undefined,
): LlmAnalysisState {
  return thread
    ? {
        status: thread.status,
        messages: thread.messages,
        draft: thread.draft,
        error: thread.error,
        retryPrompt: thread.retryPrompt,
      }
    : createEmptyLlmAnalysisState()
}

function createGameTree(startFen: string, playerRatings: PlayerRatings = {}): GameTree {
  const rootId = createNodeId()
  return {
    rootId,
    nodes: {
      [rootId]: {
        id: rootId,
        fen: startFen,
        parentId: null,
        childIds: [],
      },
    },
    selectedChildIds: {},
    playerRatings,
  }
}

const INITIAL_GAME_TREE = createGameTree(DEFAULT_START_FEN)

function getMaiaRatingForChessComRating(chessComRating: number) {
  return (
    RATING_OPTIONS.find((option) => option.chessCom === chessComRating)?.maia ?? 1500
  )
}

function normalizeFen(rawFen: string) {
  return rawFen.trim().split(/\s+/).join(' ')
}

function parseFen(rawFen: string) {
  return new Chess(normalizeFen(rawFen))
}

function extractPgnStartFen(rawPgn: string) {
  const match = rawPgn.match(/\[FEN\s+"([^"]+)"\]/i)
  return match?.[1] ?? DEFAULT_START_FEN
}

function extractPgnTag(rawPgn: string, tag: string) {
  const match = rawPgn.match(new RegExp(`\\[${tag}\\s+"([^"]+)"\\]`, 'i'))
  return match?.[1] ?? null
}

function parsePgnRating(rawValue: string | null) {
  if (!rawValue) {
    return undefined
  }

  const parsed = Number.parseInt(rawValue, 10)
  return Number.isFinite(parsed) ? parsed : undefined
}

function extractPgnPlayerRatings(rawPgn: string): PlayerRatings {
  return {
    w: parsePgnRating(extractPgnTag(rawPgn, 'WhiteElo')),
    b: parsePgnRating(extractPgnTag(rawPgn, 'BlackElo')),
  }
}

function extractChessComUsername(rawInput: string) {
  const trimmed = rawInput.trim()
  if (!trimmed) {
    return null
  }

  const normalized = trimmed.replace(/^@/, '')

  try {
    const parsed = new URL(
      /^https?:\/\//i.test(normalized) ? normalized : `https://${normalized}`,
    )
    const host = parsed.hostname.replace(/^www\./i, '').toLowerCase()
    if (host === 'chess.com' || host.endsWith('.chess.com')) {
      const segments = parsed.pathname.split('/').filter(Boolean)
      const memberIndex = segments.findIndex(
        (segment) => segment.toLowerCase() === 'member',
      )
      if (memberIndex !== -1) {
        const username = segments[memberIndex + 1]
        if (username) {
          return decodeURIComponent(username)
        }
      }
    }
  } catch {
    // Fall back to treating the input as a raw username.
  }

  return normalized.split('/')[0] || null
}

function formatChessComGameSummary(game: ChessComApiGame, username: string) {
  const lowerUsername = username.toLowerCase()
  const whiteUsername = game.white.username ?? 'White'
  const blackUsername = game.black.username ?? 'Black'
  const isWhite = whiteUsername.toLowerCase() === lowerUsername
  const color = isWhite ? 'White' : 'Black'
  const opponent = isWhite ? blackUsername : whiteUsername
  const dateLabel = game.end_time
    ? new Date(game.end_time * 1000).toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
      })
    : 'Unknown date'
  const timeClass = (game.time_class ?? 'game').replace(/_/g, ' ')
  const ratedLabel = game.rated ? 'rated' : 'casual'

  return `${dateLabel} · ${color} vs ${opponent} · ${timeClass} · ${ratedLabel}`
}

function getChessComPlayerColor(
  game: ChessComApiGame,
  username: string,
): 'white' | 'black' {
  return (game.white.username ?? '').toLowerCase() === username.toLowerCase()
    ? 'white'
    : 'black'
}

async function fetchRecentChessComGames(
  profileOrUsername: string,
  limit = 5,
): Promise<ChessComRecentGame[]> {
  const username = extractChessComUsername(profileOrUsername)
  if (!username) {
    throw new Error('Enter a Chess.com username or profile URL.')
  }

  const archivesResponse = await fetch(
    `https://api.chess.com/pub/player/${encodeURIComponent(username)}/games/archives`,
  )
  if (!archivesResponse.ok) {
    if (archivesResponse.status === 404) {
      throw new Error(`Chess.com profile not found: ${username}`)
    }

    throw new Error(`Chess.com request failed (${archivesResponse.status}).`)
  }

  const archivesPayload = (await archivesResponse.json()) as {
    archives?: string[]
  }
  const archiveUrls = [...(archivesPayload.archives ?? [])].reverse()
  const collectedGames: ChessComRecentGame[] = []

  for (const archiveUrl of archiveUrls) {
    if (collectedGames.length >= limit) {
      break
    }

    const archiveResponse = await fetch(archiveUrl)
    if (!archiveResponse.ok) {
      continue
    }

    const archivePayload = (await archiveResponse.json()) as {
      games?: ChessComApiGame[]
    }
    const monthGames = [...(archivePayload.games ?? [])]
      .filter((game) => typeof game.pgn === 'string' && game.pgn.trim())
      .sort((left, right) => (right.end_time ?? 0) - (left.end_time ?? 0))

    for (const game of monthGames) {
      collectedGames.push({
        id: game.url ?? `${game.end_time ?? 0}-${collectedGames.length}`,
        pgn: game.pgn as string,
        url: game.url ?? null,
        summary: formatChessComGameSummary(game, username),
        playerColor: getChessComPlayerColor(game, username),
      })

      if (collectedGames.length >= limit) {
        break
      }
    }
  }

  return collectedGames.slice(0, limit)
}

function buildGameTreeFromPgn(rawPgn: string) {
  const trimmedPgn = rawPgn.trim()
  if (!trimmedPgn) {
    throw new Error('Paste a PGN before importing.')
  }

  const parser = new Chess()
  parser.loadPgn(trimmedPgn)

  const startFen = extractPgnStartFen(trimmedPgn)
  const playerRatings = extractPgnPlayerRatings(trimmedPgn)
  const replay = new Chess(startFen)
  const tree = createGameTree(startFen, playerRatings)
  let parentId = tree.rootId

  for (const san of parser.history()) {
    const move = replay.move(san)
    if (!move) {
      throw new Error(`Could not replay imported PGN move: ${san}`)
    }

    const childId = createNodeId()
    tree.nodes[childId] = {
      id: childId,
      fen: replay.fen(),
      parentId,
      childIds: [],
      moveUci: `${move.from}${move.to}${move.promotion ?? ''}`,
      san: move.san,
      side: move.color,
    }
    tree.nodes[parentId] = {
      ...tree.nodes[parentId],
      childIds: [...tree.nodes[parentId].childIds, childId],
    }
    tree.selectedChildIds[parentId] = childId
    parentId = childId
  }

  return tree
}

function uciToSquares(uci: string) {
  return {
    from: uci.slice(0, 2),
    to: uci.slice(2, 4),
  }
}

function formatPercent(probability: number) {
  return `${(probability * 100).toFixed(probability >= 0.1 ? 1 : 2)}%`
}

function formatScore(line: ScoreLike, sideToMove: 'w' | 'b') {
  const whitePerspective = sideToMove === 'w' ? 1 : -1

  if (line.mate !== undefined) {
    const whiteMate = line.mate * whitePerspective
    return whiteMate > 0
      ? `Mate for White in ${whiteMate}`
      : `Mate for Black in ${Math.abs(whiteMate)}`
  }

  const pawns = (line.cp * whitePerspective) / 100
  return `${pawns >= 0 ? '+' : ''}${pawns.toFixed(2)}`
}

function getLineScoreValue(line: ScoreLike) {
  if (line.mate !== undefined) {
    return line.mate > 0 ? 100000 - line.mate : -100000 - line.mate
  }

  return line.cp
}

function getHumanRarityDescriptor(maiaProbability: number) {
  if (maiaProbability < 0.0001) {
    return 'inexplicable'
  }

  if (maiaProbability < 0.01) {
    return 'very strange'
  }

  if (maiaProbability < 0.05) {
    return 'strange'
  }

  return null
}

function withIndefiniteArticle(phrase: string) {
  return /^[aeiou]/i.test(phrase) ? `an ${phrase}` : `a ${phrase}`
}

function describeMistakePhrase(
  maiaProbability: number,
  suffix = '',
) {
  if (maiaProbability >= 0.16) {
    return `was an understandable mistake${suffix}`
  }

  const rarity = getHumanRarityDescriptor(maiaProbability)
  if (rarity) {
    return `was ${withIndefiniteArticle(`${rarity} mistake`)}${suffix}`
  }

  return `was a mistake${suffix}`
}

function describePlayedMoveVerdict(
  cpLoss: number,
  maiaProbability: number,
  line: ScoreLike,
  engineBest: ScoreLike | null,
) {
  const playedScore = getLineScoreValue(line)
  const bestScore = engineBest ? getLineScoreValue(engineBest) : null

  if (bestScore !== null) {
    if (bestScore >= 900 && playedScore >= 700) {
      if (cpLoss < 50) {
        const rarity = getHumanRarityDescriptor(maiaProbability)
        if (rarity) {
          return `was ${withIndefiniteArticle(`${rarity} move`)}`
        }

        return maiaProbability >= 0.16
          ? 'was a strong, natural move'
          : 'was very close to best'
      }

      const rarity = getHumanRarityDescriptor(maiaProbability)
      return rarity
        ? `was ${withIndefiniteArticle(`${rarity} way to keep a won game`)}`
        : cpLoss <= 220
          ? 'was not best but kept a won game'
          : 'missed a cleaner win'
    }

    if (bestScore >= 700 && playedScore >= 350) {
      if (cpLoss < 50) {
        const rarity = getHumanRarityDescriptor(maiaProbability)
        if (rarity) {
          return `was ${withIndefiniteArticle(`${rarity} move`)} but kept a winning position`
        }

        return maiaProbability >= 0.16
          ? 'was a strong move that kept a winning position'
          : 'was very close to best and kept a winning position'
      }

      if (cpLoss <= 120) {
        return maiaProbability >= 0.14
          ? 'was an understandable inaccuracy but kept a winning position'
          : 'was an inaccuracy but kept a winning position'
      }

      return describeMistakePhrase(
        maiaProbability,
        ' but kept a winning position',
      )
    }
  }

  if (cpLoss === 0) {
    return maiaProbability >= 0.18
      ? 'was a perfect practical choice'
      : 'was best by engine'
  }

  if (cpLoss <= 15) {
    return maiaProbability >= 0.16
      ? 'was an excellent practical move'
      : 'was nearly best'
  }

  if (cpLoss < 50) {
    const rarity = getHumanRarityDescriptor(maiaProbability)
    if (rarity) {
      return `was ${withIndefiniteArticle(`${rarity} move`)}`
    }

    return maiaProbability >= 0.16 ? 'was a strong, natural move' : 'was very close to best'
  }

  if (cpLoss <= 120) {
    const rarity = getHumanRarityDescriptor(maiaProbability)
    if (rarity) {
      return `was ${withIndefiniteArticle(`${rarity} inaccuracy`)}`
    }

    return maiaProbability >= 0.14 ? 'was an understandable inaccuracy' : 'was an inaccuracy'
  }

  if (cpLoss <= 220) {
    return describeMistakePhrase(maiaProbability)
  }

  if (cpLoss <= 320) {
    if (maiaProbability >= 0.16) {
      return 'was a very human blunder'
    }

    const rarity = getHumanRarityDescriptor(maiaProbability)
    return rarity ? `was ${withIndefiniteArticle(`${rarity} blunder`)}` : 'was a blunder'
  }

  if (cpLoss <= 500) {
    if (maiaProbability >= 0.16) {
      return 'was a costly blunder'
    }

    const rarity = getHumanRarityDescriptor(maiaProbability)
    return rarity
      ? `was ${withIndefiniteArticle(`${rarity} blunder`)}`
      : 'was a serious blunder'
  }

  if (maiaProbability >= 0.16) {
    return 'was a very costly blunder'
  }

  const rarity = getHumanRarityDescriptor(maiaProbability)
  return rarity ? `was ${withIndefiniteArticle(`${rarity} blunder`)}` : 'was a serious blunder'
}

function describeBestMovePraise(
  maiaProbability: number,
  bestMoveMargin: number | null,
) {
  const margin = bestMoveMargin ?? 0

  if (maiaProbability >= 0.18) {
    return ''
  }

  if (maiaProbability >= 0.1) {
    return margin >= 140
      ? 'It also mattered, because the alternatives were noticeably worse.'
      : ''
  }

  if (maiaProbability >= 0.04) {
    if (margin >= 180) {
      return 'That was a very strong find in an important moment.'
    }

    if (margin >= 90) {
      return 'That was a very strong find.'
    }

    return 'That was a nice find.'
  }

  if (maiaProbability >= 0.015) {
    if (margin >= 180) {
      return 'That was an excellent find, and it really mattered.'
    }

    if (margin >= 90) {
      return 'That was an excellent find.'
    }

    return 'That was a very sharp find.'
  }

  if (margin >= 180) {
    return 'That was a remarkable find, and the position really demanded it.'
  }

  if (margin >= 90) {
    return 'That was a remarkable find.'
  }

  return 'That was a superb find.'
}

function buildPlayedMoveExplanation({
  san,
  moveUci,
  maiaProbability,
  cpLoss,
  bestMoveMargin,
  line,
  suggestion,
  engineBest,
}: {
  san: string
  moveUci: string
  maiaProbability: number
  cpLoss: number
  bestMoveMargin: number | null
  line: ScoreLike
  suggestion: HumanAwareRecommendation['suggestion']
  engineBest: HumanAwareRecommendation['engineBest']
}) {
  const matchedEngine = engineBest?.move === moveUci
  const matchedHuman = suggestion?.move === moveUci

  if (matchedEngine) {
    const praise = describeBestMovePraise(maiaProbability, bestMoveMargin)
    return praise ? `${san} was the best move. ${praise}` : `${san} was the best move.`
  }

  if (matchedHuman) {
    if (cpLoss < 50) {
      return `${san} was the strongest realistic move.`
    }

    if (cpLoss <= 120) {
      return `${san} was an understandable inaccuracy. Many players would choose it.`
    }

    const verdict = describePlayedMoveVerdict(
      cpLoss,
      maiaProbability,
      line,
      engineBest,
    )
    return `${san} ${verdict}. Many players would choose it.`
  }

  const verdict = describePlayedMoveVerdict(
    cpLoss,
    maiaProbability,
    line,
    engineBest,
  )

  if (cpLoss < 50) {
    return `${san} ${verdict}.`
  }

  return `${san} ${verdict}.`
}

function describeMoveAnnotation({
  cpLoss,
  maiaProbability,
  line,
  engineBest,
  bestMoveMargin,
}: {
  cpLoss: number
  maiaProbability: number
  line: ScoreLike
  engineBest: ScoreLike | null
  bestMoveMargin: number | null
}): MoveAnnotation {
  const margin = bestMoveMargin ?? 0
  const playedScore = getLineScoreValue(line)
  const bestScore = engineBest ? getLineScoreValue(engineBest) : null
  const clearlyWon = bestScore !== null && bestScore >= 900 && playedScore >= 700
  const winning = bestScore !== null && bestScore >= 700 && playedScore >= 350

  if (cpLoss === 0) {
    if (maiaProbability <= 0.015 && margin >= 140) {
      return '!!'
    }

    if (maiaProbability <= 0.05 && margin >= 70) {
      return '!'
    }

    if (maiaProbability <= 0.025 && margin >= 35) {
      return '!'
    }

    return ''
  }

  if (cpLoss <= 15) {
    return maiaProbability <= 0.02 && margin >= 90 ? '!?' : ''
  }

  if (cpLoss < 50) {
    return maiaProbability <= 0.02 ? '!?' : ''
  }

  if (clearlyWon) {
    if (cpLoss <= 140) {
      return ''
    }

    return cpLoss <= 280 ? '?!' : '?'
  }

  if (winning) {
    return cpLoss <= 120 ? '?!' : '?'
  }

  if (cpLoss <= 120) {
    return '?!'
  }

  if (cpLoss <= 260) {
    return '?'
  }

  return '??'
}

function getMoveAnnotationVisualClass(annotation: MoveAnnotation) {
  if (annotation === '!!') {
    return 'annotation-brilliant'
  }

  if (annotation === '!' || annotation === '!?') {
    return 'annotation-great'
  }

  if (annotation === '?!') {
    return 'annotation-inaccuracy'
  }

  if (annotation === '?') {
    return 'annotation-mistake'
  }

  if (annotation === '??') {
    return 'annotation-blunder'
  }

  return ''
}

function AnnotatedMoveText({
  san,
  annotation,
  className,
}: {
  san: string
  annotation: MoveAnnotation
  className?: string
}) {
  const visualClass = getMoveAnnotationVisualClass(annotation)
  const joinedClassName = [
    'annotated-move',
    annotation === '!!' ? visualClass : '',
    className ?? '',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <span className={joinedClassName}>
      <span className={annotation === '!!' ? visualClass : undefined}>{san}</span>
      {annotation ? (
        <span className={`annotated-move-annotation ${visualClass}`}>{annotation}</span>
      ) : null}
    </span>
  )
}

function buildPlayedMoveCommentaryFromAnalysis({
  node,
  parentAnalysis,
  currentAnalysis,
  allowApproximateFallback = true,
}: {
  node: GameNode
  parentAnalysis: AnalysisState | null | undefined
  currentAnalysis: AnalysisState | null | undefined
  allowApproximateFallback?: boolean
}): PlayedMoveCommentary | null {
  if (
    !node.moveUci ||
    !node.san ||
    !node.side ||
    !parentAnalysis ||
    parentAnalysis.status !== 'ready'
  ) {
    return null
  }

  const parentRecommendation = parentAnalysis.recommendation
  const parentCandidates = [
    parentRecommendation.engineBest,
    ...parentRecommendation.stockfishCandidates,
    ...parentRecommendation.humanCandidates,
  ].filter((line): line is NonNullable<typeof parentRecommendation.engineBest> => Boolean(line))

  const playedLineFromParent = parentCandidates.find(
    (line) => line.move === node.moveUci,
  )

  let line: ScoreLike | null = playedLineFromParent
    ? {
        cp: playedLineFromParent.cp,
        mate: playedLineFromParent.mate,
      }
    : null
  const maiaProbability =
    playedLineFromParent?.maiaProbability ??
    parentAnalysis.maia.policy[node.moveUci] ??
    0
  let cpLoss = playedLineFromParent?.cpLoss ?? 0

  if (!line) {
    if (!allowApproximateFallback) {
      return null
    }

    if (!currentAnalysis || currentAnalysis.status !== 'ready') {
      return null
    }

    const currentBestLine =
      currentAnalysis.recommendation.engineBest ??
      currentAnalysis.recommendation.stockfishCandidates[0] ??
      null

    if (!currentBestLine) {
      return null
    }

    line = {
      cp: -currentBestLine.cp,
      mate:
        currentBestLine.mate !== undefined ? -currentBestLine.mate : undefined,
    }
    cpLoss = parentRecommendation.engineBest
      ? Math.max(
          0,
          getLineScoreValue(parentRecommendation.engineBest) -
            getLineScoreValue(line),
        )
      : 0
  }

  const suggestion = parentRecommendation.suggestion
  const engineBest = parentRecommendation.engineBest
  const nextBestAlternative =
    parentRecommendation.stockfishCandidates.find(
      (candidate) => candidate.move !== engineBest?.move,
    ) ?? null
  const bestMoveMargin = nextBestAlternative?.cpLoss ?? null
  const annotation = describeMoveAnnotation({
    cpLoss,
    maiaProbability,
    line,
    engineBest,
    bestMoveMargin,
  })

  return {
    san: node.san,
    annotation,
    moveUci: node.moveUci,
    maiaProbability,
    cpLoss,
    line,
    explanation: buildPlayedMoveExplanation({
      san: node.san,
      moveUci: node.moveUci,
      maiaProbability,
      cpLoss,
      bestMoveMargin,
      line,
      suggestion,
      engineBest,
    }),
    bestMove: engineBest?.san ?? null,
    bestMoveUci: engineBest?.move ?? null,
    bestRealisticMove: suggestion?.san ?? engineBest?.san ?? null,
    bestRealisticMoveUci: suggestion?.move ?? engineBest?.move ?? null,
  }
}

function getNextNodeId(tree: GameTree, nodeId: string) {
  const node = tree.nodes[nodeId]
  if (!node || node.childIds.length === 0) {
    return null
  }

  return tree.selectedChildIds[nodeId] ?? node.childIds[0]
}

function getMainLineNextNodeId(tree: GameTree, nodeId: string) {
  const node = tree.nodes[nodeId]
  return node?.childIds[0] ?? null
}

function getSelectedLineNodeIds(tree: GameTree) {
  const ids = [tree.rootId]
  const visited = new Set(ids)
  let cursorId = tree.rootId

  while (true) {
    const nextId = getNextNodeId(tree, cursorId)
    if (!nextId || visited.has(nextId)) {
      break
    }

    ids.push(nextId)
    visited.add(nextId)
    cursorId = nextId
  }

  return ids
}

function getMoveRows(
  lineNodes: GameNode[],
  startFen: string,
  annotationsByNodeId: Map<string, MoveAnnotation>,
): MoveRow[] {
  const rows: MoveRow[] = []
  let moveNumber = Number.parseInt(startFen.split(' ')[5] ?? '1', 10)

  for (let index = 1; index < lineNodes.length; index += 1) {
    const node = lineNodes[index]
    const annotation = annotationsByNodeId.get(node.id) ?? ''
    const cell = {
      nodeId: node.id,
      san: node.san ?? node.moveUci ?? '--',
      annotation,
    }

    if (node.side === 'w') {
      rows.push({
        moveNumber,
        white: cell,
      })
      continue
    }

    const lastRow = rows.at(-1)
    if (lastRow && lastRow.moveNumber === moveNumber && !lastRow.black) {
      lastRow.black = cell
    } else {
      rows.push({
        moveNumber,
        black: cell,
      })
    }
    moveNumber += 1
  }

  return rows
}

function isTextEntryTarget(target: EventTarget | null) {
  return (
    target instanceof HTMLElement &&
    target.closest('input, textarea, select, [contenteditable="true"]') !== null
  )
}

function buildAnalysisCacheKey(
  fen: string,
  whiteRating: number,
  blackRating: number,
) {
  return `${normalizeFen(fen)}|w:${whiteRating}|b:${blackRating}|d:${DEFAULT_ENGINE_DEPTH}`
}

function cacheAnalysisState(
  cache: Map<string, AnalysisState>,
  key: string,
  state: AnalysisState,
) {
  if (cache.has(key)) {
    cache.delete(key)
  }

  cache.set(key, state)

  if (cache.size > ANALYSIS_CACHE_LIMIT) {
    const oldestKey = cache.keys().next().value
    if (oldestKey !== undefined) {
      cache.delete(oldestKey)
    }
  }
}

function isCompletedAnalysisState(state: AnalysisState | undefined) {
  return state?.status === 'ready' || state?.status === 'game-over'
}

function buildSelectedChildIdsToNode(tree: GameTree, targetNodeId: string) {
  if (!tree.nodes[targetNodeId]) {
    return tree.selectedChildIds
  }

  const selectedChildIds = { ...tree.selectedChildIds }
  let currentNode: GameNode | null = tree.nodes[targetNodeId]

  while (currentNode?.parentId) {
    selectedChildIds[currentNode.parentId] = currentNode.id
    currentNode = tree.nodes[currentNode.parentId]
  }

  return selectedChildIds
}

function buildNotificationPreview(text: string) {
  const normalized = text.trim().replace(/\s+/g, ' ')
  if (!normalized) {
    return 'Reply ready.'
  }

  return normalized.length > 120 ? `${normalized.slice(0, 117)}...` : normalized
}

async function analyzePosition(
  fen: string,
  whiteRating: number,
  blackRating: number,
): Promise<AnalysisState> {
  const chess = new Chess(fen)

  if (chess.isGameOver() || chess.moves().length === 0) {
    return { status: 'game-over' }
  }

  const moverRating = chess.turn() === 'w' ? whiteRating : blackRating
  const opponentRating = chess.turn() === 'w' ? blackRating : whiteRating

  try {
    const [maiaEvaluation, stockfishLines, humanStats] = await Promise.all([
      maiaEngine.evaluate(fen, moverRating, opponentRating),
      stockfishEngine.analyze(fen, DEFAULT_ENGINE_DEPTH, ANALYSIS_MULTI_PV),
      fetchHumanPositionStats(fen),
    ])
    const databasePolicy =
      humanStats !== null && humanStats.totalGames >= humanStats.minSampleGames
        ? humanStats
        : null
    const effectiveHumanEvaluation: MaiaEvaluation = databasePolicy
      ? {
          ...maiaEvaluation,
          policy: databasePolicy.policy,
          policySource: 'database',
          sampleSize: databasePolicy.totalGames,
        }
      : {
          ...maiaEvaluation,
          policySource: 'maia',
          sampleSize: humanStats?.totalGames ?? null,
        }

    const recommendation = await buildHumanAwareRecommendation({
      fen,
      maiaPolicy: effectiveHumanEvaluation.policy,
      stockfishLines,
      stockfishEngine,
      depth: DEFAULT_ENGINE_DEPTH,
    })

    return {
      status: 'ready',
      maia: effectiveHumanEvaluation,
      recommendation,
    }
  } catch (error) {
    return {
      status: 'error',
      error:
        error instanceof Error
          ? error.message
          : 'Analysis failed unexpectedly.',
    }
  }
}

function getTreePrefetchNodeIds(
  tree: GameTree,
  currentNodeId: string,
  selectedLineNodeIds: string[],
) {
  const orderedIds: string[] = []
  const seen = new Set<string>()
  const variationRoots: string[] = []

  const addNode = (nodeId: string | null | undefined) => {
    if (!nodeId || seen.has(nodeId) || !tree.nodes[nodeId]) {
      return false
    }

    seen.add(nodeId)
    orderedIds.push(nodeId)
    return true
  }

  const currentLineIndex = selectedLineNodeIds.indexOf(currentNodeId)
  if (currentLineIndex !== -1) {
    for (let offset = 1; offset < selectedLineNodeIds.length; offset += 1) {
      addNode(selectedLineNodeIds[currentLineIndex + offset])
      addNode(selectedLineNodeIds[currentLineIndex - offset])
    }
  } else {
    for (const nodeId of selectedLineNodeIds) {
      addNode(nodeId)
    }
  }

  const lineVariationOrder =
    currentLineIndex === -1
      ? selectedLineNodeIds
      : selectedLineNodeIds
          .map((nodeId, index) => ({
            nodeId,
            distance: Math.abs(index - currentLineIndex),
          }))
          .sort((left, right) => left.distance - right.distance)
          .map(({ nodeId }) => nodeId)

  for (const lineNodeId of lineVariationOrder) {
    const lineNode = tree.nodes[lineNodeId]
    const selectedChildId = getNextNodeId(tree, lineNodeId)
    for (const childId of lineNode.childIds) {
      if (childId !== selectedChildId && addNode(childId)) {
        variationRoots.push(childId)
      }
    }
  }

  const queue = [...variationRoots]
  while (queue.length > 0) {
    const nodeId = queue.shift()
    if (!nodeId) {
      continue
    }

    const node = tree.nodes[nodeId]
    const selectedChildId = getNextNodeId(tree, nodeId)

    if (selectedChildId && addNode(selectedChildId)) {
      queue.push(selectedChildId)
    }

    for (const childId of node.childIds) {
      if (childId !== selectedChildId && addNode(childId)) {
        queue.push(childId)
      }
    }
  }

  return orderedIds
}

function appendBoxShadow(style: CSSProperties | undefined, shadow: string): CSSProperties {
  const boxShadow =
    typeof style?.boxShadow === 'string' && style.boxShadow.trim()
      ? `${style.boxShadow}, ${shadow}`
      : shadow

  return {
    ...style,
    boxShadow,
  }
}

function getSquareStyles(
  currentMove: GameNode,
  recommendation: HumanAwareRecommendation | null,
  selectedSquare: Square | null,
  selectedTargetSquares: Square[],
): Record<string, CSSProperties> {
  const styles: Record<string, CSSProperties> = {}

  if (currentMove.moveUci) {
    const { from, to } = uciToSquares(currentMove.moveUci)
    styles[from] = {
      background:
        'linear-gradient(135deg, rgba(197, 106, 27, 0.30), rgba(197, 106, 27, 0.10))',
    }
    styles[to] = {
      background:
        'linear-gradient(135deg, rgba(197, 106, 27, 0.48), rgba(197, 106, 27, 0.18))',
    }
  }

  if (recommendation?.suggestion) {
    const { from, to } = uciToSquares(recommendation.suggestion.move)
    styles[from] = appendBoxShadow(
      styles[from],
      'inset 0 0 0 3px rgba(47, 122, 98, 0.70)',
    )
    styles[to] = appendBoxShadow(
      styles[to],
      'inset 0 0 0 3px rgba(47, 122, 98, 0.95)',
    )
  }

  if (selectedSquare) {
    styles[selectedSquare] = appendBoxShadow(
      styles[selectedSquare],
      'inset 0 0 0 3px rgba(63, 92, 168, 0.95)',
    )
  }

  for (const square of selectedTargetSquares) {
    styles[square] = appendBoxShadow(
      styles[square],
      'inset 0 0 0 3px rgba(63, 92, 168, 0.42)',
    )
  }

  return styles
}

function moveToUciObject(move: string) {
  return {
    from: move.slice(0, 2),
    to: move.slice(2, 4),
    promotion: move[4] as 'q' | 'r' | 'b' | 'n' | undefined,
  }
}

function pvToSanSequence(fen: string, pv: string[]) {
  const chess = new Chess(fen)
  const sanMoves: string[] = []

  for (const move of pv) {
    let result: ReturnType<Chess['move']>

    try {
      result = chess.move(moveToUciObject(move))
    } catch {
      break
    }

    if (!result) {
      break
    }

    sanMoves.push(result.san)
  }

  return sanMoves
}

function summarizeDecoratedLine(fen: string, line: DecoratedLine) {
  const sideToMove = new Chess(fen).turn()

  return {
    san: line.san,
    uci: line.move,
    eval: formatScore(line, sideToMove),
    humanLikelihoodPercent: Number((line.maiaProbability * 100).toFixed(1)),
    cpLoss: line.cpLoss,
    pvSan: pvToSanSequence(fen, line.pv),
    pvUci: line.pv,
  }
}

function buildLlmAnalysisSnapshot(fen: string, state: AnalysisState) {
  if (state.status === 'error') {
    return {
      status: 'error',
      fen,
      error: state.error,
    }
  }

  if (state.status === 'game-over') {
    const chess = new Chess(fen)
    return {
      status: 'game-over',
      fen,
      sideToMove: chess.turn() === 'w' ? 'White' : 'Black',
      legalMoves: chess.moves().length,
      checkmate: chess.isCheckmate(),
      draw: chess.isDraw(),
    }
  }

  if (state.status !== 'ready') {
    return {
      status: state.status,
      fen,
    }
  }

  const recommendation = state.recommendation
  const chess = new Chess(fen)

  return {
    status: 'ready',
    fen,
    sideToMove: chess.turn() === 'w' ? 'White' : 'Black',
    bestMove: recommendation.engineBest
      ? summarizeDecoratedLine(fen, recommendation.engineBest)
      : null,
    bestRealisticMove: recommendation.suggestion
      ? summarizeDecoratedLine(fen, recommendation.suggestion)
      : null,
    engineLines: recommendation.stockfishCandidates
      .slice(0, 4)
      .map((line) => summarizeDecoratedLine(fen, line)),
    realisticLines: recommendation.humanCandidates
      .slice(0, 4)
      .map((line) => summarizeDecoratedLine(fen, line)),
  }
}

function listLegalMovesSnapshot(fen: string) {
  const chess = new Chess(fen)

  return {
    fen,
    sideToMove: chess.turn() === 'w' ? 'White' : 'Black',
    legalMoves: chess.moves({ verbose: true }).map((move) => ({
      san: move.san,
      uci: `${move.from}${move.to}${move.promotion ?? ''}`,
    })),
  }
}

function playMovesSnapshot(fen: string, moves: string[]) {
  const chess = new Chess(fen)
  const appliedMoves: Array<{ san: string; uci: string }> = []

  for (const move of moves) {
    let result: ReturnType<Chess['move']>

    try {
      result = chess.move(moveToUciObject(move))
    } catch (error) {
      return {
        status: 'error',
        fen: chess.fen(),
        error:
          error instanceof Error ? error.message : `Could not play move ${move}.`,
        appliedMoves,
      }
    }

    if (!result) {
      return {
        status: 'error',
        fen: chess.fen(),
        error: `Could not play move ${move}.`,
        appliedMoves,
      }
    }

    appliedMoves.push({
      san: result.san,
      uci: `${result.from}${result.to}${result.promotion ?? ''}`,
    })
  }

  return {
    status: 'ready',
    startFen: fen,
    finalFen: chess.fen(),
    sideToMove: chess.turn() === 'w' ? 'White' : 'Black',
    appliedMoves,
    inCheck: chess.inCheck(),
    checkmate: chess.isCheckmate(),
    draw: chess.isDraw(),
    gameOver: chess.isGameOver(),
  }
}

function App() {
  const [boardOrientation, setBoardOrientation] = useState<'white' | 'black'>(
    'white',
  )
  const [showGreenArrow, setShowGreenArrow] = useState(true)
  const [showOrangeArrow, setShowOrangeArrow] = useState(true)
  const [rating, setRating] = useState<number>(1200)
  const [gameTree, setGameTree] = useState<GameTree>(INITIAL_GAME_TREE)
  const [currentNodeId, setCurrentNodeId] = useState(INITIAL_GAME_TREE.rootId)
  const [fenDraft, setFenDraft] = useState(DEFAULT_START_FEN)
  const [pgnDraft, setPgnDraft] = useState('')
  const [chessComProfileInput, setChessComProfileInput] = useState('notnotnotjacob')
  const [chessComGames, setChessComGames] = useState<ChessComRecentGame[]>([])
  const [chessComLoading, setChessComLoading] = useState(false)
  const [fenError, setFenError] = useState<string | null>(null)
  const [pgnError, setPgnError] = useState<string | null>(null)
  const [chessComError, setChessComError] = useState<string | null>(null)
  const [promotionChoice, setPromotionChoice] = useState<PromotionChoice | null>(
    null,
  )
  const [selectedSquare, setSelectedSquare] = useState<Square | null>(null)
  const [maiaState, setMaiaState] = useState<EngineLoadState>(
    maiaEngine.getState(),
  )
  const [stockfishState, setStockfishState] = useState<EngineLoadState>(
    stockfishEngine.getState(),
  )
  const [humanBookState, setHumanBookState] = useState<EngineLoadState>({
    status: 'idle',
    detail: 'Waiting for local game file…',
  })
  const [initialBootComplete, setInitialBootComplete] = useState(false)
  const [analysis, setAnalysis] = useState<AnalysisState>({ status: 'idle' })
  const [playedMoveAnalysis, setPlayedMoveAnalysis] = useState<AnalysisState | null>(
    null,
  )
  const [llmAnalysis, setLlmAnalysis] = useState<LlmAnalysisState>(
    createEmptyLlmAnalysisState,
  )
  const [llmNotifications, setLlmNotifications] = useState<LlmNotification[]>([])
  const [analysisCacheVersion, setAnalysisCacheVersion] = useState(0)
  const [analysisQueueActive, setAnalysisQueueActive] = useState(false)
  const [activeLlmRequestCount, setActiveLlmRequestCount] = useState(0)
  const analysisCacheRef = useRef<Map<string, AnalysisState>>(new Map())
  const moveAnnotationCacheRef = useRef<Map<string, MoveAnnotation>>(new Map())
  const analysisPromiseRef = useRef<Map<string, Promise<AnalysisState>>>(new Map())
  const analysisQueueRef = useRef<AnalysisQueueItem[]>([])
  const analysisQueueRunningRef = useRef(false)
  const activeAnalysisRef = useRef<AnalysisQueueItem | null>(null)
  const llmAnalysisCacheRef = useRef<Map<string, StoredLlmThread>>(
    new Map(),
  )
  const llmRequestIdsRef = useRef<Map<string, number>>(new Map())
  const activeLlmRequestsRef = useRef<Set<string>>(new Set())
  const currentMoveLlmKeyRef = useRef<string | null>(null)

  useEffect(() => {
    const unsubscribeMaia = maiaEngine.subscribe(() => {
      setMaiaState(maiaEngine.getState())
    })
    const unsubscribeStockfish = stockfishEngine.subscribe(() => {
      setStockfishState(stockfishEngine.getState())
    })

    maiaEngine.ensureReady().catch(() => undefined)
    stockfishEngine.ensureReady().catch(() => undefined)

    return () => {
      unsubscribeMaia()
      unsubscribeStockfish()
      stockfishEngine.shutdown()
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    let timer: number | null = null

    const pollHumanBookState = async () => {
      try {
        const nextState = await fetchHumanBookLoadState()
        if (cancelled) {
          return
        }

        startTransition(() => {
          setHumanBookState(nextState)
        })

        if (nextState.status === 'idle' || nextState.status === 'loading') {
          timer = window.setTimeout(() => {
            void pollHumanBookState()
          }, 350)
        }
      } catch (error) {
        if (cancelled) {
          return
        }

        startTransition(() => {
          setHumanBookState({
            status: 'error',
            detail: 'Local game file failed to load',
            error:
              error instanceof Error
                ? error.message
                : 'The local game file could not be loaded.',
          })
        })
      }
    }

    void pollHumanBookState()

    return () => {
      cancelled = true
      if (timer !== null) {
        window.clearTimeout(timer)
      }
    }
  }, [])

  useEffect(() => {
    if (
      !initialBootComplete &&
      maiaState.status === 'ready' &&
      stockfishState.status === 'ready' &&
      humanBookState.status === 'ready'
    ) {
      setInitialBootComplete(true)
    }
  }, [
    humanBookState.status,
    initialBootComplete,
    maiaState.status,
    stockfishState.status,
  ])

  useEffect(() => {
    moveAnnotationCacheRef.current = new Map()
  }, [gameTree.rootId, rating])

  const maiaRating = getMaiaRatingForChessComRating(rating)
  const whiteRating = maiaRating
  const blackRating = maiaRating
  const currentNode = gameTree.nodes[currentNodeId]
  const startFen = gameTree.nodes[gameTree.rootId].fen
  const currentFen = currentNode.fen
  const currentChess = useMemo(() => new Chess(currentFen), [currentFen])
  const currentTurn = currentChess.turn()
  useEffect(() => {
    setSelectedSquare(null)
  }, [currentFen])
  const selectedTargetSquares = useMemo(() => {
    if (!selectedSquare) {
      return [] as Square[]
    }

    const piece = currentChess.get(selectedSquare)
    if (!piece || piece.color !== currentTurn) {
      return [] as Square[]
    }

    return currentChess
      .moves({ square: selectedSquare, verbose: true })
      .map((move) => move.to)
  }, [currentChess, currentTurn, selectedSquare])
  const selectedLineNodeIds = useMemo(
    () => getSelectedLineNodeIds(gameTree),
    [gameTree],
  )
  const selectedLineNodes = useMemo(
    () => selectedLineNodeIds.map((nodeId) => gameTree.nodes[nodeId]),
    [gameTree.nodes, selectedLineNodeIds],
  )
  const moveAnnotationsByNodeId = useMemo(() => {
    void analysisCacheVersion
    const annotations = new Map(moveAnnotationCacheRef.current)

    for (const node of Object.values(gameTree.nodes)) {
      if (!node.parentId) {
        continue
      }

      const parentNode = gameTree.nodes[node.parentId]
      if (!parentNode) {
        continue
      }

      const parentAnalysis = analysisCacheRef.current.get(
        buildAnalysisCacheKey(parentNode.fen, whiteRating, blackRating),
      )
      const currentAnalysis = analysisCacheRef.current.get(
        buildAnalysisCacheKey(node.fen, whiteRating, blackRating),
      )
      const commentary = buildPlayedMoveCommentaryFromAnalysis({
        node,
        parentAnalysis,
        currentAnalysis,
        allowApproximateFallback: false,
      })

      if (commentary?.annotation) {
        annotations.set(node.id, commentary.annotation)
      }
    }

    moveAnnotationCacheRef.current = annotations
    return annotations
  }, [analysisCacheVersion, blackRating, gameTree.nodes, whiteRating])
  const moveRows = useMemo(
    () => getMoveRows(selectedLineNodes, startFen, moveAnnotationsByNodeId),
    [moveAnnotationsByNodeId, selectedLineNodes, startFen],
  )
  const currentChildren = currentNode.childIds.map((childId) => gameTree.nodes[childId])
  const selectedChildId = getNextNodeId(gameTree, currentNodeId)
  const mainLineChildId = getMainLineNextNodeId(gameTree, currentNodeId)
  const currentAnalysisRequest = useMemo(
    () => ({
      key: buildAnalysisCacheKey(currentFen, whiteRating, blackRating),
      fen: currentFen,
      whiteRating,
      blackRating,
    }),
    [blackRating, currentFen, whiteRating],
  )
  const previousAnalysisRequest = useMemo(() => {
    if (!currentNode.parentId) {
      return null
    }

    const parentFen = gameTree.nodes[currentNode.parentId].fen

    return {
      key: buildAnalysisCacheKey(parentFen, whiteRating, blackRating),
      fen: parentFen,
      whiteRating,
      blackRating,
    }
  }, [blackRating, currentNode.parentId, gameTree.nodes, whiteRating])
  const branchPrefetchRequests = useMemo(() => {
    void analysisCacheVersion
    const seenKeys = new Set<string>()
    const pendingKeys = new Set(analysisPromiseRef.current.keys())
    const requests: Array<AnalysisRequest & { priority: number }> = []
    const nodeIds = getTreePrefetchNodeIds(gameTree, currentNodeId, selectedLineNodeIds)

    for (const [index, nodeId] of nodeIds.entries()) {
      const node = gameTree.nodes[nodeId]
      const key = buildAnalysisCacheKey(node.fen, whiteRating, blackRating)
      if (
        key === currentAnalysisRequest.key ||
        seenKeys.has(key) ||
        pendingKeys.has(key) ||
        isCompletedAnalysisState(analysisCacheRef.current.get(key))
      ) {
        continue
      }

      seenKeys.add(key)
      requests.push({
        key,
        fen: node.fen,
        whiteRating,
        blackRating,
        priority: Math.max(1, PREFETCH_PRIORITY_START - index),
      })
    }

    return requests
  }, [
    blackRating,
    currentAnalysisRequest.key,
    currentNodeId,
    gameTree,
    analysisCacheVersion,
    selectedLineNodeIds,
    whiteRating,
  ])
  const squareStyles = useMemo(
    () =>
      getSquareStyles(
        currentNode,
        analysis.status === 'ready' ? analysis.recommendation : null,
        selectedSquare,
        selectedTargetSquares,
      ),
    [analysis, currentNode, selectedSquare, selectedTargetSquares],
  )
  const arrows = useMemo(() => {
    if (analysis.status !== 'ready') {
      return [] as Array<{ startSquare: string; endSquare: string; color: string }>
    }

    const boardArrows: Array<{
      startSquare: string
      endSquare: string
      color: string
    }> = []
    const suggestion = analysis.recommendation.suggestion
    const engineBest = analysis.recommendation.engineBest
    const sharedMove =
      suggestion && engineBest && suggestion.move === engineBest.move

    if (sharedMove && (showGreenArrow || showOrangeArrow)) {
      const { from, to } = uciToSquares(suggestion.move)
      boardArrows.push({
        startSquare: from,
        endSquare: to,
        color: 'rgba(63, 92, 168, 0.94)',
      })
      return boardArrows
    }

    if (suggestion && showGreenArrow) {
      const { from, to } = uciToSquares(suggestion.move)
      boardArrows.push({
        startSquare: from,
        endSquare: to,
        color: 'rgba(47, 122, 98, 0.96)',
      })
    }

    if (engineBest && engineBest.move !== suggestion?.move && showOrangeArrow) {
      const { from, to } = uciToSquares(engineBest.move)
      boardArrows.push({
        startSquare: from,
        endSquare: to,
        color: 'rgba(197, 106, 27, 0.86)',
      })
    }

    return boardArrows
  }, [analysis, showGreenArrow, showOrangeArrow])

  const pumpAnalysisQueue = useCallback(() => {
    if (analysisQueueRunningRef.current) {
      return
    }

    analysisQueueRunningRef.current = true
    startTransition(() => {
      setAnalysisQueueActive(true)
    })

    void (async () => {
      try {
        while (analysisQueueRef.current.length > 0) {
          analysisQueueRef.current.sort((left, right) => right.priority - left.priority)
          const nextItem = analysisQueueRef.current.shift()
          if (!nextItem) {
            continue
          }

          activeAnalysisRef.current = nextItem

          let state: AnalysisState

          try {
            state = await analyzePosition(
              nextItem.fen,
              nextItem.whiteRating,
              nextItem.blackRating,
            )
          } catch (error) {
            state = {
              status: 'error',
              error:
                error instanceof Error
                  ? error.message
                  : 'Analysis failed unexpectedly.',
            }
          }

          activeAnalysisRef.current = null

          if (nextItem.interrupted === 'requeue') {
            nextItem.interrupted = false
            analysisQueueRef.current.push(nextItem)
            continue
          }

          if (nextItem.interrupted === 'drop') {
            analysisPromiseRef.current.delete(nextItem.key)
            nextItem.resolve({
              status: 'error',
              error: 'Superseded by a newer analysis request.',
            })
            continue
          }

          if (state.status === 'ready' || state.status === 'game-over') {
            cacheAnalysisState(analysisCacheRef.current, nextItem.key, state)
            startTransition(() => {
              setAnalysisCacheVersion((version) => version + 1)
            })
          }

          analysisPromiseRef.current.delete(nextItem.key)
          nextItem.resolve(state)
        }
      } finally {
        analysisQueueRunningRef.current = false

        if (analysisQueueRef.current.length > 0) {
          pumpAnalysisQueue()
        } else {
          startTransition(() => {
            setAnalysisQueueActive(false)
          })
        }
      }
    })()
  }, [])

  const ensureAnalysis = useCallback(
    (
      request: AnalysisRequest,
      priority: number,
      kind: 'current' | 'background' = 'background',
    ) => {
      const cached = analysisCacheRef.current.get(request.key)
      if (cached) {
        cacheAnalysisState(analysisCacheRef.current, request.key, cached)
        return Promise.resolve(cached)
      }

      const existingPromise = analysisPromiseRef.current.get(request.key)
      if (existingPromise) {
        const queuedItem = analysisQueueRef.current.find((item) => item.key === request.key)
        if (queuedItem && priority > queuedItem.priority) {
          queuedItem.priority = priority
        }
        if (queuedItem && kind === 'current') {
          queuedItem.kind = 'current'
        }

        const activeItem = activeAnalysisRef.current
        if (activeItem && activeItem.key === request.key && kind === 'current') {
          activeItem.kind = 'current'
        }
        return existingPromise
      }

      const promise = new Promise<AnalysisState>((resolve) => {
        analysisQueueRef.current.push({
          ...request,
          priority,
          kind,
          interrupted: false,
          resolve,
        })

        if (kind === 'current') {
          const activeItem = activeAnalysisRef.current
          if (
            activeItem &&
            activeItem.key !== request.key &&
            activeItem.interrupted === false
          ) {
            activeItem.interrupted =
              activeItem.kind === 'background' ? 'requeue' : 'drop'
            stockfishEngine.stop()
          }
        }

        pumpAnalysisQueue()
      })

      analysisPromiseRef.current.set(request.key, promise)
      return promise
    },
    [pumpAnalysisQueue],
  )

  useEffect(() => {
    let active = true
    const cached = analysisCacheRef.current.get(currentAnalysisRequest.key)

    if (cached) {
      cacheAnalysisState(analysisCacheRef.current, currentAnalysisRequest.key, cached)
      startTransition(() => {
        setAnalysis(cached)
      })
      return () => {
        active = false
      }
    }

    startTransition(() => {
      setAnalysis({ status: 'loading' })
    })

    ensureAnalysis(currentAnalysisRequest, CURRENT_ANALYSIS_PRIORITY, 'current').then(
      (state) => {
        if (!active) {
          return
        }

        startTransition(() => {
          setAnalysis(state)
        })
      },
    )

    return () => {
      active = false
    }
  }, [currentAnalysisRequest, ensureAnalysis])

  useEffect(() => {
    if (!previousAnalysisRequest) {
      startTransition(() => {
        setPlayedMoveAnalysis(null)
      })
      return
    }

    let active = true
    const cached = analysisCacheRef.current.get(previousAnalysisRequest.key)

    if (cached) {
      startTransition(() => {
        setPlayedMoveAnalysis(cached)
      })
      return () => {
        active = false
      }
    }

    startTransition(() => {
      setPlayedMoveAnalysis({ status: 'loading' })
    })

    ensureAnalysis(previousAnalysisRequest, CURRENT_ANALYSIS_PRIORITY - 1).then(
      (state) => {
        if (!active) {
          return
        }

        startTransition(() => {
          setPlayedMoveAnalysis(state)
        })
      },
    )

    return () => {
      active = false
    }
  }, [ensureAnalysis, previousAnalysisRequest])

  useEffect(() => {
    if (
      maiaState.status === 'error' ||
      stockfishState.status === 'error' ||
      branchPrefetchRequests.length === 0
    ) {
      return
    }

    const timer = window.setTimeout(() => {
      for (const request of branchPrefetchRequests) {
        void ensureAnalysis(request, request.priority, 'background')
      }
    }, PREFETCH_DELAY_MS)

    return () => {
      window.clearTimeout(timer)
    }
  }, [branchPrefetchRequests, ensureAnalysis, maiaState.status, stockfishState.status])

  useEffect(() => {
    if (
      !initialBootComplete ||
      analysisQueueActive ||
      branchPrefetchRequests.length > 0 ||
      activeLlmRequestCount > 0
    ) {
      return
    }

    const timer = window.setTimeout(() => {
      stockfishEngine.shutdown()
    }, STOCKFISH_IDLE_SHUTDOWN_MS)

    return () => {
      window.clearTimeout(timer)
    }
  }, [
    activeLlmRequestCount,
    analysisQueueActive,
    branchPrefetchRequests.length,
    initialBootComplete,
  ])

  const replaceGameTree = useCallback((nextTree: GameTree, nextFenDraft?: string) => {
    const nextStartFen = nextTree.nodes[nextTree.rootId].fen
    startTransition(() => {
      setGameTree(nextTree)
      setCurrentNodeId(nextTree.rootId)
      setFenDraft(nextFenDraft ?? nextStartFen)
    })
  }, [])

  const commitMove = useCallback(
    (from: string, to: string, promotion?: 'q' | 'r' | 'b' | 'n') => {
      const chess = new Chess(currentFen)
      let move: ReturnType<Chess['move']>

      try {
        move = chess.move({
          from,
          to,
          promotion,
        })
      } catch {
        return false
      }

      if (!move) {
        return false
      }

      const moveUci = `${move.from}${move.to}${move.promotion ?? ''}`
      const parentNode = gameTree.nodes[currentNodeId]
      const existingChildId = parentNode.childIds.find(
        (childId) => gameTree.nodes[childId].moveUci === moveUci,
      )

      if (existingChildId) {
        startTransition(() => {
          setGameTree((previousTree) => ({
            ...previousTree,
            selectedChildIds: {
              ...previousTree.selectedChildIds,
              [currentNodeId]: existingChildId,
            },
          }))
          setCurrentNodeId(existingChildId)
        })
        return true
      }

      const childId = createNodeId()
      const nextNode: GameNode = {
        id: childId,
        fen: chess.fen(),
        parentId: currentNodeId,
        childIds: [],
        moveUci,
        san: move.san,
        side: move.color,
      }

      startTransition(() => {
        setGameTree((previousTree) => ({
          ...previousTree,
          nodes: {
            ...previousTree.nodes,
            [currentNodeId]: {
              ...previousTree.nodes[currentNodeId],
              childIds: [...previousTree.nodes[currentNodeId].childIds, childId],
            },
            [childId]: nextNode,
          },
          selectedChildIds: {
            ...previousTree.selectedChildIds,
            [currentNodeId]: childId,
          },
        }))
        setCurrentNodeId(childId)
      })

      return true
    },
    [currentFen, currentNodeId, gameTree.nodes],
  )

  const tryBoardMove = useCallback(
    (from: string, to: string) => {
      const piece = currentChess.get(from as Square)
      if (!piece) {
        return false
      }

      const isPromotion =
        piece.type === 'p' && (to.endsWith('1') || to.endsWith('8'))

      if (isPromotion) {
        setPromotionChoice({
          from,
          to,
          color: piece.color,
        })
        setSelectedSquare(null)
        return false
      }

      const committed = commitMove(from, to)
      if (committed) {
        setSelectedSquare(null)
      }

      return committed
    },
    [commitMove, currentChess],
  )

  const onPieceDrop = useCallback(
    ({
      sourceSquare,
      targetSquare,
    }: {
      piece: unknown
      sourceSquare: string
      targetSquare: string | null
    }) => {
      setSelectedSquare(null)

      if (!targetSquare) {
        return false
      }

      return tryBoardMove(sourceSquare, targetSquare)
    },
    [tryBoardMove],
  )

  const onSquareClick = useCallback(
    ({ square }: { piece: unknown; square: string }) => {
      const clickedSquare = square as Square
      const clickedPiece = currentChess.get(clickedSquare)

      if (!selectedSquare) {
        if (clickedPiece && clickedPiece.color === currentTurn) {
          setSelectedSquare(clickedSquare)
        }
        return
      }

      if (selectedSquare === clickedSquare) {
        setSelectedSquare(null)
        return
      }

      const selectedPiece = currentChess.get(selectedSquare)
      if (!selectedPiece || selectedPiece.color !== currentTurn) {
        setSelectedSquare(
          clickedPiece && clickedPiece.color === currentTurn ? clickedSquare : null,
        )
        return
      }

      if (clickedPiece && clickedPiece.color === currentTurn) {
        setSelectedSquare(clickedSquare)
        return
      }

      void tryBoardMove(selectedSquare, clickedSquare)
    },
    [currentChess, currentTurn, selectedSquare, tryBoardMove],
  )

  const loadStartingFen = useCallback(() => {
    try {
      const chess = parseFen(fenDraft)
      const normalizedFen = chess.fen()
      setFenError(null)
      setPgnError(null)
      replaceGameTree(createGameTree(normalizedFen), normalizedFen)
    } catch (error) {
      setFenError(
        error instanceof Error ? error.message : 'That FEN could not be parsed.',
      )
    }
  }, [fenDraft, replaceGameTree])

  const loadPgnIntoTree = useCallback(
    (rawPgn: string) => {
      try {
        const importedTree = buildGameTreeFromPgn(rawPgn)
        setFenError(null)
        setPgnError(null)
        setPgnDraft(rawPgn)
        replaceGameTree(importedTree)
      } catch (error) {
        setPgnError(
          error instanceof Error ? error.message : 'The PGN could not be imported.',
        )
      }
    },
    [replaceGameTree],
  )

  const importPgn = useCallback(() => {
    loadPgnIntoTree(pgnDraft)
  }, [loadPgnIntoTree, pgnDraft])

  const fetchChessComRecentGames = useCallback(async () => {
    setChessComLoading(true)
    setChessComError(null)

    try {
      const games = await fetchRecentChessComGames(chessComProfileInput)
      setChessComGames(games)
      if (games.length === 0) {
        setChessComError('No recent PGNs were found for that profile.')
      }
    } catch (error) {
      setChessComGames([])
      setChessComError(
        error instanceof Error
          ? error.message
          : 'Chess.com games could not be fetched.',
      )
    } finally {
      setChessComLoading(false)
    }
  }, [chessComProfileInput])

  const importChessComGame = useCallback(
    (game: ChessComRecentGame) => {
      setChessComError(null)
      setBoardOrientation(game.playerColor)
      loadPgnIntoTree(game.pgn)
    },
    [loadPgnIntoTree],
  )

  const resetToStartingPosition = useCallback(() => {
    startTransition(() => {
      setCurrentNodeId(gameTree.rootId)
    })
  }, [gameTree.rootId])

  const navigateBack = useCallback(() => {
    if (!currentNode.parentId) {
      return
    }

    startTransition(() => {
      setCurrentNodeId(currentNode.parentId as string)
    })
  }, [currentNode.parentId])

  const navigateForward = useCallback(() => {
    if (!mainLineChildId) {
      return
    }

    startTransition(() => {
      setGameTree((previousTree) => ({
        ...previousTree,
        selectedChildIds: {
          ...previousTree.selectedChildIds,
          [currentNodeId]: mainLineChildId,
        },
      }))
      setCurrentNodeId(mainLineChildId)
    })
  }, [currentNodeId, mainLineChildId])

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (
        event.defaultPrevented ||
        event.altKey ||
        event.ctrlKey ||
        event.metaKey ||
        event.shiftKey ||
        isTextEntryTarget(event.target)
      ) {
        return
      }

      if (event.key === 'ArrowLeft' && currentNode.parentId) {
        event.preventDefault()
        navigateBack()
      }

      if (event.key === 'ArrowRight' && mainLineChildId) {
        event.preventDefault()
        navigateForward()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [currentNode.parentId, mainLineChildId, navigateBack, navigateForward])

  const selectVariation = useCallback(
    (childId: string) => {
      startTransition(() => {
        setGameTree((previousTree) => ({
          ...previousTree,
          selectedChildIds: {
            ...previousTree.selectedChildIds,
            [currentNodeId]: childId,
          },
        }))
        setCurrentNodeId(childId)
      })
    },
    [currentNodeId],
  )

  const copyCurrentFen = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(currentFen)
    } catch {
      // Ignore clipboard failures in environments without permission.
    }
  }, [currentFen])

  const recommendation = analysis.status === 'ready' ? analysis.recommendation : null
  const playedMoveCommentary = useMemo<PlayedMoveCommentary | null>(() => {
    if (!currentNode.parentId) {
      return null
    }

    return buildPlayedMoveCommentaryFromAnalysis({
      node: currentNode,
      parentAnalysis: playedMoveAnalysis,
      currentAnalysis: analysis,
    })
  }, [analysis, currentNode, playedMoveAnalysis])
  const currentMoveLlmContext = useMemo<MoveAnalysisContext | null>(() => {
    if (!currentNode.parentId || !playedMoveCommentary) {
      return null
    }

    return {
      rating,
      playerRating:
        currentNode.side ? gameTree.playerRatings[currentNode.side] ?? null : null,
      parentFen: gameTree.nodes[currentNode.parentId].fen,
      currentFen,
      playedMoveSan: playedMoveCommentary.san,
      playedMoveUci: playedMoveCommentary.moveUci,
      playedMoveEval: formatScore(
        playedMoveCommentary.line,
        currentNode.side ?? 'w',
      ),
      cpLoss: playedMoveCommentary.cpLoss,
      humanLikelihood: playedMoveCommentary.maiaProbability,
      bestMoveSan: playedMoveCommentary.bestMove,
      bestMoveUci: playedMoveCommentary.bestMoveUci,
      bestRealisticMoveSan: playedMoveCommentary.bestRealisticMove,
      bestRealisticMoveUci: playedMoveCommentary.bestRealisticMoveUci,
    }
  }, [
    currentFen,
    currentNode.parentId,
    currentNode.side,
    gameTree.nodes,
    gameTree.playerRatings,
    playedMoveCommentary,
    rating,
  ])
  const currentMoveLlmKey = currentMoveLlmContext
    ? `${currentNode.id}|rating:${rating}`
    : null
  const currentMoveLlmMeta = useMemo(
    () =>
      currentMoveLlmContext
        ? {
            nodeId: currentNode.id,
            moveLabel: currentNode.san ?? '...',
          }
        : null,
    [currentMoveLlmContext, currentNode.id, currentNode.san],
  )

  const syncVisibleLlmThread = useCallback(
    (threadKey: string, thread: StoredLlmThread | undefined) => {
      if (currentMoveLlmKeyRef.current !== threadKey) {
        return
      }

      startTransition(() => {
        setLlmAnalysis(toVisibleLlmAnalysisState(thread))
      })
    },
    [],
  )

  const writeLlmThread = useCallback(
    (
      threadKey: string,
      meta: { nodeId: string; moveLabel: string },
      updater: (previous: StoredLlmThread) => StoredLlmThread | null,
    ) => {
      const existing = llmAnalysisCacheRef.current.get(threadKey)
      const previous =
        existing ??
        ({
          ...createEmptyLlmAnalysisState(),
          nodeId: meta.nodeId,
          moveLabel: meta.moveLabel,
        } satisfies StoredLlmThread)
      const next = updater(previous)

      if (next) {
        llmAnalysisCacheRef.current.set(threadKey, next)
      } else {
        llmAnalysisCacheRef.current.delete(threadKey)
      }

      syncVisibleLlmThread(threadKey, next ?? undefined)
    },
    [syncVisibleLlmThread],
  )

  const enqueueLlmNotification = useCallback(
    (
      threadKey: string,
      meta: { nodeId: string; moveLabel: string },
      threadRating: number,
      text: string,
    ) => {
      setLlmNotifications((previous) => [
        {
          id: createNodeId(),
          threadKey,
          nodeId: meta.nodeId,
          moveLabel: meta.moveLabel,
          rating: threadRating,
          preview: buildNotificationPreview(text),
        },
        ...previous.filter((notification) => notification.threadKey !== threadKey),
      ].slice(0, 4))
    },
    [],
  )

  const beginLlmRequest = useCallback((requestToken: string) => {
    activeLlmRequestsRef.current.add(requestToken)
    setActiveLlmRequestCount(activeLlmRequestsRef.current.size)
  }, [])

  const endLlmRequest = useCallback((requestToken: string) => {
    if (!activeLlmRequestsRef.current.delete(requestToken)) {
      return
    }

    setActiveLlmRequestCount(activeLlmRequestsRef.current.size)
  }, [])

  const navigateToNode = useCallback((targetNodeId: string) => {
    startTransition(() => {
      setGameTree((previousTree) => {
        if (!previousTree.nodes[targetNodeId]) {
          return previousTree
        }

        return {
          ...previousTree,
          selectedChildIds: buildSelectedChildIdsToNode(previousTree, targetNodeId),
        }
      })
      setCurrentNodeId(targetNodeId)
    })
  }, [])

  useEffect(() => {
    currentMoveLlmKeyRef.current = currentMoveLlmKey

    if (!currentMoveLlmKey) {
      startTransition(() => {
        setLlmAnalysis(createEmptyLlmAnalysisState())
      })
      return
    }

    const cached = llmAnalysisCacheRef.current.get(currentMoveLlmKey)
    startTransition(() => {
      setLlmAnalysis(toVisibleLlmAnalysisState(cached))
    })
  }, [currentMoveLlmKey])

  useEffect(() => {
    if (!currentMoveLlmKey) {
      return
    }

    setLlmNotifications((previous) =>
      previous.filter((notification) => notification.threadKey !== currentMoveLlmKey),
    )
  }, [currentMoveLlmKey])

  const runLlmTurn = useCallback(
    ({
      prompt,
      includeUserMessage,
    }: {
      prompt: string
      includeUserMessage: boolean
    }) => {
      if (!currentMoveLlmContext || !currentMoveLlmKey || !currentMoveLlmMeta) {
        return
      }

      const threadKey = currentMoveLlmKey
      const threadContext = currentMoveLlmContext
      const threadMeta = currentMoveLlmMeta
      const requestId = (llmRequestIdsRef.current.get(threadKey) ?? 0) + 1
      llmRequestIdsRef.current.set(threadKey, requestId)
      const requestToken = `${threadKey}:${requestId}`
      const priorMessages =
        llmAnalysisCacheRef.current.get(threadKey)?.messages ?? llmAnalysis.messages
      const assistantPlaceholder = createThreadMessage('assistant', '')
      const nextMessages = [...priorMessages]

      if (includeUserMessage) {
        nextMessages.push(createThreadMessage('user', prompt))
      }

      nextMessages.push(assistantPlaceholder)

      beginLlmRequest(requestToken)
      writeLlmThread(threadKey, threadMeta, (previous) => ({
        ...previous,
        status: 'loading',
        messages: nextMessages,
        draft: includeUserMessage ? '' : previous.draft,
        error: null,
        retryPrompt: prompt,
      }))

      const analyzePositionForLlm = async (fen: string) => {
        const request = {
          key: buildAnalysisCacheKey(fen, maiaRating, maiaRating),
          fen,
          whiteRating: maiaRating,
          blackRating: maiaRating,
        }
        const priority =
          fen === threadContext.currentFen
            ? CURRENT_ANALYSIS_PRIORITY
            : fen === threadContext.parentFen
              ? CURRENT_ANALYSIS_PRIORITY - 1
              : CURRENT_ANALYSIS_PRIORITY - 4
        const kind =
          fen === threadContext.currentFen ||
          fen === threadContext.parentFen
            ? 'current'
            : 'background'
        const state = await ensureAnalysis(request, priority, kind)

        return buildLlmAnalysisSnapshot(fen, state)
      }

      void runMoveAnalysisWithLlm({
        context: threadContext,
        transcript: priorMessages.map((message) => ({
          role: message.role,
          text: message.text,
        })),
        turnPrompt: prompt,
        onTextDelta: (delta) => {
          if ((llmRequestIdsRef.current.get(threadKey) ?? 0) !== requestId) {
            return
          }

          writeLlmThread(threadKey, threadMeta, (previous) => ({
            ...previous,
            status: 'loading',
            error: null,
            messages: previous.messages.map((message) =>
              message.id === assistantPlaceholder.id
                ? { ...message, text: message.text + delta }
                : message,
            ),
          }))
        },
        toolset: {
          analyzePosition: analyzePositionForLlm,
          listLegalMoves: async (fen: string) => listLegalMovesSnapshot(fen),
          playMoves: async (fen: string, moves: string[]) =>
            playMovesSnapshot(fen, moves),
        },
      })
        .then((result) => {
          if ((llmRequestIdsRef.current.get(threadKey) ?? 0) !== requestId) {
            return
          }

          const finalizedMessages = nextMessages.map((message) =>
            message.id === assistantPlaceholder.id
              ? {
                  ...message,
                  text: result.text,
                  model: result.model,
                }
              : message,
          )

          writeLlmThread(threadKey, threadMeta, (previous) => ({
            ...previous,
            status: 'ready',
            messages: finalizedMessages,
            error: null,
          }))

          if (currentMoveLlmKeyRef.current !== threadKey) {
            enqueueLlmNotification(threadKey, threadMeta, rating, result.text)
          }
        })
        .catch((error) => {
          if ((llmRequestIdsRef.current.get(threadKey) ?? 0) !== requestId) {
            return
          }

          writeLlmThread(threadKey, threadMeta, (previous) => ({
            ...previous,
            status: 'error',
            messages: previous.messages.filter(
              (message) =>
                message.id !== assistantPlaceholder.id || message.text.trim(),
            ),
            error:
              error instanceof Error
                ? error.message
                : 'LLM move analysis failed unexpectedly.',
          }))
        })
        .finally(() => {
          endLlmRequest(requestToken)
        })
    },
    [
      beginLlmRequest,
      currentMoveLlmContext,
      currentMoveLlmKey,
      currentMoveLlmMeta,
      endLlmRequest,
      enqueueLlmNotification,
      ensureAnalysis,
      llmAnalysis.messages,
      maiaRating,
      rating,
      writeLlmThread,
    ],
  )

  const resetLlmAnalysis = useCallback(() => {
    if (currentMoveLlmKey) {
      llmRequestIdsRef.current.set(
        currentMoveLlmKey,
        (llmRequestIdsRef.current.get(currentMoveLlmKey) ?? 0) + 1,
      )
      llmAnalysisCacheRef.current.delete(currentMoveLlmKey)
      setLlmNotifications((previous) =>
        previous.filter((notification) => notification.threadKey !== currentMoveLlmKey),
      )
    }

    startTransition(() => {
      setLlmAnalysis(createEmptyLlmAnalysisState())
    })
  }, [currentMoveLlmKey])

  const submitLlmReply = useCallback(() => {
    const prompt = llmAnalysis.draft.trim()
    if (!prompt || llmAnalysis.status === 'loading') {
      return
    }

    runLlmTurn({
      prompt,
      includeUserMessage: true,
    })
  }, [llmAnalysis.draft, llmAnalysis.status, runLlmTurn])

  const retryLlmReply = useCallback(() => {
    if (!llmAnalysis.retryPrompt || llmAnalysis.status === 'loading') {
      return
    }

    runLlmTurn({
      prompt: llmAnalysis.retryPrompt,
      includeUserMessage: false,
    })
  }, [llmAnalysis.retryPrompt, llmAnalysis.status, runLlmTurn])

  const openLlmNotification = useCallback(
    (notification: LlmNotification) => {
      setLlmNotifications((previous) =>
        previous.filter((item) => item.id !== notification.id),
      )
      setRating(notification.rating)
      navigateToNode(notification.nodeId)
    },
    [navigateToNode],
  )
  const isBooting =
    !initialBootComplete &&
    (
      maiaState.status !== 'ready' ||
      stockfishState.status !== 'ready' ||
      humanBookState.status !== 'ready'
    )

  if (isBooting) {
    return (
      <div className="loading-shell">
        <section className="panel loading-screen" aria-label="Engine loading screen">
          <div className="loading-copy">
            <p className="panel-eyebrow">Loading</p>
            <h1>Preparing analysis engines</h1>
            <p>
              Stockfish, the human-play model, and the local game file need to
              finish booting before the board is ready.
            </p>
          </div>
          <div className="status-stack">
            <StatusCard label="Human model" state={maiaState} />
            <StatusCard label="Stockfish" state={stockfishState} />
            <StatusCard
              label="Local game file"
              state={humanBookState}
              readyDescription="Local human game data is ready."
            />
          </div>
        </section>
      </div>
    )
  }

  return (
    <div className="app-shell">
      {llmNotifications.length > 0 ? (
        <div className="notification-stack" aria-live="polite">
          {llmNotifications.map((notification) => (
            <button
              key={notification.id}
              type="button"
              className="notification-card"
              onClick={() => {
                openLlmNotification(notification)
              }}
            >
              <span className="notification-kicker">LLM reply ready</span>
              <strong>{notification.moveLabel}</strong>
              <span className="notification-preview">{notification.preview}</span>
            </button>
          ))}
        </div>
      ) : null}

      <main className="workspace">
        <section className="board-column">
          <div className="panel board-panel">
            <div className="board-frame">
              <Chessboard
                options={{
                  id: 'analysis-board',
                  allowDrawingArrows: true,
                  animationDurationInMs: 170,
                  arrows,
                  boardOrientation,
                  onPieceDrop,
                  onSquareClick,
                  position: currentFen,
                  showNotation: true,
                  squareStyles,
                }}
              />
            </div>

            <div className="board-toolbar">
              <button
                className="toolbar-button"
                type="button"
                onClick={navigateBack}
                disabled={!currentNode.parentId}
              >
                Back
              </button>
              <button
                className="toolbar-button"
                type="button"
                onClick={navigateForward}
                disabled={!mainLineChildId}
              >
                Forward
              </button>
              <button
                className="toolbar-button"
                type="button"
                onClick={resetToStartingPosition}
                disabled={currentNodeId === gameTree.rootId}
              >
                Reset to start
              </button>
              <button
                className="ghost-button"
                type="button"
                onClick={() =>
                  setBoardOrientation((value) =>
                    value === 'white' ? 'black' : 'white',
                  )
                }
              >
                Flip board
              </button>
              <button
                className="ghost-button"
                type="button"
                onClick={copyCurrentFen}
              >
                Copy current FEN
              </button>
            </div>
          </div>

          <div className="panel controls-panel">
            <div className="panel-heading">
              <div>
                <p className="panel-eyebrow">Setup</p>
                <h2>Rating, imports, and start position</h2>
              </div>
            </div>

            <div className="controls-grid">
              <label className="field">
                <span>Rating (Chess.com)</span>
                <select
                  value={rating}
                  onChange={(event) => {
                    setRating(Number.parseInt(event.target.value, 10))
                  }}
                >
                  {RATING_OPTIONS.map((option) => (
                    <option key={option.chessCom} value={option.chessCom}>
                      {option.chessCom}
                    </option>
                  ))}
                </select>
              </label>

              <label className="field checkbox-field">
                <span>Green arrow</span>
                <span className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={showGreenArrow}
                    onChange={(event) => {
                      setShowGreenArrow(event.target.checked)
                    }}
                  />
                  <span>Show</span>
                </span>
              </label>

              <label className="field checkbox-field">
                <span>Orange arrow</span>
                <span className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={showOrangeArrow}
                    onChange={(event) => {
                      setShowOrangeArrow(event.target.checked)
                    }}
                  />
                  <span>Show</span>
                </span>
              </label>

            </div>
            <label className="field field-full">
              <span>Chess.com profile</span>
              <input
                type="text"
                value={chessComProfileInput}
                onChange={(event) => {
                  setChessComProfileInput(event.target.value)
                  if (chessComError) {
                    setChessComError(null)
                  }
                }}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && chessComProfileInput.trim()) {
                    event.preventDefault()
                    void fetchChessComRecentGames()
                  }
                }}
                placeholder="Username or https://www.chess.com/member/..."
                spellCheck={false}
              />
            </label>

            <div className="inline-controls">
              <button
                className="toolbar-button"
                type="button"
                onClick={() => {
                  void fetchChessComRecentGames()
                }}
                disabled={chessComLoading || !chessComProfileInput.trim()}
              >
                {chessComLoading ? 'Fetching…' : 'Fetch last 5 games'}
              </button>
            </div>

            {chessComError ? <p className="field-error">{chessComError}</p> : null}

            {chessComGames.length > 0 ? (
              <ul className="chesscom-game-list">
                {chessComGames.map((game) => (
                  <li key={game.id} className="chesscom-game-card">
                    <div className="chesscom-game-copy">
                      <strong>{game.summary}</strong>
                      {game.url ? (
                        <a
                          className="chesscom-game-link"
                          href={game.url}
                          target="_blank"
                          rel="noreferrer"
                        >
                          Open on Chess.com
                        </a>
                      ) : null}
                    </div>
                    <button
                      className="toolbar-button"
                      type="button"
                      onClick={() => {
                        importChessComGame(game)
                      }}
                    >
                      Import
                    </button>
                  </li>
                ))}
              </ul>
            ) : null}

            <label className="field field-full">
              <span>Import PGN (main line)</span>
              <textarea
                rows={7}
                value={pgnDraft}
                onChange={(event) => {
                  setPgnDraft(event.target.value)
                  if (pgnError) {
                    setPgnError(null)
                  }
                }}
                placeholder='Paste a PGN here, then import it as the current game tree.'
                spellCheck={false}
              />
            </label>

            <div className="inline-controls">
              <button className="toolbar-button emphasis" type="button" onClick={importPgn}>
                Import PGN as tree
              </button>
              <button
                className="toolbar-button"
                type="button"
                onClick={() => {
                  setPgnDraft('')
                  setPgnError(null)
                }}
              >
                Clear PGN
              </button>
            </div>

            {pgnError ? <p className="field-error">{pgnError}</p> : null}

            <label className="field field-full">
              <span>Alternate starting FEN</span>
              <textarea
                rows={3}
                value={fenDraft}
                onChange={(event) => {
                  setFenDraft(event.target.value)
                  if (fenError) {
                    setFenError(null)
                  }
                }}
                spellCheck={false}
              />
            </label>

            <div className="inline-controls">
              <button className="toolbar-button emphasis" type="button" onClick={loadStartingFen}>
                Load FEN as new start
              </button>
              <button
                className="toolbar-button"
                type="button"
                onClick={() => {
                  setFenDraft(DEFAULT_START_FEN)
                  setFenError(null)
                }}
              >
                Standard start
              </button>
            </div>

            {fenError ? <p className="field-error">{fenError}</p> : null}
          </div>
        </section>

        <section className="insight-column">
          <div className="panel insight-panel">
            {currentNode.parentId && playedMoveAnalysis?.status === 'loading' ? (
              <article className="recommendation-card played">
                <div className="card-kicker">Played move</div>
                <div className="played-move-header">
                  <strong className="played-move-title">
                    <AnnotatedMoveText san={currentNode.san ?? '...'} annotation="" />
                  </strong>
                </div>
                <div className="played-move-summary">
                  <p className="recommendation-note">
                    Evaluating the played move.
                  </p>
                </div>
              </article>
            ) : null}

            {currentNode.parentId && playedMoveAnalysis?.status === 'error' ? (
              <div className="empty-state error">
                <p className="empty-title">Played move commentary failed</p>
                <p>{playedMoveAnalysis.error}</p>
              </div>
            ) : null}

            {playedMoveCommentary ? (
              <article className="recommendation-card played">
                <div className="card-kicker">Played move</div>
                <div className="played-move-header">
                  <strong className="played-move-title">
                    <AnnotatedMoveText
                      san={playedMoveCommentary.san}
                      annotation={playedMoveCommentary.annotation}
                    />
                  </strong>
                </div>
                <div className="played-move-summary">
                  <p className="recommendation-note">
                    {playedMoveCommentary.explanation}
                  </p>
                </div>
                <dl className="metric-grid">
                  <div>
                    <dt>Human likelihood</dt>
                    <dd>{formatPercent(playedMoveCommentary.maiaProbability)}</dd>
                  </div>
                  <div>
                    <dt>Eval</dt>
                    <dd>
                      {formatScore(
                        playedMoveCommentary.line,
                        currentNode.side ?? 'w',
                      )}
                    </dd>
                  </div>
                  <div>
                    <dt>CP loss</dt>
                    <dd>{playedMoveCommentary.cpLoss} cp</dd>
                  </div>
                </dl>
                <dl className="metric-grid metric-grid-secondary">
                  <div>
                    <dt>Best move</dt>
                    <dd>{playedMoveCommentary.bestMove ?? '-'}</dd>
                  </div>
                  <div>
                    <dt>Best realistic move</dt>
                    <dd>{playedMoveCommentary.bestRealisticMove ?? '-'}</dd>
                  </div>
                </dl>
                {llmAnalysis.messages.length > 0 ? (
                  <div className="inline-controls llm-controls">
                    <button
                      className="ghost-button"
                      type="button"
                      onClick={resetLlmAnalysis}
                      disabled={llmAnalysis.status === 'loading'}
                    >
                      New chat
                    </button>
                  </div>
                ) : null}
              </article>
            ) : null}

            {playedMoveCommentary ? (
              <article className="mini-panel llm-panel">
                <div className="panel-heading llm-panel-heading">
                  <div>
                    <h3>LLM chat</h3>
                  </div>
                </div>
                {llmAnalysis.messages.length > 0 ? (
                  <div className="llm-thread">
                    {llmAnalysis.messages.map((message) => (
                      <article
                        key={message.id}
                        className={`llm-message ${message.role === 'user' ? 'user' : 'assistant'}`}
                      >
                        <div className="llm-message-label">
                          {message.role === 'user' ? 'You' : 'Coach'}
                        </div>
                        <div className="llm-copy">
                          <p>{message.text || (llmAnalysis.status === 'loading' ? 'Thinking...' : '')}</p>
                        </div>
                      </article>
                    ))}
                  </div>
                ) : null}
                {llmAnalysis.error ? (
                  <div className="empty-state error compact llm-chat-error">
                    <p>{llmAnalysis.error}</p>
                    {llmAnalysis.retryPrompt ? (
                      <div className="inline-controls">
                        <button
                          className="toolbar-button"
                          type="button"
                          onClick={retryLlmReply}
                          disabled={llmAnalysis.status === 'loading'}
                        >
                          Retry
                        </button>
                      </div>
                    ) : null}
                  </div>
                ) : null}
                <div className="llm-composer">
                  <textarea
                    rows={3}
                    value={llmAnalysis.draft}
                    onChange={(event) => {
                      const value = event.target.value
                      if (currentMoveLlmKey && currentMoveLlmMeta) {
                        writeLlmThread(currentMoveLlmKey, currentMoveLlmMeta, (previous) => ({
                          ...previous,
                          draft: value,
                          error: null,
                        }))
                      }
                    }}
                    onKeyDown={(event) => {
                      if (
                        event.key === 'Enter' &&
                        !event.shiftKey &&
                        !event.nativeEvent.isComposing
                      ) {
                        event.preventDefault()
                        submitLlmReply()
                      }
                    }}
                    placeholder="Ask a follow-up about this move."
                    disabled={llmAnalysis.status === 'loading'}
                  />
                  <div className="inline-controls">
                    <button
                      className="toolbar-button"
                      type="button"
                      onClick={submitLlmReply}
                      disabled={
                        llmAnalysis.status === 'loading' || !llmAnalysis.draft.trim()
                      }
                    >
                      Send
                    </button>
                  </div>
                </div>
              </article>
            ) : null}

            {analysis.status === 'loading' ? (
              <div className="empty-state">
                <p className="empty-title">Updating analysis…</p>
                <p>Recomputing this position.</p>
              </div>
            ) : null}

            {analysis.status === 'error' ? (
              <div className="empty-state error">
                <p className="empty-title">Analysis failed</p>
                <p>{analysis.error}</p>
              </div>
            ) : null}

            {analysis.status === 'game-over' ? (
              <div className="empty-state">
                <p className="empty-title">No move to recommend</p>
                <p>The position is finished.</p>
              </div>
            ) : null}

            {analysis.status === 'ready' && recommendation ? (
              <article className="mini-panel">
                <h3>Engine lines</h3>
                <ul className="line-list">
                  {recommendation.stockfishCandidates.slice(0, 3).map((line) => (
                    <li key={`stockfish-${line.move}`}>
                      <div className="line-static">
                        <span className="line-head">
                          <strong>{line.san}</strong>
                        </span>
                        <span className="line-meta">
                          {formatScore(line, currentTurn)} | Human{' '}
                          {formatPercent(line.maiaProbability)}
                        </span>
                      </div>
                    </li>
                  ))}
                </ul>
              </article>
            ) : null}
          </div>

          <div className="panel move-panel">
            <div className="panel-heading">
              <div>
                <p className="panel-eyebrow">History</p>
                <h2>Selected line and branches</h2>
              </div>
            </div>

            <div className="variation-panel">
              <h3>Continuations from this position</h3>
              {currentChildren.length === 0 ? (
                <p className="variation-empty">
                  No saved continuations yet. Step forward through an imported
                  PGN or make a move here to create a branch.
                </p>
              ) : (
                <div className="variation-grid">
                  {currentChildren.map((childNode, index) => (
                    <button
                      key={childNode.id}
                      type="button"
                      className={`variation-chip ${getMoveAnnotationVisualClass(
                        moveAnnotationsByNodeId.get(childNode.id) ?? '',
                      )} ${selectedChildId === childNode.id ? 'active' : ''}`}
                      onClick={() => {
                        selectVariation(childNode.id)
                      }}
                    >
                      <span>{index === 0 ? 'Main' : `Var ${index}`}</span>
                      <strong className="variation-move-label">
                        <AnnotatedMoveText
                          san={childNode.san ?? '--'}
                          annotation={moveAnnotationsByNodeId.get(childNode.id) ?? ''}
                        />
                      </strong>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {moveRows.length === 0 ? (
              <div className="empty-state compact">
                <p>No moves yet. Play from the board or import a PGN to begin.</p>
              </div>
            ) : (
              <div className="move-table">
                {moveRows.map((row) => (
                  <div
                    key={`move-row-${row.moveNumber}-${row.white?.nodeId ?? 'w'}-${row.black?.nodeId ?? 'b'}`}
                    className="move-row"
                  >
                    <span className="move-number">{row.moveNumber}.</span>
                    <button
                      type="button"
                      className={`move-chip ${getMoveAnnotationVisualClass(
                        row.white?.annotation ?? '',
                      )} ${currentNodeId === row.white?.nodeId ? 'active' : ''}`}
                      onClick={() => {
                        if (row.white) {
                          setCurrentNodeId(row.white.nodeId)
                        }
                      }}
                      disabled={!row.white}
                    >
                      {row.white ? (
                        <AnnotatedMoveText
                          san={row.white.san}
                          annotation={row.white.annotation}
                        />
                      ) : (
                        '...'
                      )}
                    </button>
                    <button
                      type="button"
                      className={`move-chip ${getMoveAnnotationVisualClass(
                        row.black?.annotation ?? '',
                      )} ${currentNodeId === row.black?.nodeId ? 'active' : ''}`}
                      onClick={() => {
                        if (row.black) {
                          setCurrentNodeId(row.black.nodeId)
                        }
                      }}
                      disabled={!row.black}
                    >
                      {row.black ? (
                        <AnnotatedMoveText
                          san={row.black.san}
                          annotation={row.black.annotation}
                        />
                      ) : (
                        '...'
                      )}
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>
      </main>

      {promotionChoice ? (
        <div className="promotion-backdrop" role="presentation">
          <div className="promotion-modal" role="dialog" aria-modal="true">
            <p className="panel-eyebrow">Promotion</p>
            <h2>Choose the promotion piece</h2>
            <div className="promotion-options">
              {(['q', 'r', 'b', 'n'] as const).map((piece) => (
                <button
                  key={piece}
                  type="button"
                  className="promotion-button"
                  onClick={() => {
                    commitMove(promotionChoice.from, promotionChoice.to, piece)
                    setPromotionChoice(null)
                  }}
                >
                  {promotionChoice.color === 'w' ? piece.toUpperCase() : piece}
                </button>
              ))}
            </div>
            <button
              type="button"
              className="ghost-button"
              onClick={() => {
                setPromotionChoice(null)
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      ) : null}
    </div>
  )
}

function StatusCard({
  label,
  state,
  readyDescription = 'Local browser engine is live.',
}: {
  label: string
  state: EngineLoadState
  readyDescription?: string
}) {
  const progress = Math.max(0, Math.min(100, Math.round(state.progress ?? 0)))

  return (
    <div className="status-card">
      <span className="status-label">{label}</span>
      <strong>{state.status === 'ready' ? 'Ready' : state.detail}</strong>
      <span className="status-detail">
        {state.status === 'loading' && state.progress !== undefined
          ? `${progress}% complete`
          : state.status === 'error'
            ? state.error
            : state.status === 'ready'
              ? readyDescription
              : 'Preparing browser runtime…'}
      </span>
      {state.status === 'loading' && state.progress !== undefined ? (
        <div
          className="status-progress"
          aria-hidden="true"
        >
          <span style={{ width: `${progress}%` }} />
        </div>
      ) : null}
    </div>
  )
}

export default App
