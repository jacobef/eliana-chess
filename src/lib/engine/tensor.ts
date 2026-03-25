import { Chess } from 'chess.js'
import type { Move } from 'chess.js'

import allPossibleMovesDict from './data/all_moves.json'
import allPossibleMovesReversedDict from './data/all_moves_reversed.json'

const allPossibleMoves = allPossibleMovesDict as Record<string, number>
const allPossibleMovesReversed = allPossibleMovesReversedDict as Record<
  string,
  string
>
const eloDict = createEloDict()

function boardToTensor(fen: string) {
  const tokens = fen.split(' ')
  const piecePlacement = tokens[0]
  const activeColor = tokens[1]
  const castlingAvailability = tokens[2]
  const enPassantTarget = tokens[3]

  const pieceTypes = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
  const tensor = new Float32Array(18 * 8 * 8)
  const rows = piecePlacement.split('/')

  for (let rank = 0; rank < 8; rank += 1) {
    const row = 7 - rank
    let file = 0

    for (const char of rows[rank]) {
      if (Number.isNaN(Number.parseInt(char, 10))) {
        const index = pieceTypes.indexOf(char)
        tensor[index * 64 + row * 8 + file] = 1
        file += 1
      } else {
        file += Number.parseInt(char, 10)
      }
    }
  }

  const turnChannelStart = 12 * 64
  tensor.fill(activeColor === 'w' ? 1 : 0, turnChannelStart, turnChannelStart + 64)

  const castlingRights = [
    castlingAvailability.includes('K'),
    castlingAvailability.includes('Q'),
    castlingAvailability.includes('k'),
    castlingAvailability.includes('q'),
  ]

  for (let index = 0; index < castlingRights.length; index += 1) {
    if (!castlingRights[index]) {
      continue
    }

    const channelStart = (13 + index) * 64
    tensor.fill(1, channelStart, channelStart + 64)
  }

  if (enPassantTarget !== '-') {
    const file = enPassantTarget.charCodeAt(0) - 'a'.charCodeAt(0)
    const rank = Number.parseInt(enPassantTarget[1], 10) - 1
    tensor[17 * 64 + rank * 8 + file] = 1
  }

  return tensor
}

export function preprocess(fen: string, eloSelf: number, eloOppo: number) {
  let board = new Chess(fen)

  if (board.turn() === 'b') {
    board = new Chess(mirrorFen(board.fen()))
  }

  const boardInput = boardToTensor(board.fen())
  const eloSelfCategory = mapToCategory(eloSelf, eloDict)
  const eloOppoCategory = mapToCategory(eloOppo, eloDict)
  const legalMoves = new Float32Array(Object.keys(allPossibleMoves).length)

  for (const move of board.moves({ verbose: true }) as Move[]) {
    const promotion = move.promotion ?? ''
    const moveIndex = allPossibleMoves[`${move.from}${move.to}${promotion}`]

    if (moveIndex !== undefined) {
      legalMoves[moveIndex] = 1
    }
  }

  return {
    boardInput,
    eloSelfCategory,
    eloOppoCategory,
    legalMoves,
  }
}

function mapToCategory(elo: number, categories: Record<string, number>) {
  const interval = 100
  const start = 1100
  const end = 2000

  if (elo < start) {
    return categories[`<${start}`]
  }

  if (elo >= end) {
    return categories[`>=${end}`]
  }

  for (let lowerBound = start; lowerBound < end; lowerBound += interval) {
    const upperBound = lowerBound + interval
    if (elo >= lowerBound && elo < upperBound) {
      return categories[`${lowerBound}-${upperBound - 1}`]
    }
  }

  throw new Error('Elo value is out of range.')
}

function createEloDict() {
  const interval = 100
  const start = 1100
  const end = 2000
  const categories: Record<string, number> = { [`<${start}`]: 0 }
  let rangeIndex = 1

  for (let lowerBound = start; lowerBound < end; lowerBound += interval) {
    const upperBound = lowerBound + interval
    categories[`${lowerBound}-${upperBound - 1}`] = rangeIndex
    rangeIndex += 1
  }

  categories[`>=${end}`] = rangeIndex

  return categories
}

function mirrorSquare(square: string) {
  const file = square[0]
  const rank = String(9 - Number.parseInt(square[1], 10))
  return `${file}${rank}`
}

export function mirrorMove(moveUci: string) {
  const promotion = moveUci.length > 4 ? moveUci.slice(4) : ''
  const start = mirrorSquare(moveUci.slice(0, 2))
  const end = mirrorSquare(moveUci.slice(2, 4))
  return `${start}${end}${promotion}`
}

function swapColorsInRank(rank: string) {
  let swappedRank = ''

  for (const char of rank) {
    if (/[A-Z]/.test(char)) {
      swappedRank += char.toLowerCase()
    } else if (/[a-z]/.test(char)) {
      swappedRank += char.toUpperCase()
    } else {
      swappedRank += char
    }
  }

  return swappedRank
}

function swapCastlingRights(castling: string) {
  if (castling === '-') {
    return '-'
  }

  const rights = new Set(castling.split(''))
  const swapped = new Set<string>()

  if (rights.has('K')) swapped.add('k')
  if (rights.has('Q')) swapped.add('q')
  if (rights.has('k')) swapped.add('K')
  if (rights.has('q')) swapped.add('Q')

  return ['K', 'Q', 'k', 'q'].filter((right) => swapped.has(right)).join('') || '-'
}

function mirrorFen(fen: string) {
  const [placement, activeColor, castling, enPassant, halfmove, fullmove] =
    fen.split(' ')
  const mirroredPlacement = placement
    .split('/')
    .reverse()
    .map((rank) => swapColorsInRank(rank))
    .join('/')

  const mirroredEnPassant = enPassant === '-' ? '-' : mirrorSquare(enPassant)
  const mirroredActiveColor = activeColor === 'w' ? 'b' : 'w'

  return [
    mirroredPlacement,
    mirroredActiveColor,
    swapCastlingRights(castling),
    mirroredEnPassant,
    halfmove,
    fullmove,
  ].join(' ')
}

export { allPossibleMovesReversed }
