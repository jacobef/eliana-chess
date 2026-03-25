import { stat } from 'node:fs/promises'

import {
  buildAndPersistHumanBookArtifact,
  HUMAN_BOOK_ARTIFACT_RELATIVE_PATH,
} from '../server/lichessHumanBook'

const SOURCE_FILENAME = 'lichess_db_standard_rated_2013-01.pgn.zst'

async function main() {
  const force = process.argv.includes('--force')

  if (!force && (await isArtifactCurrent())) {
    console.log(`Human-book artifact is already current: ${HUMAN_BOOK_ARTIFACT_RELATIVE_PATH}`)
    return
  }

  console.log(`Building human-book artifact from ${SOURCE_FILENAME}...`)

  const result = await buildAndPersistHumanBookArtifact((state) => {
    if (state.status !== 'loading') {
      return
    }

    const progressSuffix =
      state.progress !== undefined ? ` (${Math.round(state.progress)}%)` : ''
    console.log(`${state.detail}${progressSuffix}`)
  })

  console.log(
    `Saved ${result.positionCount.toLocaleString()} indexed positions to ${HUMAN_BOOK_ARTIFACT_RELATIVE_PATH}`,
  )
}

async function isArtifactCurrent() {
  try {
    const [sourceStat, artifactStat] = await Promise.all([
      stat(SOURCE_FILENAME),
      stat(HUMAN_BOOK_ARTIFACT_RELATIVE_PATH),
    ])
    return artifactStat.mtimeMs >= sourceStat.mtimeMs
  } catch {
    return false
  }
}

main().catch((error) => {
  console.error(
    error instanceof Error ? error.message : 'Failed to build the human-book artifact.',
  )
  process.exitCode = 1
})
