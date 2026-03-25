import type { IncomingMessage, ServerResponse } from 'node:http'

import { handleHumanStats } from '../server/humanStatsProxy'

export default async function handler(req: IncomingMessage, res: ServerResponse) {
  const handled = await handleHumanStats(req, res)
  if (!handled && !res.writableEnded) {
    res.statusCode = 404
    res.end('Not found')
  }
}
