import type { IncomingMessage, ServerResponse } from 'node:http'

import { handleOpenAIResponses } from '../../server/openaiResponsesProxy.js'

export default async function handler(req: IncomingMessage, res: ServerResponse) {
  const handled = await handleOpenAIResponses(req, res)
  if (!handled && !res.writableEnded) {
    res.statusCode = 404
    res.end('Not found')
  }
}
