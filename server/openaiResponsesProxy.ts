import type { IncomingMessage, ServerResponse } from 'node:http'
import { Readable } from 'node:stream'

export const OPENAI_RESPONSES_PATH = '/api/openai/responses'

export async function handleOpenAIResponses(
  req: IncomingMessage,
  res: ServerResponse,
) {
  const pathname = new URL(req.url ?? OPENAI_RESPONSES_PATH, 'http://127.0.0.1').pathname
  if (pathname !== OPENAI_RESPONSES_PATH) {
    return false
  }

  if (req.method !== 'POST') {
    res.statusCode = 405
    res.setHeader('Allow', 'POST')
    res.setHeader('Content-Type', 'application/json')
    res.end(JSON.stringify({ error: { message: 'Method not allowed.' } }))
    return true
  }

  const apiKey = process.env.OPENAI_API_KEY
  if (!apiKey) {
    res.statusCode = 500
    res.setHeader('Content-Type', 'application/json')
    res.end(
      JSON.stringify({
        error: {
          message: 'OPENAI_API_KEY is not set in the server environment.',
        },
      }),
    )
    return true
  }

  let body: string

  try {
    body = await readRequestBody(req)
  } catch (error) {
    res.statusCode = 400
    res.setHeader('Content-Type', 'application/json')
    res.end(
      JSON.stringify({
        error: {
          message:
            error instanceof Error ? error.message : 'Could not read request body.',
        },
      }),
    )
    return true
  }

  try {
    const upstream = await fetch('https://api.openai.com/v1/responses', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body,
    })

    res.statusCode = upstream.status
    const contentType = upstream.headers.get('content-type') ?? 'application/json'
    res.setHeader('Content-Type', contentType)
    res.setHeader('Cache-Control', 'no-store')

    if (contentType.includes('text/event-stream') && upstream.body) {
      res.setHeader('Connection', 'keep-alive')
      const stream = Readable.fromWeb(upstream.body as globalThis.ReadableStream)
      stream.on('error', () => {
        if (!res.writableEnded) {
          res.end()
        }
      })
      stream.pipe(res)
      return true
    }

    const responseText = await upstream.text()
    res.end(responseText)
    return true
  } catch (error) {
    res.statusCode = 502
    res.setHeader('Content-Type', 'application/json')
    res.end(
      JSON.stringify({
        error: {
          message:
            error instanceof Error
              ? error.message
              : 'Failed to reach the OpenAI API.',
        },
      }),
    )
    return true
  }
}

function readRequestBody(req: IncomingMessage) {
  return new Promise<string>((resolve, reject) => {
    let body = ''

    req.setEncoding('utf8')
    req.on('data', (chunk: string) => {
      body += chunk

      if (body.length > 1_000_000) {
        reject(new Error('Request body was too large.'))
      }
    })
    req.on('end', () => {
      resolve(body)
    })
    req.on('error', reject)
  })
}
