const DEFAULT_MODEL = 'gpt-5.4'
const FALLBACK_MODELS = ['gpt-5.4', 'gpt-5.4-mini', 'gpt-5-mini']
const MAX_TOOL_ROUNDS = 24
const FINAL_OUTPUT_TOKENS = 900

type JsonRecord = Record<string, unknown>

type ResponseFunctionCall = {
  type: 'function_call'
  name: string
  call_id: string
  arguments: string
}

type ResponseMessageContent =
  | {
      type: 'output_text' | 'text'
      text?:
        | string
        | {
            value?: string
          }
    }
  | {
      type: string
      text?: string | { value?: string }
    }

type ResponseMessage = {
  type: 'message'
  content?: ResponseMessageContent[]
}

type ResponsesApiResponse = {
  id?: string
  status?: string
  error?: {
    message?: string
  } | null
  incomplete_details?: {
    reason?: string
  } | null
  output?: Array<ResponseFunctionCall | ResponseMessage | JsonRecord>
  output_text?: string
}

export type MoveAnalysisContext = {
  rating: number
  playerRating: number | null
  parentFen: string
  currentFen: string
  playedMoveSan: string
  playedMoveUci: string
  playedMoveEval: string
  cpLoss: number
  humanLikelihood: number
  bestMoveSan: string | null
  bestMoveUci: string | null
  bestRealisticMoveSan: string | null
  bestRealisticMoveUci: string | null
}

export type MoveAnalysisToolset = {
  analyzePosition: (fen: string) => Promise<unknown>
  listLegalMoves: (fen: string) => Promise<unknown>
  playMoves: (fen: string, moves: string[]) => Promise<unknown>
}

export type MoveChatMessage = {
  role: 'user' | 'assistant'
  text: string
}

export type MoveAnalysisResult = {
  model: string
  text: string
}

export async function runMoveAnalysisWithLlm({
  context,
  transcript,
  turnPrompt,
  toolset,
  model = DEFAULT_MODEL,
  onTextDelta,
}: {
  context: MoveAnalysisContext
  transcript: MoveChatMessage[]
  turnPrompt: string
  toolset: MoveAnalysisToolset
  model?: string
  onTextDelta?: (delta: string) => void
}): Promise<MoveAnalysisResult> {
  const responsesTools = [
    {
      type: 'function',
      name: 'analyze_position',
      description:
        'Analyze a chess position with the local engine stack. Use this to inspect the current position or compare branches.',
      parameters: {
        type: 'object',
        additionalProperties: false,
        properties: {
          fen: {
            type: 'string',
            description: 'The FEN of the position to analyze.',
          },
        },
        required: ['fen'],
      },
    },
    {
      type: 'function',
      name: 'list_legal_moves',
      description:
        'List legal moves from a position in both SAN and UCI so you can explore branches.',
      parameters: {
        type: 'object',
        additionalProperties: false,
        properties: {
          fen: {
            type: 'string',
            description: 'The FEN of the position to inspect.',
          },
        },
        required: ['fen'],
      },
    },
    {
      type: 'function',
      name: 'play_moves',
      description:
        'Play one or more UCI moves from a given FEN and return the resulting position.',
      parameters: {
        type: 'object',
        additionalProperties: false,
        properties: {
          fen: {
            type: 'string',
            description: 'The starting FEN.',
          },
          moves: {
            type: 'array',
            items: {
              type: 'string',
            },
            description: 'A sequence of UCI moves to play in order.',
          },
        },
        required: ['fen', 'moves'],
      },
    },
  ]

  const conversationHistory = buildConversationHistory(
    context,
    transcript,
    turnPrompt,
  )

  let streamedText = ''
  let continuationCount = 0
  const appendTextDelta = (delta: string) => {
    if (!delta) {
      return
    }

    streamedText += delta
    onTextDelta?.(delta)
  }

  let activeModel = model
  let response = await createResponse(
    {
      instructions: buildSystemPrompt(),
      input: conversationHistory,
      max_output_tokens: FINAL_OUTPUT_TOKENS,
      reasoning: {
        effort: 'medium',
      },
      store: false,
      tools: responsesTools,
    },
    activeModel,
    appendTextDelta,
  )
  activeModel = response.model

  for (let round = 0; round < MAX_TOOL_ROUNDS; round += 1) {
    const functionCalls = getFunctionCalls(response.payload)

    if (functionCalls.length === 0) {
      if (
        continuationCount === 0 &&
        response.payload.status === 'incomplete' &&
        response.payload.incomplete_details?.reason === 'max_output_tokens'
      ) {
        continuationCount += 1
        conversationHistory.push(...sanitizeOutputItems(response.payload.output))
        conversationHistory.push({
          role: 'user',
          content:
            'Finish your answer briefly without restarting or repeating yourself.',
        })

        response = await createResponse(
          {
            instructions: buildSystemPrompt(),
            input: conversationHistory,
            max_output_tokens: FINAL_OUTPUT_TOKENS,
            reasoning: {
              effort: 'medium',
            },
            store: false,
            tools: responsesTools,
          },
          activeModel,
          appendTextDelta,
        )
        activeModel = response.model
        continue
      }

      const text = streamedText.trim() || extractResponseText(response.payload)
      if (!text) {
        throw new Error('The model returned no explanation.')
      }

      return {
        model: activeModel,
        text,
      }
    }
    conversationHistory.push(...sanitizeOutputItems(response.payload.output))

    const toolOutputs = await Promise.all(
      functionCalls.map(async (call) => {
        const output = await runToolCall(call, toolset)
        return {
          type: 'function_call_output',
          call_id: call.call_id,
          output: typeof output === 'string' ? output : JSON.stringify(output),
        }
      }),
    )
    conversationHistory.push(...toolOutputs)

    response = await createResponse(
      {
        instructions: buildSystemPrompt(),
        input: conversationHistory,
        max_output_tokens: FINAL_OUTPUT_TOKENS,
        reasoning: {
          effort: 'medium',
        },
      store: false,
      tools: responsesTools,
    },
    activeModel,
    appendTextDelta,
  )
    activeModel = response.model
  }

  throw new Error('The model kept calling tools without finishing.')
}

async function runToolCall(
  call: ResponseFunctionCall,
  toolset: MoveAnalysisToolset,
) {
  const args = parseArguments(call.arguments)

  switch (call.name) {
    case 'analyze_position':
      return await toolset.analyzePosition(readString(args, 'fen'))
    case 'list_legal_moves':
      return await toolset.listLegalMoves(readString(args, 'fen'))
    case 'play_moves':
      return await toolset.playMoves(
        readString(args, 'fen'),
        readStringArray(args, 'moves'),
      )
    default:
      throw new Error(`Unsupported tool call: ${call.name}`)
  }
}

async function createResponse(
  body: JsonRecord,
  model: string,
  onTextDelta?: (delta: string) => void,
) {
  let lastError: Error | null = null

  for (const candidateModel of resolveModelCandidates(model)) {
    const response = await fetch('/api/openai/responses', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        ...body,
        model: candidateModel,
        stream: true,
      }),
    })

    if (response.ok) {
      const payload = await readStreamedResponse(response, onTextDelta)
      return {
        model: candidateModel,
        payload,
      }
    }

    const message = await getErrorMessage(response)
    const error = new Error(message)

    if (shouldRetryWithFallback(candidateModel, message)) {
      lastError = error
      continue
    }

    throw error
  }

  throw lastError ?? new Error('The OpenAI request failed.')
}

async function readStreamedResponse(
  response: Response,
  onTextDelta?: (delta: string) => void,
) {
  if (!response.body) {
    return (await response.json()) as ResponsesApiResponse
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let finalPayload: ResponsesApiResponse | null = null

  while (true) {
    const { value, done } = await reader.read()
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done })

    while (true) {
      const separatorIndex = buffer.search(/\r?\n\r?\n/)
      if (separatorIndex === -1) {
        break
      }

      const separatorLength = buffer[separatorIndex] === '\r' ? 4 : 2
      const rawEvent = buffer.slice(0, separatorIndex)
      buffer = buffer.slice(separatorIndex + separatorLength)

      const parsedEvent = parseSseEvent(rawEvent)
      if (!parsedEvent || parsedEvent.data === '[DONE]') {
        continue
      }

      let eventPayload: unknown

      try {
        eventPayload = JSON.parse(parsedEvent.data)
      } catch {
        continue
      }

      if (!isJsonRecord(eventPayload)) {
        continue
      }

      const eventType =
        typeof eventPayload.type === 'string' ? eventPayload.type : parsedEvent.event

      if (eventType === 'response.output_text.delta') {
        const delta = eventPayload.delta
        if (typeof delta === 'string') {
          onTextDelta?.(delta)
        }
        continue
      }

      if (eventType === 'error') {
        throw new Error(getStreamEventMessage(eventPayload))
      }

      if (
        eventType === 'response.completed' ||
        eventType === 'response.incomplete' ||
        eventType === 'response.failed'
      ) {
        finalPayload = extractResponsePayload(eventPayload)

        if (eventType === 'response.failed') {
          throw new Error(
            finalPayload?.error?.message ??
              getStreamEventMessage(eventPayload) ??
              'The streamed response failed.',
          )
        }
      }
    }

    if (done) {
      break
    }
  }

  if (finalPayload) {
    return finalPayload
  }

  throw new Error('The streamed response ended without a final payload.')
}

function parseSseEvent(rawEvent: string) {
  const lines = rawEvent.replace(/\r/g, '').split('\n')
  let event = ''
  const dataLines: string[] = []

  for (const line of lines) {
    if (!line || line.startsWith(':')) {
      continue
    }

    if (line.startsWith('event:')) {
      event = line.slice(6).trim()
      continue
    }

    if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trimStart())
    }
  }

  if (!event && dataLines.length === 0) {
    return null
  }

  return {
    event,
    data: dataLines.join('\n'),
  }
}

function extractResponsePayload(eventPayload: JsonRecord) {
  const response = eventPayload.response
  return isJsonRecord(response) ? (response as ResponsesApiResponse) : null
}

function getStreamEventMessage(eventPayload: JsonRecord) {
  const error = eventPayload.error
  if (isJsonRecord(error) && typeof error.message === 'string') {
    return error.message
  }

  const message = eventPayload.message
  return typeof message === 'string' ? message : 'The stream returned an error.'
}

function isJsonRecord(value: unknown): value is JsonRecord {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function resolveModelCandidates(model: string) {
  if (model === DEFAULT_MODEL) {
    return FALLBACK_MODELS
  }

  return [model]
}

function shouldRetryWithFallback(model: string, message: string) {
  return (
    model === DEFAULT_MODEL &&
    /model|unknown|not found|does not exist|unavailable/i.test(message)
  )
}

async function getErrorMessage(response: Response) {
  try {
    const payload = (await response.json()) as {
      error?: {
        message?: string
      }
      message?: string
    }

    return (
      payload.error?.message ??
      payload.message ??
      `OpenAI request failed with status ${response.status}.`
    )
  } catch {
    return `OpenAI request failed with status ${response.status}.`
  }
}

function getFunctionCalls(payload: ResponsesApiResponse) {
  return (payload.output ?? []).filter(
    (item): item is ResponseFunctionCall =>
      item.type === 'function_call' &&
      typeof item.name === 'string' &&
      typeof item.call_id === 'string' &&
      typeof item.arguments === 'string',
  )
}

function extractResponseText(payload: ResponsesApiResponse) {
  if (typeof payload.output_text === 'string' && payload.output_text.trim()) {
    return payload.output_text.trim()
  }

  const parts: string[] = []

  for (const item of payload.output ?? []) {
    if (item.type !== 'message' || !Array.isArray(item.content)) {
      continue
    }

    for (const content of item.content) {
      if (content.type !== 'output_text' && content.type !== 'text') {
        continue
      }

      const text =
        typeof content.text === 'string'
          ? content.text
          : content.text?.value ?? ''

      if (text.trim()) {
        parts.push(text.trim())
      }
    }
  }

  return parts.join('\n\n').trim()
}

function sanitizeOutputItems(output: ResponsesApiResponse['output']) {
  return (output ?? [])
    .map((item) => sanitizeJsonValue(item))
    .filter(
      (item): item is JsonRecord =>
        Boolean(item) && typeof item === 'object' && !Array.isArray(item),
    )
}

function sanitizeJsonValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map((item) => sanitizeJsonValue(item))
  }

  if (!value || typeof value !== 'object') {
    return value
  }

  const cleanedEntries = Object.entries(value as JsonRecord)
    .filter(([key]) => key !== 'id' && key !== 'status')
    .map(([key, entryValue]) => [key, sanitizeJsonValue(entryValue)] as const)

  return Object.fromEntries(cleanedEntries)
}

function parseArguments(rawArguments: string) {
  try {
    const value = JSON.parse(rawArguments) as unknown
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      throw new Error('Tool arguments must be an object.')
    }

    return value as JsonRecord
  } catch (error) {
    throw new Error(
      error instanceof Error
        ? `Invalid tool arguments: ${error.message}`
        : 'Invalid tool arguments.',
    )
  }
}

function readString(record: JsonRecord, key: string) {
  const value = record[key]
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`Tool argument "${key}" must be a non-empty string.`)
  }

  return value.trim()
}

function readStringArray(record: JsonRecord, key: string) {
  const value = record[key]
  if (!Array.isArray(value) || value.some((item) => typeof item !== 'string')) {
    throw new Error(`Tool argument "${key}" must be an array of strings.`)
  }

  return value.map((item) => item.trim()).filter(Boolean)
}

function buildSystemPrompt() {
  return [
    'You are a chess coach answering questions about one played move and its surrounding position.',
    'Use the tools when needed to inspect the parent position, the current position, and any forcing or exploratory branches.',
    'You may make repeated tool calls and pursue branches as deeply as needed before answering.',
    'Write for a strong club player, not an engine developer.',
    'Do not mention Maia, policy models, or raw model probabilities.',
    'Do not use markdown, bullets, or backticks.',
    'If the played move is within 50 centipawns of the top engine move, do not explain why the engine move was better; explain why the played move makes sense instead.',
    'Do not call a move a blunder just because the evaluation dropped inside an already won position.',
    'Be concrete about tactics, king safety, pawn structure, piece activity, and move-order points when relevant.',
    'Keep the final answer concise: usually 2 to 5 sentences and rarely more than about 120 words unless the user explicitly asks for more detail.',
  ].join(' ')
}

function buildConversationHistory(
  context: MoveAnalysisContext,
  transcript: MoveChatMessage[],
  turnPrompt: string,
) {
  return [
    {
      role: 'user',
      content: buildContextPrompt(context),
    },
    ...transcript.map((message) => ({
      role: message.role,
      content: message.text,
    })),
    {
      role: 'user',
      content: turnPrompt,
    },
  ] as JsonRecord[]
}

function buildContextPrompt(context: MoveAnalysisContext) {
  const bestMove =
    context.bestMoveSan && context.bestMoveUci
      ? `${context.bestMoveSan} (${context.bestMoveUci})`
      : 'none'
  const bestRealisticMove =
    context.bestRealisticMoveSan && context.bestRealisticMoveUci
      ? `${context.bestRealisticMoveSan} (${context.bestRealisticMoveUci})`
      : 'none'

  return [
    `Analysis rating context: ${context.rating}.`,
    context.playerRating !== null
      ? `PGN rating for the player who made this move: ${context.playerRating}.`
      : 'PGN rating for the player who made this move: unavailable.',
    `Explain the move ${context.playedMoveSan} (${context.playedMoveUci}) that led from this parent FEN to the current FEN.`,
    `Parent FEN: ${context.parentFen}`,
    `Current FEN: ${context.currentFen}`,
    `Evaluation after the played move: ${context.playedMoveEval}.`,
    `Centipawn loss versus the top engine move: ${context.cpLoss}.`,
    `Human-likelihood signal for the played move: ${(context.humanLikelihood * 100).toFixed(1)}%.`,
    `Best move from the parent position: ${bestMove}.`,
    `Best realistic move from the parent position: ${bestRealisticMove}.`,
    'This context stays fixed across the conversation. Answer the next user message using it and any tool calls you need.',
  ].join('\n')
}
