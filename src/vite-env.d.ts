/// <reference types="vite/client" />

declare module 'lila-stockfish-web/sf171-79.js' {
  import type StockfishWeb from 'lila-stockfish-web'

  type StockfishModuleFactory = (options: {
    wasmMemory: WebAssembly.Memory
    locateFile: (name: string) => string
  }) => Promise<StockfishWeb>

  const createStockfish: StockfishModuleFactory
  export default createStockfish
}
