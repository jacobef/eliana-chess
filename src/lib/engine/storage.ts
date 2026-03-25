type CacheEntry = {
  id: string
  version: string
  data: Blob
  timestamp: number
  size: number
}

export class BinaryAssetCache {
  private readonly dbName: string
  private readonly storeName: string
  private readonly version: number
  private db: IDBDatabase | null = null

  constructor(dbName: string, storeName: string, version = 1) {
    this.dbName = dbName
    this.storeName = storeName
    this.version = version
  }

  async requestPersistentStorage() {
    try {
      if ('storage' in navigator && 'persist' in navigator.storage) {
        return await navigator.storage.persist()
      }
    } catch {
      return false
    }

    return false
  }

  async get(id: string, version: string) {
    try {
      const db = await this.open()
      const store = db.transaction([this.storeName], 'readonly').objectStore(this.storeName)
      const entry = await new Promise<CacheEntry | null>((resolve, reject) => {
        const request = store.get(id)
        request.onsuccess = () => resolve((request.result as CacheEntry | undefined) ?? null)
        request.onerror = () => reject(request.error)
      })

      if (!entry) {
        return null
      }

      if (entry.version !== version) {
        await this.delete(id)
        return null
      }

      return await entry.data.arrayBuffer()
    } catch {
      return null
    }
  }

  async set(id: string, version: string, buffer: ArrayBuffer) {
    const db = await this.open()
    const store = db.transaction([this.storeName], 'readwrite').objectStore(this.storeName)

    await new Promise<void>((resolve, reject) => {
      const request = store.put({
        id,
        version,
        data: new Blob([buffer]),
        timestamp: Date.now(),
        size: buffer.byteLength,
      } satisfies CacheEntry)

      request.onsuccess = () => resolve()
      request.onerror = () => reject(request.error)
    })
  }

  async delete(id: string) {
    const db = await this.open()
    const store = db.transaction([this.storeName], 'readwrite').objectStore(this.storeName)

    await new Promise<void>((resolve, reject) => {
      const request = store.delete(id)
      request.onsuccess = () => resolve()
      request.onerror = () => reject(request.error)
    })
  }

  private async open(): Promise<IDBDatabase> {
    if (this.db) {
      return this.db
    }

    return await new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.version)

      request.onerror = () => reject(request.error)
      request.onsuccess = () => {
        this.db = request.result
        resolve(request.result)
      }
      request.onupgradeneeded = () => {
        const db = request.result
        if (!db.objectStoreNames.contains(this.storeName)) {
          db.createObjectStore(this.storeName, { keyPath: 'id' })
        }
      }
    })
  }
}
