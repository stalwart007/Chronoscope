/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Override the API origin when the SPA is hosted separately from the API. */
  readonly VITE_API_BASE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
