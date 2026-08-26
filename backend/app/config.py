"""Application settings.

All configuration is environment-driven and read once into a frozen
``Settings`` instance shared through :func:`get_settings`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", REPO_ROOT / "backend/.env"),
        env_file_encoding="utf-8",
        env_prefix="CS_",
        extra="ignore",
        frozen=True,
    )

    # ---------------------------------------------------------------- runtime
    app_name: str = "Chronoscope"
    env: Literal["dev", "prod", "test"] = "dev"
    log_level: str = "INFO"
    log_json: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:5173,http://localhost:4173,http://localhost:8080"

    # --------------------------------------------------------------- security
    #: Set to require `Authorization: Bearer <key>` (or `X-API-Key`) on every
    #: request. Empty means open access, correct for localhost, never for a
    #: deployment reachable from anywhere else.
    api_key: str | None = None
    #: Honour X-Forwarded-For only behind a reverse proxy. Otherwise clients
    #: can spoof it to escape rate limiting.
    trust_proxy_headers: bool = False
    allowed_hosts: str = "*"
    rate_limit_enabled: bool = True
    rl_default_per_min: int = 240
    rl_query_per_min: int = 40
    rl_upload_per_min: int = 24
    rl_stream_per_min: int = 60
    max_request_bytes: int = 2 * 1024 * 1024  # non-upload request bodies
    max_transcript_kb: int = 4096
    storage_quota_gb: float = 100.0
    max_video_duration_s: float = 6 * 3600
    max_video_pixels: int = 4096 * 2160
    enable_docs: bool = True

    # ------------------------------------------------------------------ paths
    data_dir: Path = REPO_ROOT / "data"
    model_cache_dir: Path | None = None

    # --------------------------------------------------------------- ingestion
    max_upload_mb: int = 4096
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    #: Worker slots for CPU-bound stages. Bounded because model memory, not
    #: CPU time, is the scarce resource.
    ingest_concurrency: int = 2
    #: How often the in-process vector index is checkpointed to disk.
    checkpoint_interval_s: float = 30.0
    frame_workers: int = 4

    # scene detection
    scene_threshold: float = 27.0
    scene_min_len_s: float = 0.8
    #: Hard ceiling on keyframes per video, enforced by an information-gain
    #: driven budget allocator (see ``ingest.keyframes``).
    max_keyframes: int = 400
    keyframe_dedupe_hamming: int = 6
    frame_max_dim: int = 896

    # transcription
    whisper_model: str = "base"
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = 5
    whisper_vad: bool = True
    language: str | None = None

    # diarization
    diarization_enabled: bool = True
    hf_token: str | None = None
    max_speakers: int = 8

    # ---------------------------------------------------------------- chunking
    chunk_target_s: float = 22.0
    chunk_max_s: float = 45.0
    chunk_overlap_s: float = 2.5

    # -------------------------------------------------------------- embeddings
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    embed_batch_size: int = 16

    # ------------------------------------------------------------ vector store
    vector_backend: Literal["qdrant", "memory"] = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "chronoscope_chunks"
    qdrant_prefer_grpc: bool = False
    hnsw_m: int = 24
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 96

    # -------------------------------------------------------------- retrieval
    rrf_k: float = 60.0
    retrieval_candidates: int = 48
    final_k: int = 8
    mmr_lambda: float = 0.72
    temporal_decay_s: float = 45.0
    neighbour_bonus: float = 0.18

    # -------------------------------------------------------------------- llm
    llm_provider_chain: str = "ollama,openrouter,groq"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_vision_model: str = "llava:7b"
    openrouter_key: str | None = None
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    openrouter_vision_model: str = "meta-llama/llama-3.2-11b-vision-instruct:free"
    groq_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    llm_timeout_s: float = 120.0
    llm_max_retries: int = 2
    llm_temperature: float = 0.15
    agent_max_steps: int = 24

    # ------------------------------------------------------------------ db
    database_url: str = ""

    # -------------------------------------------------------------- validators
    @field_validator("data_dir", mode="before")
    @classmethod
    def _expand(cls, v: object) -> object:
        return Path(str(v)).expanduser() if v else v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def artifact_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def allowed_host_list(self) -> list[str]:
        return [h.strip() for h in self.allowed_hosts.split(",") if h.strip()] or ["*"]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def provider_chain(self) -> list[str]:
        return [p.strip().lower() for p in self.llm_provider_chain.split(",") if p.strip()]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sqlalchemy_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite+aiosqlite:///{self.data_dir / 'chronoscope.db'}"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.upload_dir, self.artifact_dir):
            p.mkdir(parents=True, exist_ok=True)
        if self.model_cache_dir:
            self.model_cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(self.model_cache_dir))
            os.environ.setdefault("TORCH_HOME", str(self.model_cache_dir))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


settings = get_settings()
