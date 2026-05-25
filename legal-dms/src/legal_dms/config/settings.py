"""Load application settings from environment variables and a local .env file."""

from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    inbox_path: Path = Path.home() / "legal_dms" / "inbox"
    library_path: Path = Path.home() / "legal_dms" / "library"
    work_path: Path = Path.home() / "legal_dms" / "work"
    vector_db_path: Path = Path.home() / "legal_dms" / "vector_db"
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    llm_api_key: str = ""
    ollama_url: str = "http://127.0.0.1:11434"
    embedding_model: str = "BAAI/bge-m3"
    ocr_languages: List[str] = ["en", "hi", "mr"]
    auto_confirm: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("inbox_path", "library_path", "work_path", "vector_db_path", mode="before")
    def expand_and_create_path(cls, value: Path | str) -> Path:
        path = Path(value).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = AppSettings()
