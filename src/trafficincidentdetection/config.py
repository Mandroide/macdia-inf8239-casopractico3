import pathlib

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):  # type: ignore[misc]
    model_config = SettingsConfigDict(
        env_file=pathlib.Path(__file__).parents[2]/".env", env_file_encoding="utf-8",extra="ignore"
    )
    raw_data_dir_path: pathlib.Path = Field(default=pathlib.Path(__file__).parents[2] / "data/raw", alias="RAW_DATA_DIR_PATH")
    x_bearer_token: str = Field(default="", alias="X_BEARER_TOKEN")
