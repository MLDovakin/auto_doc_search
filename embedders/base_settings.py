import configparser
import logging
from datetime import datetime
from pathlib import Path

from loguru import logger
import os
import sys
from os.path import dirname
from typing import Dict, Set, Union

from pydantic import field_validator
from pydantic_core.core_schema import FieldValidationInfo
from pydantic_settings import BaseSettings as PydanticBaseSettings
#: Базовый путь приложения

PATH_VAULT = os.getenv("PATH_VAULT", "")

os.environ['TZ'] = os.getenv('TZ', 'Europe/Moscow')
# time.tzset()


BASE_APP_PATH = Path(dirname(dirname(__file__)))

#: Корневой путь приложения
BASE_PATH = BASE_APP_PATH.parent


class BaseSettings(PydanticBaseSettings):
    class Config:
        # Атрибуты для локальной разработки, вне конфигураций docker-compose.
        # Переменные, указанные в окружении непосредственно,
        # имеют больший приоритет, нежели переменные, указанные в файле.
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class Settings(BaseSettings):
    """
    Базовые настройки приложения

settings = Settings()