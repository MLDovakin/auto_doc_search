import os 
import sys
from typing import Union

from pydantic import Field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from embedders.base_settings import BaseSettings


class DevicesSettings(BaseSettings):

    DEVICES_URL: str = "https://gigachat.devices.sberbank.ru/"

    USE_CERTS: bool = Field(alias="DEVICES_USE_CERTS", default=False)

    KEY_PATH: str = Field(alias="DEVICES_KEY_PATH", default="")

    CERT_PATH: str = Field(alias="DEVICES_CERT_PATH", default="")

    CACERT_PATH: Union[bool, str] = Field(alias="DEVICES_CACERT_PATH", default=False)

    GIGACHAT_MAX_TOKENS: int = 512

    GIGACHAT_MODEL_NAME: str = "GigaChat-preview"

    GIGACHAT_TEMPERATURE: float = 0.0001

    GIGACHAT_CONTEXT_LEN: int = 4096
    REQUEST_TIMEOUT: float = 120.0
    TIMEOUT: float = 120
devices_settings = DevicesSettings()