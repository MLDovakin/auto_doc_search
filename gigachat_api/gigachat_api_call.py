import json
from typing import List, TypedDict, Optional
from urllib.parse import urljoin

from requests import HTTPError
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from loguru import logger
from langchain.chat_models.gigachat import GigaChat

import uuid


class Message(TypedDict):
    role: str
    content: str

from fastapi import FastAPI, HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse

from loguru import logger


api_key = "token"

def giga_chat_call(model_name: str):

    llm = GigaChat(
        model=model_name,
        credentials=api_key,

        scope="GIGACHAT_API_CORP",
        verify_ssl_certs=False,
        temperature=0.0001,
        profanity_check=False, )
    print('[GIGA CHAT LLM]',llm)

    return llm

llm = giga_chat_call('GigaChat-Pro')
llm.invoke('rgerr')