import base64
import json
from typing import Optional
import requests

import sys
import os 

from pydantic import Field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class ModelException(Exception):
    def __init__(self, text: str, exc: Optional[Exception] = None):
        self.text = text
        self.exc = exc




