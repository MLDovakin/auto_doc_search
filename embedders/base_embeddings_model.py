import os
import sys
import json
import base64
from typing import Optional
import numpy as np
from langchain_core.embeddings import Embeddings
# from app.config.base import logger, settings

from urllib.parse import urljoin

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import uuid
from typing import List, Optional
import requests
import torch
from tqdm import tqdm

from torch import Tensor
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
# model  =  SentenceTransformer("deepvk/USER-bge-m3")
import torch
from transformers import BitsAndBytesConfig

import gc

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GigaChachatEmbeddingsInstruct:
    def __init__(self):
        self.model = AutoModel.from_pretrained('ai-sage/Giga-Embeddings-instruct',
                                               quantization_config=quantization_config, trust_remote_code=True).to(
            device)

    def encode_passage(self, passage):
        passage_prefix = ""
        passage_embeddings = self.model.encode(passage, instruction=passage_prefix).cpu()
        passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)

        print('[PASSAGE SHAPE]', passage_embeddings.shape)
        torch.cuda.empty_cache()
        gc.collect()

        return passage_embeddings

    def encode_query(self, query):
        task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question", }
        query_prefix = task_name_to_instruct["example"] + "\nquestion: "
        query_embeddings = self.model.encode([query], instruction=query_prefix, ).cpu()
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

        torch.cuda.empty_cache()
        gc.collect()

        return query_embeddings

    def get_score(self, query_embeddings, passage_embeddings):
        scores = (query_embeddings @ passage_embeddings.T) * 100
        return scores


class CustomEmbedModel:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name, )

        # Добавьте слой Dense для изменения размернос

    def encode_batch(self, input_strings: List[str]) -> List[List[float]]:
        # res = [self.encode(s) for s in input_strings]
        if isinstance(input_strings, str):
            input_strings = [input_strings]
        res = self.model.encode(input_strings)

        return res