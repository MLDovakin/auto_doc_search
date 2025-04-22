from typing import List
from urllib.parse import urljoin
from tqdm import tqdm


import os
import sys
from gigachat import GigaChat

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
from urllib3.exceptions import InsecureRequestWarning
from langchain_gigachat.embeddings import GigaChatEmbeddings

# Отключение InsecureRequestWarning
warnings.filterwarnings("ignore", category=InsecureRequestWarning)



from loguru import logger
import numpy as np

class EmbedModelDevicesV1Client:
    def __init__(self):

        self.embedder = GigaChatEmbeddings(
            credentials="token",
            scope="GIGACHAT_API",
            verify_ssl_certs=False,)


    def encode_batch(self, input_strings: List[str]) -> List[List[float]]:

        res = []

        for s in tqdm(input_strings, desc="Создание эмбедингов", unit="string"):
            if not(isinstance(s, str)):
                print('[EMBEDDER ERROR]', s)
                pass
            else:
                res.append(self.embedder.embed_query(s))

        print('[EMBEDDING DEVICES RES]',  type(res[0]))
        return res

#embedder = EmbedModelDevicesV1Client()
#print(embedder.encode_batch('queery'))

