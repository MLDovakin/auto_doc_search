from typing import List, Optional
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import AutoTokenizer

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from gigachat_api.gigachat_api_call import GigaChatLLM, GigaChatV1Client
from embedders.devices_embedder import EmbedModelDevicesV1Client
from embedders.base_embeddings_model import Q2QEmbedModel
from gigachat_api.devices_settings import  devices_settings
from gigachat_api.tokenizer_api import TokenizerV1Client
from embedders.base_settings import settings

class ContextMetricsEvaluator:
    def __init__(self, embedding_model_api: Optional[EmbedModelDevicesV1Client],
                 gigachat_llm_api: Optional[GigaChatLLM],
                 tokenizer: Optional[TokenizerV1Client]):
        """
        Инициализация Evaluator для ContextRecall и ContextPrecision.
        :param embedding_model_api: API клиент для получения эмбеддингов.
        :param gigachat_llm_api: API клиент для работы с GigaChatLLM.
        :param tokenizer: API клиент токенизации.
        """
        print("[Инициализация] Настройка API клиентов...")
        self.embedding_model_api = embedding_model_api
        self.gigachat_llm_api = gigachat_llm_api
        self.tokenizer = tokenizer
        print("[Инициализация] Клиенты API успешно настроены!")

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Получает эмбеддинги для списка текстов с использованием API клиента.
        :param texts: Список текстов для преобразования в эмбеддинги.
        :return: Массив numpy с эмбеддингами.
        """
        response = self.embedding_model_api.encode_batch(texts)
        return np.squeeze(np.array(response, dtype="float32"), axis=1)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Вычисляет косинусное сходство между двумя векторами.
        :param vec1: Первый вектор.
        :param vec2: Второй вектор.
        :return: Значение косинусного сходства.
        """
        vec1 = vec1.reshape(1, -1) if len(vec1.shape) == 1 else vec1
        vec2 = vec2.reshape(1, -1) if len(vec2.shape) == 1 else vec2
        return float(cosine_similarity(vec1, vec2)[0][0])

    def context_recall(self, answer: str, context_chunks: List[str]) -> float:
        """
        Вычисляет Context Recall (доля контекста, покрытого в ответе).
        :param answer: Сгенерированный ответ.
        :param context_chunks: Список retrieved чанков.
        :return: Значение Context Recall.
        """
        print("[Context Recall] Вычисляем...")
        embeddings = self._get_embeddings([answer] + context_chunks)
        answer_emb = embeddings[0]
        context_embs = embeddings[1:]

        total_relevant = sum(
            1 for chunk_emb in context_embs if self._cosine_similarity(answer_emb, chunk_emb) > 0.7
        )

        recall_score = total_relevant / len(context_chunks) if context_chunks else 0.0
        print(f"[Context Recall] Результат: {recall_score:.2%}")
        return recall_score

    def context_precision(self, answer: str, context_chunks: List[str]) -> float:
        """
        Вычисляет Context Precision (доля ответа, относящаяся к retrieved контексту).
        :param answer: Сгенерированный ответ.
        :param context_chunks: Список retrieved чанков.
        :return: Значение Context Precision.
        """
        print("[Context Precision] Вычисляем...")
        
        # Порог сходства для Precision
        THRESHOLD = 0.7

        # Получение эмбеддингов для ответа и контекста
        embeddings = self._get_embeddings([answer] + context_chunks)
        answer_emb = embeddings[0]
        context_embs = embeddings[1:]

        # Подсчет релевантных чанков
        total_relevant = sum(
            1 for chunk_emb in context_embs if self._cosine_similarity(answer_emb, chunk_emb) > THRESHOLD
        )

        # Токенизация ответа для подсчета токенов
        answer_token_count = self.tokenizer.encode(answer)
        if not isinstance(answer_token_count, int):
            raise ValueError("`encode` должен возвращать количество токенов как int.")

        # Precision: сколько из ответа релевантно контексту
        precision_score = total_relevant / answer_token_count if answer_token_count else 0.0
        print(f"[Context Precision] Результат: {precision_score:.2%}")
        return precision_score

'''
embedding_model_api = EmbedModelDevicesV1Client(settings)  # API клиента для эмбеддингов
gigachat_llm_api = GigaChatLLM(api_client=GigaChatV1Client(devices_settings, 'GigaChat'))
tokenizer = TokenizerV1Client(settings=devices_settings, model_name='GigaChat')

evaluator = ContextMetricsEvaluator(embedding_model_api, gigachat_llm_api, tokenizer)

answer = "Кредит можно получить в отделении банка."
retrieved_chunks = [
    "Для получения кредита обратитесь в отделение банка.",
    "Вам потребуется паспорт и документы для заявки."
]

recall = evaluator.context_recall(answer, retrieved_chunks)
precision = evaluator.context_precision(answer, retrieved_chunks)

print(f"Context Recall: {recall:.2%}")
print(f"Context Precision: {precision:.2%}")
'''