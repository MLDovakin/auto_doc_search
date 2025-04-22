import faiss
import json
import numpy as np
from typing import List

class FaissSearchIndex:
    def __init__(self):
        self.index = None
        self.mapping = {}

    def create_index(self, embeddings: np.ndarray, chunks: List[str]):

        if len(embeddings) != len(chunks):
            raise ValueError("Количество эмбеддингов и чанков должно совпадать.")

        dimension = embeddings.shape[1]
        #print('[EMBEDDING DIMENSION]', embeddings.shape[1], embeddings.T.shape, embeddings.shape)

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
            
        self.mapping = {i: chunk for i, chunk in enumerate(chunks)}

    def save_index_and_mapping(self, index_path="faiss_index.index", mapping_path="chunk_mapping.json"):

    
        if self.index is None or not self.mapping:
            raise ValueError("Индекс или маппинг не созданы. Используйте create_index() перед сохранением.")

        faiss.write_index(self.index, index_path)

        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(self.mapping, f, ensure_ascii=False, indent=4)

    def load_index_and_mapping(self, index_path="faiss_index.index", mapping_path="chunk_mapping.json"):

        self.index = faiss.read_index(index_path)
        #print('[LOAD INDEX]', index_path)
        with open(mapping_path, "r", encoding="utf-8") as f:
            self.mapping = json.load(f)

    def search(self, query_embedding: np.ndarray, k=5):
        """
        Выполняет поиск ближайших векторов в FAISS индексе.

        :param query_embedding: Вектор запроса (должен быть размерности (1, dim)).
        :param k: Количество ближайших соседей для поиска.
        :return: Список словарей с чанками и расстояниями.
        """
        #print('[QUERY EMBEDDING]',query_embedding.shape)
        if self.index is None or not self.mapping:
            raise ValueError("Индекс или маппинг не загружены. Используйте create_index() или load_index_and_mapping().")

        distances, indices = self.index.search(np.array(query_embedding, dtype="float32").reshape(1, -1), k)
        
        # Проверка на случай, если ничего не найдено
        if indices.size == 0 or all(idx == -1 for idx in indices[0]):
            return "Информация по данному запросу не найдена."
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue  # Пропустить "пустые" индексы
            results.append({"chunk": self.mapping.get(str(idx), "Чанк не найден"), "distance": dist})
        
        return results