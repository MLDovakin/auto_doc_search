import nltk
nltk.download('punkt_tab')

import pandas as pd
import numpy as np
from nltk.translate.meteor_score import single_meteor_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter

from sklearn.metrics import jaccard_score
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu


class RAGMetrics:
    def __init__(self, tokenizer, model):
        """Инициализация класса для расчета метрик."""
        self.tokenizer = tokenizer
        self.model = model

    def compute_embeddings(self, texts):
        """Вычисление эмбеддингов для текстов."""
        return self.model.encode(texts, convert_to_tensor=True)

    def cosine_similarity_metric(self, embeddings_1, embeddings_2):
        """Cosine Similarity."""
        scores = cosine_similarity(embeddings_1.cpu().numpy(), embeddings_2.cpu().numpy())
        return np.mean(np.diag(scores))

    def euclidean_distance_metric(self, embeddings_1, embeddings_2):
        """Euclidean Distance."""
        distances = euclidean_distances(embeddings_1.cpu().numpy(), embeddings_2.cpu().numpy())
        return np.mean(np.diag(distances))

    def meteor_metric(self, references, hypotheses):
        """Вычисление METEOR."""
        scores = [
            single_meteor_score(self.tokenizer(ref), self.tokenizer(hyp))
            for ref, hyp in zip(references, hypotheses)
        ]
        return np.mean(scores)

    def embedding_average_cosine_similarity(self, embeddings_1, embeddings_2):
        """Embedding Average Cosine Similarity."""
        avg_embeddings_1 = embeddings_1.mean(dim=0).cpu().numpy().reshape(1, -1)
        avg_embeddings_2 = embeddings_2.mean(dim=0).cpu().numpy().reshape(1, -1)
        scores = cosine_similarity(avg_embeddings_1, avg_embeddings_2)
        return scores[0][0]

    def coverage_metric(self, references, hypotheses):
        """Coverage (доля общих токенов между текстами)."""
        scores = []
        for ref, hyp in zip(references, hypotheses):
            ref_tokens = set(self.tokenizer(ref))
            hyp_tokens = set(self.tokenizer(hyp))
            coverage = len(ref_tokens & hyp_tokens) / len(ref_tokens) if len(ref_tokens) > 0 else 0
            scores.append(coverage)
        return np.mean(scores)

    def distinct_n_metric(self, hypotheses, n=2):
        """Distinct-n (разнообразие n-грамм)."""
        all_ngrams = []
        for text in hypotheses:
            tokens = self.tokenizer(text)
            ngrams_list = list(ngrams(tokens, n))
            all_ngrams.extend(ngrams_list)
        distinct_ngrams = len(set(all_ngrams))
        total_ngrams = len(all_ngrams)
        return distinct_ngrams / total_ngrams if total_ngrams > 0 else 0

    def diversity_metric(self, hypotheses):
        """Token Entropy (оценка разнообразия токенов)."""
        all_tokens = [token for text in hypotheses for token in self.tokenizer(text)]
        token_counts = Counter(all_tokens)
        total_tokens = sum(token_counts.values())
        entropy = -sum((count / total_tokens) * np.log2(count / total_tokens) for count in token_counts.values())
        return entropy

    def bleu_metric(self, references, hypotheses):
        """BLEU Score."""
        scores = [
            sentence_bleu([self.tokenizer(ref)], self.tokenizer(hyp))
            for ref, hyp in zip(references, hypotheses)
        ]
        return np.mean(scores)

    def jaccard_similarity_metric(self, references, hypotheses):
        """Jaccard Similarity."""
        scores = []
        for ref, hyp in zip(references, hypotheses):
            ref_tokens = set(self.tokenizer(ref))
            hyp_tokens = set(self.tokenizer(hyp))
            if ref_tokens or hyp_tokens:
                score = len(ref_tokens & hyp_tokens) / len(ref_tokens | hyp_tokens)
            else:
                score = 0
            scores.append(score)
        return np.mean(scores)

    def recall_at_k(self, embeddings_1, embeddings_2, k=5):
        """Recall@K."""
        # Расстояния между всеми парами
        distances = euclidean_distances(embeddings_1.cpu().numpy(), embeddings_2.cpu().numpy())
        # Получаем индексы топ-K ближайших
        top_k_indices = np.argsort(distances, axis=1)[:, :k]
        # Считаем релевантные (диагональные совпадения в топ-K)
        recall = sum(1 for i, row in enumerate(top_k_indices) if i in row) / embeddings_1.shape[0]
        return recall

    def mips_metric(self, embeddings_1, embeddings_2):
        """MIPS (Maximum Inner Product Search)."""
        inner_products = (embeddings_1 * embeddings_2).sum(dim=1)
        return inner_products.mean().item()

    def split_texts(self, text, start="Content:", end="***КОНЕЦ СТАТЬИ***"):
        """
        Разделяет строку на тексты по шаблону начала и конца текста.

        :param text: строка, содержащая тексты
        :param start: начало каждого текста
        :param end: конец каждого текста
        :return: список текстов
        """
        return [t.strip() for t in text.split(start) if t.strip() and end in t]

    def compute_metrics_for_multitext_row(self, text1, text2):
        """
        Вычисляет метрики между всеми текстами в строке text1 и текстом text2.

        :param text1: строка с несколькими текстами
        :param text2: строка для сравнения
        :return: словарь средних метрик
        """
        texts = self.split_texts(text1)
        if not texts:
            return None

        metrics_list = []
        for t in texts:
            embeddings_1 = self.compute_embeddings([t])
            embeddings_2 = self.compute_embeddings([text2])

            metrics = {
                "Cosine Similarity": self.cosine_similarity_metric(embeddings_1, embeddings_2),
                "Euclidean Distance": self.euclidean_distance_metric(embeddings_1, embeddings_2),
                "METEOR": self.meteor_metric([t], [text2]),
                "Embedding Average Cosine Similarity": self.embedding_average_cosine_similarity(embeddings_1, embeddings_2),
                "Coverage": self.coverage_metric([t], [text2]),
                "Distinct-2": self.distinct_n_metric([t]),
                "Token Entropy": self.diversity_metric([t]),
                "BLEU Score": self.bleu_metric([t], [text2]),
                "Jaccard Similarity": self.jaccard_similarity_metric([t], [text2]),
                "Recall@5": self.recall_at_k(embeddings_1, embeddings_2, k=5),
                "MIPS": self.mips_metric(embeddings_1, embeddings_2),
            }
            metrics_list.append(metrics)

        # Усредняем метрики
        avg_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0].keys()}
        return avg_metrics

    def compute_metrics_between_dfs(self, df1, col1, df2, col2):
        """
        Вычисление всех метрик между двумя датафреймами.

        :param df1: первый DataFrame
        :param col1: колонка с текстами, содержащими несколько текстов
        :param df2: второй DataFrame
        :param col2: колонка с текстами для сравнения
        :return: DataFrame с итоговыми метриками в длинном формате
        """
        if len(df1) != len(df2):
            raise ValueError("Оба датафрейма должны иметь одинаковое количество строк!")

        # Сохраняем результаты метрик
        all_metrics = []
        for text1, text2 in zip(df1[col1], df2[col2]):
            avg_metrics = self.compute_metrics_for_multitext_row(text1, text2)
            if avg_metrics:
                for metric_name, metric_value in avg_metrics.items():
                    all_metrics.append({"metrics": metric_name, "metric_value": metric_value})

        # Преобразуем список в DataFrame
        return pd.DataFrame(all_metrics)


# Пример использования
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    # Пример датафреймов
    df1 = pd.DataFrame({'text1': [
        """Content:Текст 1.1***КОНЕЦ СТАТЬИ***Content:Текст 1.2***КОНЕЦ СТАТЬИ***Content:Текст 1.3***КОНЕЦ СТАТЬИ***"""
    ]})
    df2 = pd.DataFrame({'text2': ["Это текст для сравнения."]})

    # Инициализация модели и класса
    model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = word_tokenize
    metrics = RAGMetrics(tokenizer=tokenizer, model=model)

    # Расчет метрик между датафреймами
    results = metrics.compute_metrics_between_dfs(df1, 'text1', df2, 'text2')
    print(results)
