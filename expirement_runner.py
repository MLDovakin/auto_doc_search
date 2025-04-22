import os
import json
import re
from typing import Optional, List, Dict, Union
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from uuid import uuid4
# from loguru import logger

from faiss_index_creation import FaissSearchIndex
from langchain_core.documents import Document

from gigachat_api.gigachat_api_call import giga_chat_call

from splitters.markdown_splitter import MarkdownSplitter

from rag_chains.llm_qa_chain import get_qa_answer, get_mmlu_answer, SYSTEM_PROMPT_TEMPLATE, MMLU_SYSTEM_PROMPT
from embedders.devices_embedder import EmbedModelDevicesV1Client
from autovalidation.autoval_chain import validate_user_query, get_validation_report
from embedders.base_embeddings_model import CustomEmbedModel, GigaChachatEmbeddingsInstruct

import shutil
import GPUtil


class EmbedderManager:
    def __init__(self, embedder_name: str, custom_model_name: str = None):
        """
        Инициализирует менеджер эмбеддеров.

        :param embedder_name: Название эмбеддера ('custom' или 'GigaEmbeddingsInstruct').
        :param custom_model_name: Название кастомной модели (для 'custom').
        """
        if embedder_name == 'custom':
            self.embedder = CustomEmbedModel(custom_model_name)
        elif embedder_name == 'GigaEmbeddingsInstruct':
            self.embedder = GigaChachatEmbeddingsInstruct()
        else:
            raise ValueError(f"Unsupported embedder: {embedder_name}")

        self.embedder_type = embedder_name

    def encode_query(self, query: str):
        """

        :param query: Текст запроса.
        :return: Эмбеддинг запроса.
        """
        if self.embedder_type == 'GigaEmbeddingsInstruct':
            return self.embedder.encode_query(query)
        elif self.embedder_type == 'custom':
            return self.embedder.encode_batch([query])
        else:
            raise NotImplementedError("encode_query is not implemented for the selected embedder.")

    def encode_passage(self, passages: List[str]):
        """
        Кодирует чанк (или несколько). Вызывает соответствующий метод в зависимости от типа эмбеддера.

        :param passages: Список текстов чанков.
        :return: Эмбеддинги чанков.
        """
        if self.embedder_type == 'GigaEmbeddingsInstruct':
            return self.embedder.encode_passage(passages)
        elif self.embedder_type == 'custom':
            return self.embedder.encode_batch(passages)
        else:
            raise NotImplementedError("encode_passage is not implemented for the selected embedder.")


class FaissIndexManager:
    def __init__(self, experiment_path: Path):
        os.makedirs(str(experiment_path) + '/faiss_index', exist_ok=True)
        self.experiment_path = Path(str(experiment_path) + '/faiss_index')

    def create_and_save_index(self, index_name: str, embeddings: np.ndarray, chunks: List[str]):
        index_path = self.experiment_path / f"{index_name}_faiss.index"
        mapping_path = self.experiment_path / f"{index_name}_chunk_mapping.json"

        faiss_index = FaissSearchIndex()
        faiss_index.create_index(embeddings, chunks)
        faiss_index.save_index_and_mapping(str(index_path), str(mapping_path))

    def search_all_indexes(self, query_embedding: np.ndarray, top_n: int) -> List[Dict[str, str]]:
        results = []
        for index_file in self.experiment_path.glob("*_faiss.index"):
            index_name = index_file.stem.replace("_faiss", "")
            mapping_file = self.experiment_path / f"{index_name}_chunk_mapping.json"
            if not mapping_file.exists():
                continue

            faiss_index = FaissSearchIndex()
            faiss_index.load_index_and_mapping(str(index_file), str(mapping_file))
            index_results = faiss_index.search(query_embedding, top_n)
            #print(len(index_results))
            for result in index_results:
                result["source"] = index_name
            results.extend(index_results)
        #print('[LEN INDEXES],', len(results))
        return sorted(results, key=lambda x: x["distance"])[:top_n]


class QuestionAnsweringManager:
    def __init__(self, llm_model_name: str):
        self.llm = giga_chat_call(model_name=llm_model_name)

    def get_answer(self, query: str, context: str) -> str:
        return get_qa_answer(query=query, documents=context, system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
                             llm=self.llm)

    def get_mmlu_answer(self, query: str, context: str) -> str:
        return get_mmlu_answer(query=query, documents=context, system_prompt_template=MMLU_SYSTEM_PROMPT,
                             llm=self.llm)

class AutoValidationManager:
    def __init__(self, llm_model_name: str, experiment_path: Path):
        """
        :param llm_model_name: Название модели LLM для валидации.
        :param experiment_path: Путь к директории эксперимента для сохранения результатов.
        """
        self.llm = giga_chat_call(model_name=llm_model_name)
        self.experiment_path = experiment_path


    def validate_answers(self, questions: List[str], answers: List[str],
                         faiss_chunks: List[List[Dict[str, str]]], mode=None) -> pd.DataFrame:
        """
        Валидирует ответы на вопросы с использованием метрик и LLM.

        :param questions: Список вопросов.
        :param answers: Список ответов на вопросы.
        :param faiss_chunks: Список чанков из FAISS для каждого вопроса.
        :return: DataFrame с результатами валидации.
        """
        if len(questions) != len(answers) or len(questions) != len(faiss_chunks):
            raise ValueError("Количество вопросов, ответов и чанков должно совпадать.")

        print("[AutoValidationManager] Начало валидации ответов.")

        # Передаем faiss_chunks без изменения
        validated_df = validate_user_query(
            queries=questions,
            input_answers=answers,
            context_chunks=faiss_chunks,
            fact_extraction=True,
            llm=self.llm
        )

        validation_report = get_validation_report(validated_df, val_mark_column='validation_mark')

        output_path = self.experiment_path / "validated_answers.xlsx"
        validated_df.to_excel(output_path, index=False)

        print(f"[AutoValidationManager] Валидация завершена. Результаты сохранены в: {output_path}")

        return validation_report


class FileProcessingManager:
    def __init__(self, experiment_path: Path):
        self.experiment_path = experiment_path

    def extract_files(self, zip_path: str) -> List[Path]:
        data_files_dir = self.experiment_path / "data_files"
        data_files_dir.mkdir(exist_ok=True)

        extracted_paths = []
        with zipfile.ZipFile(zip_path, 'r') as archive:
            for name in archive.namelist():
                if name.endswith('.docx'):
                    extracted_path = self._extract_file(archive, name, data_files_dir)
                    extracted_paths.append(extracted_path)
        return extracted_paths

    def _extract_file(self, archive, name, output_dir):
        try:
            name_corrected = name.encode('cp437', errors='ignore').decode('cp866', errors='ignore')
        except UnicodeDecodeError:
            name_corrected = name

        extracted_path = output_dir / Path(name_corrected).name
        with archive.open(name) as source_file, open(extracted_path, "wb") as target_file:
            shutil.copyfileobj(source_file, target_file)
        return extracted_path

    def process_file(self, file_path: Path) -> List[str]:
        splitter = MarkdownSplitter()
        chunks = splitter._create_documents_from_docx(file_path)
        if isinstance(chunks[0], Document):
            return [chunk.page_content for chunk in chunks]
        return chunks


class QuestionGeneration:
    def __init__(self, query_gen_llm):
        self.query_gen_llm = giga_chat_call(model_name=query_gen_llm)

    def generate_questions(self, chunks_df: pd.DataFrame, chunk_column_name='chunks', style_question='short',
                           query_count=None):
        if chunk_column_name not in chunks_df.columns:
            raise ValueError(f"Входной датафрейм должен содержать колонку {chunk_column_name}.")

        questions_dict = {}
        results = []

        # Ограничиваем количество чанков для генерации вопросов
        limited_chunks_df = chunks_df.head(query_count or len(chunks_df))

        for idx, row in tqdm(limited_chunks_df.iterrows(), desc="Генерация вопросов", unit="chunk",
                             total=len(limited_chunks_df)):
            chunk_text = row[chunk_column_name]

            # Если стиль вопросов "closed", проверяем наличие чисел
            if style_question == 'closed' and not re.search(r'\d+', chunk_text):
                continue  # Пропускаем чанки без чисел

            # Генерация prompt в зависимости от стиля вопроса
            if style_question == 'short':
                prompt = gen_query_prompt(chunk_text)
            elif style_question == 'long':
                prompt = gen_query_long(chunk_text)
            elif style_question == 'mmlu':
                prompt = gen_mmlu_query(chunk_text)
            elif style_question == 'closed':
                prompt = gen_closed_query(chunk_text)
            else:
                raise ValueError(f"Неподдерживаемый стиль вопроса: {style_question}")

            try:
                question = self.query_gen_llm.invoke(prompt).content.strip()
                if not question.endswith("?"):
                    question += "?"

                questions_dict[question] = chunk_text
                results.append({"query": question, "chunk": chunk_text})

            except Exception as e:
                print(f"[Генерация вопросов] Ошибка при генерации вопроса для чанка {idx}: {e}")
                results.append({"query": f"Ошибка при генерации вопроса для чанка {idx}", "chunk": chunk_text})

        return pd.DataFrame(results)


class ExperimentManager:
    def __init__(self, config: Dict[str, Union[str, int, bool]], experiment_dir: str = "experiments_chunk_syntetic"):
        self.config = config

        self.experiment_path = Path(experiment_dir) / f"experiment_1"
        #self.experiment_path = 'experiment_1'
        self.experiment_path.mkdir(parents=True, exist_ok=True)

        self.embedder_manager = EmbedderManager(config['embedder_name'], config['custom_embedder_model_name'])
        self.index_manager = FaissIndexManager(self.experiment_path)

        self.qa_manager = QuestionAnsweringManager(config['question_answering_model_name'])
        self.file_manager = FileProcessingManager(self.experiment_path)
        self.validation_manager = AutoValidationManager(config['validation_model_name'], self.experiment_path)

        self.question_generator = QuestionGeneration(config["question_generation_model_name"])

    def load_chunks_from_index(self, index_name):
        index_path = self.experiment_path / f"{index_name}.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Индекс {index_name} не найден по пути {index_path}.")

        return pd.read_json(index_path)

    def run(self):
        zip_path = self.config.get("word_file_path")
        if not zip_path or not os.path.exists(zip_path):
            raise FileNotFoundError(f"Файл {zip_path} не найден.")

        extracted_files = self.file_manager.extract_files(zip_path)
        for file_path in extracted_files:
            chunks = self.file_manager.process_file(file_path)
            embeddings = self.embedder_manager.encode_passage(chunks)
            index_name = file_path.stem
            self.index_manager.create_and_save_index(index_name, embeddings, chunks)

    def answer_user_queries(self, queries: List[str], top_n: int = 5, mode = None):
        """
        Обрабатывает запросы пользователя и возвращает ответы.

        :param queries: Список текстовых запросов.
        :param top_n: Количество лучших результатов из индекса.
        :return: DataFrame с вопросами, ответами и источниками.
        """
        results = []

        for query in queries:
            # Получаем эмбеддинг запроса
            if self.embedder_manager.embedder_type == 'GigaEmbeddingsInstruct':
                query_embedding = self.embedder_manager.encode_query(query)
            else:
                query_embedding = self.embedder_manager.embedder.encode_batch([query])

            # Поиск наиболее релевантных чанков в индексах
            top_chunks = self.index_manager.search_all_indexes(query_embedding, top_n)
            print('[QUERY]', query, len(top_chunks))
            if not top_chunks:
                results.append({
                    "query": query,
                    "answer": "No relevant data found",
                    "source": "—"
                })
                continue

            # Формирование контекста из найденных чанков
            context = "".join([f"Content:{chunk['chunk']}***КОНЕЦ СТАТЬИ***" for chunk in top_chunks])

            # Получение ответа на основе контекста
            if mode == 'mmlu':
                answer = self.qa_manager.get_mmlu_answer(query, context)
                results.append({
                    "query": query,
                    "answer": answer,
                    "source": context
                })

            else:
                answer = self.qa_manager.get_answer(query, context)
                results.append({
                    "query": query,
                    "answer": answer,
                    "source": context
                })

        return pd.DataFrame(results)

    def answer_from_excel(self, excel_file_path: str, question_column: str, top_n: int = 5) -> pd.DataFrame:
        if not Path(excel_file_path).exists():
            raise FileNotFoundError(f"Файл {excel_file_path} не найден.")

        questions_df = pd.read_excel(excel_file_path)
        if question_column not in questions_df.columns:
            raise ValueError(f"Указанная колонка '{question_column}' не найдена в файле.")

        questions = questions_df[question_column].dropna().tolist()
        print(f"Начинаю обработку вопросов из файла: {excel_file_path}")
        qa_answers_df = self.answer_user_queries(questions, top_n=top_n)

        output_file_path = Path(excel_file_path).with_name(f"answered_{Path(excel_file_path).name}")
        qa_answers_df.to_excel(output_file_path, index=False)
        print(f"Результаты сохранены в файл: {output_file_path}")

        return qa_answers_df

    def validate_user_answers(self, validation_df: pd.DataFrame) -> pd.DataFrame:
        """
        Метод для вызова валидации пользовательских вопросов и ответов.

        :param validation_df: DataFrame с вопросами, ответами и контекстом для валидации.
        :return: DataFrame с результатами валидации.
        """
        if not {'query', 'answer'}.issubset(validation_df.columns):
            raise ValueError("DataFrame должен содержать колонки 'query' и 'answer'.")

        questions = validation_df['query'].tolist()
        answers = validation_df['answer'].tolist()

        faiss_chunks = []
        for query in questions:
            query_embedding = self.embedder_manager.encode_query(query)
            chunks = self.index_manager.search_all_indexes(query_embedding, top_n=5)
            faiss_chunks.append(chunks)

        print("[ExperimentManager] Вызов валидации ответов.")
        return self.validation_manager.validate_answers(questions, answers, faiss_chunks)

    def load_chunks_from_index(self, index_name):
        print("expiriment_path", index_name)
        index_path = self.experiment_path / f"{index_name}.json"
        print(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"Индекс {index_name} не найден по пути {index_path}")
        return index_path

    def generate_and_save_questions(self, example_index=None, style_question="short", query_count=None):
        if query_count is None:
            query_count = self.config.get("query_count", 10)

        if example_index is None:
            index_files = [f for f in os.listdir(self.experiment_path) if f.endswith(".json")]
            if not index_files:
                print("[Генерация вопросов] Индексы для генерации не найдены.")
                return

            for index_file in index_files:
                index_name = index_file.replace(".json", "")
                print(index_name)
                self._process_index(index_name, style_question, query_count)
        else:
            self._process_index(example_index, style_question, query_count)

    def _process_index(self, index_name, style_question, query_count):
        try:
            with open(self.load_chunks_from_index(index_name), 'r', encoding='utf-8') as f:
                chunks_json = json.load(f)

            chunks_df = pd.DataFrame({"chunks": chunks_json.values()})

            print(f"[INFO] Загружено {len(chunks_df)} чанков из индекса {index_name}.")

            questions_df = self.question_generator.generate_questions(
                chunks_df=chunks_df,
                chunk_column_name="chunks",
                style_question=style_question,
                query_count=query_count
            )

            os.makedirs(str(self.experiment_path)+'/generated_questions', exist_ok=True)
            output_dir = self.experiment_path / f"generated_questions/{style_question}"
            output_dir.mkdir(exist_ok=True)

            output_file = output_dir / f"{index_name}_generated_questions.xlsx"
            questions_df.to_excel(output_file, index=False)

            print(f"[Генерация вопросов] Вопросы для индекса {index_name} сохранены в файл {output_file}.")
        except Exception as e:
            print(f"[Генерация вопросов] Ошибка при обработке индекса {index_name}: {e}")


def gen_closed_query(chunk_text):
  CLOSED_QUERY_PROMPT = f"""
  Вы являетесь экспертом в создании точных вопросов, основанных на заданном текстовом фрагменте. Ваша задача — составить вопросы, которые требуют указания точных чисел или фактов, явно упомянутых в предоставленном тексте. 
  Эти вопросы должны быть направлены на числовые данные, содержащиеся в тексте.

  Инструкции:
  Прочитайте текст:

  Тщательно проанализируйте данный текстовый фрагмент, чтобы выявить ключевые числа, факты или явные детали.
  Сформулируйте вопросы:

  Составьте один вопрос, которы1 специально запрашивают точное число, факт или деталь из текста.
  Убедитесь, что вопрос нельзя правильно ответить, если не знать информацию, содержащуюся в тексте.
  Форматирование:

  Начинайте каждый вопрос с "Q:" и текста вопроса.
  Завершайте "Ответ:" и указывайте правильный ответ.
  Фокус вопросов:

  Числовые данные (например, даты, измерения, статистика).
  Явно указанные факты (например, имена, локации, описания).
  Избегайте субъективных или интерпретируемых вопросов.
  Пример ввода:
  "Эйфелева башня, расположенная в Париже, была построена в 1889 году и имеет высоту 324 метра. Она принимает более 7 миллионов посетителей ежегодно, что делает её одной из самых популярных туристических достопримечательностей мира."

  Пример вывода:
  Q: В каком году была построена Эйфелева башня?
  Ответ: 1889

  Q: Какова высота Эйфелевой башни?
  Ответ: 324 метра

  Q: Сколько посетителей Эйфелева башня принимает ежегодно?
  Ответ: 7 миллионов

  Q: В каком городе находится Эйфелева башня?
  Ответ: Париж

  Проанализируйте предоставленный ниже текстовый фрагмент и создайте один точный вопрос к числовым данным, основанный на фактах, с правильными ответами.

  Пример ввода:
  {chunk_text}
  """
  return CLOSED_QUERY_PROMPT


def gen_mmlu_query(chunk_text):
    MMLU_QUERY_GEN = f"""
  Вы эксперт в создании вопросов с несколькими вариантами ответов на основе заданного текста. Ваша задача — проанализировать предоставленный текст с фактами и создать ясный, фактический и сложные  вопросы в стиле MMLU (Massive Multitask Language Understanding) по данному чанку текста.
  
  Внимательно прочитайте входной текст:
  Создайте один вопрос с несколькими вариантами ответа для каждого факта или концепции в чанке текста.
  Убедитесь, что вопросы ясны, релевантны и недвусмысленны.
  Варианты ответов:

  Предоставьте только четыре варианта ответа (A, B, C, D) для каждого вопроса.
  Только один вариант должен быть правильным. Остальные варианты должны быть правдоподобными, но неверными (отвлекающими).
  Форматирование:

  Начинайте каждый вопрос с "Q:" и текста вопроса.
  Перечисляйте варианты ответов на отдельных строках, обозначая их как "A:", "B:", "C:", "D:".   Вариантов ответа должно быть ровно 4. .
  Укажите правильный ответ с помощью "Ответ: [Правильный вариант]".
  Сгенерируйте только один вопрос и ответ к нему не добавляя чанк текста в вывод.
  
  Типы вопросов:

  Сосредоточьтесь на фактической информации, понимании и умозаключениях.
  Принимай во внимание числовую и фактическую информацию.   
  
  ПРИМЕР1
  Чанк текста: "Тихий океан — самый большой и глубокий океан в мире, занимающий более 63 миллионов квадратных миль. Он граничит с пятью континентами и содержит тысячи островов."

  Вопрос: Q: Какой океан является самым большим и глубоким в мире?
  A: Атлантический океан
  B: Индийский океан
  C: Северный Ледовитый океан
  D: Тихий океан
  Ответ: D

  ПРИМЕР 2
  Q: Примерно сколько квадратных миль занимает Тихий океан?
  A: 20 миллионов
  B: 40 миллионов
  C: 63 миллиона
  D: 80 миллионов
  Ответ: C

  Строго соблюдайте: формат вопроса
  **Формат вопроса который должен соблюдаться**
  Q: Вопрос
  
  A: Вариант ответа 1
  B: Вариант ответа 2
  C: Вариант ответа 3
  D: Вариант ответа 4
  Ответ: (Буква A или B или C  или D)
  Вариантов ответа должно быть ровно 4. 
  
  Проанализируйте следующий текст и создайте только один вопрос в стиле MMLU с вариантами ответов:
  
  Чанк текста: {chunk_text}
  """
    return MMLU_QUERY_GEN


GEN_QUERY_PROMPT_LONG = f"""
Ты - начинающий работник банка. Тебе дан текст отрывка банковского документа.
Придумай короткий вопрос на который можно ответить используя этот текст.
Вопрос должен соответствовать контексту текста.
Выводи только придуманный вопрос и ничего более. 

Примечание: Вопросы 
Примеры:
1. Как начисляются бонусы по карте?
2. Какая история по кредитной карте Сбера?
3. Как оформить программу ПДС?
4. Условия и ставки по кредитной карте в Сбере? 
5. Проценты по карте? 
6. Как работает льготный период? 
7. Есть ли комиссия за обслуживание карты тинькофф? 
8. Как работает молодежная карта? 
9. Какая ставка на снятие будет по кредитной карте Сбера?
10. Какие проценты начисляются за снятие?

[Начало текста]

[Конец текста]
Вопрос:
"""


def gen_query_prompt(chunk):
    GEN_QUERY_PROMPT = f"""Ты - эксперт который генерирует вопросы по банковским документам.
    Тебе дан отрывок текста по банковскому продукту. 
    Сгенерируй по  ключевой информации и фактов из этого отрывка текста 10 вопросов (каждый вопрос должен быть до 5 слов) которые соответствуют сдержанию текста.
    
    **Требования к сгенерированным вопросам **
        1. Вопрос должен быть до  5  слов. 
        2. Вопрос должен спрашивать ключевую информацию из документа.
        3. Вопрос должен быть не слишком подробным но в тоже время соответствовать содержанию документа (как-будто его задал  пользователь который пришел в банк)
        4. Вопросы должны быть разнообразными и по разному сформулированы  
        

    **Примеры вопросов которые задают пользователи**
        снятие залога с земельного участка после погашения ипотеки?
        можно ли кредитовать если в смете не прописана информация об инженерных коммуникациях?
        Что за скидка на строительство?
        Диапазон ставок по средневзвешанной ставке по ипотеке для семей с детьми?
        к какому региону относится г Тамбов?
        какие объекты по неипотечной сделке?
        зачем мне аккредитив?
        горячая линия домклик?
        В недофинансировании можно принимать доверенность от продавца?
        Возможно купить дом на вторичном рынке от физического лица в днр?
        Как получить заёмщику второй транш по ипотеки?
        Можно ли передать первоначальный взнос по семейной ипотеки до сделки?
        При какой программе кредитования есть сикдка 03 на ставку при РиР?
        заклданая продана залоговый объект иного банка?
        какой первоначальный взнос по семейной ипотеке при покупке дома с земельным участком от застройщика индивидуального предпринимателя?
    

    ДОКУМЕНТ:  {chunk}
    Вопрос:
    Вопросы должны быть разнообразные по формулировке.
    """
    return GEN_QUERY_PROMPT


def gen_query_long(chunk):
    S_GEN_PROMPT = f"""Ты - эксперт по задаче вопросов по текстовому отрывку. Задвавай простые и короткие вопросы по заданному тексту. 
    Тебе дан чанк ответа , сформулируй по нему вопрос который соответствует контексту данного чанка так, чтобы из этого чанка можно было найти ответ на этот вопрос.
    Задавай краткий вопрос который соответствует чанку по контексту.
    Начинай вопрос с разных слов если они подходят по контексту например: Почему? Как? Могу ли? Вопросы лолжны быть разнообразны по формулировке.

    **ПРИМЕР 1**
    Чанк: У ВТБ кешбэк в рекламе  до 25%, а в реальности 25% на покупки только в первые 30 дней. Максимальный размер кешбэка в месяц за покупки в категориях – 3000 рублей. В дальнейшем клиент получает кешбэк 2% в категориях «Транспорт и такси», «Кафе и рестораны».
    Вопрос: Какой кешбэк в банке ВТБ? 

    **ПРИМЕР 2**
    Чанк: В СберБанке клиент может оформить Кредитную СберКарту с вечным бесплатным обслуживанием и беспроцентным периодом до 120 дней. Кредитная СберКарта выпускается на пластике международной платежной системы МИР. 
    Вопрос: Какой беспроцетный период по кредитной СберКарте? 

    **ПРИМЕР 3**
    Чанк: Для блокировки карты Сбера есть несколько вариантов. Наиболее быстрый – зайти в мобильное приложение Сбербанк Онлайн, выбрать нужную карту, затем в настройках выбрать пункт меню «Заблокировать» и следовать дальнейшим инструкциям.
    Вопрос: Как заблокировать кредитную карту? 

    Сгенерируй вопрос который соответствует контексту чанка.
    Чанк: {chunk}
    Вопрос:"""
    return S_GEN_PROMPT