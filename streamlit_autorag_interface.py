import streamlit as st
import os
import tempfile
import hashlib
import pandas as pd
from expirement_runner import ExperimentManager
import re
import streamlit as st
from pathlib import Path
import pandas as pd
from gigachat_api.gigachat_api_call import giga_chat_call
import streamlit as st
from pathlib import Path
import pandas as pd
import os
from gigachat_api.gigachat_api_call import giga_chat_call
import streamlit as st
from pathlib import Path
import pandas as pd
import os
from gigachat_api.gigachat_api_call import giga_chat_call
def process_validation_mark(mark):
    import re

    # Извлечение числовой оценки
    score_match = re.search(r'\*\*Оценка\*\*: \[\[([0-9]+)\]\]', mark) or re.search(r'\[\[([0-9]+)\]\]', mark)
    score = int(score_match.group(1)) if score_match else None

    # Извлечение причины или обоснования
    reason_match = re.search(r'\*\*(Причина|Обоснование)\*\*: (.*?)$', mark, re.DOTALL)
    if reason_match:
        reason = reason_match.group(2).strip()
    else:
        # Альтернативный случай, когда причина/обоснование не указаны под специальным заголовком
        reason = mark.split(']],')[-1].strip() if ']],' in mark else "Обоснование отсутствует"

    return score, reason

# Инициализация базового конфига
config = {
    "word_file_path": None,
    "embedder_name": "custom",
    "top_n_chunks": 5,
    "query_excel_file": None,
    "query_column_excel_file": "query",
    "autoval_queries": False,
    "query": None,
    "long_gen_queries": False,  # Новое поле для подробных вопросов
    "generate_questions_for_files": False,
    "auto_validate_gen_query": False,
    "gen_query_count": 5,
    "cert_devices_path": "C:/Users/22060352/Desktop/dev__test/tls/tls.crt.txt",
    "key_devices_path": "C:/Users/22060352/Desktop/dev__test/tls/tls.key.txt",
    "question_answering_model_name": "GigaChat",
    "query_generation_model_name": "GigaChat-Max",
    "auto_validation_model_name": "GigaChat-Max",
}
llm = giga_chat_call(model_name='GigaChat-Max')
# Инициализация ExperimentManager
experiment_manager = ExperimentManager(config)

# Инициализация состояния сессии для сохранения переменных
if 'state' not in st.session_state:
    st.session_state.state = {
        "word_file_path": None,
        "query": None,
        "qa_answers": None,
        "validated_answers": None,
        "uploaded_file": None,
        "df_questions": None,
        "answers_df": None,
        "output_path": None,
        "generate_questions": False,  # Новое поле для галочки

    }

# Интерфейс загрузки архива
st.title("Experiment Interface")

# Загрузка архива .zip
st.header("1. Загрузка архива .zip с документами")
uploaded_file = st.file_uploader("Загрузите .zip файл", type="zip")

if uploaded_file is not None:
    # Сохранение пути к файлу в сессии и конфиге
    word_file_path = Path("temp.zip")
    with open(word_file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.session_state.state["word_file_path"] = str(word_file_path)
    config["word_file_path"] = str(word_file_path)
    st.success("Файл успешно загружен!")

# Галочка для генерации вопросов
generate_questions = st.checkbox("Сгенерировать вопросы для файлов", value=st.session_state.state.get("generate_questions", False))
st.session_state.state["generate_questions"] = generate_questions
config["generate_questions_for_files"] = generate_questions

# Галочка для генерации подробных вопросов
long_gen_queries = st.checkbox("Сгенерировать подробные вопросы", value=st.session_state.state.get("long_gen_queries", False))
st.session_state.state["long_gen_queries"] = long_gen_queries
config["long_gen_queries"] = long_gen_queries

# Кнопка для создания индекса
if uploaded_file:
    if st.button("Создать индекс"):
        experiment_manager = ExperimentManager(config)
        experiment_manager.run()
        st.success("Индекс успешно создан!")
    if config["long_gen_queries"] or config["generate_questions_for_files"]:
        st.text('Сгенерированы тестовыe вопросы в папке generated_questions ')
# Поле ввода вопроса
st.header("2. Ввод вопроса для поиска ответа")
query = st.text_input("Введите ваш вопрос", value=st.session_state.state.get("query", ""))

if query:
    st.session_state.state["query"] = query
    config["query"] = query

    if st.button("Получить ответ"):
        # Получение ответа
        qa_answers = experiment_manager.answer_user_query(query)
        st.session_state.state["qa_answers"] = qa_answers

        # Извлечение ответа и используемых чанков
        answer = qa_answers['Ответ'][0]
        used_chunks = qa_answers['Источник'][0]  # Предполагается, что это поле существует

        # Вывод результата
        st.subheader("Результат")
        st.write(f"**Вопрос:** {query}")
        st.write(f"**Ответ:** {answer}")

        # Парсинг и вывод чанков
        st.subheader("Использованные чанки")
        chunk_list = used_chunks.split("***КОНЕЦ СТАТЬИ***")  # Разделяем на отдельные чанки
        for i, chunk in enumerate(chunk_list, start=1):
            if chunk.strip():  # Пропускаем пустые строки
                st.markdown(f"**Чанк {i}:** {chunk.strip()}")

# Автоматическая валидация
if st.button("Автоматическая валидация вопроса"):
    qa_answers_path = os.path.join(experiment_manager.experiment_path, 'qa_answers.xlsx')

    if os.path.exists(qa_answers_path):
        qa_answers = pd.read_excel(qa_answers_path)

        # Устанавливаем конфигурацию для автовалидации
        config["autoval_queries"] = True
        st.text('[ONE QUERY VALIDATION]',)

        validated_answers = experiment_manager.evaluate_with_metrics_and_llm(
            questions=qa_answers['Вопрос'], answers=qa_answers['Ответ'], llm=llm
        )
        # Генерируем текст для каждого вопроса
        valid_path = os.path.join(experiment_manager.experiment_path, 'validated_answers.xlsx')

        if os.path.exists(valid_path):
            valid_answers = pd.read_excel(valid_path)

            for ind in range(len(valid_answers)):
                score, reason = process_validation_mark(valid_answers['validation_mark'][ind])

                # Если оценка ниже 4, добавляем "неправильно"
                if score is not None and score < 4:
                    score_text = f"{score} (неправильно)"
                else:
                    score_text = f"{score}"

                st.write(f"**Вопрос:** {valid_answers['query'][ind]}")
                st.write(f"**Ответ:** {valid_answers['answer'][ind]}")
                st.write(f"**Оценка:** {score_text}")
                st.write(f"**Причина:** {reason}")
                st.write("---")
    else:
        st.warning("Файл qa_answers.xlsx не найден. Пожалуйста, сгенерируйте ответы на вопросы.")

# Загрузка файла с вопросами (Excel)
uploaded_file = st.file_uploader("Загрузите файл с вопросами (Excel)", type=["xlsx"])

if uploaded_file:
    st.session_state.state["uploaded_file"] = uploaded_file
    try:
        # Читаем файл и сохраняем в состояние
        st.session_state.state["df_questions"] = pd.read_excel(uploaded_file)

        # Проверяем наличие колонки query
        if "query" not in st.session_state.state["df_questions"].columns:
            st.error("Файл должен содержать колонку 'query' с вопросами.")
        else:
            if st.button("Ответить по вопросам из файла"):
                st.session_state.state["output_path"] = os.path.join(
                    experiment_manager.experiment_path, "answers_from_file.xlsx"
                )

                # Выполняем метод answer_from_excel
                st.session_state.state["answers_df"] = experiment_manager.answer_from_excel(
                    excel_file_path=uploaded_file.name, question_column='query', top_n=5,
                )

                # Сохраняем результаты в файл
                st.session_state.state["answers_df"].to_excel(st.session_state.state["output_path"], index=False)
                st.success(f"Результаты сохранены в файл: {st.session_state.state['output_path']}")

    except Exception as e:
        st.error(f"Ошибка при обработке файла: {e}")
else:
    st.info("Пожалуйста, загрузите файл с вопросами.")

# Восстановление состояния после обновления интерфейса
if st.session_state.state["df_questions"] is not None:
    st.write("Предпросмотр загруженного файла (восстановлено):")
    st.dataframe(st.session_state.state["df_questions"].head())

if st.session_state.state["answers_df"] is not None:
    st.write("Результаты обработки (восстановлено):")
    st.dataframe(st.session_state.state["answers_df"])

st.title("Автоматическая валидация вопросов")

# Путь к файлу для валидации
experiment_path = experiment_manager.experiment_path
answered_file_path = os.path.join(experiment_path, "answers_from_file.xlsx")
validated_file_path = os.path.join(experiment_path, "validated_answers.xlsx")


# Кнопка для автоматической валидации
if st.button("Запустить автовалидацию"):
    if os.path.exists(answered_file_path):
        # Читаем файл с вопросами и ответами
        qa_answers = pd.read_excel(answered_file_path)
        st.text('[AUTOVALID PROCESSED]')
        # Выполняем автоматическую валидацию
        validated_answers = experiment_manager.evaluate_with_metrics_and_llm(
            questions=qa_answers['Вопрос'],
            answers=qa_answers['Ответ'],
            llm=llm,
        )
        st.text('[VALIDATION REPORT]')
        st.text(validated_answers)

        # Проверяем наличие файла с результатами валидации
        if os.path.exists(validated_file_path):
            validated_df = pd.read_excel(validated_file_path)

            # Добавляем новые колонки для оценки и причины
            validated_df[['Оценка', 'Причина']] = validated_df['validation_mark'].apply(
                lambda mark: pd.Series(process_validation_mark(mark))
            )

            # Выводим датафрейм с результатами
            st.write("Результаты автоматической валидации:")
            st.dataframe(validated_df[['query', 'answer', 'Оценка', 'Причина']])
        else:
            st.error("Файл validated_answers.xlsx не найден.")
    else:
        st.error(f"Файл {answered_file_path} не найден. Пожалуйста, убедитесь, что он существует.")
