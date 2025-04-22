import pandas as pd
import re
import os

import pandas as pd
import re
import os

# Регулярное выражение для проверки формата вопроса и извлечения ответа
pattern = re.compile(r"""
    (Q:.*?\n                # Вопрос начинается с Q:
    A:.*?\n                # Вариант A
    B:.*?\n                # Вариант B
    C:.*?\n                # Вариант C
    D:.*?\n)                # Вариант D
    (?:Ответ|Answer):\s*([ABCD]) # Ответ с буквой A, B, C или D
""", re.VERBOSE | re.DOTALL)

def extract_and_clean_query(query):
    match = pattern.search(query)
    if match:
        cleaned_query = match.group(1).strip()  # Оставляем только текст вопроса и вариантов
        answer = match.group(2)  # Извлекаем букву ответа
        return cleaned_query, answer
    return None, None  # Если не совпадает с форматом, возвращаем None

def clear_mmlu(folder_path):
    all_data = []

    # Перебираем все файлы в указанной папке
    for filename in os.listdir(folder_path):
        if filename.endswith(".xlsx"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_excel(file_path)  # Читаем файл Excel

            if 'query' in df.columns:
                cleaned_data = df['query'].apply(lambda x: extract_and_clean_query(x))
                df['query'] = cleaned_data.apply(lambda x: x[0])
                df['answer'] = cleaned_data.apply(lambda x: x[1])

                # Добавляем только строки с валидными ответами
                valid_rows = df.dropna(subset=['query', 'answer'])[['query', 'answer']]
                all_data.append(valid_rows)

    # Объединяем все данные в один DataFrame
    queries_to_validate = pd.concat(all_data, ignore_index=True)
    return queries_to_validate

def calculate_accuracy(df_with_answers, df_with_model):
    # Объединяем два датафрейма по колонке query
    merged_df = pd.merge(df_with_answers, df_with_model, on='query', how='inner')

    # Считаем количество совпадающих ответов
    correct_answers = (merged_df['answer'] == merged_df['model_answer']).sum()
    total_answers = len(merged_df)

    # Вычисляем точность
    accuracy = (correct_answers / total_answers) * 100 if total_answers > 0 else 0
    return accuracy

# Пример вызова функций
folder_path = "/generated_questions/mmlu"

