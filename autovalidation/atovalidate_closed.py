import re
import os
import pandas as pd

def process_closed_questions(folder_path):
    all_data = []

    # Регулярное выражение для разделения вопроса и ответа
    closed_pattern = re.compile(r"""
        Q:\s*(.*?)\n         # Вопрос после Q:
        Ответ:\s*(.*)          # Ответ после Ответ:
    """, re.VERBOSE | re.DOTALL)

    # Перебираем все файлы в указанной папке
    for filename in os.listdir(folder_path):
        if filename.endswith(".xlsx"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_excel(file_path)  # Читаем файл Excel

            if 'query' in df.columns:
                cleaned_data = df['query'].apply(lambda x: closed_pattern.search(x) if isinstance(x, str) else None)
                df['query'] = cleaned_data.apply(lambda x: x.group(1).strip() if x else None)
                df['ground_truth_answer'] = cleaned_data.apply(lambda x: x.group(2).strip() if x else None)

                # Добавляем только строки с валидными данными
                valid_rows = df.dropna(subset=['query', 'ground_truth_answer'])[['query', 'ground_truth_answer']]
                all_data.append(valid_rows)

    # Объединяем все данные в один DataFrame
    closed_questions = pd.concat(all_data, ignore_index=True)
    return closed_questions