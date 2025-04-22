import re
from io import BytesIO

import pypandoc
import unicodedata

import numpy as np
from docx import Document
from razdel import sentenize
from typing import List, Generator, Union
from string import punctuation
from scipy.signal import argrelmax
from itertools import islice
from os.path import dirname

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


#from app.clients.embeding_model import EmbedModelV1Client

# class that detects chunks in the original document and saves the updated document
from typing.io import IO


# class that detects chunks in the original document and saves the updated document
class DetectChunks:
    def __init__(self) -> None:
        self.separator = 'START OF CHUNK'

    def extract_subarticles(self, starting_points: List[int], paragraphs: List[str], title: str):
        subarticles = []
        for idx, start in enumerate(starting_points):
            # определяем границы именно самих подстатей
            if start != starting_points[-1]:
                end = starting_points[idx + 1]
                subarticle = "\n".join([elem.text.strip() for elem in paragraphs[start:end]])
                subarticles.append("\n\n".join([title, subarticle]).strip()) if idx > 0 else subarticles.append(
                    subarticle.strip())
            elif start == starting_points[-1]:
                subarticle = "\n".join([elem.text.strip() for elem in paragraphs[start:]])
                subarticles.append("\n\n".join([title, subarticle]).strip()) if idx > 0 else subarticles.append(
                    subarticle.strip())
        return subarticles

    def detect_subarticles(self, file_stream: BytesIO, save_path=None) -> Document:
        subarticles_starts = []
        document = Document(file_stream)
        paragraphs = [elem for elem in document.paragraphs if elem.text not in ["", " "]]
        article_title = unicodedata.normalize("NFKD", paragraphs[0].text).strip()
        for i, paragraph in enumerate(paragraphs):
            sentences = [elem.text for elem in list(sentenize(unicodedata.normalize("NFKD", paragraph.text)))]
            pieces = [run for run in paragraph.runs if re.match(r'\S+', run.text.strip())]
            if len(pieces) > 0:
                paragraph_start = pieces[0]
                if i == 0:
                    subarticles_starts.append(i)
                    paragraph.insert_paragraph_before(text=self.separator)
                elif i < len(paragraphs) - 1:
                    # предыдущий элемент
                    prev_node = i - 1
                    prev_node_runs = [run for run in paragraphs[prev_node].runs if re.match(r'\S+', run.text.strip())]
                    # условие на выполнение: предыдущий paragraph не должен состоять только из одного предложения
                    prev_sentences = [elem.text for elem in
                                      list(sentenize(unicodedata.normalize("NFKD", paragraphs[prev_node].text)))]
                    # следующий элемент
                    next_node = i + 1
                    next_node_runs = [run for run in paragraphs[next_node].runs if re.match(r'\S+', run.text.strip())]
                    next_sentences = [elem.text for elem in
                                      list(sentenize(unicodedata.normalize("NFKD", paragraphs[next_node].text)))]
                    if paragraph.alignment == 1 and len(sentences) == 1 and paragraphs[prev_node].alignment != 1:
                        subarticles_starts.append(i)
                        paragraph.insert_paragraph_before(text=self.separator)
                    elif paragraph_start.bold or paragraph_start.italic:
                        if len(sentences) > 1 and len(pieces) > 1 and len(prev_sentences) > 1:
                            subarticles_starts.append(i)
                            paragraph.insert_paragraph_before(text=self.separator)

        # now we extract subarticles - everything which is enclosed into the limits
        subarticles = self.extract_subarticles(subarticles_starts, paragraphs, article_title)

        # saving updated document if needed
        if save_path:
            document.save(save_path)
        return document


# class that makes lists-aware segmentation based on html
class ListAwareSegmentation:
    def __init__(self) -> None:
        self.separator = 'START OF CHUNK'

    def check_inside_items(self, el, is_last=False):
        sentences = [elem.text for elem in list(sentenize(el))]
        if len(sentences) >= 1 and sentences[0] != '':
            for j in range(len(sentences)):
                if sentences[j][-1] in punctuation and len(sentences[j]) > 5:
                    sentences[j] = sentences[j][:-1] + ','
            el = ' '.join(sentences)
            if el.strip() == '':
                return el
            if el[-1] in punctuation and not is_last:
                el = el[:-1] + ';'
            elif el[-1] in punctuation and is_last:
                el = el[:-1] + '.'
            elif not is_last:
                el += ';'
            elif is_last:
                el += '.'

        return el

    def fix_sentences(self, soup):
        output = []
        prev_in_list = False

        for element in soup:
            if element.name in ['p', 'blockquote', 'h1', 'h2', 'h3', 'h4']:
                count = 0
                text = element.get_text().strip()
                text = text.replace("\r\n", "").replace("\n", " ")
                if text != '':
                    if text[0].isdigit() or text[0] in punctuation:
                        text = self.check_inside_items(text, True)

                        if output[-1][:-1] != ':' and not prev_in_list:
                            prev_in_list = True
                        elif output[-1][:-1] != ':' and prev_in_list:
                            output[-1] = output[-1][:-1] + ';'

                        output[-1] += ' ' + text
                    else:
                        # Добавляем текст элемента в выходной список
                        output.append(text)
                        prev_in_list = False

            elif element.name == 'ol' or element.name == 'ul':
                list_items = [li.get_text().strip().replace(';', ',').replace("\r\n", "").replace("\n", " ") for li in element.find_all('li')]
                for i in range(len(list_items)):
                    if i == len(list_items) - 1:
                        list_items[i] = self.check_inside_items(list_items[i], True)
                    else:
                        list_items[i] = self.check_inside_items(list_items[i])

                output[-1] += ' ' + ' '.join(list_items)
        # Объединяем список в строку с разделителем "\n"
        result_1 = '\n'.join(output)
        if self.separator in result_1 and result_1.startswith(self.separator):
            result = result_1.split(self.separator)[1:]
        else:
            result = [result_1]

        for i in range(len(result)):
            if result[i].startswith('\n'):
                result[i] = result[i].replace('\n', '', 1)

        return result

    def get_blocks(self, doc_stream: BytesIO):
        docx_to_html_data = pypandoc.convert_text(doc_stream.read(), to='html', format="docx")
        soup = BeautifulSoup(docx_to_html_data, 'html.parser')
        blocks = self.fix_sentences(soup)
        total_header = blocks[0].split('\n')[0].strip()
        new_block = []
        for i, el in enumerate(blocks):
            if el == '':
                continue
            if i > 0: 
                el = total_header + '\n\n' + el
                new_block.append(el.strip())
            elif i == 0:
                elements = [elem.strip() for elem in el.split('\n')]
                title = elements[0]
                full_text = '\n'.join(elements[1:])
                el = '\n\n'.join([title, full_text])
                new_block.append(el.strip())

        return new_block


class LMLongDocumentSplitter:
    def __init__(self,
                 backbone,
                 embed_model: Union[Q2QEmbedModel]
                 ) -> None:
        self.embed_model = embed_model
        self.embed_tokenizer = AutoTokenizer.from_pretrained(backbone)

    def window(self, seq, n=3) -> Generator:
        """https://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator
        Returns a sliding window of width n over data from the iterable seq"""
        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result

    def get_window_pairs(self,
                         text: str,
                         event_dt: str,
                         event_id: str,
                         WINDOW_SIZE: int = 3
                         ) -> Union[List[str], List[List[float]]]:
        sentences = [elem.text for elem in list(sentenize(text))]
        window_sents = list(self.window(sentences, WINDOW_SIZE))
        window_sents = [' '.join(window) for window in window_sents]

        sents_embeddings = self.embed_model.encode_batch(
            input_strings=[f"query: {sent}" for sent in window_sents], 

        )
        return sentences, sents_embeddings

    def climb(self, seq, i, mode='left') -> int:
        """Given a sequence seq of values and index i, advance the index either to the right or left while the
        value keeps increasing, then return the value at new index
        """
        if mode == 'left':
            while True:
                curr = seq[i]
                if i == 0:
                    return curr
                i = i - 1
                if not seq[i] > curr:
                    return curr
        if mode == 'right':
            while True:
                curr = seq[i]
                if i == (len(seq) - 1):
                    return curr
                i = i + 1
                if seq[i] <= curr:
                    return curr

    def get_depths(self, scores) -> np.ndarray:
        """
        Given a sequence of coherence scores of length n, compute a sequence of depth scores of similar length
        """
        depths = []
        for i in range(len(scores)):
            score = scores[i]
            l_peak = self.climb(scores, i, mode='left')
            r_peak = self.climb(scores, i, mode='right')
            depth = 0.5 * (l_peak + r_peak - (2 * score))
            depths.append(depth)
        return np.array(depths)

    def get_local_maxima(self, depth_scores, order=1) -> np.ndarray:
        """
        Given a sequence of depth scores, return a filtered sequence where only local maxima
        selected based on the given order
        """
        maxima_ids = argrelmax(depth_scores, order=order)[0]
        filtered_scores = np.zeros(len(depth_scores))
        filtered_scores[maxima_ids] = depth_scores[maxima_ids]
        return filtered_scores

    def compute_threshold(self, scores) -> float:
        """
        Based on: https://aclanthology.org/J97-1003.pdf
        Automatically compute an appropriate threshold given a sequence of depth scores
        """
        s = scores[np.nonzero(scores)]
        threshold = np.mean(s) - (np.std(s) / 2)
        return threshold

    def get_threshold_segments(self, scores, threshold=0.1) -> np.ndarray:
        """
        Given a sequence of depth scores, return indexes where the value is greater than the threshold
        """
        segment_ids = np.where(scores >= threshold)[0]
        return segment_ids

    def get_segmeted_text(self,
                          text: str,
                          event_dt: str,
                          event_id: str,
                          WINDOW_SIZE: int = 3
                          ) -> List[str]:
        sentences, sentences_embeddings = self.get_window_pairs(
            text,
            event_dt,
            event_id,
            WINDOW_SIZE
        )

        coherence_scores = [cosine_similarity([pair[0]], [pair[1]])[0][0] for pair in
                            zip(sentences_embeddings, sentences_embeddings[1:])]
        depth_scores = self.get_depths(coherence_scores)
        filtered_scores = self.get_local_maxima(depth_scores, order=1)
        threshold = self.compute_threshold(filtered_scores)
        segment_ids = self.get_threshold_segments(filtered_scores, threshold)

        segment_indices = segment_ids + WINDOW_SIZE
        segment_indices = [0] + segment_indices.tolist() + [len(sentences)]
        slices = list(zip(segment_indices[:-1], segment_indices[1:]))
        segmented = [sentences[s[0]: s[1]] for s in slices]

        return segmented

    def get_chunks_batched(self, article_chunks: List[str], event_id: str, event_dt: str) -> List[str]:
        final_blocks = []
        for chunk in article_chunks:
            header = chunk.split('\n\n')[0].strip()
            chunk_without_header = ' '.join(chunk.split('\n\n')[1:])

            # длина в токенах больше 100, но не больше 512
            if 100 < len(self.embed_tokenizer(chunk)['input_ids']) <= 512:
                final_blocks.append(chunk)
            # длина в токенах меньше 100
            elif len(self.embed_tokenizer(chunk)['input_ids']) <= 100 and len(final_blocks) > 0:
                prev_block_length = len(self.embed_tokenizer(final_blocks[-1])['input_ids'])
                prev_block_title = final_blocks[-1].split('\n\n')[0].strip()
                text_updated_length = prev_block_length + len(self.embed_tokenizer(chunk)['input_ids'])

                if header == prev_block_title and text_updated_length <= 650:
                    final_blocks[-1] += f" {chunk_without_header}"
                else:
                    final_blocks.append(chunk)
            else:
                if chunk_without_header.strip() == '':
                    continue

                segmented_text = self.get_segmeted_text(chunk_without_header, event_dt, event_id)

                for i, block in enumerate(segmented_text):
                    join_block = header + '\n' + ' '.join(block)

                    if len(final_blocks) > 0 and i > 0 and len(self.embed_tokenizer(join_block)['input_ids']) < 100:
                        prev_block_title = final_blocks[-1].split('\n\n')[0].strip()
                        prev_block_length = len(self.embed_tokenizer(final_blocks[-1])['input_ids'])
                        text_updated_length = prev_block_length + len(self.embed_tokenizer(join_block)['input_ids'])
                        if prev_block_length <= 512 and text_updated_length <= 650:
                            final_blocks[-1] += ' '.join(block)
                        else:
                            final_blocks.append(join_block)
                    else:
                        final_blocks.append(join_block)

        return final_blocks
    




