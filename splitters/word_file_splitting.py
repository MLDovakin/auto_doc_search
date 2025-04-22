
import io
import zipfile
from zipfile import ZipFile, ZipExtFile

import uuid
from typing import Union, List
from docx import Document
from fastapi import UploadFile

#from app.clients.devices_embeddings import EmbedModelDevicesV1Client
#from app.clients.embeding_model import EmbedModelV1Client
#from app.models.articles import ArticleChunk, ArticleIn, Status
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from splitters.split_documents import DetectChunks, ListAwareSegmentation, LMLongDocumentSplitter
import datetime 


def get_plain_text(file: Union[UploadFile, zipfile.ZipExtFile]) -> str:

    if isinstance(file, zipfile.ZipFile):
        docx_filename = next(name for name in file.namelist() if name.endswith('.docx'))
        with file.open(docx_filename) as docx_file:
            docx_bytes = docx_file.read()
    else:
        docx_bytes = file.read()
        file.seek(0)

    docx = io.BytesIO(docx_bytes)
    document = Document(docx)
    
    
    paragraphs = [elem for elem in document.paragraphs]
    txt = '\n'.join([paragraph.text for paragraph in paragraphs])
    return txt


def split_docx_to_chunks(
        file: Union[UploadFile, zipfile.ZipExtFile],  event_id: str, event_dt: str
) -> List[str]:

    reader = DetectChunks()

    list_reader = ListAwareSegmentation()

    if isinstance(file, zipfile.ZipExtFile):
        docx_bytes = file.read()
    else:
        docx_bytes =  file.read()
    docx = io.BytesIO(docx_bytes)
    doc = reader.detect_subarticles(docx)
    doc_stream = io.BytesIO()
    doc.save(doc_stream)
    doc_stream.seek(0)
    # updating chunks list's aware
    all_blocks = list_reader.get_blocks(doc_stream)

    return all_blocks
