"""
Purpose: Utilities for pipeline module.
"""

from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import tqdm
from typing import Protocol, runtime_checkable


class _EmbeddingsProtocol(Protocol):

    def embed_documents():
        ...
    
    def embed_query():
        ...


class _TextSplitterProtocol(Protocol):

    def split_text():
        ...


def load_and_split_from_str(
    content: list[str],
    text_splitter,
    show_progress: bool = True
):
    """
    Load and split str into document objects.
    """
    if isinstance(content, str):
        content = [content]
    if isinstance(content, list):
        chunks_list = [
            text_splitter.split_text(ele)
            for ele in content
        ]
        if show_progress:
            return [
                Document(page_content=chunk, metadata={"source": "memory"})
                for chunks in tqdm(chunks_list)
                for chunk in chunks
            ]
        else:
            return [
                Document(page_content=chunk, metadata={"source": "memory"})
                for chunks in chunks_list
                for chunk in chunks
            ]
    else:
        raise TypeError("`content` must be one of `str` or `list[str]`.")


def load_and_split_from_dir(
    content: str,
    text_splitter,
    show_progress: bool = True
):
    """
    Load and split files in directory into document objects.
    """
    loader = DirectoryLoader(str(content), show_progress=True)
    return loader.load_and_split(text_splitter)


def create_vectorstore(
    docs,
    embeddings
):
    """
    Create a FAISS vectorstore 
    """
    return FAISS.from_documents(documents=docs, embedding=embeddings)