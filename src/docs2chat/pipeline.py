"""
Langchain LLM pipeline for generative QA pipeline.
"""


from dataclasses import dataclass, field, InitVar
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Iterable, Optional, Union


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_formatter = logging.Formatter(
    "%(asctime)s:%(levelname)s:%(module)s: %(message)s"
)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)
_logger.addHandler(_console_handler)


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


LOADER_FACTORY = {
    "text": load_and_split_from_str,
    "dir": load_and_split_from_dir
}


@dataclass
class PreProcessor:

    LOADER_FACTORY = {
        "text": load_and_split_from_str,
        "dir": load_and_split_from_dir
    }
    
    content: Union[str, list[str]] = field()
    docs: Optional[list] = field(default=None)
    load_from_type: str = field(default="dir")
    
    def __post_init__(self):
        if self.load_from_type not in ["text", "dir"]:
            raise ValueError(
                "`load_from_type` must be one of `text` or `dir`."
            )
    
    def load_and_split(self, text_splitter, show_progress=True, store=False):
        load_func = self.LOADER_FACTORY[self.load_from_type]
        docs = load_func(
            content=self.content,
            text_splitter=text_splitter,
            show_progress=show_progress
        )
        if store:
            setattr(self, "docs", docs)
        return docs
    
    def create_vectorstore(self, docs, embeddings, store=False):
        vectorstore = create_vectorstore(docs=docs, embeddings=embeddings)
        if store:
            setattr(self, "vectorstore", vectorstore)
        return vectorstore

    def preprocess(
        self,
        text_splitter,
        embeddings,
        show_progress,
        store_docs: bool = False,
        store_vectorstore: bool = False
    ):
        docs = self.load_and_split(
            text_splitter=text_splitter,
            show_progress=show_progress,
            store=store_docs
        )
        vectorstore = self.create_vectorstore(
            docs=docs,
            embeddings=embeddings,
            store=store_vectorstore
        )
        return vectorstore
