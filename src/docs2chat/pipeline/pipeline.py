"""
Langchain LLM pipeline for generative QA pipeline.
"""


from dataclasses import dataclass, field, InitVar
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
import logging
from pathlib import Path
from typing import Iterable, Optional, Union


from docs2chat.config import config
from docs2chat.pipeline.utils import (
    load_and_split_from_dir,
    load_and_split_from_str,
    _EmbeddingsProtocol,
    _TextSplitterProtocol
)


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_formatter = logging.Formatter(
    "%(asctime)s:%(levelname)s:%(module)s: %(message)s"
)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)
_logger.addHandler(_console_handler)


@dataclass
class PreProcessor:

    LOADER_FACTORY = {
        "text": load_and_split_from_str,
        "dir": load_and_split_from_dir
    }
    
    content: Union[str, list[str]] = field()
    docs: Optional[list] = field(default=None)
    embeddings: Optional[_EmbeddingsProtocol] = field(
        default=HuggingFaceInstructEmbeddings(
            model_name=config.EMBEDDING_DIR
        )
    )
    load_from_type: str = field(default="dir")
    text_splitter: Optional[_TextSplitterProtocol] = field(
        default=CharacterTextSplitter(        
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    )
    
    def __post_init__(self):
        if self.load_from_type not in ["text", "dir"]:
            raise ValueError(
                "`load_from_type` must be one of `text` or `dir`."
            )
    
    def load_and_split(self, text_splitter, show_progress=True, store=False):
        load_func = PreProcessor.LOADER_FACTORY[self.load_from_type]
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
