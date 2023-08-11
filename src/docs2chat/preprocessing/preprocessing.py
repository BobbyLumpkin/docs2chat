"""
Langchain LLM pipeline for generative QA pipeline.
"""


from dataclasses import dataclass, field, InitVar
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
import logging
from pathlib import Path
import sys
from typing import Iterable, Optional, Union


from docs2chat.config import Config, config
from docs2chat.preprocessing.utils import (
    create_vectorstore,
    load_and_split_from_dir,
    load_and_split_from_str,
    _EmbeddingsProtocol,
    _RetrieverProtocol,
    _TextSplitterProtocol
)


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_formatter = logging.Formatter(
    "%(asctime)s:%(levelname)s:%(module)s: %(message)s"
)
_console_handler = logging.StreamHandler(sys.stdout)
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
    embeddings: Optional[_EmbeddingsProtocol] = field(default=None)
    load_from_type: str = field(default="dir")
    text_splitter: Optional[_TextSplitterProtocol] = field(default=None)
    
    def __post_init__(self):
        if self.load_from_type not in ["text", "dir"]:
            raise ValueError(
                "`load_from_type` must be one of `text` or `dir`."
            )
        if self.text_splitter is None:
            _logger.info(
                "Generating text splitter."
            )
            text_splitter = CharacterTextSplitter(        
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            setattr(self, "text_splitter", text_splitter)
        if self.embeddings is None:
            _logger.info(
                f"Loading embedding model from {config.EMBEDDING_DIR}."
            )
            embeddings = HuggingFaceInstructEmbeddings(
                model_name=config.EMBEDDING_DIR
            )
            setattr(self, "embeddings", embeddings)
    
    def load_and_split(self, show_progress=True, store=False):
        load_func = PreProcessor.LOADER_FACTORY[self.load_from_type]
        docs = load_func(
            content=self.content,
            text_splitter=self.text_splitter,
            show_progress=show_progress
        )
        if store:
            setattr(self, "docs", docs)
        return docs
    
    def create_vectorstore(self, docs, store=False):
        vectorstore = create_vectorstore(
            docs=docs,
            embeddings=self.embeddings
        )
        if store:
            setattr(self, "vectorstore", vectorstore)
        return vectorstore

    def preprocess(
        self,
        show_progress,
        store_docs: bool = False,
        store_vectorstore: bool = False
    ):
        _logger.info(
            "Loading documents into vectorstore. This may take a few minutes ..."
        )
        docs = self.load_and_split(
            show_progress=show_progress,
            store=store_docs
        )
        vectorstore = self.create_vectorstore(
            docs=docs,
            store=store_vectorstore
        )
        return vectorstore


def get_conversation_chain(
    config_obj: Config,
    docs_dir: str
) -> ConversationalRetrievalChain:
    preprocessor = PreProcessor(content=docs_dir)
    vectorstore = preprocessor.preprocess(show_progress=False)
    _logger.info(
        f"Loading LLM from {config_obj.MODEL_PATH}."
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    llm = LlamaCpp(
        model_path=config_obj.MODEL_PATH,
        n_gpu_layers=40,
        n_ctx=2048,
        n_batch=512,
        input={"temperature": 0.75, "max_length": 2000, "top_p": 1},
        verbose=False
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory, 
        return_source_documents=True
    )