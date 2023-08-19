"""
Purpose: Extractive QA functionality.
"""


from dataclasses import dataclass, field, InitVar
from haystack.nodes import (
    EmbeddingRetriever,
    FARMReader,
    SentenceTransformersRanker
)
from haystack.pipelines import ExtractiveQAPipeline, Pipeline
from langchain.docstore.document import Document
import logging
import math
import sys
from typing import Literal, Optional, Union


from docs2chat.config import config
from docs2chat.preprocessing import PreProcessor
from docs2chat.extract.utils import (
    _RankerReaderProtocol,
    _HaystackPipelineProtocol,
    _HaystackRetrieverProtocol,
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
class SnipExtractivePipeline:

    content: InitVar[Optional[str]] = field(default=None)
    hs_pipeline: Optional[_HaystackPipelineProtocol] = field(default=None)
    preprocessor: Optional[PreProcessor] = field(default=None)
    num_return_docs: int = field(default=4)
    reader: Optional[_RankerReaderProtocol] = field(default=None)
    retriever: Optional[_HaystackRetrieverProtocol] = field(default=None)
    return_threshold: float = field(default=0)

    def __post_init__(self, content):
        if self.hs_pipeline is None:
            if self.preprocessor is None:
                _logger.info(
                    "PreProcessor was not passed. "
                    "Initializing a PreProcessor object."
                )
                preprocessor = PreProcessor(
                    chain_type="snip",
                    content=content
                )
                setattr(self, "preprocessor", preprocessor)
            if not hasattr(self.preprocessor, "vectorstore"):
                _logger.info(
                    "No vectorstore detected."
                )
                self.preprocessor.preprocess(
                    return_vectorstore=False,
                    show_progress=False,
                    store_vectorstore=True
                )
            if self.retriever is None:
                _logger.info(
                    "Generating a HS Retriever."
                )
                retriever = EmbeddingRetriever(
                    document_store=self.preprocessor.vectorstore,
                    embedding_model=config.EMBEDDING_DIR
                )
                setattr(self, "retriever", retriever)
                self.preprocessor.vectorstore.update_embeddings(retriever)
            if self.reader is None:
                _logger.info(
                    "Generating a HS Reader."
                )
                reader = FARMReader(
                    model_name_or_path=config.HS_READER_DIR,
                    use_gpu=False
                )
                setattr(self, "reader", reader)
            _logger.info(
                "Constructing snip pipeline."
            )
            hs_pipeline = ExtractiveQAPipeline(reader, retriever)
            setattr(self, "hs_pipeline", hs_pipeline)
    
    def __call__(self, query: str):
        return self.run(query=query)
    
    def run(self, query: str) -> tuple[Document, float]:
        results = self.hs_pipeline.run(
            query=query,
            params={
                "Retriever": {
                    "top_k": min(100, math.floor((1.5 * self.num_return_docs)))
                },
                "Reader": {"top_k": self.num_return_docs}
            }
        )
        return [
            result for result in results["answers"]
            if result.score >= self.return_threshold
        ]


@dataclass
class SearchExtractivePipeline:
    
    content: InitVar[Optional[str]] = field(default=None)
    hs_pipeline: Optional[_HaystackPipelineProtocol] = field(default=None)
    preprocessor: Optional[PreProcessor] = field(default=None)
    num_return_docs: int = field(default=4)
    ranker: Optional[_RankerReaderProtocol] = field(default=None)
    retriever: Optional[_HaystackRetrieverProtocol] = field(default=None)
    return_threshold: float = field(default=0)

    def __post_init__(self, content):
        if self.hs_pipeline is None:
            if self.preprocessor is None:
                _logger.info(
                    "PreProcessor was not passed. "
                    "Initializing a PreProcessor object."
                )
                preprocessor = PreProcessor(
                    chain_type="search",
                    content=content
                )
                setattr(self, "preprocessor", preprocessor)
            if not hasattr(self.preprocessor, "vectorstore"):
                _logger.info(
                    "No vectorstore detected."
                )
                self.preprocessor.preprocess(
                    return_vectorstore=False,
                    show_progress=False,
                    store_vectorstore=True
                )
            if self.retriever is None:
                _logger.info(
                    "Generating a HS Retriever."
                )
                retriever = EmbeddingRetriever(
                    document_store=self.preprocessor.vectorstore,
                    embedding_model=config.EMBEDDING_DIR
                )
                setattr(self, "retriever", retriever)
                self.preprocessor.vectorstore.update_embeddings(retriever)
            if self.ranker is None:
                _logger.info(
                    "Generating a HS Ranker."
                )
                ranker = SentenceTransformersRanker(
                    model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2",
                    use_gpu=False
                )
                setattr(self, "ranker", ranker)
            _logger.info(
                "Constructing search pipeline."
            )
            hs_pipeline = Pipeline()
            hs_pipeline.add_node(
                component=self.retriever, name="Retriever", inputs=["Query"])
            hs_pipeline.add_node(
                component=self.ranker, name="Ranker", inputs=["Retriever"])
            setattr(self, "hs_pipeline", hs_pipeline)
    
    def __call__(self, query: str):
        return self.run(query=query)
    
    def run(self, query: str) -> tuple[Document, float]:
        results = self.hs_pipeline.run(
            query=query,
            params={
                "Retriever": {
                    "top_k": min(100, math.floor((1.5 * self.num_return_docs)))
                },
                "Ranker": {"top_k": self.num_return_docs}
            }
        )
        return [
            result for result in results["documents"]
            if result.score >= self.return_threshold
        ]


class ExtractivePipeline:

    extractive_pipeline_dict = {
        "search": SearchExtractivePipeline,
        "snip": SnipExtractivePipeline
    }

    def __new__(
        cls,
        chain_type: Literal["search", "snip"] = "snip",
        **kwargs
    ):
        extractive_pipeline_cls = cls.extractive_pipeline_dict[chain_type]
        return extractive_pipeline_cls(**kwargs)