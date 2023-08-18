"""
Purpose: Extractive QA functionality.
"""


from dataclasses import dataclass, field, InitVar
from langchain.docstore.document import Document
import logging
import sys
from typing import Optional, Union


from docs2chat.preprocessing import PreProcessor
from docs2chat.extract.utils import (
    _RankerReaderProtocol,
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
class SearchExtractivePipeline:
    
    content: InitVar[Optional[str]] = field(default=None)
    preprocessor: Optional[PreProcessor] = field(default=None)
    num_return_docs: int = field(default=4)
    ranker: Optional[_RankerReaderProtocol] = field(default=None)
    # reader: Optional[_RankerReaderProtocol] = field(default=None)
    retriever: Optional[_HaystackRetrieverProtocol] = field(default=None)
    return_threshold: float = field(default=0)

    def __post_init__(self, content):
        if self.preprocessor is None:
            _logger.info(
                "PreProcessor was not passed. "
                "Initializing a PreProcessor object."
            )
            preprocessor = PreProcessor(
                chain_type="extractive",
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
                embedding_model=config.HS_EMBEDDING_DIR
            )
            setattr(self, "retriever", retriever)
            self.preprocessor.vectorstore.update_embeddings(retriever)
    
    def __call__(self, query: str):
        return self.run(query=query)
    
    def run(self, query: str) -> tuple[Document, float]:
        results = self.preprocessor\
            .vectorstore.similarity_search_with_score(
                query=query,
                k=self.num_return_docs
            )
        return [
            result for result in results
            if result[1] >= self.return_threshold
        ]
