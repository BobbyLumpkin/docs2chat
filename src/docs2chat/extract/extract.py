"""
Purpose: Extractive QA functionality.
"""


from dataclasses import dataclass, field, InitVar
from langchain.docstore.document import Document
import logging
import sys
from typing import Optional, Union


from docs2chat.preprocessing import PreProcessor


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_formatter = logging.Formatter(
    "%(asctime)s:%(levelname)s:%(module)s: %(message)s"
)
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_formatter)
_logger.addHandler(_console_handler)


@dataclass
class ExtractivePipeline:
    
    preprocessor_kwargs: InitVar[dict] = field(default={})
    preprocessor: Optional[PreProcessor] = field(default=None)
    num_return_docs: int = field(default=4)
    return_threshold: float = field(default=0)

    def __post_init__(self, preprocessor_kwargs):
        if self.preprocessor is None:
            _logger.info(
                "PreProcessor was not passed. "
                "Initializing a PreProcessor object."
            )
            preprocessor = PreProcessor(**preprocessor_kwargs)
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
    
    def __call__(self, query: str):
        return self.run(query=query)
    
    def run(self, query: str) -> tuple[Document, float]:
        results = extractive_pipeline.preprocessor\
            .vectorstore.similarity_search_with_score(
                query=query,
                k=self.num_return_docs
            )
        return [
            result for result in results
            if result[1] >= self.return_threshold
        ]
