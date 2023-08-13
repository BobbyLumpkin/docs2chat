"""
Purpose: Utilities for apps subpackage of docs2chat.
"""


from docs2chat.config import Config
from docs2chat.chat import get_conversation_chain
from docs2chat.extract import ExtractivePipeline


class ChainFactory:

    def __new__(
        self,
        chain_type: str,
        docs_dir: str,
        config_obj: Config = None,
        num_return_docs: int = None,
        return_threshold: float = None
    ):
        if chain_type == "generative":
            if config_obj is None:
                raise ValueError(
                    "When `chain_type` is `generative` "
                    "a config_obj must be provided!"
                )
            chain = get_conversation_chain(
                docs_dir=docs_dir,
                config_obj=config_obj
            )
            format_func = format_conversation_chain_output
        elif chain_type == "extractive":
            for kwarg in [num_return_docs, return_threshold]:
                if kwarg is None:
                    raise ValueError(
                        "When `chain_type` is extractive "
                        f"`{kwarg}` must be provided!"
                    )
            chain = ExtractivePipeline(
                content=docs_dir,
                num_return_docs=num_return_docs,
                return_threshold=return_threshold
            )
            format_func = format_extractive_pipeline_output
        return (chain, format_func)


def format_conversation_chain_output(output):
    sources = list({
        source_doc.metadata["source"]
        for source_doc in output["source_documents"]
    })
    print(
        f"\nAI Answer: {output['answer']}"
        f"\nAI Answer Sources: {sources}"
    )
    return


def format_extractive_pipeline_output(output):
    for idx, tup in enumerate(output):
        print(
            f"\nDocument {idx + 1} -- Score: {1 - tup[1]}\n"
            f"Content: {tup[0].page_content}\n"
            f"Source: {tup[0].metadata['source']}"
        )
    return


def load_bool(value):
    if value.lower() == "true":
        return True
    return False


def load_none_or_str(value):
    if value == "None":
        return None
    return value