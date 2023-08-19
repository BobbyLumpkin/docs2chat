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
        elif chain_type in ["search", "snip"]:
            for kwarg in [num_return_docs, return_threshold]:
                if kwarg is None:
                    raise ValueError(
                        "When `chain_type` is extractive "
                        f"`{kwarg}` must be provided!"
                    )
            chain = ExtractivePipeline(
                chain_type=chain_type,
                content=docs_dir,
                num_return_docs=num_return_docs,
                return_threshold=return_threshold
            )
        else:
            raise ValueError(
                "`chain_type` must be one of 'generative', "
                "'search' or 'snip."
            )
        format_func = FORMAT_FUNC_FACTORY[chain_type]
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


def format_search_pipeline_output(output):
    for idx, doc in enumerate(output):
        print(
            f"\nDocument {idx + 1} -- Score: {doc.score}\n"
            f"Content: {doc.content}\n"
            f"Source: {doc.meta['source']}"
        )
    return


def format_snip_pipeline_output(output):
    for idx, doc in enumerate(output):
        print(
            f"\nSnippet: {idx + 1} -- Score: {doc.score}\n"
            f"Content: {doc.answer}\n"
            f"content: {doc.context}\n"
            f"Source: {doc.meta['source']}"
        )
    return


FORMAT_FUNC_FACTORY = {
    "generative": format_conversation_chain_output,
    "search": format_search_pipeline_output,
    "snip": format_snip_pipeline_output
}


def load_bool(value):
    if value.lower() == "true":
        return True
    return False


def load_none_or_str(value):
    if value == "None":
        return None
    return value