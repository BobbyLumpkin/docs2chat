"""
Purpose: Conversational (generative) QA functionality.
"""


from langchain.chains import ConversationalRetrievalChain
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import logging
import sys


from docs2chat.config import Config, config
from docs2chat.preprocessing import PreProcessor


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_formatter = logging.Formatter(
    "%(asctime)s:%(levelname)s:%(module)s: %(message)s"
)
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_formatter)
_logger.addHandler(_console_handler)


PROMPT_TEMPLATE = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
Your answer should be concise, helpful and accurate.
If you don't know the answer, just say 'I'm sorry, but I lack the information to assist with this'.
Do not try to make up an answer.

{context}

Question: {question}
Answer:"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)


def get_conversation_chain(
    docs_dir: str,
    config_obj: Config = config,
) -> ConversationalRetrievalChain:
    preprocessor = PreProcessor(chain_type="generative", content=docs_dir)
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
        n_ctx=2048,
        input={"temperature": 0.75, "max_length": 2000, "top_p": 1},
        verbose=False
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory, 
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )