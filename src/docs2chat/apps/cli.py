"""
Purpose: CLI application for docs2chat.
"""


import argparse
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
import logging
import os
import readline
import sys


from docs2chat.apps.utils import load_bool, load_none_or_str
from docs2chat.config import config
from docs2chat.preprocessing import PreProcessor


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_formatter = logging.Formatter(
    "%(asctime)s:%(levelname)s:%(module)s: %(message)s"
)
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_formatter)
_logger.addHandler(_console_handler)


COLOR_RESET = "\033[0m"
GREEN = "\033[32m"
BANNER = f"""
{GREEN}
    ____  ____  ______________   ________  _____  ______
   / __ \/ __ \/ ____/ ___/__ \ / ____/ / / /   |/_  __/
  / / / / / / / /    \__ \__/ // /   / /_/ / /| | / /   
 / /_/ / /_/ / /___ ___/ / __// /___/ __  / ___ |/ /    
/_____/\____/\____//____/____/\____/_/ /_/_/  |_/_/     
                                                                     
"""


def run_cli_application(
    docs_dir: str = None,
    config_yaml: str = None
):
    if docs_dir is None:
        docs_dir = config.DOCUMENTS_DIR
    if config_yaml is not None:
        config.reset_config(config_yaml)

    print(BANNER, COLOR_RESET)
    
    preprocessor = PreProcessor(content=docs_dir)
    vectorstore = preprocessor.preprocess(show_progress=False)
    _logger.info(
        f"Loading LLM from {config.MODEL_PATH}."
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    llm = LlamaCpp(
        model_path=config.MODEL_PATH,
        n_ctx=2048,
        input={"temperature": 0.75, "max_length": 2000, "top_p": 1},
        verbose=False
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    print("\n----------Enter a Question Below----------\n")
    question = input("User Question: ")
    while question != "quit":
        response = conversation_chain({"question": question})
        print(
            "--------------"
            f"\nAI Answer: {response['answer']}\n"
            "--------------")
        question = input("User Question: ")
    print("Quitting chat. Goodbye!")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Launch docs2chat app.")

    parser.add_argument(
        "--config_yaml",
        type=load_none_or_str,
        help="Absolute path to yaml config file.",
        default="None",
        required=False
    )

    parser.add_argument(
        "--docs_dir",
        type=str,
        help="Full path to directory containing documents.",
        default=config.DOCUMENTS_DIR,
        required=False
    )

    args = parser.parse_args()

    run_cli_application(
        docs_dir=args.docs_dir,
        config_yaml=args.config_yaml
    )
