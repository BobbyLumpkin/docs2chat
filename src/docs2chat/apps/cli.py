"""
Purpose: CLI application for docs2chat.
"""


import argparse
import logging
import os
import readline
import sys


from docs2chat.apps.utils import ChainFactory, load_bool, load_none_or_str
from docs2chat.config import config
from docs2chat.chat import get_conversation_chain


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
    chain_type: str = "generative",
    config_yaml: str = None,
    docs_dir: str = None,
    num_return_docs: int = None,
    return_threshold: float = None
):
    if docs_dir is None:
        docs_dir = config.DOCUMENTS_DIR
    if config_yaml is not None:
        config.reset_config(config_yaml)

    print(BANNER, COLOR_RESET)
    
    chain, format_func = ChainFactory(
        chain_type=chain_type,
        docs_dir=docs_dir,
        config_obj=config,
        num_return_docs=num_return_docs,
        return_threshold=return_threshold
    )

    print(f"\n----------{GREEN}Enter a Question Below{COLOR_RESET}----------{GREEN}\n")
    question = input("User Question: ")
    while question != "quit":
        response = chain(question)
        format_func(response)
        print(f"{COLOR_RESET}--------------{GREEN}")
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

    parser.add_argument(
        "--chain_type",
        type=str,
        help=(
            "What type of QA to perform. "
            "One of `extractive`, `generative`."),
        default="generative",
        required=False
    )

    parser.add_argument(
        "--num_return_docs",
        type=int,
        help=(
            "The number of documents to return "
            "(if `chain_type` is `extractive`)."),
        default=4,
        required=False
    )

    parser.add_argument(
        "--return_threshold",
        type=float,
        help=(
            "The confidence threshold in [0,1] to use as a cutoff "
            "(if `chain_type` is `extractive`.)"),
        default=0,
        required=False
    )

    args = parser.parse_args()

    run_cli_application(
        chain_type=args.chain_type,
        config_yaml=args.config_yaml,
        docs_dir=args.docs_dir,
        num_return_docs=args.num_return_docs,
        return_threshold=args.return_threshold
    )
