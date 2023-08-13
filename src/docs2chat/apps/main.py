"""
Purpose: Launch the appropriate version of the application.
"""


import argparse
import os
from pathlib import Path
import subprocess


from docs2chat.apps.utils import load_bool, load_none_or_str
from docs2chat.config import config


APPS_PATH = Path(os.path.realpath(__file__)).parents[1].absolute() / "apps"
CLI_SCRIPT_PATH = APPS_PATH / "cli.py"


def main():
    parser = argparse.ArgumentParser(description="Launch docs2chat app.")

    parser.add_argument(
        "--type",
        type=str,
        help=(
            "One of 'cli', 'gui' or 'web'. "
            "Determines the type of app to launch."
        ),
        default="cli",
        required=False
    )

    parser.add_argument(
        "--config_yaml",
        type=load_none_or_str,
        help="Absolute path to yaml config file.",
        default="None",
        required=False
    )

    parser.add_argument(
        "--debug",
        type=load_bool,
        help="Whether or not to redirect stderr to 'dev/null'.",
        default=False,
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
    
    if args.type == "cli":
        run_kwargs = {
            "args": [
                "python3",
                str(CLI_SCRIPT_PATH),
                f"--docs_dir={args.docs_dir}",
                f"--config_yaml={args.config_yaml}",
                f"--chain_type={args.chain_type}",
                f"--num_return_docs={args.num_return_docs}",
                f"--return_threshold={args.return_threshold}"
            ]
        }
        if not args.debug:
            run_kwargs["stderr"] = subprocess.DEVNULL
        subprocess.run(**run_kwargs)
    

if __name__ == "__main__":
    main()