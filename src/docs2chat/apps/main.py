"""
Purpose: Launch the appropriate version of the application.
"""


import argparse


from docs2chat.apps.cli import run_cli_application
from docs2chat.config import config


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
        "--docs_dir",
        type=str,
        help="Full path to directory containing documents.",
        default=config.DOCUMENTS_DIR,
        required=False
    )

    args = parser.parse_args()
    
    print(
        "-----------------------------------------"
        "\n----------Welcome To Docs2Chat----------\n"
        "-----------------------------------------"
    )

    if args.type == "cli":
        run_cli_application(
            docs_dir=args.docs_dir,
            config_yaml=args.config_yaml
        )


def load_none_or_str(value):
    if value == "None":
        return None
    return value
    

if __name__ == "__main__":
    main()