"""
Purpose: Utilities for apps subpackage of docs2chat.
"""


def load_bool(value):
    if value.lower() == "true":
        return True
    return False


def load_none_or_str(value):
    if value == "None":
        return None
    return value