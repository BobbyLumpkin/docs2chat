"""
Purpose: Utilities for extract subpackage of docs2chat.
"""


from typing import Protocol


class _HaystackRetrieverProtocol(Protocol):

    def retrieve():
        ...
    
    def retrieve_batch():
        ...


class _RankerReaderProtocol(Protocol):

    def predict():
        ...
    
    def predict_batch():
        ...
