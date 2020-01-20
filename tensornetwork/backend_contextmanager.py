from typing import Text, Union
from tensornetwork.backends.base_backend import BaseBackend

class DefaultBackend():

    with_statement = False
    backend = None

    def __init__(self, backend: Union[Text, BaseBackend]) -> None:
        DefaultBackend.backend = backend

    def __enter__(self):
        DefaultBackend.with_statement = True

    def __exit__(self, exc_type, exc_value, traceback):
        DefaultBackend.with_statement = False
        DefaultBackend.backend = None
