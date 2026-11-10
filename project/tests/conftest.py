import sys
from pathlib import Path

import keyring
from keyring.backend import KeyringBackend
from keyring.errors import PasswordDeleteError
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


class MemoryKeyring(KeyringBackend):
    priority = 1

    def __init__(self):
        super().__init__()
        self.passwords: dict[tuple[str, str], str] = {}

    def get_password(self, service, username):
        return self.passwords.get((service, username))

    def set_password(self, service, username, password):
        self.passwords[(service, username)] = password

    def delete_password(self, service, username):
        if self.passwords.pop((service, username), None) is None:
            raise PasswordDeleteError("not found")


# Never touch the developer's real Windows Credential Manager from tests, including at
# import time (api.main builds settings when imported).
keyring.set_keyring(MemoryKeyring())


@pytest.fixture(autouse=True)
def memory_keyring():
    backend = MemoryKeyring()
    keyring.set_keyring(backend)
    return backend
