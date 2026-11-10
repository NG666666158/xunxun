import pytest
from fastapi import HTTPException

from api import main


class ReadyStore:
    backend = "embedded"

    def list_documents(self):
        return []


class ReadySystem:
    initialized = True
    store = ReadyStore()

    def initialize(self):
        return self


class FailingSystem:
    initialized = False

    def initialize(self):
        raise ConnectionError("internal dependency detail")


def test_liveness_does_not_initialize_dependencies():
    assert main.health_live()["status"] == "alive"


def test_readiness_checks_initialized_store(monkeypatch):
    monkeypatch.setattr(main, "system", ReadySystem())
    payload = main.health_ready()

    assert payload["status"] == "ready"
    assert payload["checks"] == {
        "vector_store": "ok",
        "vector_backend": "embedded",
        "model_api": "configured",
    }


def test_readiness_reports_milvus_backend(monkeypatch):
    class MilvusReadyStore(ReadyStore):
        backend = "milvus"

    class MilvusReadySystem(ReadySystem):
        store = MilvusReadyStore()

    monkeypatch.setattr(main, "system", MilvusReadySystem())
    assert main.health_ready()["checks"]["vector_backend"] == "milvus"


def test_readiness_fails_closed_without_leaking_internal_error(monkeypatch):
    monkeypatch.setattr(main, "system", FailingSystem())

    with pytest.raises(HTTPException) as error:
        main.health_ready()

    assert error.value.status_code == 503
    assert "internal dependency detail" not in error.value.detail
