import asyncio
from types import SimpleNamespace

import pytest  # type: ignore[import]
from fastapi import HTTPException

import src.server.main as server_main
from src.server.models.registration import ModelUnloadConfig


def test_unload_model_missing_model_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_main, "_registry", SimpleNamespace(register_unload=lambda _name: asyncio.sleep(0, result=False)))

    async def _run() -> None:
        with pytest.raises(HTTPException) as exc_info:
            await server_main.unload_model(ModelUnloadConfig(model_name="missing-model"))

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Model 'missing-model' not found"

    asyncio.run(_run())


def test_unload_model_preserves_http_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _raise_http_exception(_model_name: str) -> bool:
        raise HTTPException(status_code=404, detail="not found from registry")

    monkeypatch.setattr(server_main, "_registry", SimpleNamespace(register_unload=_raise_http_exception))

    async def _run() -> None:
        with pytest.raises(HTTPException) as exc_info:
            await server_main.unload_model(ModelUnloadConfig(model_name="missing-model"))

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "not found from registry"

    asyncio.run(_run())
