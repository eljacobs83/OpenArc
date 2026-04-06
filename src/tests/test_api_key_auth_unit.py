import asyncio

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

import src.server.main as server_main


def test_verify_api_key_allows_requests_when_api_key_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_main, "API_KEY", "")

    result = asyncio.run(server_main.verify_api_key(credentials=None))

    assert result is None


def test_verify_api_key_rejects_missing_bearer_when_api_key_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_main, "API_KEY", "secret-key")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(server_main.verify_api_key(credentials=None))

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Unauthorized"
    assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}


def test_verify_api_key_rejects_invalid_bearer_when_api_key_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_main, "API_KEY", "secret-key")
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="wrong")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(server_main.verify_api_key(credentials=credentials))

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Unauthorized"
    assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}


def test_verify_api_key_accepts_valid_bearer_when_api_key_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_main, "API_KEY", "secret-key")
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="secret-key")

    result = asyncio.run(server_main.verify_api_key(credentials=credentials))

    assert result == "secret-key"
