from dataclasses import dataclass
from typing import Any

import pytest  # type: ignore[import]
from fastapi import HTTPException

import src.server.main as server_main
from src.server.model_registry import ModelRecord
from src.server.models.registration import ModelType


@dataclass
class _DummyUploadFile:
    content: bytes

    async def read(self) -> bytes:
        return self.content


class _Workers:
    async def transcribe_qwen3_asr(self, model_name: str, generation_config: Any) -> dict[str, Any]:
        return {"text": f"qwen:{model_name}", "metrics": {"input_token": 1}}

    async def transcribe_whisper(self, model_name: str, generation_config: Any) -> dict[str, Any]:
        return {"text": f"whisper:{model_name}", "metrics": {"input_token": 1}}


def _set_registry_model(model_name: str, model_type: ModelType, monkeypatch: pytest.MonkeyPatch) -> None:
    record = ModelRecord(model_name=model_name, model_type=model_type.value)

    class _Registry:
        def __init__(self) -> None:
            self._lock = server_main.asyncio.Lock()
            self._models = {"dummy": record}

    monkeypatch.setattr(server_main, "_registry", _Registry())


@pytest.mark.asyncio
async def test_openai_audio_transcriptions_rejects_unsupported_model_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_registry_model("demo-llm", ModelType.LLM, monkeypatch)
    monkeypatch.setattr(server_main, "_workers", _Workers())

    with pytest.raises(HTTPException) as exc_info:
        await server_main.openai_audio_transcriptions(
            file=_DummyUploadFile(b"audio"),
            model="demo-llm",
            response_format="json",
            openarc_asr=None,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Model type llm does not support transcription"


@pytest.mark.asyncio
async def test_openai_audio_transcriptions_only_requires_openarc_asr_for_qwen3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_registry_model("demo-whisper", ModelType.WHISPER, monkeypatch)
    monkeypatch.setattr(server_main, "_workers", _Workers())

    response = await server_main.openai_audio_transcriptions(
        file=_DummyUploadFile(b"audio"),
        model="demo-whisper",
        response_format="json",
        openarc_asr=None,
    )

    assert response == {"text": "whisper:demo-whisper"}


@pytest.mark.asyncio
async def test_openai_audio_transcriptions_requires_openarc_asr_for_qwen3_asr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_registry_model("demo-qwen3", ModelType.QWEN3_ASR, monkeypatch)
    monkeypatch.setattr(server_main, "_workers", _Workers())

    with pytest.raises(HTTPException) as exc_info:
        await server_main.openai_audio_transcriptions(
            file=_DummyUploadFile(b"audio"),
            model="demo-qwen3",
            response_format="json",
            openarc_asr=None,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "openarc_asr required for Qwen3 ASR models"
