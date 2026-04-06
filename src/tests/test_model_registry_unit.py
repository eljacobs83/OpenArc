import asyncio
from types import SimpleNamespace

import pytest  # type: ignore[import]

import src.server.model_registry as registry_module
from src.server.model_registry import ModelRegistry, create_model_instance
from src.server.models.registration import (
    EngineType,
    ModelLoadConfig,
    ModelStatus,
    ModelType,
)


def _sample_load_config(name: str = "mock-model") -> ModelLoadConfig:
    return ModelLoadConfig(
        model_path="/models/mock",
        model_name=name,
        model_type=ModelType.LLM,
        engine=EngineType.OV_GENAI,
        device="CPU",
        runtime_config={},
    )


def test_register_load_sets_status_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ModelRegistry()
    load_config = _sample_load_config()

    async def _noop_unload(*_args, **_kwargs):
        return None

    dummy_model = SimpleNamespace(unload_model=_noop_unload)

    async def fake_create(config):  # type: ignore[override]
        assert config is load_config
        return dummy_model

    monkeypatch.setattr(registry_module, "create_model_instance", fake_create)

    async def _run():
        model_id = await registry.register_load(load_config)
        status = await registry.status()
        return model_id, status

    model_id, status = asyncio.run(_run())

    assert model_id
    assert status["total_loaded_models"] == 1
    entry = status["models"][0]
    assert entry["model_name"] == load_config.model_name
    assert entry["status"] == ModelStatus.LOADED.value


def test_register_load_duplicate_name_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ModelRegistry()
    load_config = _sample_load_config()

    async def _noop_unload(*_args, **_kwargs):
        return None

    dummy_model = SimpleNamespace(unload_model=_noop_unload)

    async def fake_create(config):  # type: ignore[override]
        return dummy_model

    monkeypatch.setattr(registry_module, "create_model_instance", fake_create)

    async def _run():
        await registry.register_load(load_config)
        with pytest.raises(ValueError):
            await registry.register_load(load_config)

    asyncio.run(_run())


def test_register_load_duplicate_name_concurrent_only_one_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ModelRegistry()
    load_config = _sample_load_config("concurrent-model")

    async def _noop_unload(*_args, **_kwargs):
        return None

    dummy_model = SimpleNamespace(unload_model=_noop_unload)

    async def fake_create(config):  # type: ignore[override]
        await asyncio.sleep(0.05)
        return dummy_model

    monkeypatch.setattr(registry_module, "create_model_instance", fake_create)

    async def _run():
        task_a = asyncio.create_task(registry.register_load(load_config))
        task_b = asyncio.create_task(registry.register_load(load_config))
        results = await asyncio.gather(task_a, task_b, return_exceptions=True)
        status = await registry.status()
        return results, status

    results, status = asyncio.run(_run())

    success_count = sum(1 for result in results if isinstance(result, str))
    errors = [result for result in results if isinstance(result, Exception)]

    assert success_count == 1
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    assert "already registered" in str(errors[0])
    assert status["total_loaded_models"] == 1
    assert status["models"][0]["model_name"] == load_config.model_name


def test_register_unload_invokes_model_unload(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ModelRegistry()
    load_config = _sample_load_config()

    unload_calls = []

    class DummyModel:
        async def unload_model(self):
            unload_calls.append(True)

    async def fake_create(config):  # type: ignore[override]
        return DummyModel()

    monkeypatch.setattr(registry_module, "create_model_instance", fake_create)

    async def _run():
        await registry.register_load(load_config)
        result = await registry.register_unload(load_config.model_name)
        assert result is True
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        status = await registry.status()
        return status

    status = asyncio.run(_run())

    assert unload_calls
    assert status["total_loaded_models"] == 0


def test_register_unload_unknown_model_returns_false() -> None:
    registry = ModelRegistry()

    async def _run():
        return await registry.register_unload("missing-model")

    result = asyncio.run(_run())
    assert result is False


def test_get_model_instance_returns_loaded_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ModelRegistry()
    load_config = _sample_load_config()
    dummy_model = SimpleNamespace()

    async def fake_create(config):  # type: ignore[override]
        assert config is load_config
        return dummy_model

    monkeypatch.setattr(registry_module, "create_model_instance", fake_create)

    async def _run():
        await registry.register_load(load_config)
        return await registry.get_model_instance(load_config.model_name)

    model_instance = asyncio.run(_run())
    assert model_instance is dummy_model


def test_create_model_instance_rejects_unknown_combination() -> None:
    load_config = ModelLoadConfig(
        model_path="/models/mock",
        model_name="unsupported",
        model_type=ModelType.VLM,
        engine=EngineType.OV_OPTIMUM,
        device="CPU",
        runtime_config={},
    )

    async def _run():
        with pytest.raises(ValueError) as exc:
            await create_model_instance(load_config)
        return str(exc.value)

    message = asyncio.run(_run())
    assert "not supported" in message


def test_model_class_registry_includes_qwen3_asr() -> None:
    key = (EngineType.OPENVINO, ModelType.QWEN3_ASR)
    assert registry_module.MODEL_CLASS_REGISTRY[key] == "src.engine.openvino.qwen3_asr.infer.OVQwen3ASR"
