import asyncio

import pytest  # type: ignore[import]

import src.server.worker_registry as worker_module
from src.server.model_registry import ModelRecord, ModelRegistry
from src.server.models.registration import ModelType
from src.server.models.openvino import KokoroLanguage, KokoroVoice, OV_KokoroGenConfig, OV_Qwen3ASRGenConfig
from src.server.models.optimum import PreTrainedTokenizerConfig, RerankerConfig
from src.server.models.ov_genai import OVGenAI_GenConfig, OVGenAI_WhisperGenConfig


def _make_worker(response_value, metrics_value, supports_stream: bool = False):
    async def _worker(model_name, model_queue, model_instance, registry):
        while True:
            packet = await model_queue.get()
            if packet is None:
                break
            if getattr(packet.gen_config, "stream", False) and supports_stream and packet.stream_queue is not None:
                await packet.stream_queue.put(response_value)
                if metrics_value is not None:
                    await packet.stream_queue.put({"metrics": metrics_value})
                await packet.stream_queue.put(None)
            packet.response = response_value
            packet.metrics = metrics_value
            if packet.result_future is not None and not packet.result_future.done():
                packet.result_future.set_result(packet)
            model_queue.task_done()
    return _worker


@pytest.fixture
def worker_registry(monkeypatch: pytest.MonkeyPatch) -> worker_module.WorkerRegistry:
    class DummyLLM:  # noqa: D401
        pass

    class DummyVLM:  # noqa: D401
        pass

    class DummyWhisper:  # noqa: D401
        pass

    class DummyKokoro:  # noqa: D401
        pass

    class DummyQwen3ASR:  # noqa: D401
        pass

    class DummyEmb:  # noqa: D401
        pass

    class DummyRR:  # noqa: D401
        pass

    monkeypatch.setattr(worker_module, "OVGenAI_LLM", DummyLLM)
    monkeypatch.setattr(worker_module, "OVGenAI_VLM", DummyVLM)
    monkeypatch.setattr(worker_module, "OVGenAI_Whisper", DummyWhisper)
    monkeypatch.setattr(worker_module, "OV_Kokoro", DummyKokoro)
    monkeypatch.setattr(worker_module, "OVQwen3ASR", DummyQwen3ASR)
    monkeypatch.setattr(worker_module, "Optimum_EMB", DummyEmb)
    monkeypatch.setattr(worker_module, "Optimum_RR", DummyRR)

    monkeypatch.setattr(
        worker_module.QueueWorker,
        "queue_worker_llm",
        _make_worker("llm-full", {"tokens": 3}, supports_stream=True),
    )
    monkeypatch.setattr(
        worker_module.QueueWorker,
        "queue_worker_vlm",
        _make_worker("vlm-full", {"tokens": 2}, supports_stream=True),
    )
    monkeypatch.setattr(
        worker_module.QueueWorker,
        "queue_worker_whisper",
        _make_worker("whisper-text", {"words": 2}),
    )
    monkeypatch.setattr(
        worker_module.QueueWorker,
        "queue_worker_kokoro",
        _make_worker("audio-base64", {"chunks_processed": 1}),
    )
    monkeypatch.setattr(
        worker_module.QueueWorker,
        "queue_worker_qwen3_asr",
        _make_worker("qwen3-text", {"chunks": 1}),
    )
    monkeypatch.setattr(
        worker_module.QueueWorker,
        "queue_worker_emb",
        _make_worker([[0.1, 0.2]], {"dim": 2}),
    )
    monkeypatch.setattr(
        worker_module.QueueWorker,
        "queue_worker_rr",
        _make_worker([{"doc": "A", "score": 0.9}], {"total": 1}),
    )

    model_registry = ModelRegistry()
    return worker_module.WorkerRegistry(model_registry)


def _make_record(model_type: ModelType, model_name: str) -> ModelRecord:
    engine_map = {
        ModelType.LLM: "ov_genai",
        ModelType.VLM: "ov_genai",
        ModelType.WHISPER: "ov_genai",
        ModelType.QWEN3_ASR: "openvino",
        ModelType.KOKORO: "openvino",
        ModelType.EMB: "ov_optimum",
        ModelType.RERANK: "ov_optimum",
    }
    instance_factory = {
        ModelType.LLM: worker_module.OVGenAI_LLM,
        ModelType.VLM: worker_module.OVGenAI_VLM,
        ModelType.WHISPER: worker_module.OVGenAI_Whisper,
        ModelType.QWEN3_ASR: worker_module.OVQwen3ASR,
        ModelType.KOKORO: worker_module.OV_Kokoro,
        ModelType.EMB: worker_module.Optimum_EMB,
        ModelType.RERANK: worker_module.Optimum_RR,
    }
    record = ModelRecord(
        model_path="/models/mock",
        model_name=model_name,
        model_type=model_type,
        engine=engine_map[model_type],
        device="CPU",
    )
    record.model_instance = instance_factory[model_type]()  # type: ignore[call-arg]
    return record


async def _load_and_call(worker_registry, record, coro):
    await worker_registry._on_model_loaded(record)
    await asyncio.sleep(0)
    result = await coro
    await worker_registry._on_model_unloaded(record)
    await asyncio.sleep(0)
    return result


def test_llm_generate(worker_registry: worker_module.WorkerRegistry) -> None:
    record = _make_record(ModelType.LLM, "llm-model")
    config = OVGenAI_GenConfig(prompt="hello")

    async def _run():
        return await _load_and_call(worker_registry, record, worker_registry.generate("llm-model", config))

    result = asyncio.run(_run())
    assert result == {"text": "llm-full", "metrics": {"tokens": 3}}


def test_llm_stream(worker_registry: worker_module.WorkerRegistry) -> None:
    record = _make_record(ModelType.LLM, "llm-stream")
    config = OVGenAI_GenConfig(prompt="hi", stream=True)

    async def _run():
        await worker_registry._on_model_loaded(record)
        await asyncio.sleep(0)
        outputs = []
        async for item in worker_registry.stream_generate("llm-stream", config):
            outputs.append(item)
        await worker_registry._on_model_unloaded(record)
        await asyncio.sleep(0)
        return outputs

    outputs = asyncio.run(_run())
    assert outputs == ["llm-full", {"metrics": {"tokens": 3}}]


def test_vlm_generate(worker_registry: worker_module.WorkerRegistry) -> None:
    record = _make_record(ModelType.VLM, "vlm-model")
    config = OVGenAI_GenConfig(prompt="describe image")

    async def _run():
        return await _load_and_call(worker_registry, record, worker_registry.generate("vlm-model", config))

    result = asyncio.run(_run())
    assert result == {"text": "vlm-full", "metrics": {"tokens": 2}}


def test_whisper_transcribe(worker_registry: worker_module.WorkerRegistry) -> None:
    record = _make_record(ModelType.WHISPER, "whisper-model")
    config = OVGenAI_WhisperGenConfig(audio_base64="AAA")

    async def _run():
        return await _load_and_call(worker_registry, record, worker_registry.transcribe_whisper("whisper-model", config))

    result = asyncio.run(_run())
    assert result == {"text": "whisper-text", "metrics": {"words": 2}}


def test_kokoro_generate_speech(worker_registry: worker_module.WorkerRegistry) -> None:
    record = _make_record(ModelType.KOKORO, "kokoro-model")
    config = OV_KokoroGenConfig(
        input="Hello",
        voice=KokoroVoice.AF_SARAH,
        lang_code=KokoroLanguage.AMERICAN_ENGLISH,
        speed=1.0,
        character_count_chunk=50,
        response_format="wav",
    )

    async def _run():
        return await _load_and_call(worker_registry, record, worker_registry.generate_speech_kokoro("kokoro-model", config))

    result = asyncio.run(_run())
    assert result == {"audio_base64": "audio-base64", "metrics": {"chunks_processed": 1}}


def test_qwen3_asr_transcribe(worker_registry: worker_module.WorkerRegistry) -> None:
    record = _make_record(ModelType.QWEN3_ASR, "qwen3-asr-model")
    config = OV_Qwen3ASRGenConfig(audio_base64="AAA")

    async def _run():
        return await _load_and_call(worker_registry, record, worker_registry.transcribe_qwen3_asr("qwen3-asr-model", config))

    result = asyncio.run(_run())
    assert result == {"text": "qwen3-text", "metrics": {"chunks": 1}}


def test_embed(worker_registry: worker_module.WorkerRegistry) -> None:
    record = _make_record(ModelType.EMB, "emb-model")
    config = PreTrainedTokenizerConfig(text=["embed me"])

    async def _run():
        return await _load_and_call(worker_registry, record, worker_registry.embed("emb-model", config))

    result = asyncio.run(_run())
    assert result == {"data": [[0.1, 0.2]], "metrics": {"dim": 2}}


def test_rerank(worker_registry: worker_module.WorkerRegistry) -> None:
    record = _make_record(ModelType.RERANK, "rerank-model")
    config = RerankerConfig(query="Paris", documents=["Paris", "Berlin"])

    async def _run():
        return await _load_and_call(worker_registry, record, worker_registry.rerank("rerank-model", config))

    result = asyncio.run(_run())
    assert result == {"data": [{"doc": "A", "score": 0.9}], "metrics": {"total": 1}}


def test_missing_model_queue(worker_registry: worker_module.WorkerRegistry) -> None:
    with pytest.raises(ValueError):
        worker_registry._get_model_queue("missing")


@pytest.mark.parametrize(
    ("infer_method_name", "worker_method_name", "config"),
    [
        ("infer_llm", "queue_worker_llm", OVGenAI_GenConfig(prompt="boom")),
        ("infer_vlm", "queue_worker_vlm", OVGenAI_GenConfig(prompt="boom")),
        ("infer_whisper", "queue_worker_whisper", OVGenAI_WhisperGenConfig(audio_base64="AAA")),
        ("infer_qwen3_asr", "queue_worker_qwen3_asr", OV_Qwen3ASRGenConfig(audio_base64="AAA")),
        (
            "infer_kokoro",
            "queue_worker_kokoro",
            OV_KokoroGenConfig(
                input="boom",
                voice=KokoroVoice.AF_SARAH,
                lang_code=KokoroLanguage.AMERICAN_ENGLISH,
                speed=1.0,
                character_count_chunk=50,
                response_format="wav",
            ),
        ),
        ("infer_qwen3_tts", "queue_worker_qwen3_tts", object()),
        ("infer_emb", "queue_worker_emb", PreTrainedTokenizerConfig(text=["boom"])),
        ("infer_rerank", "queue_worker_rr", RerankerConfig(query="boom", documents=["a"])),
    ],
)
def test_queue_workers_complete_future_and_trigger_unload_on_inference_exception(
    monkeypatch: pytest.MonkeyPatch,
    infer_method_name: str,
    worker_method_name: str,
    config,
) -> None:
    unload_event = asyncio.Event()
    unload_calls: list[str] = []
    sentinel = object()

    class DummyRegistry:
        async def register_unload(self, model_name: str) -> None:
            unload_calls.append(model_name)
            unload_event.set()

    async def _raise_infer(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(worker_module.InferWorker, infer_method_name, _raise_infer)

    async def _run() -> None:
        model_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        packet = worker_module.WorkerPacket(
            request_id="req-1",
            id_model="model-1",
            gen_config=config,
            result_future=loop.create_future(),
        )
        await model_queue.put(packet)

        worker_fn = getattr(worker_module.QueueWorker, worker_method_name)
        worker_task = asyncio.create_task(worker_fn("model-1", model_queue, sentinel, DummyRegistry()))

        completed_packet = await asyncio.wait_for(packet.result_future, timeout=1.0)
        await asyncio.wait_for(unload_event.wait(), timeout=1.0)
        await asyncio.wait_for(model_queue.join(), timeout=1.0)
        await asyncio.wait_for(worker_task, timeout=1.0)

        assert completed_packet is packet
        assert completed_packet.metrics is not None
        assert "error" in completed_packet.metrics
        assert unload_calls == ["model-1"]
        assert model_queue.qsize() == 0
        assert model_queue._unfinished_tasks == 0

    asyncio.run(_run())
