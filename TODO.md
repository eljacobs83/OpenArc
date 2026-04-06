# OpenArc QA TODO

## 1) CLI & Configuration Management (`src/cli`, `src/cli/modules`)

### Bugs found
- [ ] `serve start --load-models` parses a single option value separately from positional startup models, which can accidentally treat a space-delimited model list as one model name.
  - **Where:** `src/cli/groups/serve.py` (`--load-models`, `startup_models`, `models_to_load` assembly).
  - **Why this is a bug:** Help text says “space-separated model names,” but the option itself only captures one token; the rest are only captured if passed as positionals.

- [ ] `openarc_config.json` path is inferred from source tree location, which can break when installed as a package and executed outside the repo root.
  - **Where:** `src/cli/modules/server_config.py` (`Path(__file__).parent.parent.parent.parent / "openarc_config.json"`).
  - **Why this is a bug:** Runtime behavior depends on installation layout rather than explicit user config path, causing config files to land in unexpected locations.

## 2) API Layer & Error Handling (`src/server/main.py`)

### Bugs found
- [ ] Several endpoints convert intentionally raised `HTTPException` (e.g., 404/400) into generic 500 responses due to broad `except Exception` blocks.
  - **Where:** `src/server/main.py` (e.g., `/openarc/unload`, plus similar `except Exception as exc` patterns around endpoint handlers).
  - **Why this is a bug:** Client-facing status codes are corrupted and users receive incorrect server-error responses.

- [ ] API key verification allows startup with a missing `OPENARC_API_KEY` but does not fail fast; every authenticated endpoint then fails at request time.
  - **Where:** `src/server/main.py` (`API_KEY = os.getenv("OPENARC_API_KEY")`, `verify_api_key`).
  - **Why this is a bug:** Misconfiguration is discovered late and manifests as repeated auth failures instead of a clear startup error.

## 3) Model Registry & Worker Orchestration (`src/server/model_registry.py`, `src/server/worker_registry.py`)

### Bugs found
- [ ] Unload path reports success immediately after scheduling background unload, even though unload can still fail asynchronously.
  - **Where:** `src/server/model_registry.py` (`register_unload` + `_unload_task`).
  - **Why this is a bug:** API can report “unloading”/success while the underlying unload operation errors later, leading to state drift.

- [ ] Background unload errors are logged at `info` level with minimal context.
  - **Where:** `src/server/model_registry.py` (`_unload_task` exception handler).
  - **Why this is a bug:** Operational failures are easy to miss in production logs and harder to debug.

## 4) OV GenAI Inference Engines (`src/engine/ov_genai`)

### Bugs found
- [ ] Streaming generation path emits debug internals using `logger.error`, even for healthy requests.
  - **Where:** `src/engine/ov_genai/llm.py` (`[DEBUG]` log lines in `generate_stream`).
  - **Why this is a bug:** Normal traffic produces false error logs, degrading observability and alert quality.

- [ ] `generate_stream` always awaits the generation task in `finally`, which can leak unexpected exceptions after client disconnect/cancellation handling paths.
  - **Where:** `src/engine/ov_genai/llm.py` and `src/engine/ov_genai/vlm.py` (`result = await gen_task` in `finally`).
  - **Why this is a bug:** Cleanup paths can still raise late exceptions and destabilize streaming response completion.

## 5) OpenVINO Engine Surface (`src/engine/openvino`)

### Bugs found
- [ ] Kitten TTS surface is present but completely unimplemented.
  - **Where:** `src/engine/openvino/kitten.py`.
  - **Why this is a bug:** Placeholder module can be discovered/imported without delivering functionality, creating dead/incomplete engine surface.

## 6) Docs, Env Contract & Deployment (`README.md`, `docs/`, `docker-compose.yaml`)

### Bugs found
- [ ] Startup model environment variable is inconsistent across runtime and deployment docs/config (`OPENARC_STARTUP_MODELS` vs `OPENARC_AUTOLOAD_MODEL`).
  - **Where:**
    - Runtime reads: `src/server/main.py` (`OPENARC_STARTUP_MODELS`).
    - CLI writes: `src/cli/groups/serve.py` (`OPENARC_STARTUP_MODELS`).
    - Docker/docs still reference: `docker-compose.yaml`, `README.md`, `docs/install.md` (`OPENARC_AUTOLOAD_MODEL`).
  - **Why this is a bug:** Autoload behavior silently fails in documented/dockerized workflows unless users discover the new variable name.
