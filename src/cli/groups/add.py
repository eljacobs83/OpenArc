"""
Add command - Add a model configuration to the config file.
"""
import json
from typing import Any

import click

from ..main import cli, console
from ..utils import validate_model_path


def _coerce_runtime_value(value: str) -> Any:
    """Coerce runtime config value from string when possible."""
    cleaned = value.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return cleaned


def parse_runtime_config(runtime_config: str | None) -> dict[str, Any]:
    """Parse runtime config from JSON or KEY=VALUE pairs.

    Accepted formats:
    - JSON object: '{"MODEL_DISTRIBUTION_POLICY": "PIPELINE_PARALLEL"}'
    - Comma-separated key/value pairs: 'MODEL_DISTRIBUTION_POLICY=PIPELINE_PARALLEL,NUM_STREAMS=2'
    """
    if not runtime_config:
        return {}

    raw = runtime_config.strip()
    if not raw:
        return {}

    # JSON is still the preferred format
    if raw.startswith("{"):
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError(
                f"runtime_config must be a JSON object (dictionary), got {type(parsed).__name__}"
            )
        return parsed

    # Fallback parser for shell-friendly KEY=VALUE pairs
    parsed: dict[str, Any] = {}
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                "runtime_config key/value format must use KEY=VALUE pairs separated by commas"
            )
        key, value = token.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("runtime_config contains an empty key")
        parsed[key] = _coerce_runtime_value(value)

    return parsed


@cli.command()
@click.option('--model-name', '--mn',
    required=True,
    help='Public facing name of the model.')
@click.option('--model-path', '--m',
    required=True, 
    help='Path to OpenVINO IR converted model.')
@click.option('--engine', '--en',
    type=click.Choice(['ovgenai', 'openvino', 'optimum']),
    required=True,
    help='Engine used to load the model (ovgenai, openvino, optimum)')
@click.option('--model-type', '--mt',
    type=click.Choice([
        'llm', 'vlm', 'whisper', 'qwen3_asr', 'kokoro',
        'qwen3_tts_custom_voice', 'qwen3_tts_voice_design', 'qwen3_tts_voice_clone',
        'emb', 'rerank',
    ]),
    required=True,
    help='Model type (llm, vlm, whisper, qwen3_asr, kokoro, qwen3_tts_custom_voice, qwen3_tts_voice_design, qwen3_tts_voice_clone, emb, rerank)')
@click.option('--device', '--d',
    required=True,
    help='Device(s) to load the model on.')
@click.option("--runtime-config", "--rtc",
    default=None,
    help=(
        'OpenVINO runtime configuration as JSON string '
        '(e.g., \'{"MODEL_DISTRIBUTION_POLICY": "PIPELINE_PARALLEL"}\') '
        'or KEY=VALUE pairs '
        '(e.g., MODEL_DISTRIBUTION_POLICY=PIPELINE_PARALLEL,NUM_STREAMS=2).'
    ))
@click.option('--vlm-type', '--vt',
    type=click.Choice(['internvl2', 'llava15', 'llavanext', 'minicpmv26', 'phi3vision', 'phi4mm', 'qwen2vl', 'qwen25vl', 'gemma3']),
    required=False,
    default=None,
    help='Vision model type. Used to map correct vision tokens.')
@click.option('--draft-model-path', '--dmp',
    required=False,
    default=None,
    help='Path to draft model for speculative decoding.')
@click.option('--draft-device', '--dd',
    required=False,
    default=None,
    help='Draft model device.')
@click.option('--num-assistant-tokens', '--nat',
    required=False,
    default=None,
    type=int,
    help='Number of tokens draft model generates per step.')
@click.option('--assistant-confidence-threshold', '--act',
    required=False,
    default=None,
    type=float,
    help='Confidence threshold for accepting draft tokens.')
@click.pass_context
def add(ctx, model_path, model_name, engine, model_type, device, runtime_config, vlm_type, draft_model_path, draft_device, num_assistant_tokens, assistant_confidence_threshold):
    """- Add a model configuration to the config file."""
    
    # Validate model path
    if not validate_model_path(model_path):
        console.print(f"[red]Model file check failed! {model_path} does not contain openvino model files OR your chosen path is malformed. Verify chosen path is correct and acquired model files match source on the hub, or the destination of converted model.[/red]")
        ctx.exit(1)
    
    # Parse runtime_config if provided
    try:
        parsed_runtime_config = parse_runtime_config(runtime_config)
    except (json.JSONDecodeError, ValueError) as e:
        console.print(f"[red]Error parsing runtime_config:[/red] {e}")
        console.print('[yellow]JSON example: \'{"MODEL_DISTRIBUTION_POLICY": "PIPELINE_PARALLEL"}\'[/yellow]')
        console.print('[yellow]KEY=VALUE example: MODEL_DISTRIBUTION_POLICY=PIPELINE_PARALLEL,NUM_STREAMS=2[/yellow]')
        ctx.exit(1)
    
    # Build and save configuration
    load_config = {
        "model_name": model_name,
        "model_path": model_path,  
        "model_type": model_type,  
        "engine": engine,    
        "device": device,
        "runtime_config": parsed_runtime_config,
        "vlm_type": vlm_type if vlm_type else None
    }
    
    # Add speculative decoding options if provided
    if draft_model_path:
        load_config["draft_model_path"] = draft_model_path
    if draft_device:
        load_config["draft_device"] = draft_device
    if num_assistant_tokens is not None:
        load_config["num_assistant_tokens"] = num_assistant_tokens
    if assistant_confidence_threshold is not None:
        load_config["assistant_confidence_threshold"] = assistant_confidence_threshold
    
    ctx.obj.server_config.save_model_config(model_name, load_config)
    console.print(f"[green]Model configuration saved:[/green] {model_name}")
    console.print(f"[dim]Use 'openarc load {model_name}' to load this model.[/dim]")
