import pytest

from src.cli.groups.add import parse_runtime_config


def test_parse_runtime_config_json_object() -> None:
    config = parse_runtime_config('{"MODEL_DISTRIBUTION_POLICY": "PIPELINE_PARALLEL", "NUM_STREAMS": 2}')

    assert config["MODEL_DISTRIBUTION_POLICY"] == "PIPELINE_PARALLEL"
    assert config["NUM_STREAMS"] == 2


def test_parse_runtime_config_key_value_pairs() -> None:
    config = parse_runtime_config("MODEL_DISTRIBUTION_POLICY=PIPELINE_PARALLEL, NUM_STREAMS=2, ENABLE=true")

    assert config == {
        "MODEL_DISTRIBUTION_POLICY": "PIPELINE_PARALLEL",
        "NUM_STREAMS": 2,
        "ENABLE": True,
    }


def test_parse_runtime_config_raises_on_bad_pair() -> None:
    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_runtime_config("MODEL_DISTRIBUTION_POLICY")
