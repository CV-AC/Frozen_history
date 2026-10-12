import argparse
import json
from pathlib import Path

import pytest

from diffsr import engine_seq
from train import command_for

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs/train").glob("*.json"))


class Parsed(Exception):
    pass


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_training_config_is_accepted_by_real_parser(monkeypatch, path):
    original = argparse.ArgumentParser.parse_args
    config = json.loads(path.read_text())

    def capture(parser, args=None, namespace=None):
        parsed = original(parser, command_for(path)[3:], namespace)
        assert parsed.name == config["name"]
        assert parsed.scale == config["scale"]
        assert parsed.temporal_weight == config["temporal_weight"]
        raise Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    with pytest.raises(Parsed):
        engine_seq.main()
