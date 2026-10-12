import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def command_for(path, overrides=()):
    config = json.loads(Path(path).read_text())
    command = [sys.executable, "-m", "diffsr.engine_seq"]
    for key, value in config.items():
        if value is None:
            continue
        flag = key.replace("_", "-")
        if isinstance(value, bool):
            if key == "resume":
                if value:
                    command.append("--resume")
            else:
                command.append("--" + ("" if value else "no-") + flag)
        elif isinstance(value, list):
            command.extend(["--" + flag, *map(str, value)])
        elif isinstance(value, (str, int, float)):
            command.extend(["--" + flag, str(value)])
        else:
            raise ValueError(f"Unsupported training option: {key}")
    return command + list(overrides)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args, overrides = parser.parse_known_args()
    command = command_for(args.config, overrides)
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
