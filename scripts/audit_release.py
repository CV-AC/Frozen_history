import argparse
import ast
import io
import json
from pathlib import Path
import re
import tokenize

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = {
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "THIRD_PARTY.md",
    "pyproject.toml",
    ".gitignore",
    "requirements.txt",
    "requirements-metrics.txt",
    "requirements-cuda.txt",
    "requirements-dev.txt",
    "SHA256SUMS",
}
DIRECTORIES = ("diffsr", "scripts", "configs", "docs", "tests", ".github")
EXTENSIONS = {".py", ".md", ".json", ".toml", ".txt", ".yml", ".yaml"}
PUBLIC_MEDIA = {
    "assets/figures/teaser.png",
    "assets/figures/pipeline.png",
    "assets/figures/temporal-loss.png",
    "assets/figures/reconstruction.png",
    "assets/figures/temporal-stability.png",
}


def public_files(root=ROOT):
    root = Path(root).resolve()
    selected = [root / name for name in sorted(TOP_LEVEL) if (root / name).is_file()]
    selected.extend(
        root / name for name in sorted(PUBLIC_MEDIA) if (root / name).is_file()
    )
    for name in DIRECTORIES:
        selected.extend(
            p
            for p in (root / name).rglob("*")
            if p.is_file()
            and p.suffix in EXTENSIONS
            and not any(
                part in {"__pycache__", ".pytest_cache", ".ruff_cache"}
                for part in p.parts
            )
        )
    return sorted(set(selected))


def audit(root=ROOT):
    root = Path(root).resolve()
    issues = []
    paths = public_files(root)
    private_path = re.compile(
        r"/(?:Users|home|scratch|appl|lustre|gpfs|work|mnt|hpc)/|[A-Za-z]:\\(?:Users|work)\\",
        re.IGNORECASE,
    )
    email = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    identifiers = re.compile(
        r"\b(?:paper\d{3,}|SLURM_[A-Z_]+)\b|"
        + "#"
        + r"SBATCH|(?:--account|--partition|--nodelist)(?:=|\s)",
        re.IGNORECASE,
    )
    credentials = re.compile(
        r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)"
    )
    for path in paths:
        relative = str(path.relative_to(root))
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            issues.append({"file": relative, "issue": "symlink or path escape"})
            continue
        if relative in PUBLIC_MEDIA:
            from PIL import Image

            try:
                with Image.open(path) as image:
                    if image.format != "PNG":
                        issues.append(
                            {"file": relative, "issue": "invalid media format"}
                        )
                    if image.info:
                        issues.append({"file": relative, "issue": "image metadata"})
                    image.verify()
            except (OSError, ValueError, SyntaxError):
                issues.append({"file": relative, "issue": "invalid image"})
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeError:
            issues.append({"file": relative, "issue": "non-text payload"})
            continue
        for label, pattern in [
            ("private absolute path", private_path),
            ("personal email", email),
            ("internal identifier", identifiers),
            ("credential", credentials),
        ]:
            if pattern.search(source):
                issues.append({"file": relative, "issue": label})
        if path.suffix == ".py":
            try:
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if (
                        isinstance(
                            node,
                            (
                                ast.Module,
                                ast.FunctionDef,
                                ast.AsyncFunctionDef,
                                ast.ClassDef,
                            ),
                        )
                        and ast.get_docstring(node) is not None
                    ):
                        issues.append({"file": relative, "issue": "Python docstring"})
                if any(
                    token.type == tokenize.COMMENT
                    for token in tokenize.generate_tokens(io.StringIO(source).readline)
                ):
                    issues.append({"file": relative, "issue": "Python comment"})
            except (SyntaxError, tokenize.TokenError) as error:
                issues.append({"file": relative, "issue": f"invalid Python: {error}"})
        if path.suffix == ".json":
            try:
                json.loads(source)
            except json.JSONDecodeError:
                issues.append({"file": relative, "issue": "invalid JSON"})
    return {
        "files": len(paths),
        "python_files": sum(p.suffix == ".py" for p in paths),
        "bytes": sum(p.stat().st_size for p in paths),
        "issues": issues,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = audit(args.root)
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(report["issues"]))


if __name__ == "__main__":
    main()
