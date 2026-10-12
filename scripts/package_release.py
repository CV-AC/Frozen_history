import argparse
import hashlib
from pathlib import Path
import zipfile

from audit_release import ROOT, audit, public_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "frozen-history-source.zip")
    args = parser.parse_args()
    report = audit()
    if report["issues"]:
        raise SystemExit(f"Release audit failed: {report['issues']}")
    files = [p for p in public_files() if p.name != "SHA256SUMS"]
    sums = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT).as_posix()}\n"
        for path in files
    )
    (ROOT / "SHA256SUMS").write_text(sums)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        args.out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in public_files():
            name = "frozen-history/" + path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    print(f"Packaged {len(public_files())} release files: {args.out.name}")
    print(f"SHA256: {hashlib.sha256(args.out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
