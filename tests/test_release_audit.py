import pytest

from audit_release import audit


@pytest.mark.parametrize(
    "payload",
    [
        "/" + "Users" + "/private/data",
        "/" + "home" + "/private/data",
        "C:" + "\\Users" + "\\private\\data",
        "person" + "@" + "example.org",
        "SL" + "URM_JOB_ID=123",
        "--" + "account=private",
        "github_" + "pat_" + "a" * 30,
        "AK" + "IA" + "A" * 16,
    ],
)
def test_private_metadata_is_rejected(tmp_path, payload):
    (tmp_path / "README.md").write_text(payload)
    assert audit(tmp_path)["issues"]


@pytest.mark.parametrize("source", ["# note\nx = 1\n", '""""""\nx = 1\n'])
def test_comments_and_empty_docstrings_are_rejected(tmp_path, source):
    directory = tmp_path / "diffsr"
    directory.mkdir()
    (directory / "example.py").write_text(source)
    assert audit(tmp_path)["issues"]


def test_public_attribution_and_portable_source_are_accepted(tmp_path):
    (tmp_path / "README.md").write_text("https://github.com/pytorch/pytorch")
    directory = tmp_path / "diffsr"
    directory.mkdir()
    (directory / "example.py").write_text('DATA = "data/qrisp"\n')
    assert audit(tmp_path)["issues"] == []


def test_audit_script_has_no_metadata_exemption(tmp_path):
    directory = tmp_path / "scripts"
    directory.mkdir()
    (directory / "audit_release.py").write_text('PATH = "' + "/" + 'Users/private"\n')
    assert audit(tmp_path)["issues"]


def test_symlink_escape_is_rejected(tmp_path):
    directory = tmp_path / "diffsr"
    directory.mkdir()
    (directory / "example.py").symlink_to(tmp_path.parent / "outside.py")
    (tmp_path.parent / "outside.py").write_text("x = 1\n")
    assert audit(tmp_path)["issues"]


def test_public_figure_is_included_and_verified(tmp_path):
    from PIL import Image

    directory = tmp_path / "assets/figures"
    directory.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(directory / "teaser.png")
    report = audit(tmp_path)
    assert report["files"] == 1
    assert report["issues"] == []


def test_figure_metadata_is_rejected(tmp_path):
    from PIL import Image, PngImagePlugin

    directory = tmp_path / "assets/figures"
    directory.mkdir(parents=True)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Comment", "private metadata")
    Image.new("RGB", (8, 8)).save(directory / "teaser.png", pnginfo=metadata)
    assert audit(tmp_path)["issues"] == [
        {"file": "assets/figures/teaser.png", "issue": "image metadata"}
    ]


def test_invalid_figure_is_rejected(tmp_path):
    directory = tmp_path / "assets/figures"
    directory.mkdir(parents=True)
    (directory / "teaser.png").write_bytes(b"not an image")
    assert audit(tmp_path)["issues"] == [
        {"file": "assets/figures/teaser.png", "issue": "invalid image"}
    ]
