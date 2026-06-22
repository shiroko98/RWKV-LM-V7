from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from src.binidx import MMapIndexedDataset
from src.sft_binidx import EncodedDocument, write_documents

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "fill_sft_mask.py"

spec = importlib.util.spec_from_file_location("fill_sft_mask", SCRIPT_PATH)
fill_sft_mask = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(fill_sft_mask)


def read_docs(prefix: str) -> list[list[int]]:
    dataset = MMapIndexedDataset(prefix)
    return [dataset[index].astype(np.int64).tolist() for index in range(len(dataset))]


def test_resolve_mask_prefix_accepts_dataset_mask_or_bin_path():
    assert fill_sft_mask.resolve_mask_prefix("data/sft") == Path("data/sft.mask")
    assert fill_sft_mask.resolve_mask_prefix("data/sft.mask") == Path("data/sft.mask")
    assert fill_sft_mask.resolve_mask_prefix("data/sft.mask.bin") == Path("data/sft.mask")
    assert fill_sft_mask.resolve_mask_prefix("data/sft.mask.idx") == Path("data/sft.mask")


def test_fill_mask_bin_writes_new_sidecar_without_touching_tokens(tmp_path):
    prefix = str(tmp_path / "tiny_sft")
    write_documents(
        prefix,
        [
            EncodedDocument(input_ids=[10, 11, 12], loss_mask=[0, 1, 0]),
            EncodedDocument(input_ids=[20, 21], loss_mask=[1, 0]),
        ],
        token_dtype=np.uint16,
        mask_dtype=np.uint8,
    )
    original_tokens = read_docs(prefix)
    original_masks = read_docs(prefix + ".mask")

    output_prefix = tmp_path / "tiny_sft_full.mask"
    stats = fill_sft_mask.fill_mask_bin(prefix + ".mask", output_prefix=output_prefix)

    assert stats["dtype"] == "uint8"
    assert stats["documents"] == 2
    assert stats["elements"] == 5
    assert stats["in_place"] is False
    assert read_docs(str(output_prefix)) == [[1, 1, 1], [1, 1]]
    assert read_docs(prefix) == original_tokens
    assert read_docs(prefix + ".mask") == original_masks


def test_fill_mask_bin_default_output_prefix_and_custom_value(tmp_path):
    prefix = str(tmp_path / "tiny_sft")
    write_documents(
        prefix,
        [EncodedDocument(input_ids=[1, 2, 3], loss_mask=[0, 1, 0])],
        token_dtype=np.uint16,
        mask_dtype=np.uint8,
    )

    stats = fill_sft_mask.fill_mask_bin(prefix, value=0, chunk_elements=1)

    assert stats["output_prefix"] == prefix + ".mask.all1"
    assert stats["value"] == 0
    assert read_docs(prefix + ".mask.all1") == [[0, 0, 0]]


def test_fill_mask_bin_can_update_mask_in_place(tmp_path):
    prefix = str(tmp_path / "tiny_sft")
    write_documents(
        prefix,
        [EncodedDocument(input_ids=[1, 2, 3, 4], loss_mask=[0, 1, 0, 1])],
        token_dtype=np.uint16,
        mask_dtype=np.uint8,
    )

    stats = fill_sft_mask.fill_mask_bin(prefix + ".mask.bin", in_place=True, chunk_elements=2)

    assert stats["output_prefix"] == prefix + ".mask"
    assert stats["in_place"] is True
    assert read_docs(prefix + ".mask") == [[1, 1, 1, 1]]
    assert read_docs(prefix) == [[1, 2, 3, 4]]


def test_fill_mask_bin_rejects_invalid_inputs(tmp_path):
    prefix = str(tmp_path / "tiny_sft")
    write_documents(
        prefix,
        [EncodedDocument(input_ids=[1, 2], loss_mask=[0, 1])],
        token_dtype=np.uint16,
        mask_dtype=np.uint8,
    )

    with pytest.raises(FileNotFoundError, match="missing mask bin"):
        fill_sft_mask.fill_mask_bin(tmp_path / "missing")
    bin_path = Path(prefix + ".mask.bin")
    idx_path = Path(prefix + ".mask.idx")
    renamed_idx = idx_path.with_suffix(".idx.bak")
    idx_path.rename(renamed_idx)
    try:
        with pytest.raises(FileNotFoundError, match="missing mask idx"):
            fill_sft_mask.fill_mask_bin(prefix)
    finally:
        renamed_idx.rename(idx_path)
    with pytest.raises(ValueError, match="output-prefix"):
        fill_sft_mask.fill_mask_bin(prefix, output_prefix=tmp_path / "out.mask", in_place=True)
    with pytest.raises(ValueError, match="chunk_elements"):
        fill_sft_mask.fill_mask_bin(prefix, chunk_elements=0)
    with pytest.raises(ValueError, match="safely stored"):
        fill_sft_mask.fill_mask_bin(prefix, value=256)
    with open(bin_path, "ab") as stream:
        stream.write(b"\x00")
    with pytest.raises(ValueError, match="mask bin size mismatch"):
        fill_sft_mask.fill_mask_bin(prefix)


def test_parse_args_and_main_write_output_prefix(tmp_path, monkeypatch, capsys):
    prefix = str(tmp_path / "tiny_sft")
    output_prefix = str(tmp_path / "tiny_sft_cli.mask")
    write_documents(
        prefix,
        [EncodedDocument(input_ids=[1, 2], loss_mask=[0, 1])],
        token_dtype=np.uint16,
        mask_dtype=np.uint8,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fill_sft_mask.py",
            prefix + ".mask.idx",
            "--output-prefix",
            output_prefix,
            "--value",
            "1",
            "--chunk-elements",
            "1",
        ],
    )

    assert fill_sft_mask.main() == 0

    output = capsys.readouterr().out
    assert "### SFT mask fill complete" in output
    assert f"output_prefix={output_prefix}" in output
    assert read_docs(output_prefix) == [[1, 1]]


def test_script_entrypoint_guard_is_present():
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__"' in text
    assert "raise SystemExit(main())" in text
