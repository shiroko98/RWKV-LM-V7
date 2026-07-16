from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.make_data import DEFAULT_VOCAB, build_arg_parser  # noqa: E402
from data.tokenizer.rwkv_tokenizer import TRIE_TOKENIZER  # noqa: E402
from src.binidx import MMapIndexedDataset  # noqa: E402


def test_make_data_defaults_to_special_first_vocab():
    args = build_arg_parser().parse_args(["sample.jsonl", "1", "4096"])

    assert Path(args.vocab) == DEFAULT_VOCAB


def test_make_data_writes_atomic_special_tokens_by_default(tmp_path: Path):
    source = tmp_path / "special.jsonl"
    text = "Assistant: <think>\nanswer.<|im_end|>\n><tool_call>"
    source.write_text(json.dumps({"text": text}, ensure_ascii=False) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, str(ROOT / "data" / "make_data.py"), str(source), "1", "4096"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    tokenizer = TRIE_TOKENIZER(str(DEFAULT_VOCAB), strict_length=True)
    dataset = MMapIndexedDataset(str(tmp_path / source.stem))
    token_ids = dataset[0].astype(int).tolist()

    assert token_ids[-1] == 0
    assert tokenizer.decode(token_ids[:-1]) == text
    assert 65533 in token_ids
    assert 65531 in token_ids
    assert 65534 in token_ids
