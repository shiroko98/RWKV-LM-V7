# SFT Packing Demo

## Goal

This demo shows the current SFT preprocessing behavior for:

- chat-template rendering
- appending one sample-final `<|endoftext|>` during preprocessing
- fixed-length packing
- `<|endoftext|>` padding
- mask behavior for real stop tokens vs padding tokens

## Input

Input file:

- [input.jsonl](D:/codes/RWKV-LM-V7-12B-train/tests/artifacts/sft_packing_demo/input.jsonl)

Contents:

```jsonl
{"messages":[{"role":"user","content":"请回答 1+1"},{"role":"assistant","content":"1+1=2。"}]}
{"messages":[{"role":"user","content":"请回答 2+2"},{"role":"assistant","content":"2+2=4。"}]}
```

## Build Command

Run:

```powershell
conda run -n model python data\make_sft_binidx.py tests\artifacts\sft_packing_demo\input.jsonl --out-prefix tests\artifacts\sft_packing_demo\packed --vocab rwkv_vocab_v20260603.txt --chat-template chat_template.jinja --pack-length 128
```

Observed output:

```text
### Built SFT binidx dataset: prefix=tests\artifacts\sft_packing_demo\packed docs=1 tokens=128 trainable_tokens=16
```

## Output Files

Generated files:

- `packed.bin`
- `packed.idx`
- `packed.mask.bin`
- `packed.mask.idx`
- [inspection.json](D:/codes/RWKV-LM-V7-12B-train/tests/artifacts/sft_packing_demo/inspection.json)

## Inspection Command

Run:

```powershell
conda run -n model python -c "import json; from pathlib import Path; from src.binidx import MMapIndexedDataset; from data.tokenizer.rwkv_tokenizer import TRIE_TOKENIZER; tok=TRIE_TOKENIZER('rwkv_vocab_v20260603.txt', strict_length=True); ds=MMapIndexedDataset('tests/artifacts/sft_packing_demo/packed'); ms=MMapIndexedDataset('tests/artifacts/sft_packing_demo/packed.mask'); ids=ds[0].astype(int).tolist(); mask=ms[0].astype(int).tolist(); text=tok.decode(ids); train=tok.decode([i for i,m in zip(ids,mask) if m==1]); eod=tok.token2idx[b'<|endoftext|>']; learned=[idx for idx,(t,m) in enumerate(zip(ids,mask)) if t==eod and m==1]; padded=[idx for idx,(t,m) in enumerate(zip(ids,mask)) if t==eod and m==0]; summary={'length':len(ids),'trainable_count':sum(mask),'decoded_prefix':text[:300],'trainable_text':train,'learned_eod_positions':learned,'padded_eod_positions_head':padded[:20],'padded_eod_count':len(padded)}; Path('tests/artifacts/sft_packing_demo/inspection.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'); print(json.dumps(summary, ensure_ascii=False, indent=2))"
```

Observed result:

```json
{
  "length": 128,
  "trainable_count": 16,
  "decoded_prefix": "<|im_start|>System: You are a helpful assistant. Your name is xiaoke and is built by CETC.<|im_end|>\n<|im_start|>User: 请回答 1+1<|im_end|>\n<|im_start|>Assistant: 1+1=2。<|im_end|><|endoftext|><|im_start|>System: You are a helpful assistant. Your name is xiaoke and is built by CETC.<|im_end|>\n<|im_start",
  "trainable_text": "1+1=2。<|im_end|><|endoftext|>2+2=4。<|im_end|><|endoftext|>",
  "learned_eod_positions": [
    49,
    99
  ],
  "padded_eod_positions_head": [
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119
  ],
  "padded_eod_count": 28
}
```

## Interpretation

Current packing behavior is:

- each rendered sample gets one appended `<|endoftext|>` during preprocessing
- that first sample-final `<|endoftext|>` is the real stop token and is trainable
- when multiple samples are packed together, each sample contributes exactly one trainable `<|endoftext|>`
- if the packed sequence is still shorter than `pack_length`, extra `<|endoftext|>` tokens are appended as padding
- those extra padding `<|endoftext|>` tokens always have `mask=0`

So the correct rule is not "all trailing EOD tokens are padding".

The correct rule is:

- the first appended `<|endoftext|>` after a real sample is a learnable stop token
- only the subsequent `<|endoftext|>` tokens added purely to fill the packed length are padding

## Related Tests

Relevant checks live in:

- [tests/test_sft_binidx.py](D:/codes/RWKV-LM-V7-12B-train/tests/test_sft_binidx.py)

In particular:

- `test_build_binidx_dataset_packs_to_fixed_length_and_masks_padding_eod`
- `test_packing_keeps_exactly_one_trainable_eod_between_samples`
- `test_single_sample_padding_keeps_first_terminal_eod_trainable`
