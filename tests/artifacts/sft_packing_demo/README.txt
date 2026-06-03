SFT packing demo

Input file
- input.jsonl

Build command
- conda run -n model python data\make_sft_binidx.py tests\artifacts\sft_packing_demo\input.jsonl --out-prefix tests\artifacts\sft_packing_demo\packed --vocab rwkv_vocab_v20260603.txt --chat-template chat_template.jinja --pack-length 128

Build result
- docs=1
- tokens=128
- trainable_tokens=16

Inspection command
- conda run -n model python -c "import json; from pathlib import Path; from src.binidx import MMapIndexedDataset; from data.tokenizer.rwkv_tokenizer import TRIE_TOKENIZER; tok=TRIE_TOKENIZER('rwkv_vocab_v20260603.txt', strict_length=True); ds=MMapIndexedDataset('tests/artifacts/sft_packing_demo/packed'); ms=MMapIndexedDataset('tests/artifacts/sft_packing_demo/packed.mask'); ids=ds[0].astype(int).tolist(); mask=ms[0].astype(int).tolist(); text=tok.decode(ids); train=tok.decode([i for i,m in zip(ids,mask) if m==1]); eod=tok.token2idx[b'<|endoftext|>']; learned=[idx for idx,(t,m) in enumerate(zip(ids,mask)) if t==eod and m==1]; padded=[idx for idx,(t,m) in enumerate(zip(ids,mask)) if t==eod and m==0]; summary={'length':len(ids),'trainable_count':sum(mask),'decoded_prefix':text[:300],'trainable_text':train,'learned_eod_positions':learned,'padded_eod_positions_head':padded[:20],'padded_eod_count':len(padded)}; Path('tests/artifacts/sft_packing_demo/inspection.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'); print(json.dumps(summary, ensure_ascii=False, indent=2))"

Expected behavior shown by inspection.json
- two source samples are packed into one fixed-length document
- the end of each real sample contributes exactly one trainable <|endoftext|>
- extra <|endoftext|> tokens used only for padding have loss mask 0
