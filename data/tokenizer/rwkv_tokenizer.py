########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

from __future__ import annotations

import ast
import warnings


# These token strings are exported by the special-first HF converter with their
# existing RWKV vocabulary IDs. They must be isolated before greedy trie
# tokenization so an ordinary token such as b" <" cannot consume the leading
# b"<" of a marker.
RWKV_SPECIAL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<think>",
    "<tool_call>",
)


class TRIE:
    __slots__ = tuple("ch,to,values,front".split(","))
    to: list
    values: set

    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for ch in range(256)]
        self.values = set()
        self.front = front

    def __repr__(self):
        fr = self
        ret = []
        while (fr is not None):
            if (fr.ch is not None):
                ret.append(fr.ch)
            fr = fr.front
        return "<TRIE %s %s>" % (ret[::-1], self.values)

    def add(self, key: bytes, idx: int = 0, val=None):
        if (idx == len(key)):
            if (val is None):
                val = key
            self.values.add(val)
            return self
        ch = key[idx]
        if (self.to[ch] is None):
            self.to[ch] = TRIE(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx+1, val=val)

    def find_longest(self, key: bytes, idx: int = 0):
        u: TRIE = self
        ch: int = key[idx]

        while (u.to[ch] is not None):
            u = u.to[ch]
            idx += 1
            if (u.values):
                ret = idx, u, u.values
            if (idx == len(key)):
                break
            ch = key[idx]
        return ret


def parse_vocab_line(line: str, *, strict_length: bool = False):
    line = line.rstrip("\r\n")
    first_space = line.index(" ")
    last_space = line.rindex(" ")

    idx = int(line[:first_space])
    token_value = ast.literal_eval(line[first_space:last_space])
    token_bytes = token_value.encode("utf-8") if isinstance(token_value, str) else token_value
    if not isinstance(token_bytes, bytes):
        raise TypeError(f"Unsupported token value type {type(token_bytes)!r} for vocab line: {line!r}")

    declared_length = int(line[last_space + 1:].strip())
    if strict_length and len(token_bytes) != declared_length:
        raise ValueError(
            f"Token length mismatch for vocab index {idx}: "
            f"declared={declared_length}, actual={len(token_bytes)}, line={line!r}"
        )

    return idx, token_bytes, declared_length


class TRIE_TOKENIZER():
    def __init__(
        self,
        file_name,
        *,
        strict_length: bool = False,
        special_first: bool = True,
    ):
        self.idx2token = {}
        mismatch_count = 0
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            idx, token_bytes, declared_length = parse_vocab_line(
                line,
                strict_length=strict_length,
            )
            if len(token_bytes) != declared_length:
                mismatch_count += 1
            self.idx2token[idx] = token_bytes

        if mismatch_count > 0:
            warnings.warn(
                f"Loaded {mismatch_count} vocab entries whose declared byte lengths did not "
                "match the parsed token bytes. Continuing because strict_length=False.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        self.special_first = bool(special_first)
        self.special_token2idx = {
            token.encode("utf-8"): self.token2idx[token.encode("utf-8")]
            for token in RWKV_SPECIAL_TOKENS
            if token.encode("utf-8") in self.token2idx
        }
        self._special_tokens_by_length = sorted(
            self.special_token2idx,
            key=len,
            reverse=True,
        )

        self.root = TRIE()
        for t, i in self.token2idx.items():
            _ = self.root.add(t, val=(t, i))

    def _encode_plain_bytes_with_spans(self, src: bytes, *, offset: int = 0):
        idx: int = 0
        tokens = []
        byte_spans = []
        while idx < len(src):
            start = idx
            idx, _, values = self.root.find_longest(src, idx)
            assert idx != start
            _, token = next(iter(values))
            tokens.append(token)
            byte_spans.append((offset + start, offset + idx))
        return tokens, byte_spans

    def _find_next_special_token(self, src: bytes, start: int):
        best_start = None
        best_token = None
        for token in self._special_tokens_by_length:
            token_start = src.find(token, start)
            if token_start < 0:
                continue
            if best_start is None or token_start < best_start:
                best_start = token_start
                best_token = token
        return best_start, best_token

    def encodeBytesWithSpans(self, src: bytes):
        if not self.special_first or not self.special_token2idx:
            return self._encode_plain_bytes_with_spans(src)

        tokens = []
        byte_spans = []
        cursor = 0
        while cursor < len(src):
            special_start, special_token = self._find_next_special_token(src, cursor)
            if special_start is None or special_token is None:
                plain_tokens, plain_spans = self._encode_plain_bytes_with_spans(
                    src[cursor:],
                    offset=cursor,
                )
                tokens.extend(plain_tokens)
                byte_spans.extend(plain_spans)
                break

            if special_start > cursor:
                plain_tokens, plain_spans = self._encode_plain_bytes_with_spans(
                    src[cursor:special_start],
                    offset=cursor,
                )
                tokens.extend(plain_tokens)
                byte_spans.extend(plain_spans)

            special_end = special_start + len(special_token)
            tokens.append(self.special_token2idx[special_token])
            byte_spans.append((special_start, special_end))
            cursor = special_end

        return tokens, byte_spans

    def encodeBytes(self, src: bytes):
        tokens, _ = self.encodeBytesWithSpans(src)
        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        try:
            return self.decodeBytes(tokens).decode('utf-8')
        except BaseException:
            return '\ufffd'  # bad utf-8

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except BaseException:
                pass
            print(f'{repr(s)}{i}', end=' ')
        print()
