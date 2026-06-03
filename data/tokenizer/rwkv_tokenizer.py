########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

from __future__ import annotations

import ast
import warnings

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
    def __init__(self, file_name, *, strict_length: bool = False):
        self.idx2token = {}
        sorted = []  # must be already sorted
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
            sorted += [token_bytes]
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

        self.root = TRIE()
        for t, i in self.token2idx.items():
            _ = self.root.add(t, val=(t, i))

    def encodeBytes(self, src: bytes):
        idx: int = 0
        tokens = []
        while (idx < len(src)):
            _idx: int = idx
            idx, _, values = self.root.find_longest(src, idx)
            assert (idx != _idx)
            _, token = next(iter(values))
            tokens.append(token)
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
