"""Regression test for the chat-template tokenization normalizer.

`apply_chat_template(tokenize=True)` returns different container types across
transformers versions: a flat list[int] (older / return_dict=False) or a
dict-like BatchEncoding (newer). Feeding the latter to torch.tensor iterates
its string keys → "'str'/'dict' object cannot be interpreted as an integer"
(observed on Colab). `chat_template_token_ids` must normalize all shapes to a
flat list[int]. No model load — fake tokenizer returns each shape.
"""

from __future__ import annotations

import torch
from transformers import BatchEncoding

from nla.schema import chat_template_token_ids

_IDS = [11, 22, 33, 44]
_MSGS = [{"role": "user", "content": "x"}]


class _FakeTokenizer:
    """Returns a fixed token sequence in a chosen container, ignoring kwargs
    (simulates a transformers version that returns that shape regardless)."""

    def __init__(self, mode: str):
        self.mode = mode

    def apply_chat_template(self, messages, **kwargs):
        if self.mode == "list":
            return list(_IDS)  # old behavior: flat list[int]
        if self.mode == "tensor":
            return torch.tensor([_IDS])  # [1, T]
        if self.mode == "batchencoding":  # the Colab failure mode
            return BatchEncoding(
                {"input_ids": torch.tensor([_IDS]), "attention_mask": torch.tensor([[1, 1, 1, 1]])}
            )
        raise ValueError(self.mode)


def test_normalizes_flat_list():
    assert chat_template_token_ids(_FakeTokenizer("list"), _MSGS) == _IDS


def test_normalizes_tensor():
    assert chat_template_token_ids(_FakeTokenizer("tensor"), _MSGS) == _IDS


def test_normalizes_batchencoding():
    # This is exactly what newer transformers returns and what broke the gate.
    assert chat_template_token_ids(_FakeTokenizer("batchencoding"), _MSGS) == _IDS


def test_returns_python_ints():
    out = chat_template_token_ids(_FakeTokenizer("tensor"), _MSGS)
    assert all(isinstance(x, int) for x in out)
