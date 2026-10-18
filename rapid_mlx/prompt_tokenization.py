# SPDX-License-Identifier: Apache-2.0
"""Tokenize a rendered prompt string without doubling its BOS.

Chat templates such as Llama 3's start with ``{{ bos_token }}``, so the rendered
chat prompt already begins with the BOS text.  Encoding that string with the
tokenizer's default ``add_special_tokens=True`` prepends a second BOS id, which
changes the model's output.  Every place that tokenizes a prompt string for
generation, context-length accounting or prefix-cache planning must apply the
same rule, otherwise the counts and the token ids the scheduler prefills drift
apart.  A prompt without a leading BOS text (raw completions, templates that
do not emit BOS) keeps the tokenizer's default special-token handling.
"""

from __future__ import annotations

from typing import Any


def adds_special_tokens(tokenizer: Any, prompt: str) -> bool:
    """Whether ``prompt`` should be encoded with the tokenizer's special tokens.

    ``False`` only when the prompt already starts with the tokenizer's BOS text.
    """

    bos = getattr(tokenizer, "bos_token", None)
    return not (isinstance(bos, str) and bos and prompt.startswith(bos))


def encode_prompt(tokenizer: Any, prompt: str) -> list[int]:
    """Encode ``prompt`` with exactly one BOS when the tokenizer adds one."""

    if adds_special_tokens(tokenizer, prompt):
        return list(tokenizer.encode(prompt))
    return list(tokenizer.encode(prompt, add_special_tokens=False))


__all__ = ["adds_special_tokens", "encode_prompt"]
