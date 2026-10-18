"""A rendered chat prompt that already starts with BOS must be tokenized with one BOS.

Llama-3-style templates start with ``{{ bos_token }}``; encoding the rendered text
with the tokenizer's default special tokens used to prepend a second BOS id at
scheduler admission, while the route's context guard (``count_prompt_tokens``)
already counted a single one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from rapid_mlx.engine.batched import BatchedEngine
from rapid_mlx.prompt_host_cache import PromptHostCache
from rapid_mlx.prompt_tokenization import adds_special_tokens, encode_prompt
from rapid_mlx.scheduler import Scheduler
from rapid_mlx.service.helpers import count_prompt_tokens

BOS = "<|begin_of_text|>"


def _tokenizer(*, bos_token=BOS, adds_bos=True):
    backend = Tokenizer(
        WordLevel({"[UNK]": 0, BOS: 1, "hello": 2, "world": 3}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = Whitespace()
    if adds_bos:
        backend.post_processor = TemplateProcessing(
            single=f"{BOS} $A", special_tokens=[(BOS, 1)]
        )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", bos_token=bos_token
    )


def _scheduler(tokenizer, *, host_cache=False):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.tokenizer = tokenizer
    scheduler.config = SimpleNamespace(model_name="test-model")
    scheduler.prompt_host_cache = PromptHostCache(enabled=host_cache)
    return scheduler


def test_default_encode_doubles_template_bos():
    """The defect this module guards against: the baseline encode of a BOS-led prompt."""
    assert _tokenizer().encode(f"{BOS}hello world") == [1, 1, 2, 3]


@pytest.mark.parametrize("host_cache", [False, True])
def test_scheduler_admission_keeps_a_single_template_bos(host_cache):
    scheduler = _scheduler(_tokenizer(), host_cache=host_cache)
    assert scheduler._encode_prompt_string(f"{BOS}hello world") == [1, 2, 3]


def test_scheduler_admission_still_adds_bos_to_prompts_without_one():
    scheduler = _scheduler(_tokenizer())
    assert scheduler._encode_prompt_string("hello world") == [1, 2, 3]


def test_scheduler_admission_through_a_wrapped_tokenizer():
    wrapper = SimpleNamespace(tokenizer=_tokenizer())
    scheduler = _scheduler(wrapper)
    assert scheduler._encode_prompt_string(f"{BOS}hello") == [1, 2]


def test_tokenizer_without_bos_text_is_untouched():
    tok = _tokenizer(bos_token=None, adds_bos=False)
    assert adds_special_tokens(tok, "hello")
    assert encode_prompt(tok, "hello world") == tok.encode("hello world")


def test_tokenizer_without_default_bos_is_not_given_one():
    """A BOS-led prompt on a tokenizer that adds nothing stays as rendered."""
    tok = _tokenizer(adds_bos=False)
    assert encode_prompt(tok, f"{BOS}hello") == [1, 2]


def test_context_guard_count_matches_scheduler_ids():
    tokenizer = _tokenizer()
    scheduler = _scheduler(tokenizer, host_cache=True)
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._is_mllm = False
    engine._tokenizer = tokenizer
    engine._engine = SimpleNamespace(engine=SimpleNamespace(scheduler=scheduler))

    for prompt in (f"{BOS}hello world", "hello world"):
        assert count_prompt_tokens(engine, prompt) == len(
            scheduler._encode_prompt_string(prompt)
        )
