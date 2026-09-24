"""Shared tokenizer for every corpus-prep script and for generation/evaluation.

StarCoder2's BPE tokenizer, not GPT-2's. Chosen after measuring GPT-2 BPE's
efficiency on this repo's actual code-heavy corpus: 2.31 chars/token, versus
3.17 chars/token for StarCoder2 (27% fewer tokens, measured on the same real
text). cl100k/o200k are more efficient still, but were rejected for this
model scale: this repo's models are small and weight-tied (the token
embedding doubles as the output head), so vocabulary size sets a large,
non-architecture-dependent slice of total parameters. At d_model=1024,
cl100k's ~100k vocab would make the embedding table alone larger than the
entire 8-layer transformer body it sits alongside, and o200k's ~200k vocab
worse still — confounding exactly the "read the param column before the
loss" comparisons this harness exists to keep clean. StarCoder2's vocabulary
(49,152) is smaller than GPT-2's padded 50,304, so this switch improves
tokenisation efficiency AND shrinks the embedding table, rather than trading
one for the other.

49,152 is already a multiple of 64, so — unlike GPT-2's 50,257 padded up to
50,304 — no padding step is needed for tensor-core alignment.

Attribution: BigCode / StarCoder2 (2024). See NOTICE.md.
"""

HF_TOKENIZER_ID = 'bigcode/starcoder2-15b'
ENCODING_NAME = 'starcoder2'

_tokenizer = None


def get_tokenizer():
    """The shared Tokenizer instance. Downloaded once (tokenizer.json only,
    not any model weights) and cached in-process."""
    global _tokenizer
    if _tokenizer is None:
        from tokenizers import Tokenizer
        _tokenizer = Tokenizer.from_pretrained(HF_TOKENIZER_ID)
    return _tokenizer


def encode_ordinary(text):
    """Token ids for `text`, with no special tokens inserted.

    The tiktoken.encode_ordinary() equivalent: for tokenising raw document
    text, where the document-boundary token is appended separately by the
    caller (see eot_token()), not injected mid-text by the tokenizer itself.
    """
    return get_tokenizer().encode(text, add_special_tokens=False).ids


def eot_token():
    """Document-boundary token id (<|endoftext|>)."""
    return get_tokenizer().token_to_id('<|endoftext|>')


def vocab_size():
    """True vocabulary size. Already a multiple of 64 — no padding needed."""
    return get_tokenizer().get_vocab_size()


def decode(ids):
    return get_tokenizer().decode(ids)
