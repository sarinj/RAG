from __future__ import annotations

import hashlib
from pathlib import Path

from tokenizers import Tokenizer


def load_tokenizer(path: Path | None, expected_version: str) -> Tokenizer:
    if path is None:
        raise ValueError("TOKENIZER_PATH is not configured")
    try:
        raw = path.read_bytes()
        tokenizer = Tokenizer.from_file(str(path))
    except Exception as exc:
        raise ValueError("The configured tokenizer file is unavailable or invalid") from exc
    actual_version = f"sha256:{hashlib.sha256(raw).hexdigest()}"
    if expected_version != actual_version:
        raise ValueError("The tokenizer does not match the shared index configuration")
    return tokenizer
