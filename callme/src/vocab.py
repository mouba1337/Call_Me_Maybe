import json
from pathlib import Path
from typing import Any, Dict, Set
from pydantic import BaseModel, ConfigDict, Field
from llm_sdk import Small_LLM_Model

class VocabFilter(BaseModel):
    """
    Manages the LLM's vocabulary using pure string operations (No Regex).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    vocab_path: Path
    llm: Small_LLM_Model
    vocab: Dict[str, int] = Field(default_factory=dict)
    text_cache: Dict[int, str] = Field(default_factory=dict)
    
    # Cache to ensure we only calculate allowed tokens once
    char_cache: Dict[str, Set[int]] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if not self.vocab_path.exists():
            raise FileNotFoundError(f"Vocab file not found at {self.vocab_path}")

        with open(self.vocab_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        normalized: Dict[str, int] = {}

    # Case A: {"token": 123}
        if raw and isinstance(next(iter(raw.keys())), str) and isinstance(next(iter(raw.values())), int):
            normalized = raw

    # Case B: {"123": "token"} or {123: "token"}
        elif raw and isinstance(next(iter(raw.values())), str):
            for k, v in raw.items():
                normalized[v] = int(k)

        else:
            raise ValueError("Unsupported vocab format")

        self.vocab = normalized
        self.text_cache = {
            token_id: token_str.replace("Ġ", " ")
            for token_str, token_id in self.vocab.items()
        }

    def get_tokens_by_chars(self, allowed_chars: str) -> Set[int]:
        """Returns tokens made ENTIRELY of the allowed characters."""
        if allowed_chars in self.char_cache:
            return self.char_cache[allowed_chars]

        allowed_set = set(allowed_chars)
        allowed_tokens = set()
        
        for token_id, clean_str in self.text_cache.items():
            if all(c in allowed_set for c in clean_str):
                allowed_tokens.add(token_id)
                
        self.char_cache[allowed_chars] = allowed_tokens
        return allowed_tokens

    def get_boolean_tokens(self) -> Set[int]:
        """Returns specific tokens valid for booleans."""
        if "boolean" in self.char_cache:
            return self.char_cache["boolean"]
            
        valid_words = {"true", "false", "True", "False", ",", "}", " ", "\n", "\t"}
        allowed_tokens = set()
        
        for token_id, clean_str in self.text_cache.items():
            if clean_str.strip() in valid_words or clean_str in valid_words:
                allowed_tokens.add(token_id)
                
        self.char_cache["boolean"] = allowed_tokens
        return allowed_tokens