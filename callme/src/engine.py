import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from enum import Enum, auto

from llm_sdk import Small_LLM_Model

# --- 1. THE TOKEN DETECTIVE ---
class VocabFilter:
    def __init__(self, vocab_path: str | Path):
        path = Path(vocab_path)
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary file not found at {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            self.vocab: Dict[str, int] = json.load(f)
            
        self.allowed_number_tokens = self._get_number_tokens()
        self.allowed_boolean_tokens = self._get_boolean_tokens()

    def _get_number_tokens(self) -> Set[int]:
        allowed_ids = set()
        number_pattern = re.compile(r'^[\d\.\-\sĠ]+$')
        for token_string, token_id in self.vocab.items():
            clean_token = token_string.replace('Ġ', '').strip()
            if clean_token == '': 
                continue
            if number_pattern.match(clean_token):
                try:
                    if clean_token in ['.', '-']:
                        allowed_ids.add(token_id)
                    else:
                        float(clean_token) 
                        allowed_ids.add(token_id)
                except ValueError:
                    pass
        return allowed_ids

    def _get_boolean_tokens(self) -> Set[int]:
        allowed_ids = set()
        valid_bool_strings = {'true', 'false', 'True', 'False', 'TRUE', 'FALSE'}
        for token_string, token_id in self.vocab.items():
            clean_token = token_string.replace('Ġ', '').strip()
            if clean_token in valid_bool_strings:
                allowed_ids.add(token_id)
        return allowed_ids

    def get_allowed_tokens(self, data_type: str) -> Set[int]:
        if data_type == "number":
            return self.allowed_number_tokens
        elif data_type == "boolean":
            return self.allowed_boolean_tokens
        elif data_type == "string":
            return set(self.vocab.values()) 
        return set()

# --- 2. THE STATE MACHINE ---
class ParserState(Enum):
    EXPECTING_BRACE_OPEN = auto()
    EXPECTING_FUNCTION_NAME_KEY = auto()
    EXPECTING_FUNCTION_NAME_VALUE = auto()
    EXPECTING_PARAMS_KEY = auto()
    EXPECTING_PARAMS_DICT_OPEN = auto()
    EXPECTING_ARG_NAME = auto()
    EXPECTING_ARG_VALUE_NUMBER = auto()
    EXPECTING_ARG_VALUE_STRING = auto()
    EXPECTING_BRACE_CLOSE = auto()
    DONE = auto()

# --- 3. THE GENERATION ENGINE ---
class ConstrainedEngine:
    def __init__(self, llm: Small_LLM_Model, vocab_filter: VocabFilter, max_tokens: int = 150):
        self.llm = llm
        self.vocab_filter = vocab_filter
        self.max_tokens = max_tokens
        self.vocab_size = len(self.vocab_filter.vocab)

    def _mask_logits(self, logits: List[float], allowed_token_ids: set[int]) -> List[float]:
        np_logits = np.array(logits, dtype=np.float32)
        mask = np.ones(self.vocab_size, dtype=bool)
        if allowed_token_ids:
            mask[list(allowed_token_ids)] = False 
        np_logits[mask] = np.NINF
        return np_logits.tolist()

    def _get_allowed_tokens_for_state(self, state: ParserState, expected_type: str = "") -> set[int]:
        if state == ParserState.EXPECTING_ARG_VALUE_NUMBER:
            return self.vocab_filter.get_allowed_tokens("number")
        elif state == ParserState.EXPECTING_ARG_VALUE_STRING:
            return self.vocab_filter.get_allowed_tokens("string")
        return set()

    def _transition_state(self, current_state: ParserState, generated_token_str: str) -> ParserState:
        if current_state == ParserState.EXPECTING_BRACE_OPEN and "{" in generated_token_str:
            return ParserState.EXPECTING_FUNCTION_NAME_KEY
        return current_state

    def generate_function_call(self, prompt: str, schema: List[Dict[str, Any]]) -> Optional[str]:
        try:
            input_ids_tensor = self.llm.encode(prompt)
            input_ids = input_ids_tensor[0].tolist() if input_ids_tensor.dim() > 1 else input_ids_tensor.tolist()
            
            generated_ids: List[int] = []
            current_state = ParserState.EXPECTING_BRACE_OPEN
            
            for _ in range(self.max_tokens):
                if current_state == ParserState.DONE:
                    break
                    
                current_context = input_ids + generated_ids
                logits = self.llm.get_logits_from_input_ids(current_context)
                
                allowed_ids = self._get_allowed_tokens_for_state(current_state)
                if allowed_ids:
                    logits = self._mask_logits(logits, allowed_ids)
                
                next_token_id = int(np.argmax(logits))
                generated_ids.append(next_token_id)
                
                next_token_str = self.llm.decode([next_token_id])
                current_state = self._transition_state(current_state, next_token_str)

            return self.llm.decode(generated_ids)

        except Exception as e:
            print(f"Generation error: {e}")
            return None