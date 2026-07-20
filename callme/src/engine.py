"""
Core Constrained Generation Engine for callmemaybe.
Handles the state machine and logit masking for strict JSON generation.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from enum import Enum, auto

# Assuming llm_sdk is placed next to src as per the subject guidelines
from llm_sdk import Small_LLM_Model
from src.vocab_filter import VocabFilter

class ParserState(Enum):
    """
    Defines the current state of the JSON generation.
    This tells the engine what type of token is legally allowed next.
    """
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

class ConstrainedEngine:
    """
    The main engine that drives the LLM generation token by token,
    enforcing JSON structure via logit manipulation.
    """

    def __init__(self, llm: Small_LLM_Model, vocab_filter: VocabFilter, max_tokens: int = 150):
        """
        Initializes the engine with the required models and tools.
        """
        self.llm = llm
        self.vocab_filter = vocab_filter
        self.max_tokens = max_tokens
        self.vocab_size = len(self.vocab_filter.vocab)

    def _mask_logits(self, logits: List[float], allowed_token_ids: set[int]) -> List[float]:
        """
        Applies constrained decoding by setting the probability of any token
        NOT in the allowed_token_ids set to negative infinity.
        """
        # Convert to numpy array for fast mathematical operations
        np_logits = np.array(logits, dtype=np.float32)
        
        # Create a boolean mask where True means the token is NOT allowed
        mask = np.ones(self.vocab_size, dtype=bool)
        if allowed_token_ids:
            allowed_list = list(allowed_token_ids)
            mask[allowed_list] = False # Unmask the allowed tokens
            
        # Apply negative infinity to all disallowed tokens
        np_logits[mask] = np.NINF
        
        return np_logits.tolist()

    def _get_allowed_tokens_for_state(self, state: ParserState, expected_type: str = "") -> set[int]:
        """
        Consults the VocabFilter based on the current parser state.
        """
        if state == ParserState.EXPECTING_ARG_VALUE_NUMBER:
            return self.vocab_filter.get_allowed_tokens("number")
        elif state == ParserState.EXPECTING_ARG_VALUE_STRING:
            return self.vocab_filter.get_allowed_tokens("string")
        
        # TODO: Expand this to handle specific JSON structural tokens 
        # like '{', '}', '"', ':', and function names from the schema.
        # For now, it returns an empty set to be built upon.
        return set()

    def _transition_state(self, current_state: ParserState, generated_token_str: str) -> ParserState:
        """
        Advances the state machine based on the token that was just generated.
        This is a simplified transition map that needs to be expanded based on schema.
        """
        # Example logic: if we just generated a '{', move to expecting a key
        if current_state == ParserState.EXPECTING_BRACE_OPEN and "{" in generated_token_str:
            return ParserState.EXPECTING_FUNCTION_NAME_KEY
            
        # Example logic: if we just finished typing a number, look for a comma or brace
        if current_state == ParserState.EXPECTING_ARG_VALUE_NUMBER:
            # You will need to build logic to detect when the number is "finished"
            pass
            
        return current_state

    def generate_function_call(self, prompt: str, schema: List[Dict[str, Any]]) -> Optional[str]:
        """
        The main generation loop. Processes the prompt and forces the LLM
        to output valid JSON matching the schema.
        """
        try:
            # 1. Tokenization: Convert prompt to input IDs
            input_ids_tensor = self.llm.encode(prompt)
            input_ids = input_ids_tensor.tolist()
            
            generated_ids: List[int] = []
            current_state = ParserState.EXPECTING_BRACE_OPEN
            
            # The Generation Pipeline Loop
            for _ in range(self.max_tokens):
                if current_state == ParserState.DONE:
                    break
                    
                # 2. Get Logits: Ask the model for next token probabilities
                current_context = input_ids + generated_ids
                logits = self.llm.get_logits_from_input_ids(current_context)
                
                # 3. Determine Constraints: What are we allowed to type right now?
                # (You will need to pass the current expected type from the schema)
                allowed_ids = self._get_allowed_tokens_for_state(current_state)
                
                # 4. Constrained Decoding: Mask invalid tokens
                if allowed_ids:
                    logits = self._mask_logits(logits, allowed_ids)
                
                # 5. Token Selection: Pick the highest probability valid token (Argmax)
                next_token_id = int(np.argmax(logits))
                generated_ids.append(next_token_id)
                
                # 6. Decode and Transition State
                next_token_str = self.llm.decode([next_token_id])
                current_state = self._transition_state(current_state, next_token_str)

            return self.llm.decode(generated_ids)

        except Exception as e:
            # Graceful error handling as explicitly required by the subject
            print(f"Generation error: {e}")
            return None