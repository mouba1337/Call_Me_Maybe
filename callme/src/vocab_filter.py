import json
import re
from pathlib import Path
from typing import Dict, Set

class VocabFilter:
    """
    The VocabFilter acts as the 'Token Detective'.
    It loads the model's BPE vocabulary and categorizes tokens into safe lists
    based on what data types they represent (numbers, booleans, strings).
    """

    def __init__(self, vocab_path: str | Path):
        """
        Initializes the filter by loading the vocab.json file into memory.
        """
        path = Path(vocab_path)
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary file not found at {path}")
        
        # Load the vocabulary dictionary: {"token_string": token_id, ...}
        with open(path, 'r', encoding='utf-8') as f:
            self.vocab: Dict[str, int] = json.load(f)
            
        # Pre-compute the allowed token lists so we don't have to do it 
        # during the high-speed LLM generation loop.
        self.allowed_number_tokens = self._get_number_tokens()
        self.allowed_boolean_tokens = self._get_boolean_tokens()
        
        print(f"Vocab loaded: {len(self.vocab)} total tokens.")
        print(f" - Found {len(self.allowed_number_tokens)} number-related tokens.")
        print(f" - Found {len(self.allowed_boolean_tokens)} boolean-related tokens.")

    def _get_number_tokens(self) -> Set[int]:
        """
        Scans the vocabulary for any token that represents a valid part of a number.
        This includes digits (0-9), decimal points (.), and negative signs (-).
        """
        allowed_ids = set()
        
        # A regex pattern that matches standard digits, decimals, or negative signs.
        # Because BPE tokenizers often add special prefixes (like 'Ġ' for space), 
        # we check if the token contains only characters valid in a float.
        number_pattern = re.compile(r'^[\d\.\-\sĠ]+$')
        
        for token_string, token_id in self.vocab.items():
            # Clean the token string (remove the special 'Ġ' space marker if it exists)
            clean_token = token_string.replace('Ġ', '').strip()
            
            if clean_token == '': 
                continue
                
            # If the clean token consists entirely of numeric characters/symbols
            if number_pattern.match(clean_token):
                # Extra check to ensure it doesn't have multiple decimals (e.g. '1.2.3')
                try:
                    # Allow structural number pieces
                    if clean_token in ['.', '-']:
                        allowed_ids.add(token_id)
                    else:
                        # If this parses as a float, it's a valid number chunk
                        float(clean_token) 
                        allowed_ids.add(token_id)
                except ValueError:
                    # Fails if it's something like '1.2.3'
                    pass
                    
        return allowed_ids

    def _get_boolean_tokens(self) -> Set[int]:
        """
        Scans the vocabulary for BPE tokens that represent true or false.
        """
        allowed_ids = set()
        valid_bool_strings = {'true', 'false', 'True', 'False', 'TRUE', 'FALSE'}
        
        for token_string, token_id in self.vocab.items():
            # Clean the BPE space marker and standard spaces
            clean_token = token_string.replace('Ġ', '').strip()
            # We match exactly to avoid matching random words containing "true"
            if clean_token in valid_bool_strings:
                allowed_ids.add(token_id)
                
        return allowed_ids

    def get_allowed_tokens(self, data_type: str) -> Set[int]:
        """
        Returns the set of allowed token IDs for a given JSON schema data type.
        """
        if data_type == "number":
            return self.allowed_number_tokens
        elif data_type == "boolean":
            return self.allowed_boolean_tokens
        elif data_type == "string":
            # For strings, we typically allow everything except structural JSON tokens 
            # like unescaped quotes. In an advanced implementation, you dynamically 
            # manage quotes inside the state machine.
            return set(self.vocab.values()) 
        else:
            # Fallback for unknown types
            return set()

if __name__ == "__main__":
    # Quick local test (runs only if you execute this specific file directly)
    # Note: You will need a real vocab.json file to run this locally!
    test_path = Path("../data/input/vocab.json")
    
    if test_path.exists():
        filter = VocabFilter(test_path)
        print("Success! Tokens categorized.")
    else:
        print("Note: Create a dummy vocab.json in data/input/ to test this directly.")