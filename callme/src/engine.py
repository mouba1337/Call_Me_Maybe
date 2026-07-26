import json
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set

import numpy as np
from pydantic import BaseModel, ConfigDict
from llm_sdk import Small_LLM_Model

from src.vocab import VocabFilter


class ParserState(Enum):
    EXPECTING_STRUCTURE = auto()
    EXPECTING_PARAM_VALUE = auto()


class ConstrainedEngine(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Small_LLM_Model
    vocab_filter: VocabFilter
    max_tokens: int = 150

    def _mask_logits(self, logits: List[float], allowed_ids: Set[int]) -> List[float]:
        if not allowed_ids:
            return logits

        np_logits = np.array(logits, dtype=np.float32)
        vocab_len = len(logits)

        mask = np.ones(vocab_len, dtype=bool)
        valid_indices = [tid for tid in allowed_ids if tid < vocab_len]
        
        if valid_indices:
            mask[valid_indices] = False

        np_logits[mask] = -float("inf")
        return np_logits.tolist()


    def _determine_expected_type(self, current_text: str, schema: List[Dict[str, Any]]) -> str:
        """
        Determine expected token type while generating JSON.
        Important: distinguish whether we're typing a KEY or a VALUE.
        """

    # -----------------------------
    # 0) If we're currently inside an open quote, detect if this is a KEY
    # -----------------------------
        last_quote = current_text.rfind('"')
        if last_quote != -1:
        # odd number of quotes => currently inside a string literal
            if current_text.count('"') % 2 == 1:
            # Find previous non-space char before the opening quote
                i = last_quote - 1
                while i >= 0 and current_text[i].isspace():
                    i -= 1

            # If quote follows { or , we're typing an object key, not a value
                if i >= 0 and current_text[i] in "{,":
                    return "string"   # allow alphabet; do NOT schema-mask as number/bool/etc.

    # -----------------------------
    # 1) Find function name
    # -----------------------------
        func_name = ""
        name_idx = current_text.rfind('"name"')
        if name_idx != -1:
            colon_idx = current_text.find(':', name_idx)
            if colon_idx != -1:
                first_q = current_text.find('"', colon_idx)
                if first_q != -1:
                    second_q = current_text.find('"', first_q + 1)
                    if second_q != -1:
                        func_name = current_text[first_q + 1:second_q]

    # -----------------------------
    # 2) Find current argument key (best effort)
    # -----------------------------
        arg_name = ""
        parts = current_text.split(':')
        if len(parts) >= 2:
            last_key_part = parts[-2]
            quotes = last_key_part.split('"')
            if len(quotes) >= 3:
                arg_name = quotes[-2]

    # -----------------------------
    # 3) Schema lookup
    # -----------------------------
        for fn in schema:
            if fn.get("name") == func_name:
                return fn.get("parameters", {}).get(arg_name, {}).get("type", "string")

        return "string"

    def _determine_allowed_tokens(self, state: ParserState, current_text: str, schema: List[Dict[str, Any]], current_step: int) -> Set[int]:
        if state == ParserState.EXPECTING_PARAM_VALUE:
            expected_type = self._determine_expected_type(current_text, schema)
            
            if expected_type in ("number", "integer", "float"):
                return self.vocab_filter.get_tokens_by_chars("0123456789.- \t\n,}")
            elif expected_type == "boolean":
                return self.vocab_filter.get_boolean_tokens()
            else:
                # String type: Allow all valid tokens, UNLESS we are running out of time.
                # If we are within 3 tokens of the maximum limit, force it to close the quote.
                if self.max_tokens - current_step <= 3:
                    return self.vocab_filter.get_tokens_by_chars('"')
                    
                return set(self.vocab_filter.vocab.values())

        # Structure mode: Only allow numbers and basic JSON syntax (No alphabet!)
        structural_chars = '0123456789_:,{}[]" \n\t.-'
        return self.vocab_filter.get_tokens_by_chars(structural_chars)

    def _update_state(self, current_text: str) -> ParserState:
        # Strip escaped quotes to safely count actual string delimiters
        clean_text = current_text.replace('\\"', '')
        if clean_text.count('"') % 2 != 0:
            return ParserState.EXPECTING_PARAM_VALUE

        # If outside a string, check if we are immediately after a colon
        if current_text.rstrip().endswith(':'):
            return ParserState.EXPECTING_PARAM_VALUE

        return ParserState.EXPECTING_STRUCTURE

    def generate_function_call(self, prompt: str, schema: List[Dict[str, Any]]) -> Optional[str]:
        try:
            sys_prompt = (
                "You are a strict JSON formatting AI. Output ONLY a valid JSON object.\n"
                'Format: {"name": "function_name", "parameters": {"arg_name": arg_value}}\n\n'
                f"Available tools:\n{json.dumps(schema)}\n\n"
                f"User Request: {prompt}\n"
                "JSON Output:\n"
            )

            input_ids_tensor = self.llm.encode(sys_prompt)
            input_ids = input_ids_tensor[0].tolist() if input_ids_tensor.dim() > 1 else input_ids_tensor.tolist()
            
            generated_ids: List[int] = []
            current_state = ParserState.EXPECTING_STRUCTURE

            for _ in range(self.max_tokens):
                current_text = self.llm.decode(generated_ids) if generated_ids else ""
                
                # Instant early stopping without regex
                clean_text = current_text.strip()
                if clean_text.startswith("{") and clean_text.endswith("}") and clean_text.count("{") == clean_text.count("}") and clean_text.count("{") > 0:
                    try:
                        json.loads(clean_text)
                        break 
                    except Exception:
                        pass

                context = input_ids + generated_ids
                logits = self.llm.get_logits_from_input_ids(context)

                allowed_ids = self._determine_allowed_tokens(current_state, current_text, schema, len(generated_ids))

                if allowed_ids:
                    logits = self._mask_logits(logits, allowed_ids)

                next_token_id = int(np.argmax(logits))
                generated_ids.append(next_token_id)

                new_text = self.llm.decode(generated_ids)
                current_state = self._update_state(new_text)

            raw_output = self.llm.decode(generated_ids).strip()
            
            # Manual extraction of the JSON without regex
            start = raw_output.find("{")
            end = raw_output.rfind("}")
            if start != -1 and end != -1 and end > start:
                return raw_output[start:end+1]
                
            return raw_output

        except Exception as e:
            print(f"Engine failure: {e}")
            return None