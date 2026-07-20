import json
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError

# ==========================================
# 1. PYDANTIC MODELS (The Rulebook)
# ==========================================

class ParameterDetail(BaseModel):
    """Validates the inner dictionary of a parameter, e.g., {"type": "number"}"""
    type: str

class FunctionReturn(BaseModel):
    """Validates the return block, e.g., {"type": "string"}"""
    type: str

class FunctionDefinition(BaseModel):
    """
    Validates a complete function object from the JSON array.
    Pydantic automatically ensures 'name' and 'description' are present.
    """
    name: str
    description: str
    # A dictionary where the key is the argument name (e.g., 'a', 's') 
    # and the value is the ParameterDetail. Defaults to empty dict if missing.
    parameters: Dict[str, ParameterDetail] = Field(default_factory=dict)
    returns: Optional[FunctionReturn] = None


# ==========================================
# 2. LOADER & ERROR HANDLER
# ==========================================

def load_function_definitions(filepath: str | Path) -> List[FunctionDefinition]:
    """
    Safely loads, parses, and validates the functions_definition.json file.
    Gracefully catches and reports errors to prevent hard crashes.
    """
    path = Path(filepath)
    
    # Error Handling 1: Missing File
    if not path.exists():
        print(f"Error: The schema file '{path}' does not exist.")
        return []

    try:
        # Error Handling 2: Invalid JSON Syntax
        with open(path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        # Error Handling 3: Pydantic Validation Failures
        validated_functions = [FunctionDefinition(**item) for item in raw_data]
        return validated_functions
        
    except json.JSONDecodeError as e:
        print(f"Error: The file '{path}' is not valid JSON. Details: {e}")
        return []
    except ValidationError as e:
        print(f"Error: The data in '{path}' does not match the expected schema.")
        print(f"Pydantic Details:\n{e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading the schema: {e}")
        return []

# Quick local test (runs only if you execute this specific file directly)
if __name__ == "__main__":
    test_path = Path("../data/input/functions_definition.json")
    functions = load_function_definitions(test_path)
    for func in functions:
        print(f"Loaded: {func.name} -> Parameters: {list(func.parameters.keys())}")
