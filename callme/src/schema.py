import json
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError

class ParameterDetail(BaseModel):
    type: str

class FunctionReturn(BaseModel):
    type: str

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterDetail] = Field(default_factory=dict)
    returns: Optional[FunctionReturn] = None

def load_function_definitions(filepath: str | Path) -> List[FunctionDefinition]:
    path = Path(filepath)
    if not path.exists():
        print(f"Error: The schema file '{path}' does not exist.")
        return []

    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        return [FunctionDefinition(**item) for item in raw_data]
    except json.JSONDecodeError as e:
        print(f"Error: The file '{path}' is not valid JSON. Details: {e}")
        return []
    except ValidationError as e:
        print(f"Error: The data in '{path}' does not match the expected schema.\n{e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading the schema: {e}")
        return []