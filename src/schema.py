import json
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError


class ParameterDetail(BaseModel):
    """Represents the type of a specific parameter."""

    type: str


class FunctionReturn(BaseModel):
    """Represents the return type of the function."""

    type: str


class FunctionDefinition(BaseModel):
    """Represents a single function definition from the JSON input."""

    name: str
    description: str
    parameters: Dict[str, ParameterDetail] = Field(default_factory=dict)
    returns: Optional[FunctionReturn] = None


def load_function_definitions(
    filepath: str | Path
) -> List[FunctionDefinition]:
    """Loads and validates the function definitions JSON file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Error: The schema file '{path}' "
            "does not exist."
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        return [FunctionDefinition(**item) for item in raw_data]
    except json.JSONDecodeError as e:
        raise ValueError(f"Error: The file '{path}' "
                         f"is not valid JSON. Details: {e}")
    except ValidationError as e:
        raise ValueError(
            f"Error: The data in '{path}' "
            f"does not match the expected schema.\n{e}"
        )
