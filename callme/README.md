
## Description
This project is a constrained decoding engine designed for function calling in Large Language Models. The goal of this tool is to translate natural language prompts into strictly structured JSON function calls using a small 0.6B parameter model. Instead of relying on unpredictable raw text generation, it guides the model's output token-by-token to guarantee 100% valid JSON and perfect schema compliance.

## Instructions
This project uses `uv` for lightning-fast dependency management.

**Compilation & Installation:**
Ensure you have Python 3.10+ installed. To set up the virtual environment and install dependencies (including the provided `llm_sdk`):
`make install`

**Execution:**
The project runs as a Python module. To execute the main script:
`make run`

**Linting:**
To ensure strict typing and PEP8 compliance (while safely ignoring the provided SDK files):
`make lint`

## Resources
*   **Global Concepts:** General algorithms, tokenization principles, and Python concepts were researched via GeeksforGeeks, Medium articles, Reddit discussions, and YouTube tutorials.
*   **Peer Learning:** Extensive discussions with peers at 1337 helped solidify the architecture and clarify the expected behavior of the generation pipeline.
*   **AI Usage:** An AI assistant (Gemini) was used as a sparring partner to troubleshoot complex configuration issues. Specifically, AI helped debug Flake8 recursion crashes within the `.venv`, structure `pyproject.toml` and `.flake8` files to successfully bypass linter errors in the provided SDK without altering the mandatory Makefile commands, and clarify the roles of environment files like `uv.lock` and `.python-version`.

## Algorithm Explanation
The system uses Constrained Decoding to intervene during the LLM's token generation process. At each step, the model produces a probability distribution (logits) for the next token. Before selection, the algorithm evaluates which tokens maintain valid JSON syntax and comply with the required function schema. The logits for any invalid tokens are overridden and set to negative infinity. The model is then forced to sample only from the remaining valid tokens, guaranteeing structural integrity.

## Design Decisions
  **Module Structure:** The codebase is packaged as a module with `__main__.py` and `__init__.py` in the `src` directory to natively support the `-m src` execution required by the subject.
**Configuration Files:** Instead of modifying the Makefile (which is strictly forbidden), tool configurations (`pyproject.toml` for Mypy and `.flake8` for Flake8) are placed in the root directory to silently enforce environment rules and safely ignore the uneditable SDK folder during linting.

## Performance Analysis
By utilizing constrained decoding, the system achieves near-perfect reliability with a lightweight 0.6B model. The output accuracy for function selection and argument extraction remains high, and because invalid tokens are aggressively masked, the resulting JSON is 100% parseable without syntax errors. The processing speed is well within the 5-minute requirement for the test batch.

## Challenges Faced
A major technical hurdle was satisfying the strict, mandatory Makefile linting rules (`flake8 .` and `mypy .`) without failing on the uneditable `llm_sdk` or causing a recursion crash by accidentally scanning the `.venv` directory. This was solved by migrating configurations into a strictly defined `.flake8` file and a `pyproject.toml` override block to act as a trap-bypass for the linters.

## Testing Strategy
The implementation was validated against a variety of inputs from `function_calling_tests.json`. The testing strategy focused on edge cases such as missing keys, malformed prompt data, and type mismatches. Furthermore, the `make clean` and `make lint` targets were rigorously tested to ensure a pristine build environment and zero type-hinting violations.

## Example Usage
To run the engine with custom input and output paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json