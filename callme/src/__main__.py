import argparse
import json
from pathlib import Path

from llm_sdk import Small_LLM_Model
from src.engine import ConstrainedEngine
from src.schema import load_function_definitions
from src.vocab import VocabFilter
import time 


def main() -> None:
    start_all = time.time()
    """Main CLI entry point for the Call Me Maybe constrained generator."""
    parser = argparse.ArgumentParser(
        description="Call Me Maybe - Constrained Generation CLI"
    )

    # 1. Subject-compliant argument parsing
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the JSON file containing available functions.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the JSON file containing prompts.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path to write the resulting JSON array.",
    )
    args = parser.parse_args()

    # 2. Load and validate function definitions using Pydantic
    try:
        functions = load_function_definitions(args.functions_definition)
        if not functions:
            print("Failed to load valid function schemas. Exiting.")
            return
    except Exception as e:
        print(f"Error loading function definitions: {e}")
        return

    # 3. Load input prompts
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            tests = json.load(f)
    except Exception as e:
        print(f"Error reading input file '{args.input}': {e}")
        return

    print("Loading LLM model components...")
    t0 = time.time()

    llm = Small_LLM_Model()
    print(f"[TIME] model init: {time.time() - t0:.2f}s")

    vocab_path = Path(llm.get_path_to_vocab_file())
    vocab_filter = VocabFilter(vocab_path=vocab_path, llm=llm)
    engine = ConstrainedEngine(llm=llm, vocab_filter=vocab_filter)

    results = []

    # 4. Process each prompt through the LLM engine
    for i, test in enumerate(tests):
        t1 = time.time()

        prompt = test.get("prompt", "") if isinstance(test, dict) else ""
        if not prompt:
            continue

        raw_output = engine.generate_function_call(
            prompt,
            [f.model_dump() for f in functions],
            max_tokens=40
        )
        print(f"[TIME] prompt #{i}: {time.time() - t1:.2f}s")

        if raw_output:
            try:
                parsed_json = json.loads(raw_output)
                results.append(
                    {
                        "prompt": prompt,
                        "name": parsed_json.get("name"),
                        "parameters": parsed_json.get("parameters", {}),
                    }
                )
            except Exception:
                results.append(
                    {
                        "prompt": prompt,
                        "name": None,
                        "parameters": {},
                    }
                )
        else:
            results.append(
                {
                    "prompt": prompt,
                    "name": None,
                    "parameters": {},
                }
            )
    print(f"[TIME] total runtime: {time.time() - start_all:.2f}s")
    # 5. Write the final compliant output file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(results, out_f, indent=4)
        print(f"Successfully wrote results to {output_path}")
    except Exception as e:
        print(f"Error writing to output file: {e}")


if __name__ == "__main__":
    main()