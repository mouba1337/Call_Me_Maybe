import argparse
import json
from pathlib import Path
from llm_sdk import Small_LLM_Model
from src.schema import load_function_definitions  # Assuming your pydantic code is here
from src.vocab_filter import VocabFilter
from src.engine import ConstrainedEngine

def main():
    parser = argparse.ArgumentParser(description="Call Me Maybe - Constrained Generation CLI")
    parser.add_argument("--functions_definition", type=str, default="data/input/functions_definition.json")
    parser.add_argument("--input", type=str, default="data/input/function_calling_tests.json")
    parser.add_argument("--output", type=str, default="data/output/function_calling_results.json")
    args = parser.parse_args()

    # 1. Graceful Schema and Input Loading
    functions = load_function_definitions(args.functions_definition)
    if not functions:
        print("Failed to load function schemas. Exiting.")
        return

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            tests = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    # 2. Initialize Core components
    print("Loading LLM model components...")
    llm = Small_LLM_Model()
    vocab_path = llm.get_path_to_vocab_file()
    vocab_filter = VocabFilter(vocab_path)
    engine = ConstrainedEngine(llm, vocab_filter)

    results = []

    # 3. Main execution loop over prompts
    for test in tests:
        prompt = test.get("prompt", "")
        print(f"Processing prompt: {prompt}")
        
        # Run through your constrained state-machine
        # Note: You'll pass your pydantic schema list here
        raw_output = engine.generate_function_call(prompt, [f.model_dump() for f in functions])
        
        if raw_output:
            try:
                # Parse the generated valid string into the required final format
                parsed_json = json.loads(raw_output)
                results.append({
                    "prompt": prompt,
                    "name": parsed_json.get("name"),
                    "parameters": parsed_json.get("parameters", {})
                })
            except Exception as parse_err:
                print(f"Error structuring output JSON: {parse_err}")
                
    # 4. Save results securely using context managers
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, indent=4)
    print(f"Successfully wrote results to {output_path}")

if __name__ == "__main__":
    main()