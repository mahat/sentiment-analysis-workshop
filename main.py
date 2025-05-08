import argparse
import json
import sys

from experiments import (
    simple_prompt_experiment,
    few_shot_experiment,
    cot_experiment,
    utils,
)
from datasets import load_from_disk

# llama paths
# model names
small_model_repo_id = "bartowski/Qwen2.5-0.5B-Instruct-GGUF"
small_model_file_name = "Qwen2.5-0.5B-Instruct-IQ2_M.gguf"

big_model_repo_id = "bartowski/Qwen2.5-1.5B-Instruct-GGUF"
big_model_file_name = "Qwen2.5-1.5B-Instruct-IQ2_M.gguf"

# load in context examples
incontext_ds_small = load_from_disk("./data/incontext_small.hf")

# using lambda function for lazy loading 
models = {
    "simple_prompt_small_llm": {
        "instance": lambda: simple_prompt_experiment.SimplePromptExperiment(
            repo_id=big_model_repo_id, file_name=big_model_file_name, n_ctx=4096
        ),
        "params": {"temperature": 0.2, "min_p": 0.45},
    },
    "simple_prompt_big_llm": {
        "instance": lambda: simple_prompt_experiment.SimplePromptExperiment(
            repo_id=big_model_repo_id, file_name=big_model_file_name, n_ctx=4096
        ),
    },
    "few_shot": {
        "instance": lambda: few_shot_experiment.FewShotExperiment(
            repo_id=small_model_repo_id,
            file_name=small_model_file_name,
            n_ctx=1024 * 8,
            valid_ds=[],
            incoxtext_examples=incontext_ds_small,
            pass_k=5,
        ),
        "params": {"temperature": 0.3, "min_p": 0.05, "logit_bias": {35490: -100}},
    },
    "CoT": {
        "instance": lambda: cot_experiment.CotExperiment(
            repo_id=small_model_repo_id, file_name=small_model_file_name, n_ctx=1024 * 4
        ),
        "params": {"max_tokens": -1, "temperature": 1.0},
    },
}

def list_models():
    """
    Lists the available models.
    """
    return "\n".join(models.keys())


def run_model(model_name, text):
    """
    runs model name with text
    """
    if model_name not in models:
        return json.dumps({"error": f"Model '{model_name}' not found."})

    mdl = models[model_name]["instance"]()
    for p in models[model_name].get("params", {}):
        setattr(mdl, p, models[model_name]["params"][p])
    response = mdl.single_result_generation(text)
    return {"model_name": model_name, "response": response}


def main():
    """
    Main function to parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="CLI for interacting with models.")
    parser.add_argument("command", choices=["list", "run", "eval"], help="Command to execute.")
    parser.add_argument(
        "--model", help="Name of the model to use with the 'run' command."
    )
    parser.add_argument("--text", help="Text to process with the 'run' command.")
    parser.add_argument("--out_path", help="output directory with the 'eval' command")
    parser.add_argument("--ds_path", help="Dataset Path to process with the 'eval' command.")

    args = parser.parse_args()

    if args.command == "list":
        result = list_models()
        print(result)
    elif args.command == "run":
        if not args.model or not args.text:
            print(
                json.dumps(
                    {
                        "error": "Both --model and --text are required for the 'run' command."
                    }
                )
            )
            sys.exit(1)
        result = run_model(args.model, args.text)
        print(result)
    elif args.command == "eval":
        utils.experiment_runner(models=models,ds_path=args.ds_path, out_path=args.out_path)
        print(f'Evaluation completed you can find the results in: {args.out_path}')
    else:
        print(json.dumps({"error": "Invalid command."}))  # Should never reach here
        sys.exit(1)


if __name__ == "__main__":
    main()
