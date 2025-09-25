from pathlib import Path

import datasets

ds = datasets.load_dataset("swefficiency/swefficiency", split="test")
instance_ids = set(d["instance_id"] for d in ds)

OUTPUT_DIR = Path("/predictions/converted")

# model_name = "gemini25flash"
# model_name = "gpt5mini"
# model_name = "claude37sonnet"

for model_name in ["gemini25flash", "gpt5mini", "claude37sonnet"]:
    INPUT_FILE = f"predictions/openhands/{model_name}_raw.jsonl"
    OUTPUT_FILE = OUTPUT_DIR / f"oh_{model_name}.jsonl"

    # Read in JSONL file and convert to list of dict.
    import json

    def read_jsonl(file_path):
        data = []
        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    raw_predictions = read_jsonl(INPUT_FILE)

    predictions = []
    for item in raw_predictions:
        if not item["metadata"]:
            continue

        if item["instance_id"] not in instance_ids:
            print(
                f"Warning: instance_id {item['instance_id']} not in dataset, skipping."
            )
            continue
        eval_entry = {
            "instance_id": item["instance_id"],
            "model_patch": item["test_result"].get("git_patch", ""),
            "model_name_or_path": item["metadata"]["eval_output_dir"].split("/")[-1],
        }
        predictions.append(eval_entry)

    # Write out to JSONL file.
    def write_jsonl(data, file_path):
        with open(file_path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    write_jsonl(predictions, OUTPUT_FILE)
