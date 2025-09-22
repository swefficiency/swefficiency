from pathlib import Path

import datasets

ds = datasets.load_dataset("swefficiency-anon/swefficiency", split="test")
instance_ids = set(d["instance_id"] for d in ds)

OUTPUT_DIR = Path("predictions/converted")

# model_name = "gemini25flash"
# model_name = "gpt5mini"
# model_name = "claude37sonnet"

for model_name in ["gemini25flash", "gpt5mini", "claude37sonnet"]:
    INPUT_FILE = Path(f"predictions/sweagent/{model_name}_raw.json")
    OUTPUT_FILE = OUTPUT_DIR / f"sweagent_{model_name}.jsonl"

    if not INPUT_FILE.exists():
        print(f"Input file {INPUT_FILE} does not exist, skipping.")
        continue

    # Read in JSONL file and convert to list of dict.
    import json

    raw_predictions = json.load(open(INPUT_FILE))

    predictions = []
    for instance_id, prediction in raw_predictions.items():
        predictions.append(prediction)

    # Write out to JSONL file.
    def write_jsonl(data, file_path):
        with open(file_path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    write_jsonl(predictions, OUTPUT_FILE)
