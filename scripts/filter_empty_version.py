import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("input_path", type=str)
args = parser.parse_args()

output_path = args.input_path.replace(".json", ".non-empty.jsonl")
df = pd.read_json(args.input_path, lines=False, dtype={"version": str})
print(df.head())
print(f"Got {len(df)} instances")
df = df[df["version"].notna()]
df["version"] = df["version"].astype(str)
print(f"Got {len(df)} non-empty versions instances")

# Lowercase instance id.
df["instance_id"] = df["instance_id"].str.lower()
df["repo"] = df["repo"].str.lower()

print(df['test_patch'].apply(lambda x: len(x)).describe())

df.to_json(output_path, lines=True, orient="records")
print(f"Filtered {args.input_path} to {output_path}")

print("Version Stats:")
print(df["version"].value_counts())

print("Unique versions:")
print(sorted(list(df["version"].unique())))
