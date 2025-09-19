import datasets

ds = datasets.load_dataset("swefficiency/swefficiency", split="test")

instance_id_to_github_url = {}

for d in ds:
    instance_id = d["instance_id"]
    repo = d["repo"]

    pull_number = instance_id.split("-")[-1]

    github_url = f"https://github.com/{repo}/pull/{pull_number}"
    instance_id_to_github_url[instance_id] = github_url

# Save to two column CSV file.
import pandas as pd

df = pd.DataFrame(
    instance_id_to_github_url.items(), columns=["instance_id", "github_url"]
)
df.to_csv("instance_id_to_github_url.csv", index=False)
