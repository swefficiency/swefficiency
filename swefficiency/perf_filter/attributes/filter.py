import argparse

from swefficiency.perf_filter import utils
from swefficiency.perf_filter.attributes import constants

from pathlib import Path

import json

import tqdm

import pandas as pd

def is_perf_pr(repo_name, pr):
    if repo_name in constants.REPO_PERF_FILTERS and constants.REPO_PERF_FILTERS[repo_name](pr):
        return True
        
    
    return constants.REPO_PERF_FILTERS['default'](pr)

def main(args):
    # First, create output dir if not exists.
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load PRs.
    repo_name = Path(args.prs_path).stem.replace("-prs", "")
    prs = utils.read_jsonl(args.prs_path, to_df=False)
    
    # Convert list of prs to df.
    prs_df = pd.DataFrame(prs)
    
    prs_df = prs_df[prs_df['merged_at'].notna()]
    
    instances = utils.read_jsonl(args.instances_path, to_df=False)
    
    print(f"Filtering attributes on {repo_name}")
    
    default_filter_func = constants.REPO_PERF_FILTERS['default']
    repo_pull_filter_func = constants.REPO_PERF_FILTERS.get(repo_name)
    if repo_pull_filter_func is None:
        print("repo specific filter function not found...")
        guarunteed_perf_prs = []
    else:
        guarunteed_perf_prs = prs_df[prs_df.apply(lambda row: repo_pull_filter_func(row), axis=1)]['number']
    possible_perf_prs = prs_df[prs_df.apply(lambda row: default_filter_func(row), axis=1)]['number']
    
    guarunteed_pr_numbers = set(guarunteed_perf_prs)
    possible_perf_prs = set(possible_perf_prs)
    print(len(guarunteed_perf_prs), len(possible_perf_prs))
    
    perf_pr_numbers = guarunteed_pr_numbers.union(possible_perf_prs)
    
    output_filename = Path(args.instances_path).stem.split(".")[0] + "_attribute.jsonl"
    output_path = output_dir / output_filename
    
    counter = 0
    with open(str(output_path), 'w') as output:
        for instance in tqdm.tqdm(instances, desc="Writing instances"):
            is_in_perf_pr_numbers = instance['pull_number'] in perf_pr_numbers
            has_perf_keywords_in_text = constants.filter_content(instance['problem_statement'])
            has_perf_keywords_in_text = False
            
            if not is_in_perf_pr_numbers and not has_perf_keywords_in_text:
                continue
            
            # Filter out changes that are markdown only.
            edits = utils.extract_edits(instance['patch'])
            if all(utils.is_doc_file(file_name) for file_name, _, _ in edits):
                continue
            
            if any(utils.has_lock_file_change(file_name) for file_name, _, _ in edits):
                continue
            
            print(json.dumps(instance), flush=True, file=output)
            counter += 1
            
    print(f"Identified {counter} possible performance task instances for {repo_name}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prs_path", type=str, help="Path to PRs file.", required=True)
    parser.add_argument("--instances_path", type=str, help="Path to candidate task instances file", required=True)
    parser.add_argument("--output_dir", type=str, help="Path to save results", required=True)
    args = parser.parse_args()
    main(args)