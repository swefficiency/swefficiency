import glob
import os
from pathlib import Path
import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe

gc = gspread.service_account()

sheet_name = "global_sweperf_all_data_annotate"

gc = gspread.service_account()
sh = gc.open(sheet_name)
worksheet = sh.get_worksheet(0)

# Write the dataframe to this worksheet.
v1_df = get_as_dataframe(
    worksheet,
    header=0,
    dtype=str,
)

REPOS = [
    "astropy",
    "dask",
    "matplotlib",
    "numpy",
    "scikit-learn",
    "scipy",
    "sympy",
    "xarray",
    "pandas",
]

run_directory = "logs/run_evaluation"
folder_pattern = "perf_profiling_20250612_{repo}/gold/*"

instance_ids = "pandas-dev__pandas-40840 pandas-dev__pandas-53195"
instance_ids = set(instance_ids.split())

for repo in REPOS:
    folder_path = os.path.join(run_directory, folder_pattern.format(repo=repo))

    for instance_folder in glob.glob(folder_path):
        instance_id = os.path.basename(instance_folder)

        # Find the row with the matching instance_id, assuming it exists.
        row_status = v1_df[v1_df["instance_id"] == instance_id].iloc[0]["status"]

        # row_status != "NEEDS_ANNOTATE" and
        if instance_id not in instance_ids:
            continue

        print(f"Processing instance: {instance_id}")

        row_index = v1_df.index[v1_df["instance_id"] == instance_id]

        perf_summary_file = Path(instance_folder) / "perf_summary.txt"
        if perf_summary_file.exists():
            # Write it to the notes column in the v1_df dataframe.
            with open(perf_summary_file, "r") as f:
                content = f.read()

            # Update the 'notes' column with the content.
            v1_df.loc[row_index, "notes"] = content

        else:
            # Update the 'notes' column with the content.
            preedit_output = Path(instance_folder) / "perf_output_preedit.txt"
            result = "An unknown error occurred. Please check the local files."

            if preedit_output.exists():
                result = preedit_output.read_text()[-40000:]
            else:
                print(instance_id)

            v1_df.loc[row_index, "notes"] = f"An error occurred:\n\n{result}"

# Write the updated dataframe back to the worksheet.
set_with_dataframe(
    worksheet,
    v1_df,
    include_index=False,
    include_column_header=True,
)
