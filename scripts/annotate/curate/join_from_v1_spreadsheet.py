# %%
import time
import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from tqdm import tqdm

gc = gspread.service_account()

CURRENT_DATA_SPREADSHEET_NAME = "sweperf_all_data"

all_data_spreadsheet = gc.open(CURRENT_DATA_SPREADSHEET_NAME)
all_data_worksheet = all_data_spreadsheet.get_worksheet(0)
column_names = all_data_worksheet.row_values(1)

# Get sheet as a dataframe.
all_data_df = get_as_dataframe(
    all_data_worksheet, 
    header=0, 
    dtype=str, 
    usecols=column_names
)

print(column_names)
print(all_data_df.columns)

# %%
# Join V1 spreadsheet data with the current CSV data.

V1_SPREADSHEET_NAME = "sweperf_v1_data"

v1_spreadsheet = gc.open(V1_SPREADSHEET_NAME)
v1_worksheet = v1_spreadsheet.get_worksheet(0)

v1_df = get_as_dataframe(
    v1_worksheet, 
    header=0, 
    dtype=str, 
    usecols=["instance_id", "perf_workload", "before_mean", "before_std_dev", "after_mean", "after_std_dev", "improvement"]
)

print(all_data_df.columns)

all_data_df = all_data_df.set_index("instance_id", drop=False)
v1_df       = v1_df.set_index("instance_id", drop=False)

# Drop any rows with duplicate index values in either.
all_data_df = all_data_df[~all_data_df.index.duplicated(keep='first')]
v1_df       = v1_df[~v1_df.index.duplicated(keep='first')]

v1_df = v1_df.rename(columns={"perf_workload": "workload"})

print(v1_df)

def get_notes(row):
    before_mean = row.get("before_mean", "")
    before_std_dev = row.get("before_std_dev", "")
    after_mean = row.get("after_mean", "")
    after_std_dev = row.get("after_std_dev", "")
    improvement = row.get("improvement", "")

    return f"Before Mean: {before_mean}\nBefore Std Dev: {before_std_dev}\nAfter Mean: {after_mean}\nAfter Std Dev: {after_std_dev}\nImprovement: {improvement}"

v1_df['notes'] = v1_df.apply(get_notes, axis=1)

all_data_df.update(v1_df[["workload", "notes"]])
      # clear old rows
set_with_dataframe(
    all_data_worksheet,
    all_data_df,
    include_column_header=True,
    resize=True
)


# %%

# %%
# Join V2 (Mike's) spreadsheet data with the current CSV data.

MIKE_SPREADSHEETs = [
    "annotate_images_astropy",
    "annotate_images_sympy",
    "annotate_images_scipy",
]

for spreadsheet_name in tqdm(MIKE_SPREADSHEETs, desc="Processing Mike's spreadsheets"):
    v1_spreadsheet = gc.open(spreadsheet_name)
    v1_worksheet = v1_spreadsheet.get_worksheet(0)

    mike_df = get_as_dataframe(
        v1_worksheet, 
        header=0, 
        dtype=str, 
        usecols=["instance_id", "cleaned_workload", "notes"]
    )

    all_data_df = all_data_df.set_index("instance_id", drop=False)
    mike_df       = mike_df.set_index("instance_id", drop=False)
    mike_df = mike_df.rename(columns={"cleaned_workload": "workload"})
    
    # Keep only rows that have actual value for "workload" in mike_df.
    mike_df = mike_df[mike_df["workload"].notna()]
    print(mike_df)
    
    all_data_df.update(mike_df[["workload", "notes"]])
       # clear old rows
    set_with_dataframe(
        all_data_worksheet,
        all_data_df,
        include_column_header=True,
        resize=True
    )
# %%

# %%
