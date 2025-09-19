import json
import multiprocessing
import datasets
from pathlib import Path

import collections
import tqdm

from swefficiency.harness.log_parsers import MAP_REPO_TO_PARSER


# Take old dataset and check: some errors:
# dataset = datasets.load_dataset("swefficiency/swefficiency", split='test', revision="718e5821f73f86414fa72bf8b7f716651a3a835a")
# ground_truth_data = Path("logs/run_evaluation/perf_eval_ground_truth/gold")


# dataset = datasets.load_dataset("swefficiency/swefficiency", split='test')

new_dataset = []

# Try 2: verify any flaky tests?
dataset = datasets.load_dataset(
    "swefficiency/swefficiency_old",
    split="test",
    revision="48006d26383840d848db076517849facc2afec4d",
)

# latest? 39f0c020f75114c7e46d439bcc8a1d315f9b81ed

log_dir = Path("logs/run_evaluation/")
ground_truth_data_dirs = [
    log_dir / "jeff_perf_ground_truth1/gold",
    log_dir / "jeff_perf_ground_truth2/gold",
    log_dir / "jeff_perf_ground_truth3/gold",
    log_dir / "jeff_perf_ground_truth4/gold",
    log_dir / "jeff_perf_ground_truth5/gold",
]

flaky_rerun_data_dirs = [
    log_dir / "jeff_perf_ground_truth_flaky/gold",
    log_dir / "jeff_perf_ground_truth_flaky2/gold",
    log_dir / "jeff_perf_ground_truth_flaky3/gold",
    log_dir / "jeff_perf_ground_truth_flaky4/gold",
    log_dir / "jeff_perf_ground_truth_flaky5/gold",
]


def parse_perf_summary(perf_summary):
    perf_lines = perf_summary.strip().splitlines()

    before_mean = float(perf_lines[0].split(":")[1].strip())
    before_std = float(perf_lines[1].split(":")[1].strip())
    after_mean = float(perf_lines[2].split(":")[1].strip())
    after_std = float(perf_lines[3].split(":")[1].strip())
    improvement = (after_mean - before_mean) / before_mean * 100

    return {
        "before_mean": before_mean,
        "after_mean": after_mean,
        "before_std": before_std,
        "after_std": after_std,
        "improvement": improvement,
    }


flaky_tests_across_all = set()


import gspread
from gspread_dataframe import get_as_dataframe

gc = gspread.service_account()
annotate_spreadsheet = gc.open("global_sweperf_all_data_annotate")
annotate_worksheet = annotate_spreadsheet.get_worksheet(0)

annotate_df = get_as_dataframe(
    annotate_worksheet,
    header=0,
    dtype=str,
)

workloads_to_overwrite = [
    "scikit-learn__scikit-learn-10610",
    "scikit-learn__scikit-learn-9858",
    "pandas-dev__pandas-53731",
    "sympy__sympy-25631",
    "pydata__xarray-9429",
    "sympy__sympy-20989",
    "dask__dask-5940",
    "sympy__sympy-21501",
    "scikit-learn__scikit-learn-24856",
]

instances_to_ignore = [
    "scikit-learn__scikit-learn-14075",
    # Latest run ignore.
    "numpy__numpy-23088",
    "scipy__scipy-16449",
    "scikit-learn__scikit-learn-19934",
    # "sympy__sympy-25631", # NOTE THIS NEEDS RERUN.
    "scipy__scipy-22610",
    "pandas-dev__pandas-36611",
    "pandas-dev__pandas-52290",
    "pandas-dev__pandas-44608",
    "pandas-dev__pandas-54223",
    "scikit-learn__scikit-learn-13987",
    "matplotlib__matplotlib-20197",
    "pandas-dev__pandas-39678",
    "scipy__scipy-10574",
    "numpy__numpy-11518",
    # "pydata__xarray-9429", # NOTE this needs rerun.
    "matplotlib__matplotlib-24847",
    "matplotlib__matplotlib-18997",
    "pandas-dev__pandas-43623",
    "pandas-dev__pandas-43316",
    "pandas-dev__pandas-52132",
    "numpy__numpy-25991",
    "sympy__sympy-12748",
    "scikit-learn__scikit-learn-16499",
    "sympy__sympy-21320",
    "pandas-dev__pandas-41503",
    "pandas-dev__pandas-43737",
    "numpy__numpy-13634",
    "pandas-dev__pandas-43307",
    # slow
    # "numpy__numpy-19620",
    # "numpy__numpy-19609", # maybe fine?
    # "numpy__numpy-25299", # maybe fine?
    # "scikit-learn__scikit-learn-10610",
    # "scipy__scipy-12474", # maybe fine?
    # "scikit-learn__scikit-learn-25490", # maybe fine?
    # "scikit-learn__scikit-learn-29835", # maybe fine?
    # "scikit-learn__scikit-learn-17737", # maybe fine?
    # "pandas-dev__pandas-57560",
    # "pandas-dev__pandas-51784", # maybe fine?
    # "pandas-dev__pandas-43316",
    # "pandas-dev__pandas-43307",
    # "pandas-dev__pandas-43281",
    # "scipy__scipy-10064", # maybe fine?
    # "numpy__numpy-25299",
    # "numpy__numpy-21394",
    # "numpy__numpy-19620",
    # "numpy__numpy-19618",
    # "numpy__numpy-19609",
    # "sympy__sympy-15453",
    # Identified as non-significant slow
    # "sympy__sympy-15453",
    # "numpy__numpy-21832", # maybe fine?
    # "numpy__numpy-18203",
    # "matplotlib__matplotlib-18018", # maybe fine?
    # "numpy__numpy-12321",
    # "numpy__numpy-11518", # maybe fine?
    # "sympy__sympy-21320", # maybe fine?
    # "pandas-dev__pandas-25953", # maybe fine?
    # "matplotlib__matplotlib-24847",
    # "pandas-dev__pandas-53368", # maybe fine?
    # "scikit-learn__scikit-learn-13987", # maybe fine?
    # "numpy__numpy-25788", # maybe fine?
    # "pandas-dev__pandas-43277", # maybe fine?
    # "pandas-dev__pandas-40254",
    # "scipy__scipy-22610",
    # "numpy__numpy-23088", # maybe fine?
    # "numpy__numpy-13634",
    # "pandas-dev__pandas-40178", # maybe fine?
    # "pandas-dev__pandas-50306", # maybe fine?
    # "scikit-learn__scikit-learn-16499", # maybe fine?
    # "pandas-dev__pandas-34948", # maybe fine?
    # "pandas-dev__pandas-41567", # maybe fine?
    # "pandas-dev__pandas-30768",
    # "numpy__numpy-25991",
    # "numpy__numpy-13697",
    # "numpy__numpy-18324", # maybe fine?
    # "pandas-dev__pandas-41503",
    # "pandas-dev__pandas-43737", # maybe fine?
]

from datetime import datetime, timezone


def convert_timestamp(value):
    """
    Converts between Unix timestamp (milliseconds) and ISO8601 datetime.

    If `value` is an int/float → treat as Unix timestamp in ms → return ISO string.
    If `value` is a str (ISO datetime) → return Unix timestamp in ms.
    """
    try:
        value = int(value)
    except (ValueError, TypeError):
        return value

    dt = datetime.fromtimestamp(value / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def worker(d):
    instance_id = d["instance_id"]

    if instance_id in instances_to_ignore:
        # print(f"Skipping instance {instance_id} as it is in the ignore list")
        return None

    owner = instance_id.split("__")[0]
    repo_name, remainder = instance_id.split("__")[1].rsplit("-", 1)

    d_dict = dict(d)

    correctness_report = {}

    good_perf = True

    good_perf_runtime = True
    good_perf_value = None

    flaky_tests = collections.defaultdict(list)

    # print("==========")
    # print(instance_id)

    if any(not (run_dir / instance_id).exists() for run_dir in ground_truth_data_dirs):
        return None

    data_dirs = ground_truth_data_dirs
    if any((run_dir / instance_id).exists() for run_dir in flaky_rerun_data_dirs):
        data_dirs = flaky_rerun_data_dirs

    for ground_truth_data in data_dirs:
        covering_test_status_json = (
            ground_truth_data / instance_id / "covering_test_status.json"
        )
        perf_summary = ground_truth_data / instance_id / "perf_summary.txt"
        instance_log = ground_truth_data / instance_id / "run_instance.log"

        instance_correctness_report_raw = dict()
        reparsed_covering_test_status_file_dir = (
            ground_truth_data / instance_id / "raw_correctness_output"
        )
        for file in reparsed_covering_test_status_file_dir.glob("*.txt"):
            instance_correctness_report_raw.update(
                MAP_REPO_TO_PARSER[f"{owner}/{repo_name}"](file.read_text())
            )
        instance_correctness_report = instance_correctness_report_raw
        # instance_correctness_report = json.loads(covering_test_status_json.read_text()) if covering_test_status_json.exists() else None

        for k, v in instance_correctness_report.items():
            flaky_tests[k].append(v)

        instance_perf_report = (
            perf_summary.read_text() if perf_summary.exists() else None
        )
        if instance_perf_report is not None:
            perf_output = parse_perf_summary(instance_perf_report)

            delta = perf_output["after_mean"] - perf_output["before_mean"]
            max_std = perf_output["after_std"]

            if delta >= 0 or (
                perf_output["improvement"] > -25 and abs(delta) < max_std
            ):
                good_perf = False
                print(delta, max_std, perf_output["improvement"])
                print(str(perf_summary))
                break

        # Look for wording "Pre-edit perf runtime: X seconds"
        if instance_log.exists():
            instance_log_content = instance_log.read_text()
            if "Pre-edit perf runtime:" in instance_log_content:
                pre_edit_perf_runtime = float(
                    instance_log_content.split("Pre-edit perf runtime:")[1]
                    .split("seconds")[0]
                    .strip()
                )

                good_perf_runtime = good_perf_runtime and pre_edit_perf_runtime < 600
                good_perf_value = (
                    max(good_perf_value, pre_edit_perf_runtime)
                    if good_perf_value is not None
                    else pre_edit_perf_runtime
                )

    # Keep correctness report only if all in lists are the same
    real_flaky_tests = {k: v for k, v in flaky_tests.items() if len(set(v)) > 1}
    non_flaky_tests = {k: v[0] for k, v in flaky_tests.items() if len(set(v)) == 1}

    # for k, v in real_flaky_tests.items():
    #     print(k, flaky_tests[k])

    if not good_perf:
        # print(f"Instance {instance_id} has bad performance improvement")
        return None

    # if not good_perf_runtime:
    #     print(f"Instance {instance_id} has bad performance runtime: {good_perf_value} seconds")

    def skip_condition(name):
        conditions = [
            "test_user_agent.py",
            "sklearn/tests/test_common.py",
            "pandas/tests/extension/json/test_json.py::TestArithmeticOps::test_arith_frame_with_scalar",
            "pandas/tests/indexes/test_base.py::TestIndex::test_isin_nan_common_object",
            "pandas/tests/test_algos.py::TestIsin::test_different_nans",
            "test_nan",
            "numpy/_core/tests/test_umath_accuracy.py::TestAccuracy::test_validate_svml_fp16",
            "numpy/_core/tests/test_scalarmath.py::test_array_scalar_ufunc_dtypes",
            "test_reindex_nan",
            "pandas/tests/series/methods/test_reindex.py::test_reindex_nan",
            "pandas/tests/io/json/test_pandas.py::TestPandasContainer::test_roundtrip_simple[columns-True-True-float]",
            "pandas/tests/io/json/test_json_table_schema.py::TestTableOrient::test_build_series",
            "dropna",
        ]
        
        failed_tests = [
            # "numpy/_core/tests/test_umath_accuracy.py::TestAccuracy::test_validate_svml_fp16",
            # "numpy/_core/tests/test_scalarmath.py::test_array_scalar_ufunc_dtypes",
            # "numpy/tests/test__all__.py::test_no_duplicates_in_np__all__",
            # "numpy/_core/tests/test_umath_accuracy.py::TestAccuracy::test_validate_svml_fp16",
            # "pandas/tests/apply/test_series_apply.py::test_map_missing_mixed[vals0-mapping0-exp0]",
            "pandas/tests/test_algos.py::TestUnique::test_first_nan_kept",
            # "pandas/tests/test_algos.py::TestIsin::test_different_nan_objects",
            "pandas/tests/io/json/test_readlines.py::test_to_jsonl",
            "pandas/tests/io/json/test_readlines.py::test_readjson_chunks[1]",
            "pandas/tests/io/json/test_readlines.py::test_readjson_chunks[1.0]",
            "pandas/tests/io/json/test_readlines.py::test_readjson_each_chunk",
            "xarray/tests/test_backends.py::test_open_mfdataset_manyfiles[h5netcdf-20-True-None-5]",
            "xarray/tests/test_backends.py::test_open_mfdataset_manyfiles[h5netcdf-20-True-5-5]",
            "xarray/tests/test_backends.py::test_open_mfdataset_manyfiles[netcdf4-20-True-None-5]",
            "pandas/tests/io/test_fsspec.py::test_json_options",
            # "pandas/tests/io/json/test_readlines.py::test_readjson_chunks[1]",
            "pandas/tests/io/json/test_readlines.py::test_readjson_each_chunk",
            # "pandas/tests/series/methods/test_reindex.py::test_reindex_nan",
            # "pandas/tests/io/json/test_pandas.py::TestPandasContainer::test_roundtrip_simple[columns-True-True-float]",
            # "pandas/tests/io/json/test_json_table_schema.py::TestTableOrient::test_build_series",
            # "pandas/tests/series/methods/test_reindex.py::test_reindex_nan",
            "pandas/tests/indexes/period/test_freq_attr.py::TestFreq::test_freq_setter_deprecated",
            "pandas/tests/reshape/test_get_dummies.py::TestGetDummies::test_get_dummies_basic_drop_first_NA[sparse]",
            "pandas/tests/reshape/test_get_dummies.py::TestGetDummies::test_get_dummies_basic_drop_first_NA[dense]",
            "dask/dataframe/tests/test_dataframe.py::test_describe_empty",
            "xarray/tests/test_backends.py::test_open_mfdataset_manyfiles[netcdf4-20-True-None-5]",
            "pandas/tests/apply/test_series_apply.py::test_map_missing_mixed[vals0-mapping0-exp0]",
            "pandas/tests/tools/test_to_datetime.py::TestDatetimeParsingWrappers::test_parsers[True-2005-11-expected20-None]",
            "pandas/tests/tools/test_to_datetime.py::TestDatetimeParsingWrappers::test_parsers[True-2014-06-expected36-None]",
            "pandas/tests/tools/test_to_datetime.py::TestDatetimeParsingWrappers::test_parsers[True-2014-6-expected38-None]",
            "pandas/tests/tools/test_to_datetime.py::TestDatetimeParsingWrappers::test_parsers[False-2005-11-expected20-None]",
            "pandas/tests/tools/test_to_datetime.py::TestDatetimeParsingWrappers::test_parsers[False-2014-06-expected36-None]",
            "pandas/tests/tools/test_to_datetime.py::TestDatetimeParsingWrappers::test_parsers[False-2014-6-expected38-None]",
            "pandas/tests/apply/test_series_apply.py::TestSeriesMap::test_map_missing_mixed[vals0-mapping0-exp0]",
            "pandas/tests/scalar/timestamp/test_unary_ops.py::TestTimestampUnaryOps::test_round_sanity[ceil-18]",
            "pandas/tests/libs/test_hashtable.py::TestHashTableUnsorted::test_vector_resize[True-Int32HashTable-Int32Vector-int32-False-10]",
            "pandas/tests/computation/test_eval.py::TestAlignment::test_complex_series_frame_alignment[python-pandas-i-s-i-s0]",
            "pandas/tests/scalar/timedelta/test_timedelta.py::TestTimedeltas::test_round_sanity",
            "lib/matplotlib/tests/test_constrainedlayout.py::test_compressed1",
            "pandas/tests/test_algos.py::TestIsin::test_different_nan_objects",
            "xarray/tests/test_backends.py::TestH5NetCDFDataRos3Driver::test_get_variable_list",
            "pandas/tests/io/parser/common/test_file_buffer_url.py::test_url",
            "pandas/tests/io/json/test_pandas.py::TestPandasContainer::test_round_trip_exception_",
            "pandas/tests/io/parser/test_common.py::test_url",
        ]

        all_conditions = set(conditions + failed_tests)

        return any(c in name for c in all_conditions)

    def single_threaded_condition(name):
        conditions = [
            "pandas/tests/io",
            "xarray/tests/test_backends.py",
        ]

        return any(c in name for c in conditions)

    # HACK: Ignore test
    d_dict["covering_tests"] = [
        test for test in d_dict["covering_tests"] if not skip_condition(test)
    ]

    if len(d_dict["covering_tests"]) == 0:
        # print(f"Instance {instance_id} has no covering tests after filtering")
        return None

    if d_dict["instance_id"] in workloads_to_overwrite:
        print(f"Overwriting workload for {d_dict['instance_id']}")
        d_dict["workload"] = annotate_df[
            annotate_df["instance_id"] == d_dict["instance_id"]
        ]["workload"].values[0]

    d_dict["PASS_TO_PASS"] = [
        test
        for test, status in non_flaky_tests.items()
        if "PASS" in status and not skip_condition(test)
    ]

    d_dict["image_name"] = f"ghcr.io/swefficiency/swefficiency:{instance_id}"

    if "single_thread_tests" not in d_dict:
        d_dict["single_thread_tests"] = list(
            set(
                [
                    "/testbed/" + k.split("::")[0]
                    for k, v in real_flaky_tests.items()
                    if any("PASS" in status for status in v) and not skip_condition(k)
                ]
                + [
                    test
                    for test in d_dict["covering_tests"]
                    if single_threaded_condition(test)
                ]
            )
        )
        
    # Remove any test
            
    # Convert "created_at" and "updated_at" to ISO8601 if they are int timestamps.
    for time_field in ["created_at"]:
        if time_field in d_dict:
            d_dict[time_field] = convert_timestamp(d_dict[time_field])

    # Delete notes column
    if "notes" in d_dict:
        del d_dict["notes"]

    # TODO: use original github username
    if "image_name" in d_dict:
        d_dict["image_name"] = d_dict["image_name"].replace("TODO", "swefficiency")

    return d_dict


with multiprocessing.Pool() as pool:
    results = list(tqdm.tqdm(pool.imap(worker, dataset), total=len(dataset)))

new_dataset = [result for result in results if result is not None]

print(f"Total instances with good performance: {len(new_dataset)}")
# print(f"Instances with flaky tests:", " ".join(d["instance_id"] for d in new_dataset if "single_thread_tests" in d and d["single_thread_tests"]))

new_dataset = datasets.Dataset.from_list(new_dataset)

# Upload the new dataset
new_dataset.push_to_hub("swefficiency/swefficiency", split="test")
