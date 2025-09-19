from pathlib import Path

INSTANCE = "pandas-dev__pandas-52341"
INSTANCE = "pandas-dev__pandas-52381"
INSTANCE = "astropy__astropy-7549"
INSTANCE = "pydata__xarray-7374"

PROFILE_RUN_DIR = Path("logs/run_evaluation/profile_runs")
GOLD_PROFILE_DIR = PROFILE_RUN_DIR / "gold"
LLM_PROFILE_DIR = PROFILE_RUN_DIR / "default_sweperf_claude__anthropic--claude-3-7-sonnet-20250219__t-0.00__p-1.00__c-1.00___swefficiency_full_test"

gold_preedit_file = GOLD_PROFILE_DIR / INSTANCE / "preedit" / "workload_preedit_cprofile.prof"
gold_postedit_file = GOLD_PROFILE_DIR / INSTANCE / "postedit" / "workload_postedit_cprofile.prof"

llm_preedit_file = LLM_PROFILE_DIR / INSTANCE / "preedit" / "workload_preedit_cprofile.prof"
llm_postedit_file = LLM_PROFILE_DIR / INSTANCE / "postedit" / "workload_postedit_cprofile.prof"

