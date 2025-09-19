__version__ = "1.0.0"

from swefficiency.collect.build_dataset import main as build_dataset
from swefficiency.collect.get_tasks_pipeline import main as get_tasks_pipeline
from swefficiency.collect.print_pulls import main as print_pulls

from swefficiency.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    MAP_REPO_VERSION_TO_SPECS,
)

from swefficiency.harness.docker_build import (
    build_image,
    build_base_images,
    build_env_images,
    build_instance_images,
    build_instance_image,
    close_logger,
    setup_logger,
)

from swefficiency.harness.docker_utils import (
    cleanup_container,
    remove_image,
    copy_to_container,
    exec_run_with_timeout,
    list_images,
)

from swefficiency.harness.grading import (
    compute_fail_to_pass,
    compute_pass_to_pass,
    get_logs_eval,
    get_eval_report,
    get_resolution_status,
    ResolvedStatus,
    TestStatus,
)

from swefficiency.harness.log_parsers import (
    MAP_REPO_TO_PARSER,
)

from swefficiency.harness.run_evaluation import (
    main as run_evaluation,
)

from swefficiency.harness.utils import (
    get_environment_yml,
    get_requirements,
)

from swefficiency.versioning.constants import (
    MAP_REPO_TO_VERSION_PATHS,
    MAP_REPO_TO_VERSION_PATTERNS,
)

from swefficiency.versioning.get_versions import (
    get_version,
    map_version_to_task_instances,
    get_versions_from_build,
    get_versions_from_web,
)

from swefficiency.versioning.utils import (
    split_instances,
)