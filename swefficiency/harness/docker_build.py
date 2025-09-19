from __future__ import annotations

import logging
import re
import threading
import traceback
import docker
import docker.errors
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

USE_HOST_NETWORK = False

from swefficiency.harness.constants import (
    ANNOTATE_IMAGE_BUILD_DIR,
    BASE_IMAGE_BUILD_DIR,
    ENV_IMAGE_BUILD_DIR,
    INSTANCE_IMAGE_BUILD_DIR,
    MAP_REPO_VERSION_TO_SPECS,
)
from swefficiency.harness.test_spec import (
    get_test_specs_from_dataset,
    make_test_spec,
    TestSpec,
)
from swefficiency.harness.docker_utils import (
    cleanup_container,
    remove_image,
    find_dependent_images,
)

ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


class BuildImageError(Exception):
    def __init__(self, image_name, message, logger):
        super().__init__(message)
        self.super_str = super().__str__()
        self.image_name = image_name
        self.log_path = logger.log_file
        self.logger = logger

    def __str__(self):
        return (
            f"Error building image {self.image_name}: {self.super_str}\n"
            f"Check ({self.log_path}) for more information."
        )


def setup_logger(instance_id: str, log_file: Path, mode="w"):
    """
    This logger is used for logging the build process of images and containers.
    It writes logs to the log file.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{instance_id}.{log_file.name}")
    handler = logging.FileHandler(log_file, mode=mode)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    setattr(logger, "log_file", log_file)
    return logger


def close_logger(logger):
    # To avoid too many open files
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


def build_image(
    image_name: str,
    setup_scripts: dict,
    dockerfile: str,
    platform: str,
    client: docker.DockerClient,
    build_dir: Path,
    nocache: bool = False,
    version: str = None,
    test_spec: TestSpec = None,
    build_timeout: int | None = None,
    env_mode: bool = False,
    force_rebuild: bool = False,
    # nfs_cache: Path | None = None,
):
    """
    Builds a docker image with the given name, setup scripts, dockerfile, and platform.

    Args:
        image_name (str): Name of the image to build
        setup_scripts (dict): Dictionary of setup script names to setup script contents
        dockerfile (str): Contents of the Dockerfile
        platform (str): Platform to build the image for
        client (docker.DockerClient): Docker client to use for building the image
        build_dir (Path): Directory for the build context (will also contain logs, scripts, and artifacts)
        nocache (bool): Whether to use the cache when building
    """
    if test_spec:
        print(test_spec.instance_id)
        print(test_spec.build_timeout if test_spec.build_timeout else build_timeout)
        print(build_dir)
        print(test_spec.version)

    # HACK: If build_dir already exists and has contents, we should skip.
    # if build_dir.exists() and len(list(build_dir.glob("*"))) > 1 and env_mode and not force_rebuild:
    # raise BuildImageError(image_name, "Already built or already failed from previous run.", None)

    logger = setup_logger(image_name, build_dir / "build_image.log")
    logger.info(
        f"Building image {image_name} for version {version or test_spec.version if test_spec else 'unknown'}"
    )
    logger.info(
        f"Building image {image_name}\n"
        f"Using dockerfile:\n{dockerfile}\n"
        f"Adding ({len(setup_scripts)}) setup scripts to image build repo"
    )

    for setup_script_name, setup_script in setup_scripts.items():
        logger.info(f"[SETUP SCRIPT] {setup_script_name}:\n{setup_script}")
    try:
        # Write the setup scripts to the build directory
        for setup_script_name, setup_script in setup_scripts.items():
            setup_script_path = build_dir / setup_script_name
            with open(setup_script_path, "w") as f:
                f.write(setup_script)
            if setup_script_name not in dockerfile:
                logger.warning(
                    f"Setup script {setup_script_name} may not be used in Dockerfile"
                )

        # Write the dockerfile to the build directory
        dockerfile_path = build_dir / "Dockerfile"
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile)

        # Build the image
        logger.info(
            f"Building docker image {image_name} in {build_dir} with platform {platform}"
        )
        build_timeout = (
            test_spec.build_timeout
            if test_spec and test_spec.build_timeout
            else build_timeout
        )

        for i in range(2):
            try:
                response = client.api.build(
                    path=str(build_dir),
                    tag=image_name,
                    rm=True,
                    forcerm=True,
                    decode=True,
                    platform=platform,
                    nocache=nocache,
                    # network_mode="host" if USE_HOST_NETWORK else None,
                    # container_limits={"memory": 32 * 1024 * 1024 * 1024},
                    timeout=build_timeout,  # TODO: Verify this timeout.
                )

                # Log the build process continuously
                buildlog = ""
                for chunk in response:
                    if "stream" in chunk:
                        # Remove ANSI escape sequences from the log
                        chunk_stream = ansi_escape.sub("", chunk["stream"])
                        logger.info(chunk_stream.strip())
                        buildlog += chunk_stream
                    elif "errorDetail" in chunk:
                        # Decode error message, raise BuildError
                        logger.error(
                            f"Error {image_name}: {ansi_escape.sub('', chunk['errorDetail']['message'])}"
                        )
                        print(f"Error {image_name}: {chunk['errorDetail']['message']}")
                        raise docker.errors.BuildError(
                            chunk["errorDetail"]["message"], buildlog
                        )
                break
            except Exception as e:
                if i < 1:
                    logger.warning(
                        f"Failed to build image {image_name} (attempt {i+1}/2): {e}\nRetrying..."
                    )
                else:
                    logger.error(
                        f"Failed to build image {image_name} after retry attempts: {e}"
                    )
                    traceback.print_exc()
                    raise BuildImageError(image_name, str(e), logger) from e

        logger.info("Image built successfully!")
    except docker.errors.BuildError as e:
        logger.error(f"docker.errors.BuildError during {image_name}: {e}")
        raise BuildImageError(image_name, str(e), logger) from e
    except Exception as e:
        logger.error(f"Error building image {image_name}: {e}")
        raise BuildImageError(image_name, str(e), logger) from e
    finally:
        #  # SLURM: Save to NFS cache if not already there.
        # if nfs_cache is not None:
        #     nfs_image_path = nfs_cache / image_name.replace(":", "__")
        #     if not nfs_image_path.exists():
        #         try:
        #             image = client.images.get(image_name)
        #         except docker.errors.ImageNotFound:
        #             print(f"Image {image_name} not found after build, skipping NFS cache save.")
        #             raise BuildImageError(
        #                 image_name,
        #                 f"Base image {image_name} not found after build, skipping NFS cache save", logger)

        #         nfs_image_path.parent.mkdir(parents=True, exist_ok=True)
        #         with open(nfs_image_path, "wb") as f:
        #             for chunk in image.save():
        #                 f.write(chunk)
        #         print(f"Saved image {image_name} to NFS cache at {nfs_image_path}")

        close_logger(logger)  # functions that create loggers should close them


def build_base_images(
    client: docker.DockerClient,
    dataset: list,
    force_rebuild: bool = False,
    nfs_cache: Path | None = None,
):
    """
    Builds the base images required for the dataset if they do not already exist.

    Args:
        client (docker.DockerClient): Docker client to use for building the images
        dataset (list): List of test specs or dataset to build images for
        force_rebuild (bool): Whether to force rebuild the images even if they already exist

    Note that NFS cache is a Podman hack. It's not invalidated correctly, so it may lead to issues.

    """
    # Get the base images to build from the dataset
    test_specs = get_test_specs_from_dataset(dataset)
    base_images = {
        x.base_image_key: (x.base_dockerfile, x.platform) for x in test_specs
    }
    if force_rebuild:
        for key in base_images:
            remove_image(client, key, "quiet")

    # Build the base images
    for image_name, (dockerfile, platform) in base_images.items():
        try:
            # Check if the base image already exists
            client.images.get(image_name)
            if force_rebuild:
                # Remove the base image if it exists and force rebuild is enabled
                remove_image(client, image_name, "quiet")
            else:
                print(f"Base image {image_name} already exists, skipping build.")
                continue
        except docker.errors.ImageNotFound:
            pass
        # Build the base image (if it does not exist or force rebuild is enabled)
        print(f"Building base image ({image_name})")
        build_image(
            image_name=image_name,
            setup_scripts={},
            dockerfile=dockerfile,
            platform=platform,
            client=client,
            build_dir=BASE_IMAGE_BUILD_DIR / image_name.replace(":", "__"),
        )

    print("Base images built successfully.")


def get_env_configs_to_build(
    client: docker.DockerClient,
    dataset: list,
):
    """
    Returns a dictionary of image names to build scripts and dockerfiles for environment images.
    Returns only the environment images that need to be built.

    Args:
        client (docker.DockerClient): Docker client to use for building the images
        dataset (list): List of test specs or dataset to build images for
    """
    image_scripts = dict()
    base_images = dict()
    test_specs = get_test_specs_from_dataset(dataset)

    for test_spec in test_specs:
        # Check if the base image exists
        try:
            if test_spec.base_image_key not in base_images:
                base_images[test_spec.base_image_key] = client.images.get(
                    test_spec.base_image_key
                )
            base_image = base_images[test_spec.base_image_key]
        except docker.errors.ImageNotFound:
            raise Exception(
                f"Base image {test_spec.base_image_key} not found for {test_spec.env_image_key}\n."
                "Please build the base images first."
            )

        # Check if the environment image exists
        image_exists = False
        try:
            env_image = client.images.get(test_spec.env_image_key)
            image_exists = True

            if env_image.attrs["Created"] < base_image.attrs["Created"]:
                # Remove the environment image if it was built after the base_image
                for dep in find_dependent_images(client, test_spec.env_image_key):
                    # Remove instance images that depend on this environment image
                    remove_image(client, dep, "quiet")
                remove_image(client, test_spec.env_image_key, "quiet")
                image_exists = False
        except docker.errors.ImageNotFound:
            pass
        if not image_exists:
            # Add the environment image to the list of images to build
            image_scripts[test_spec.env_image_key] = {
                "setup_script": test_spec.setup_env_script,
                "dockerfile": test_spec.env_dockerfile,
                "platform": test_spec.platform,
                "test_spec": test_spec,
            }
    return image_scripts


def build_env_images(
    client: docker.DockerClient,
    dataset: list,
    force_rebuild: bool = False,
    max_workers: int = 4,
    batch_size: int = 1,
):
    """
    Builds the environment images required for the dataset if they do not already exist.

    Args:
        client (docker.DockerClient): Docker client to use for building the images
        dataset (list): List of test specs or dataset to build images for
        force_rebuild (bool): Whether to force rebuild the images even if they already exist
        max_workers (int): Maximum number of workers to use for building images
    """
    # Get the environment images to build from the dataset
    if force_rebuild:
        env_image_keys = {x.env_image_key for x in get_test_specs_from_dataset(dataset)}
        for key in env_image_keys:
            remove_image(client, key, "quiet")
    build_base_images(client, dataset, force_rebuild)
    configs_to_build = get_env_configs_to_build(client, dataset)
    if len(configs_to_build) == 0:
        print("No environment images need to be built.")
        return [], []
    print("Length of env dataset:", len(dataset))
    print(f"Building {len(configs_to_build)} environment images")

    print(f"Total environment images to build: {len(configs_to_build)}")

    # Build the environment images
    successful, failed = list(), list()

    # Split into batches if batch_size is specified.
    actual_batch_size = batch_size * max_workers
    batched_configs_to_build = [
        dict(list(configs_to_build.items())[i : i + actual_batch_size])
        for i in range(0, len(configs_to_build), actual_batch_size)
    ]

    with tqdm(
        total=len(configs_to_build), smoothing=0, desc="Building environment images"
    ) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for configs_to_build in batched_configs_to_build:
                # Create a future for each image to build
                futures = {
                    executor.submit(
                        build_image,
                        image_name,
                        {"setup_env.sh": config["setup_script"]},
                        config["dockerfile"],
                        config["platform"],
                        client,
                        ENV_IMAGE_BUILD_DIR / image_name.replace(":", "__"),
                        test_spec=config["test_spec"],
                        env_mode=True,  # This is just for skipping envs that won't build.
                        force_rebuild=force_rebuild,
                    ): image_name
                    for image_name, config in configs_to_build.items()
                }

                # Wait for each future to complete
                for future in as_completed(futures):
                    pbar.update(1)
                    try:
                        # Update progress bar, check if image built successfully
                        future.result()
                        successful.append(futures[future])
                    except BuildImageError as e:
                        print(f"BuildImageError {e.image_name}")
                        traceback.print_exc()
                        failed.append(futures[future])
                        continue
                    except Exception:
                        print("Error building image")
                        traceback.print_exc()
                        failed.append(futures[future])
                        continue

                # # After each batch parallel completes, run docker system prune to free up space.
                # print("Pruning batched images to free up space...")
                # client.api.prune_containers()
                # client.api.prune_images()
                # client.api.prune_volumes()

    # Show how many images failed to build
    if len(failed) == 0:
        print("All environment images built successfully.")
    else:
        print(f"{len(failed)} environment images failed to build.")

    # Return the list of (un)successfuly built images
    return successful, failed


def build_instance_images(
    client: docker.DockerClient,
    dataset: list,
    force_rebuild: bool = False,
    max_workers: int = 4,
):
    """
    Builds the instance images required for the dataset if they do not already exist.

    Args:
        dataset (list): List of test specs or dataset to build images for
        client (docker.DockerClient): Docker client to use for building the images
        force_rebuild (bool): Whether to force rebuild the images even if they already exist
        max_workers (int): Maximum number of workers to use for building images
    """
    # Build environment images (and base images as needed) first
    test_specs = list(map(make_test_spec, dataset))
    if force_rebuild:
        for spec in test_specs:
            remove_image(client, spec.instance_image_key, "quiet")
    _, env_failed = build_env_images(client, test_specs, force_rebuild, max_workers)

    if len(env_failed) > 0:
        # Don't build images for instances that depend on failed-to-build env images
        dont_run_specs = [
            spec for spec in test_specs if spec.env_image_key in env_failed
        ]
        test_specs = [
            spec for spec in test_specs if spec.env_image_key not in env_failed
        ]
        print(
            f"Skipping {len(dont_run_specs)} instances - due to failed env image builds"
        )
    print(f"Building instance images for {len(test_specs)} instances")
    successful, failed = list(), list()

    # Build the instance images
    with tqdm(
        total=len(test_specs), smoothing=0, desc="Building instance images"
    ) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a future for each image to build
            futures = {
                executor.submit(
                    build_instance_image,
                    test_spec,
                    client,
                    None,  # logger is created in build_instance_image, don't make loggers before you need them
                    False,
                ): test_spec
                for test_spec in test_specs
            }

            # Wait for each future to complete
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    # Update progress bar, check if image built successfully
                    future.result()
                    successful.append(futures[future])
                except BuildImageError as e:
                    print(f"BuildImageError {e.image_name}")
                    traceback.print_exc()
                    failed.append(futures[future])
                    continue
                except Exception:
                    print("Error building image")
                    traceback.print_exc()
                    failed.append(futures[future])
                    continue

    # Show how many images failed to build
    if len(failed) == 0:
        print("All instance images built successfully.")
    else:
        print(f"{len(failed)} instance images failed to build.")

    # Return the list of (un)successfuly built images
    return successful, failed


def build_instance_image(
    test_spec: TestSpec,
    client: docker.DockerClient,
    logger: logging.Logger | None,
    nocache: bool,
):
    """
    Builds the instance image for the given test spec if it does not already exist.

    Args:
        test_spec (TestSpec): Test spec to build the instance image for
        client (docker.DockerClient): Docker client to use for building the image
        logger (logging.Logger): Logger to use for logging the build process
        nocache (bool): Whether to use the cache when building
    """
    # Set up logging for the build process
    build_dir = INSTANCE_IMAGE_BUILD_DIR / test_spec.instance_image_key.replace(
        ":", "__"
    )
    new_logger = False
    if logger is None:
        new_logger = True
        logger = setup_logger(test_spec.instance_id, build_dir / "prepare_image.log")

    # Get the image names and dockerfile for the instance image
    image_name = test_spec.instance_image_key
    env_image_name = test_spec.env_image_key
    dockerfile = test_spec.instance_dockerfile

    # Check that the env. image the instance image is based on exists
    try:
        env_image = client.images.get(env_image_name)
    except docker.errors.ImageNotFound as e:
        raise BuildImageError(
            test_spec.instance_id,
            f"Environment image {env_image_name} not found for {test_spec.instance_id}",
            logger,
        ) from e
    logger.info(
        f"Environment image {env_image_name} found for {test_spec.instance_id}\n"
        f"Building instance image {image_name} for {test_spec.instance_id}"
    )

    # Check if the instance image already exists
    image_exists = False
    try:
        instance_image = client.images.get(image_name)
        if instance_image.attrs["Created"] < env_image.attrs["Created"]:
            # the environment image is newer than the instance image, meaning the instance image may be outdated
            remove_image(client, image_name, "quiet")
            image_exists = False
        else:
            image_exists = True
    except docker.errors.ImageNotFound:
        pass

    # Build the instance image
    if not image_exists:
        build_image(
            image_name=image_name,
            setup_scripts={
                "setup_repo.sh": test_spec.install_repo_script,
            },
            dockerfile=dockerfile,
            platform=test_spec.platform,
            client=client,
            build_dir=build_dir,
            nocache=nocache,
            version=test_spec.version,
            build_timeout=test_spec.build_timeout,
        )
    else:
        logger.info(f"Image {image_name} already exists, skipping build.")

    if new_logger:
        close_logger(logger)


def build_annotate_instance_images(
    client: docker.DockerClient,
    dataset: list,
    force_rebuild: bool = False,
    max_workers: int = 4,
    build_annotate_instance_images=None,
):
    """
    Builds the annotate instance images required for the dataset if they do not already exist.

    Args:
        dataset (list): List of test specs or dataset to build images for
        client (docker.DockerClient): Docker client to use for building the images
        force_rebuild (bool): Whether to force rebuild the images even if they already exist
        max_workers (int): Maximum number of workers to use for building images
    """
    # Build environment images (and base images as needed) first
    test_specs = list(map(make_test_spec, dataset))
    if force_rebuild:
        for spec in test_specs:
            try:
                remove_image(client, spec.env_image_key, "quiet")
                remove_image(client, spec.instance_image_key, "quiet")
            except Exception:
                pass

    _, instance_failed = build_instance_images(
        client, test_specs, force_rebuild, max_workers
    )
    successful, failed = list(), list()

    # Build the instance images
    with tqdm(
        total=len(test_specs), smoothing=0, desc="Building annotate instance images"
    ) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a future for each image to build
            futures = {
                executor.submit(
                    build_annotate_instance_image,
                    test_spec,
                    client,
                    None,  # logger is created in build_instance_image, don't make loggers before you need them
                    False,
                    instance=(
                        build_annotate_instance_images[test_spec.instance_id]
                        if build_annotate_instance_images
                        else None
                    ),
                ): test_spec
                for test_spec in test_specs
            }

            # Wait for each future to complete
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    # Update progress bar, check if image built successfully
                    future.result()
                    successful.append(futures[future])
                except BuildImageError as e:
                    print(f"BuildImageError {e.image_name}")
                    traceback.print_exc()
                    failed.append(futures[future])
                    continue
                except Exception:
                    print("Error building image")
                    traceback.print_exc()
                    failed.append(futures[future])
                    continue

    # Show how many images failed to build
    if len(failed) == 0:
        print("All annotate instance images built successfully.")
    else:
        print(f"{len(failed)} annotate instance images failed to build.")

    # Return the list of (un)successfuly built images
    return successful, failed


def build_annotate_instance_image(
    test_spec: TestSpec,
    client: docker.DockerClient,
    logger: logging.Logger | None,
    nocache: bool,
    instance=None,
):
    """
    Builds the instance image for the given test spec if it does not already exist.

    Args:
        test_spec (TestSpec): Test spec to build the instance image for
        client (docker.DockerClient): Docker client to use for building the image
        logger (logging.Logger): Logger to use for logging the build process
        nocache (bool): Whether to use the cache when building
    """
    # Set up logging for the build process
    build_dir = (
        ANNOTATE_IMAGE_BUILD_DIR
        / test_spec.annotate_instance_image_key.replace(":", "__")
    )
    new_logger = False
    if logger is None:
        new_logger = True
        logger = setup_logger(test_spec.instance_id, build_dir / "prepare_image.log")

    # Get the image names and dockerfile for the instance image
    instance_image_name = test_spec.instance_image_key
    image_name = test_spec.annotate_instance_image_key

    dockerfile = test_spec.annotate_instance_dockerfile

    # Check that the env. image the instance image is based on exists
    try:
        env_image = client.images.get(instance_image_name)
    except docker.errors.ImageNotFound as e:
        raise BuildImageError(
            test_spec.instance_id,
            f"Instance image {instance_image_name} not found for {test_spec.instance_id}",
            logger,
        ) from e
    logger.info(
        f"Instance image {instance_image_name} found for {test_spec.instance_id}\n"
        f"Building instance image {image_name} for {test_spec.instance_id}"
    )

    # Check if the instance image already exists
    image_exists = False
    try:
        instance_image = client.images.get(image_name)
        if instance_image.attrs["Created"] < env_image.attrs["Created"]:
            # the environment image is newer than the instance image, meaning the instance image may be outdated
            remove_image(client, image_name, "quiet")
            image_exists = False
        else:
            image_exists = True
    except docker.errors.ImageNotFound:
        pass

    # Build the instance image
    if not image_exists:
        build_image(
            image_name=image_name,
            setup_scripts={
                "patch.diff": instance["patch"] if instance else "",
                "perf.sh": test_spec.performance_script,
            },
            dockerfile=dockerfile,
            platform=test_spec.platform,
            client=client,
            build_dir=build_dir,
            nocache=nocache,
            version=test_spec.version,
            build_timeout=test_spec.build_timeout,
        )
    else:
        logger.info(f"Image {image_name} already exists, skipping build.")

    if new_logger:
        close_logger(logger)


def get_common_numa_node(cpus):
    """
    Given a list of CPU IDs, return the common NUMA node if all CPUs
    belong to the same node. Otherwise return None.
    """
    nodes = set()
    for cpu in cpus:
        cpu_dir = f"/sys/devices/system/cpu/cpu{cpu}"
        try:
            entries = [e for e in os.listdir(cpu_dir) if e.startswith("node")]
            if not entries:
                raise FileNotFoundError(f"No NUMA node info for CPU {cpu}")
            nodes.add(int(entries[0].replace("node", "")))
        except Exception as e:
            print(f"Warning: could not determine NUMA node for CPU {cpu}: {e}")
            return None

    return nodes.pop() if len(nodes) == 1 else None


def create_container_from_image(
    image_name: str,
    test_spec: TestSpec,
    run_id: str,
    client: docker.DockerClient,
    logger: logging.Logger,
    cpu_groups: str | None = None,
):
    """
    Builds a container from the given image name and creates it.

    Args:
        image_name (str): Name of the image to create the container from
        client (docker.DockerClient): Docker client for creating the container
        run_id (str): Run ID identifying process, used for the container name
        logger (logging.Logger): Logger to use for logging the build process
        nocache (bool): Whether to use the cache when building
        cpu_groups (str): CPU groups to assign to the container
    """
    container = None
    try:
        # Get configurations for how container should be created
        config = MAP_REPO_VERSION_TO_SPECS[test_spec.repo][test_spec.version]
        user = "root" if not config.get("execute_test_as_nonroot", False) else "nonroot"
        # nano_cpus = config.get("nano_cpus")

        extra_args = {}
        if cpu_groups:
            print(f"Using CPU groups: {cpu_groups}")
            extra_args.update(cpu_groups)
            print(f"Creating container with args: {extra_args}")

        # Create the container
        logger.info(f"Creating container for {test_spec.instance_id}...")
        container = client.containers.create(
            image=image_name,
            name=test_spec.get_instance_container_name(run_id),
            user=user,
            detach=True,
            command="tail -f /dev/null",
            # nano_cpus=nano_cpus,
            platform=test_spec.platform,
            network_mode="host" if USE_HOST_NETWORK else None,
            # mem_limit="32g",
            oom_kill_disable=False,
            oom_score_adj=1000,
            **extra_args,
        )
        logger.info(f"Container for {test_spec.instance_id} created: {container.id}")
        return container
    except Exception as e:
        # If an error occurs, clean up the container and raise an exception
        logger.error(f"Error creating container for {test_spec.instance_id}: {e}")
        logger.info(traceback.format_exc())
        cleanup_container(client, container, logger)
        raise BuildImageError(test_spec.instance_id, str(e), logger) from e


def build_container(
    test_spec: TestSpec,
    client: docker.DockerClient,
    run_id: str,
    logger: logging.Logger,
    nocache: bool,
    force_rebuild: bool = False,
    cpu_groups: str | None = None,
):
    """
    Builds the instance image for the given test spec and creates a container from the image.

    Args:
        test_spec (TestSpec): Test spec to build the instance image and container for
        client (docker.DockerClient): Docker client for building image + creating the container
        run_id (str): Run ID identifying process, used for the container name
        logger (logging.Logger): Logger to use for logging the build process
        nocache (bool): Whether to use the cache when building
        force_rebuild (bool): Whether to force rebuild the image even if it already exists
    """
    # Build corresponding instance image
    if force_rebuild:
        remove_image(client, test_spec.instance_image_key, "quiet")
    build_instance_image(test_spec, client, logger, nocache)

    return create_container_from_image(
        test_spec.instance_image_key,
        test_spec,
        run_id,
        client,
        logger,
        cpu_groups=cpu_groups,
    )
