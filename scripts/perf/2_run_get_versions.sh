
REPO_NAME=$1  # e.g., getmoto__moto

INSTANCE_PATH=artifacts/1_attributes/${REPO_NAME}-task-instances_attribute.jsonl

OUTPUT_DIR=${BASE_DIR}/artifacts/2_versioning
CONDA_PATH=~/miniforge3/bin/conda
TESTBED_PATH=~/testbed

pushd swefficiency/versioning

python get_versions.py \
    --instances_path $INSTANCE_PATH \
    --retrieval_method github \
    --conda_env temp \
    --num_workers 1 \
    --path_conda $CONDA_PATH \
    --output_dir $OUTPUT_DIR \
    --testbed $TESTBED_PATH

popd

OUTPUT_PATH=${OUTPUT_DIR}/${REPO_NAME}-task-instances_attribute_versions.json
python3 ${BASE_DIR}/scripts/filter_empty_version.py $OUTPUT_PATH
