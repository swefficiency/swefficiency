
REPO_NAME=$1  # e.g., getmoto__moto
INSTANCE_PATH=artifacts/1_attributes/${REPO_NAME}-task-instances_attribute.jsonl
OUTPUT_DIR=artifacts/2_versioning
CONDA_PATH=~/miniforge3/condabin/conda
TESTBED_PATH=~/scratch/testbed

pushd swefficiency/versioning

python get_versions.py \
    --instances_path $INSTANCE_PATH \
    --retrieval_method github \
    --conda_env temp \
    --num_workers 4 \
    --path_conda $CONDA_PATH \
    --output_dir $OUTPUT_DIR \
    --testbed $TESTBED_PATH

popd

OUTPUT_PATH=$OUTPUT_DIR/${REPO_NAME}-task-instances_attribute_versions.json
python3 scripts/filter_empty_version.py $OUTPUT_PATH
