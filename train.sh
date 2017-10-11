declare -r TARGET_LABEL=$1
echo $TARGET_LABEL
declare -r PROJECT=$(gcloud config list project --format "value(core.project)")
declare -r JOB_ID="${USER}_$(date +%Y%m%d_%H%M%S)"
declare -r BUCKET="gs://pix2pixdata"
declare -r GCS_PATH="${BUCKET}/${TARGET_LABEL}"

# Submit training job.
gcloud ml-engine jobs submit training "$JOB_ID" \
       --module-name trainer.task \
       --package-path trainer \
       --staging-bucket "$BUCKET" \
       --region us-central1 \
       --config train_config.yaml \
       -- \
       --output_path "${GCS_PATH}/training" \
       --eval_data_paths "${GCS_PATH}/eval-*" \
       --train_data_paths "${GCS_PATH}/train-*"

# Monitor training logs.
gcloud ml-engine jobs stream-logs "$JOB_ID"

