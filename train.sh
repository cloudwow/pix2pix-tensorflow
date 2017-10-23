declare -r TARGET_TOPIC=$1
echo $TARGET_TOPIC
declare -r PROJECT=$(gcloud config list project --format "value(core.project)")
declare -r JOB_ID="${USER}_$(date +%Y%m%d_%H%M%S)"
declare -r BUCKET="gs://pix2pixdata"
declare -r GCS_PATH="${BUCKET}/${TARGET_TOPIC}"
declare -r GCS_OUTPUT_PATH="${BUCKET}/${TARGET_TOPIC}/output"

# Submit training job.
gcloud ml-engine jobs submit training "$JOB_ID" \
       --module-name trainer.mypix2pix \
       --package-path trainer \
       --staging-bucket "$BUCKET" \
       --region us-central1 \
       --config train_config.yaml \
       --packages wheels/myp2p-0.0.2-py2.py3-none-any.whl \
       -- \
       --topic="$TARGET_TOPIC" \
       --training_image_dir train_images \
       --max_steps=30000 

       #       --checkpoint="$GCS_OUTPUT_PATH"
      
# Monitor training logs.
gcloud ml-engine jobs stream-logs "$JOB_ID"
