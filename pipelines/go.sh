python extract_edges_pipeline.py --cloud --setup_file=./setup.py \
       --gcs_bucket=pix2pixdata \
       --runner=DataflowRunner \
       --project=pix2pix-182420 \
       --staging_location=gs://pix2pixdata/staging \
       --temp_location=gs://pix2pixdata/temp \
       --num_workers=5
