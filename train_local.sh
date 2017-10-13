gsutil -m rm -rf gs://pix2pixdata/dev/output
python mypix2pix.py --topic=dev --mode=train --max_steps=2
