gsutil -m rm -rf gs://pix2pixdata/dev/output
python mypix2pix.py --topic=dev --max_steps=20000 --verbosity=DEBUG
