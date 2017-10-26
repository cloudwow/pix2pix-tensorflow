./build.sh
python trainer/mypix2pix.py --topic=southpark --max_steps=700000 \
       --output_dir=./output \
       --training_image_dir /ldata/data/southpark/train_images 


#       --checkpoint=./output \

