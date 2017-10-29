./build.sh
python trainer/mypix2pix.py --topic=southpark --max_steps=500 \
       --depth=3\
       --output_dir=./output \
       --training_image_dir /big/data/southpark/train_images 


#       --checkpoint=./output \

