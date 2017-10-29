./build.sh
python trainer/progress.py --topic=southpark --max_steps=500 \
       --depth=3 \
       --output_dir=./export \
       --checkpoint=./output \
       --training_image_dir /big/data/southpark/train_images 


#       --checkpoint=./output \

