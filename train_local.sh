if [ "$#" -ne 1 ]
then
    echo "pass in the depth argument"
    exit 1
fi
declare -r DEPTH=$1
./build.sh
python trainer/mypix2pix.py --topic=southpark --max_steps=500 \
       --depth="$DEPTH"\
       --steps_per_export=1000 \
       --output_dir=./output \
       --training_image_dir /big/data/southpark/train_images 

