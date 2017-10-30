if [ "$#" -ne 1 ]
then
    echo "pass in the depth argument"
    exit 1
fi
declare -r DEPTH=$1
./build.sh
python trainer/progress.py --topic=southpark --max_steps=500 \
       --depth="$DEPTH" \
       --output_dir=./export \
       --checkpoint=./output \
       --training_image_dir /big/data/southpark/train_images 


#       --checkpoint=./output \

