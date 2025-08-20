## nohup bash run_train_det.sh 0.00010 > ./output/mdx/train_det_1.log 2>&1 &
# conda activate troch_2.4.0
OutputDir=./output/md49
BatchSize=7

if [ $# -eq 1 ] ; then
torchrun --nproc_per_node=1 \
        main.py \
        --output_dir $OutputDir \
        --lr $1 \
        --batch_size $BatchSize \
        --epochs 50 \
        --lr_drop 40 \
        --coco_path ../../datasets/coco_2017/
else
torchrun --nproc_per_node=1 \
        main.py \
        --output_dir $OutputDir \
        --lr $1 \
        --batch_size $BatchSize \
        --epochs 50 \
        --lr_drop 40 \
        --resume $2 \
        --coco_path ../../datasets/coco_2017/
fi