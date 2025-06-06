## nohup bash run_train_det.sh > ./output/train_det.log 2>&1 &
# conda activate troch_2.4.0
OutputDir=./output/md1
BatchSize=6
if [ $# -eq 1 ] ; then
  torchrun \
        --nproc_per_node=1 \
        main.py \
        --batch_size $BatchSize \
        --output_dir $OutputDir \
        --set_cost_class 2 \
        --lr $1 \
        --lr_drop 30 \
        --gamma 0.3 \
        --coco_path ../../datasets/coco_2017/
else
torchrun \
        --nproc_per_node=1 \
        main.py \
        --batch_size $BatchSize \
        --output_dir $OutputDir \
        --set_cost_class 2 \
        --lr $1 \
        --resume $2 \
        --lr_drop 30 \
        --gamma 0.3 \
        --coco_path ../../datasets/coco_2017/
fi