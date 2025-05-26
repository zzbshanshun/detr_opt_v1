## nohup bash run_train_det.sh > ./output/train_det.log 2>&1 &
# conda activate troch_2.4.0
torchrun \
        --nproc_per_node=1 \
        main.py \
        --batch_size 4 \
        --output_dir ./output \
        --lr $1 \
        --lr_drop 25 \
        --gamma 0.3 \
        --coco_path ../../datasets/coco_2017/

        # --resume $2 \