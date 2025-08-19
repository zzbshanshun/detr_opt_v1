python main.py \
        --batch_size 7 \
        --no_aux_loss --eval \
        --resume ./output/md48/checkpoint0001_2.pth \
        --coco_path ../../datasets/coco_2017/

#--resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \