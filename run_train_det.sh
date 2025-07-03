## nohup bash run_train_det.sh > ./output/train_det.log 2>&1 &
# conda activate troch_2.4.0
OutputDir=./output/md16
BatchSize=6
CostClass=2
CostBbox=5
CostGiou=2
ClsLossCoef=3
BboxLossCoef=5
GiouLossCoef=2

if [ $# -eq 1 ] ; then
  torchrun \
        --nproc_per_node=1 \
        main.py \
        --batch_size $BatchSize \
        --output_dir $OutputDir \
        --set_cost_class $CostClass \
        --set_cost_bbox $CostBbox \
        --set_cost_giou $CostGiou \
        --cls_loss_coef $ClsLossCoef \
        --bbox_loss_coef $BboxLossCoef \
        --giou_loss_coef $GiouLossCoef \
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
        --set_cost_class $CostClass \
        --set_cost_bbox $CostBbox \
        --set_cost_giou $CostGiou \
        --cls_loss_coef $ClsLossCoef \
        --bbox_loss_coef $BboxLossCoef \
        --giou_loss_coef $GiouLossCoef \
        --lr $1 \
        --resume $2 \
        --lr_drop 30 \
        --gamma 0.3 \
        --coco_path ../../datasets/coco_2017/
fi
