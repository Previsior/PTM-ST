FILE_NAME=$(basename -- "$0")
EXP_NAME="${FILE_NAME%.*}"

export CUDA_VISIBLE_DEVICES=$1

# Flickr 500
python distill_tesla_SDD_ema.py --dataset=flickr \
    --buffer_path './buffer/flickr/nfnet_bert/InfoNCE/convexified_0_6' './buffer/flickr/nfnet_bert/InfoNCE/convexified_0_8' \
    --min_start_epoch 0 1 --max_start_epoch 2 3 \
    --lr_img 1000 --lr_txt 1000 --lr_lr 0.01 \
    --lr_teacher_img 0.1 --lr_teacher_txt 0.1 \
    --lr_sim 10.0 --sim_type full \
    --num_queries 200 299 --name ${EXP_NAME} \
    --Iteration 2000 2000 --subset_num 2 \
    --mini_batch_size 40 --ema_decay 0.99

# # Flickr 200
# python distill_tesla_SDD_ema.py --dataset=flickr \
#     --buffer_path './buffer/flickr/nfnet_bert/InfoNCE/convexified_0_6' './buffer/flickr/nfnet_bert/InfoNCE/convexified_0_8' \
#     --min_start_epoch 0 1 --max_start_epoch 2 3 \
#     --lr_img 1000 --lr_txt 1000 --lr_lr 0.01 \
#     --lr_teacher_img 0.1 --lr_teacher_txt 0.1 \
#     --lr_sim 10.0 --sim_type full \
#     --num_queries 99 100 --name ${EXP_NAME} \
#     --Iteration 2000 2000 --subset_num 2 --ema_decay 0.99

# # Flickr 100
# python distill_tesla_SDD_ema.py --dataset=flickr \
#     --buffer_path './buffer/flickr/nfnet_bert/InfoNCE/convexified_0_6' \
#     --min_start_epoch 0 --max_start_epoch 2 \
#     --lr_img 1000 --lr_txt 1000 --lr_lr 0.01 \
#     --lr_teacher_img 0.1 --lr_teacher_txt 0.1 \
#     --lr_sim 10.0 --sim_type full \
#     --num_queries 99 --name ${EXP_NAME} \
#     --Iteration 2000 --subset_num 2 --ema_decay 0.99

# COCO 500
python distill_tesla_SDD_ema.py --dataset=coco \
    --buffer_path './buffer/coco/nfnet_bert/InfoNCE/convexified_0_6' './buffer/coco/nfnet_bert/InfoNCE/convexified_0_8' \
    --image_root='distill_utils/data/COCO/' \
    --min_start_epoch 0 1 --max_start_epoch 2 3 \
    --lr_img 1000 --lr_txt 1000 --lr_lr 0.01 \
    --lr_teacher_img 0.1 --lr_teacher_txt 0.1 \
    --lr_sim 50 --sim_type full \
    --num_queries 200 299 --name ${EXP_NAME} \
    --Iteration 2000 2000 --subset_num 2 --ema_decay 0.99

# COCO 200
python distill_tesla_SDD_ema.py --dataset=coco \
    --buffer_path './buffer/coco/nfnet_bert/InfoNCE/convexified_0_6' './buffer/coco/nfnet_bert/InfoNCE/convexified_0_8' \
    --image_root='distill_utils/data/COCO/' \
    --min_start_epoch 0 1 --max_start_epoch 2 3 \
    --lr_img 1000 --lr_txt 1000 --lr_lr 0.01 \
    --lr_teacher_img 0.1 --lr_teacher_txt 0.1 \
    --lr_sim 50 --sim_type full \
    --num_queries 99 100 --name ${EXP_NAME} \
    --Iteration 2000 2000 --subset_num 2 --ema_decay 0.99

# COCO 100
python distill_tesla_SDD_ema.py --dataset=coco \
    --buffer_path './buffer/coco/nfnet_bert/InfoNCE/convexified_0_6' \
    --image_root='distill_utils/data/COCO/' \
    --min_start_epoch 0 --max_start_epoch 2 \
    --lr_img 1000 --lr_txt 1000 --lr_lr 0.01 \
    --lr_teacher_img 0.1 --lr_teacher_txt 0.1 \
    --lr_sim 50 --sim_type full \
    --num_queries 99 --name ${EXP_NAME} \
    --Iteration 2000 --subset_num 2 --ema_decay 0.99

# CC3M 500
python distill_tesla_SDD_ema.py --dataset=cc3m \
    --buffer_path './buffer/cc3m/nfnet_bert/InfoNCE/convexified_0_6' './buffer/cc3m/nfnet_bert/InfoNCE/convexified_0_8' \
    --min_start_epoch 0 1 --max_start_epoch 2 3 \
    --lr_img 1000 --lr_txt 1000 --lr_lr 0.01 \
    --lr_teacher_img 0.1 --lr_teacher_txt 0.1 \
    --lr_sim 50 --sim_type full \
    --num_queries 200 299 --name ${EXP_NAME} \
    --Iteration 2000 2000 --subset_num 2 --ema_decay 0.99

# CC3M 200
python distill_tesla_SDD_ema.py --dataset=cc3m \
    --buffer_path './buffer/cc3m/nfnet_bert/InfoNCE/convexified_0_6' './buffer/cc3m/nfnet_bert/InfoNCE/convexified_0_8' \
    --min_start_epoch 0 1 --max_start_epoch 2 3 \
    --lr_img 1000 --lr_txt 1000 --lr_lr 0.01 \
    --lr_teacher_img 0.1 --lr_teacher_txt 0.1 \
    --lr_sim 50 --sim_type full \
    --num_queries 99 100 --name ${EXP_NAME} \
    --Iteration 2000 2000 --subset_num 2 --ema_decay 0.99

# CC3M 100
python distill_tesla_SDD_ema.py --dataset=cc3m \
    --buffer_path './buffer/cc3m/nfnet_bert/InfoNCE/convexified_0_6' \
    --min_start_epoch 0 --max_start_epoch 2 \
    --lr_img 1000 --lr_txt 1000 --lr_lr 0.01 \
    --lr_teacher_img 0.1 --lr_teacher_txt 0.1 \
    --lr_sim 50 --sim_type full \
    --num_queries 99 --name ${EXP_NAME} \
    --Iteration 2000 --subset_num 2 --ema_decay 0.99
