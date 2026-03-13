FILE_NAME=$(basename -- "$0")
EXP_NAME="${FILE_NAME%.*}"

export CUDA_VISIBLE_DEVICES=$1
python distill_ptm-st.py --dataset=cc3m \
    --buffer_path './buffer/cc3m/nfnet_bert/InfoNCE/convexified_0_6' './buffer/cc3m/nfnet_bert/InfoNCE/convexified_0_8' './buffer/cc3m/nfnet_bert/InfoNCE/convexified_0_10' \
    --min_start_epoch 0 1 2 --max_start_epoch 3 4 5 \
    --lr_img 1000 --lr_txt 1000 --lr_lr 0.01 \
    --lr_teacher_img 0.1 --lr_teacher_txt 0.1 \
    --lr_sim 10 --sim_type full \
    --num_queries 332 332 333 --name ${EXP_NAME} \
    --Iteration 2000 2000 2000 --subset_num 3 --ema_decay 0.99 --out_embedding 2048 --eval_it 200
