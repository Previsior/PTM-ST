FILE_NAME=$(basename -- "$0")
EXP_NAME="${FILE_NAME%.*}"

export CUDA_VISIBLE_DEVICES=$1;

python eval.py --dataset=flickr --num_eval 1 \
    --ckpt_path 'logged_files/flickr/reproduce_flickr_499_liner__0629_100023/distilled_3000.pt' \
    --loss_type WBCE --image_encoder=nfnet --text_encoder=distilbert --num_queries 499 \
    --sim_type lowrank

# ['clip', 'nfnet', 'vit', 'nf_resnet50', "nf_regnet"]
# ['bert', 'clip', 'distilbert']
# --ckpt_path 'logged_files/flickr/distill_flickr2_200_299_liner_0712_163515/distilled_0_1999.pt' 'logged_files/flickr/distill_flickr2_200_299_liner_0712_163515/distilled_0_3999.pt' \
# --ckpt_path 'logged_files/flickr/reproduce_flickr_499_liner__0629_100023/distilled_3000.pt'
