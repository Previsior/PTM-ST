CUDA_VISIBLE_DEVICES=$1

# flickr
python buffer.py --dataset='flickr' --train_epochs=10 --num_experts=20 \
    --buffer_path='buffer' --image_encoder=nfnet --text_encoder=bert \
    --image_size 224 --image_trainable

# # coco
# python buffer.py --dataset='coco' --train_epochs=10 --num_experts=20 \
#     --buffer_path='buffer' --image_encoder=nfnet --text_encoder=bert \
#     --image_size 224 --image_trainable

# # cc3m
# python buffer.py --dataset='cc3m' --train_epochs=10 --num_experts=20 \
#     --buffer_path='buffer' --image_encoder=nfnet --text_encoder=bert \
#     --image_size 224 --out_embedding 2048

# # dinov2 + bge buffer
# python buffer.py --dataset='coco' --train_epochs=10 --num_experts=20 \
#     --buffer_path='buffer' --image_encoder=dinov2 --text_encoder=bge \
#     --image_size 224 --out_embedding 768