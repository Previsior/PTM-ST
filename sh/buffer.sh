CUDA_VISIBLE_DEVICES=$1

python buffer.py --dataset='coco' --train_epochs=10 --num_experts=5 \
    --buffer_path='buffer' --image_encoder=dinov2 --text_encoder=bge \
    --image_size 224 --out_embedding 768
