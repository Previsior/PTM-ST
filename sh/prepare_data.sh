set -euo pipefail

CACHE_DIR="./.cache/ptm_st_data"
FLICKR_DIR="./distill_utils/data/Flickr30k/flickr30k-images"
COCO_DIR="./distill_utils/data/COCO"
CC3M_DIR="./distill_utils/data/CC3M"

mkdir -p "$CACHE_DIR/kagglehub" "$CACHE_DIR/coco" "$CACHE_DIR/cc3m/unzip"
mkdir -p "$FLICKR_DIR" "$COCO_DIR" "$CC3M_DIR"

# # flickr
# export KAGGLEHUB_CACHE="$CACHE_DIR/kagglehub"
# # export KAGGLE_API_TOKEN=KGAT_xxxxx
# pip install --no-cache-dir kagglehub
# python - <<'PY'
# import shutil
# from pathlib import Path
# import kagglehub
# target = Path("./distill_utils/data/Flickr30k/flickr30k-images")
# src = Path(kagglehub.dataset_download("hsankesara/flickr-image-dataset", force_download=True))
# for p in src.rglob("*"):
#     if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
#         dst = target / p.name
#         if not dst.exists():
#             shutil.copy2(p, dst)
# PY

# coco
wget -c -P "$CACHE_DIR/coco" http://images.cocodataset.org/zips/train2014.zip
wget -c -P "$CACHE_DIR/coco" http://images.cocodataset.org/zips/val2014.zip
wget -c -P "$CACHE_DIR/coco" http://images.cocodataset.org/zips/test2014.zip
unzip -q -n "$CACHE_DIR/coco/train2014.zip" -d "$COCO_DIR"
unzip -q -n "$CACHE_DIR/coco/val2014.zip" -d "$COCO_DIR"
unzip -q -n "$CACHE_DIR/coco/test2014.zip" -d "$COCO_DIR"

# llava-cc3m
wget -c -O "$CACHE_DIR/cc3m/images.zip" \
  https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/resolve/main/images.zip
unzip -q -n "$CACHE_DIR/cc3m/images.zip" -d "$CACHE_DIR/cc3m/unzip"
find "$CACHE_DIR/cc3m/unzip" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
  -exec mv -n {} "$CC3M_DIR/" \;

rm -rf "$CACHE_DIR"