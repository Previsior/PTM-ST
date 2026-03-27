<div align="center">

<h1>Multimodal Dataset Distillation via Phased Teacher Models</h1>

[Shengbin Guo*](https://previsior.github.io/) &nbsp;&nbsp; [Hang Zhao*](https://openreview.net/profile?id=~Hang_Zhao4) &nbsp;&nbsp; [Senqiao Yang](https://senqiaoyang.com/) &nbsp;&nbsp;[Chenyang Jiang](https://github.com/j-cyoung/j-cyoung)<br>[Yuhang Cheng](https://openreview.net/profile?id=~Yuhang_Cheng3) &nbsp;&nbsp;[Xiangru Peng](https://openreview.net/profile?id=~Xiangru_Peng1) &nbsp;&nbsp;[Rui Shao](https://rshaojimmy.github.io/OrionLab/) &nbsp;&nbsp;[Zhuotao Tian†](https://zhuotaotian.github.io/)
<br>

<sup>* equal contribution &nbsp;&nbsp; † corresponding author</sup>

<p align="center">
    📖 <a href="https://openreview.net/forum?id=Me4AON8160"><b>OpenReview</b></a> &nbsp;&nbsp; | &nbsp;&nbsp; 🤗 <a href="https://huggingface.co/datasets/Previsior22/PTM-ST">Huggingface</a> &nbsp;&nbsp; | &nbsp;&nbsp; 📑 <a href="https://arxiv.org/abs/2603.25388">Arxiv</a> &nbsp;&nbsp;
</p>

</div>

---

TODO list:
- [x] Upload basic code.  
- [x] Upload buffers and models.  
- [x] Updated Repro Guidance (README).  
- [x] Submit paper to arxiv.  
- [ ] Add ST method.

## Getting Started
### Environment
Run the following commands to build environment:

```bash
git clone https://github.com/Previsior/PTM-ST.git
cd PTM-ST

conda create -n ptm-st python=3.8 -y
conda activate ptm-st
pip install -r requirements.txt
```

### Models and Datasets
**Pretrained models**: All pretrained weights/tokenizers used by the code are downloaded automatically on first use and stored in the default cache directory of the corresponding library (`transformers`, `timm`, or `clip`).

Reference links for the pretrained models currently used/supported by the code:

- Default image encoder `nfnet_l0` (`timm`): https://huggingface.co/timm/nfnet_l0.ra2_in1k
- Default text encoder `bert-base-uncased`: https://huggingface.co/google-bert/bert-base-uncased
- Optional text encoder `distilbert-base-uncased`: https://huggingface.co/distilbert/distilbert-base-uncased
- Optional text encoder `BAAI/bge-base-en-v1.5`: https://huggingface.co/BAAI/bge-base-en-v1.5
- Optional CLIP model `ViT-B/32`: https://huggingface.co/openai/clip-vit-base-patch32

**Datasets**: You can run the command below to download [Flickr30K](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset), [COCO](https://cocodataset.org/#download) and [LLaVA-CC3M](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/blob/main/images.zip) datasets.

```bash
bash sh/prepare_data.sh
```

Or download it manually yourself, and put them here:

```
./distill_utils/data/
├── Flickr30k/
│   ├── flickr30k-images/
│   │   ├── 1234.jpg
│   │   └── ......
│   ├── results_20130124.token
│   └── readme.txt
└── COCO/
│   ├── train2014/
│   ├── val2014/
│   └── test2014/
└── CC3M/
    ├── GCC_train_002820774.jpg
    ├── GCC_train_002805422.jpg
    └── ...
```

**Annotations**: Download the annotation files:

```bash
hf download previsor/PTM-ST --repo-type dataset --include "Annotation/*" --local-dir ./data
```

### Generate Expert Trajectories
You can generate expert trajectories by running the `sh/buffer.sh`, or alternatively, download our [pre-generated trajectories](https://huggingface.co/datasets/Previsior22/PTM-ST) for faster reproduction:

```bash
hf download previsor/PTM-ST --repo-type dataset --include "flickr/nfnet_bert/InfoNCE/normal/*" --local-dir ./buffer
hf download previsor/PTM-ST --repo-type dataset --include "coco/nfnet_bert/InfoNCE/normal/*" --local-dir ./buffer
hf download previsor/PTM-ST --repo-type dataset --include "cc3m/nfnet_bert/InfoNCE/normal/*" --local-dir ./buffer
```

### Genetate Convex Trajectories for ST
Coming soon.

### Distillation
You can distill multimodal datasets by running `sh/distill.sh`.
The file records the specific parameter settings of different datasets and distilled data pairs. For example, the Flickr 500 pair:

```bash
FILE_NAME=$(basename -- "$0")
EXP_NAME="${FILE_NAME%.*}"
export CUDA_VISIBLE_DEVICES=$1
python distill_ptm-st.py --dataset=flickr \
    --buffer_path './buffer/flickr/nfnet_bert/InfoNCE/convexified_0_6' './buffer/flickr/nfnet_bert/InfoNCE/convexified_0_8' \
    --min_start_epoch 0 1 --max_start_epoch 2 3 \
    --lr_img 1000 --lr_txt 1000 --lr_lr 0.01 \
    --lr_teacher_img 0.1 --lr_teacher_txt 0.1 \
    --lr_sim 10.0 --sim_type full \
    --num_queries 200 299 --name ${EXP_NAME} \
    --Iteration 2000 2000 --subset_num 2 \
    --mini_batch_size 40 --ema_decay 0.99 --image_trainable
```

## Citation
If you find this code useful in your research, please consider citing our work:

```bibtex
@article{guo2026multimodal,
  title={Multimodal Dataset Distillation via Phased Teacher Models},
  author={Guo, Shengbin and Zhao, Hang and Yang, Senqiao and Jiang, Chenyang and Cheng, Yuhang and Peng, Xiangru and Shao, Rui and Tian, Zhuotao},
  journal={arXiv preprint arXiv:2603.25388},
  year={2026}
}
```