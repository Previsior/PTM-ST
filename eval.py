"""Evaliuate the distilled data
Example:
    
"""

from collections import defaultdict
import os
import re

import numpy as np
import torch

import copy
import argparse
import datetime

from data import get_dataset, textprocess, textprocess_train
from src.reparam_module import ReparamModule
from src.epoch import evaluate_synset_with_similarity_PDD
from src.networks import CLIPModel_full
from src.similarity_mining import LowRankSimilarityGenerator, FullSimilarityGenerator

from src.vl_distill_utils import load_or_process_file, syn_dataset


def formatting_result_head():
    return "Img R@1  | Img R@5  | Img R@10 | Txt R@1  | Txt R@5  | Txt R@10 | Mean"


def formatting_result_content(val_result):
    return "{img_r1:9.2f} | {img_r5:9.2f} | {img_r10:9.2f} | {txt_r1:9.2f} | {txt_r5:9.2f} | {txt_r10:9.2f} | {r_mean:9.2f}".format(
        **val_result
    )

def formatting_result_content_clean(val_result):
    return "{img_r1} {img_r5} {img_r10} {txt_r1} {txt_r5} {txt_r10} {r_mean}".format(
        **val_result
    )

def formatting_result_all(val_result):
    return "Image R@1={img_r1} R@5={img_r5} R@10={img_r10} | Text R@1={txt_r1} R@5={txt_r5} R@10={txt_r10} | Mean={r_mean}".format(
        **val_result
    )



def main(args):  
    ''' organize the real train dataset '''  
    trainloader, testloader, train_dataset, test_dataset = get_dataset(args)

    train_sentences = train_dataset.get_all_captions() 

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = load_or_process_file('text', textprocess, args, testloader)
    train_caption = load_or_process_file('train_text', textprocess_train, args, train_sentences)

    bert_test_embed = torch.from_numpy(data['bert_test_embed']).cpu()
    print("The shape of bert_test_embed: {}".format(bert_test_embed.shape))
    train_caption_embed = torch.from_numpy(train_caption['bert_test_embed']).cpu()
    print("The shape of train_caption_embed: {}".format(train_caption_embed.shape))

    

    subset_num = len(args.ckpt_path)

    for path in args.ckpt_path:
        if not os.path.exists(path):
            raise ValueError(f"{path} does not exist")

    image_syn_lst = []
    text_syn_lst = []
    sim_param_lst = []
    sim_mat_lst = []
    syn_lr_img_lst = []
    syn_lr_txt_lst = []
    for i in range(subset_num):
        print("Load from", args.ckpt_path[i])
        ckpt = torch.load(args.ckpt_path[i])
        image_syn_lst.append(ckpt["image"].to(args.device))
        text_syn_lst.append(ckpt["text"].to(args.device))
        sim_param_lst.append([p.to(args.device) for p in ckpt["similarity_params"]])
        sim_mat_lst.append(ckpt["similarity_mat"].to(args.device))
        syn_lr_img_lst.append(ckpt.get("syn_lr_img"))
        syn_lr_txt_lst.append(ckpt.get("syn_lr_txt"))
    syn_lr_img = syn_lr_img_lst[-1]
    syn_lr_txt = syn_lr_txt_lst[-1]

    syn_data_list = []
    for i in range(subset_num):
        if i == subset_num - 1:
            continue
        if args.sim_type == 'lowrank':
            nsg = LowRankSimilarityGenerator(args.num_queries[i], args.sim_rank, args.alpha)
            params = sim_param_lst[i]
            nsg.load_params(params)
            current_syn_data = syn_dataset(image_syn_lst[i].detach().cpu(), text_syn_lst[i].detach().cpu(),
                                        nsg, copy.deepcopy(syn_lr_img_lst[i].item()), copy.deepcopy(syn_lr_txt_lst[i].item()))
        elif args.sim_type == 'full':
            nsg = FullSimilarityGenerator(args.num_queries[i])
            params = sim_param_lst[i]
            nsg.load_params(params)
            current_syn_data = syn_dataset(image_syn_lst[i].detach().cpu(), text_syn_lst[i].detach().cpu(),
                                        nsg, copy.deepcopy(syn_lr_img_lst[i].item()), copy.deepcopy(syn_lr_txt_lst[i].item()))
        syn_data_list.append(current_syn_data)
        
    print(syn_lr_img, syn_lr_txt)

    print('Evaluation\nimage_model_train = %s, text_model_train = %s, iteration = ?'%(args.image_encoder, args.text_encoder))

    multi_eval_aggr_result = defaultdict(list)  # aggregated results of multiple evaluations

    for it_eval in range(args.num_eval):
        net_eval = CLIPModel_full(args)

        image_syn_eval, text_syn_eval = copy.deepcopy(image_syn_lst[-1]), copy.deepcopy(text_syn_lst[-1])
        similarity_syn_eval = copy.deepcopy(sim_mat_lst[-1])

        _, _, best_val_result = evaluate_synset_with_similarity_PDD(
                    it_eval, net_eval, image_syn_eval, text_syn_eval, syn_lr_img, syn_lr_txt,
                    similarity_syn_eval, testloader, args, bert_test_embed,
                    syn_data_list)

        for k, v in best_val_result.items():
            multi_eval_aggr_result[k].append(v)


        if not args.std:
            formatting_result_content(best_val_result)
            # formatting_result_content_clean(best_val_result)
            # logged img_r1, img_r5, img_r10, txt_r1, txt_r5, txt_r10, r_mean

    print(formatting_result_head())
    if args.std:
        mean_results = {k: np.mean(v) for k, v in multi_eval_aggr_result.items()}
        std_results = {k: np.std(v) for k, v in multi_eval_aggr_result.items()}
        
        print(formatting_result_content(mean_results))
        print(formatting_result_content(std_results))
        print(formatting_result_content_clean({k: "%.2f$\\pm$%.2f"%(mean_results[k],std_results[k]) for k in std_results}))

    print(args.image_encoder)
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='flickr30k', help='dataset')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=50, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=100, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=3000, help='how many distillation steps to perform')

    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_txt', type=float, default=1000, help='learning rate for updating synthetic texts')
    parser.add_argument('--lr_lr', type=float, default=1e-03, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher_img', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--lr_teacher_txt', type=float, default=0.1, help='learning rate for updating network parameters')

    parser.add_argument('--loss_type', type=str)
    
    parser.add_argument('--batch_train', type=int, default=128, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--txt_init', type=str, default='real', choices=["noise", "real"],
                        help='noise/real: initialize synthetic texts from random noise or randomly sampled real images.')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='./data/Flickr30k/', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')

    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")

    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")

    parser.add_argument('--no_aug', action="store_true", default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')


    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')

    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parser.add_argument('--name', type=str, default=current_time, help='name of wandb run')
    parser.add_argument('--num_queries', type=int, default=[200, 299], nargs='+', help='number of queries')
    parser.add_argument('--mini_batch_size', type=int, default=100, help='number of queries')
    parser.add_argument('--basis', type=bool, default=False, help='whether use basis or not')
    parser.add_argument('--n_basis', type=int, default=64, help='n_basis')
    parser.add_argument('--recursive', type=bool, default=False, help='whether use basis or not')
    parser.add_argument('--load_npy', type=bool, default=False, help='load_npy')
    parser.add_argument('--image_size', type=int, default=224, help='image_size')
    parser.add_argument('--image_root', type=str, default='distill_utils/data/Flickr30k/', help='location of image root')
    parser.add_argument('--ann_root', type=str, default='./data/Annotation/', help='location of ann root')
    parser.add_argument('--batch_size_train', type=int, default=128, help='batch_size_train')
    parser.add_argument('--batch_size_test', type=int, default=128, help='batch_size_test')
    parser.add_argument('--image_encoder', type=str, default='nfnet',  help='image encoder') # , choices=['clip', 'nfnet', 'vit', 'nf_resnet50', "nf_regnet"]
    parser.add_argument('--text_encoder', type=str, default='bert', choices=['bert', 'clip', 'distilbert'], help='text encoder')
    parser.add_argument('--text_pretrained', type=bool, default=True, help='text_pretrained')
    parser.add_argument('--image_pretrained', type=bool, default=True, help='image_pretrained')
    parser.add_argument('--text_trainable', type=bool, default=False, help='text_trainable')
    parser.add_argument('--image_trainable', type=bool, default=True, help='image_trainable') 
    parser.add_argument('--only_has_image_projection', type=bool, default=False, help='None')
    parser.add_argument('--distill', type=bool, default=True, help='whether distill')
    parser.add_argument('--optimize', type=str, default='reparam', choices=['reparam', 'ift'], help='matching_train')
    parser.add_argument('--image_only', type=bool, default=False, help='None')
    parser.add_argument('--text_only', type=bool, default=False, help='None')
    parser.add_argument('--draw', type=bool, default=False, help='None')
    parser.add_argument('--std', type=bool, default=True, help='standard deviation')
    parser.add_argument('--disabled_wandb', type=bool, default=False, help='disable wandb')
    parser.add_argument('--test_with_norm', type=bool, default=False, help='')

    parser.add_argument('--clamp_lr', type=float, default=None, help='')


    # Arguments below are for LoRS

    parser.add_argument('--resume_from', default=None, type=str)
    
    parser.add_argument('--sim_type', type=str, default="full", choices=["full", "lowrank"], help='similarity matrix type')
    parser.add_argument('--sim_rank', type=int, default=10, help='similarity matrix rank')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha in LoRA')
    parser.add_argument('--lr_sim', type=float, default=1e-03, help='learning rate for updating similarity mat learning rate')
    parser.add_argument('--temperature', type=float, default=0.07, help="temperature of CLIP model")
    
    parser.add_argument('--momentum_lr', type=float, default=0.5)
    parser.add_argument('--momentum_syn', type=float, default=0.5)
    parser.add_argument('--momentum_sim', type=float, default=0.5)
    parser.add_argument('--merge_loss_branches', action="store_true", default=False)

    # Arguments below are for evaluation

    parser.add_argument('--ckpt_path', type=str, nargs='+')
    parser.add_argument('--syn_lr_img', type=float, default=None)
    parser.add_argument('--syn_lr_txt', type=float, default=None)
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--l2', type=float, default=0.0005)
    
    args = parser.parse_args()

    main(args)