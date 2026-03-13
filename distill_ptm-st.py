from collections import defaultdict
import glob
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.utils
import math

import wandb
import copy
import datetime
import time

from data import get_dataset, textprocess, textprocess_train
from src.epoch import evaluate_synset_PTM
from src.networks import CLIPModel_full, MultilabelContrastiveLoss
from src.reparam_module import ReparamModule
from src.utils import ParamDiffAug, get_time
from src.similarity_mining import LowRankSimilarityGenerator, FullSimilarityGenerator
from src.vl_distill_utils import shuffle_files, nearest_neighbor, get_images_texts, load_or_process_file, syn_dataset

from model_ema import ModelEmaV2
# from MCT.convexify import interp_params


def make_timestamp(prefix: str="", suffix: str="") -> str:
    tmstamp = '{:%m%d_%H%M%S}'.format(datetime.datetime.now())
    return prefix + tmstamp + suffix



def main(args, syn_data_list):
    print(f"Distilling the {args.current_subset_num}th subset.")
    
    # 1. load dataset
    ''' organize the real train dataset '''  
    trainloader, testloader, train_dataset, test_dataset = get_dataset(args)
    train_sentences = train_dataset.get_all_captions() 
    data = load_or_process_file('text', textprocess, args, testloader)
    train_caption = load_or_process_file('train_text', textprocess_train, args, train_sentences)
    bert_test_embed = torch.from_numpy(data['bert_test_embed']).cpu()
    print("The shape of bert_test_embed: {}".format(bert_test_embed.shape))
    train_caption_embed = torch.from_numpy(train_caption['bert_test_embed']).cpu()
    print("The shape of train_caption_embed: {}".format(train_caption_embed.shape))


    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / args.temperature))

    # 2. Check and process parameters
    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))
    
    if args.zca and args.texture:
        raise AssertionError("Cannot use zca and texture together")
    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.dsa:
        args.dc_aug_param = None
    args.dsa_param = ParamDiffAug()
    zca_trans = args.zca_trans if args.zca else None
    args.zca_trans = zca_trans
    args.distributed = torch.cuda.device_count() > 1


    # 3. initialize the synthetic data
    syn_lr_img = torch.tensor(args.lr_teacher_img).to(args.device).requires_grad_(True)
    syn_lr_txt = torch.tensor(args.lr_teacher_txt).to(args.device).requires_grad_(True)
    image_syn, text_syn = get_images_texts(args.num_queries, train_dataset, args)
    if args.sim_type == 'lowrank':
        sim_generator = LowRankSimilarityGenerator(args.num_queries, args.sim_rank, args.alpha)
    elif args.sim_type == 'full':
        sim_generator = FullSimilarityGenerator(args.num_queries)
    else:
        raise AssertionError("Invalid similarity type: {}".format(args.sim_type))
    sim_generator = sim_generator.to(args.device)

    if args.pix_init == 'noise':
        mean = torch.tensor([-0.0626, -0.0221,  0.0680])
        std = torch.tensor([1.0451, 1.0752, 1.0539])
        image_syn = torch.randn([args.num_queries, 3, 224, 224])
        for c in range(3):
            image_syn[:, c] = image_syn[:, c] * std[c] + mean[c]
        print(f'Subset{args.current_subset_num}: Initialized synthetic image from random noise')

    if args.txt_init == 'noise':
        text_syn = torch.normal(mean=-0.0094, std=0.5253, size=(args.num_queries, 768))
        print(f'Subset{args.current_subset_num}: Initialized synthetic text from random noise')


    ''' training '''
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    text_syn = text_syn.detach().to(args.device).requires_grad_(True)
    contrastive_criterion = MultilabelContrastiveLoss(args.loss_type)

    optimizer = torch.optim.SGD([
        {'params': [image_syn], 'lr': args.lr_img, "momentum": args.momentum_syn},
        {'params': [text_syn], 'lr': args.lr_txt, "momentum": args.momentum_syn},
        {'params': [syn_lr_img, syn_lr_txt], 'lr': args.lr_lr, "momentum": args.momentum_lr},
        {'params': sim_generator.get_indexed_parameters(), 'lr': args.lr_sim, "momentum": args.momentum_sim},
    ], lr=0)
    optimizer.zero_grad()
    
    # ema
    ema = ModelEmaV2([image_syn, text_syn] + sim_generator.get_indexed_parameters() + [syn_lr_img, syn_lr_txt], decay=args.ema_decay)
    
    if args.draw:
        sentence_list = nearest_neighbor(train_sentences, text_syn.detach().cpu(), train_caption_embed)
        wandb.log({"original_sentence_list": wandb.Html('<br>'.join(sentence_list))}, step=0)
        wandb.log({"original_synthetic_images": wandb.Image(torch.nan_to_num(image_syn.detach().cpu()))}, step=0)


    # 4. Load expert trajectory
    expert_dir = args.buffer_path
    print("Expert Dir: {}".format(expert_dir))

    img_expert_files = list(glob.glob(os.path.join(expert_dir, "img_replay_buffer_*.pt")))
    txt_expert_files = list(glob.glob(os.path.join(expert_dir, "txt_replay_buffer_*.pt")))
    if len(txt_expert_files) != len(img_expert_files) or len(txt_expert_files) == 0:
        raise AssertionError("No buffers / Error buffers detected at {}".format(expert_dir))
    
    img_expert_files, txt_expert_files = shuffle_files(img_expert_files, txt_expert_files)
    
    file_idx = 0
    expert_idx = 0
    
    img_buffer = torch.load(img_expert_files[file_idx])
    txt_buffer = torch.load(txt_expert_files[file_idx])


    # === record time and memory overhead
    _distill_iter_count = 0
    _distill_time_accum_s = 0.0
    _alloc_samples_bytes = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    # ===
    
    # 5. distillation
    progress = 0
    for it in tqdm(range(sum(args.Iteration[:args.current_subset_num]), sum(args.Iteration[:args.current_subset_num+1])+1)):
        wandb.log({"Progress": it}, step=it)
        progress += 1
        
        # 5.1 Evaluate the distilled data and record
        if progress % args.eval_it == 0:
            print('Evaluation for ema\nimage_model_train = %s, text_model_train = %s, subset_num = %d, iteration = %d'%(args.image_encoder, args.text_encoder, args.current_subset_num, it))

            # aggregated results of multiple evaluations
            ema_multi_eval_aggr_result = defaultdict(list)
            for it_eval in range(args.num_eval):
                net_eval = CLIPModel_full(args)
                with torch.no_grad():
                    ema_image_syn = ema.module[0]
                    ema_text_syn = ema.module[1]
                    if args.sim_type == 'lowrank':
                        ema_sim_params = ema.module[2:5]
                    elif args.sim_type == 'full':
                        ema_sim_params = [ema.module[2]]
                    
                    ema_image_save = ema_image_syn
                    ema_text_save = ema_text_syn
                    ema_sim_params_save = ema_sim_params

                # avoid any unaware modification
                ema_image_syn_eval, ema_text_syn_eval = copy.deepcopy(ema_image_save.detach()), copy.deepcopy(ema_text_save.detach())
                ema_sim_params = copy.deepcopy([p.detach() for p in ema_sim_params_save])
                lr_img, lr_txt = copy.deepcopy(syn_lr_img.detach().item()), copy.deepcopy(syn_lr_txt.detach().item())
                ema_similarity_syn_eval = copy.deepcopy(sim_generator.generate_with_param(ema_sim_params).detach())
                
                # eval function
                _, _, ema_best_val_result = evaluate_synset_PTM(
                    it_eval, net_eval, ema_image_syn_eval, ema_text_syn_eval, lr_img, lr_txt,
                    ema_similarity_syn_eval, testloader, args, bert_test_embed,
                    syn_data_list)
                
                # record
                for k, v in ema_best_val_result.items():
                    ema_multi_eval_aggr_result[k].append(v)
                    
            for key, values in ema_multi_eval_aggr_result.items():
                if key in ["img_r_mean", "txt_r_mean"]:
                    continue
                wandb.log({
                    "Mean/{}".format(key): np.mean(values), 
                    "Std/{}".format(key): np.std(values)
                }, step=it)
        
        
        # 5.2 Record Results, Draw (Optional)
        if progress % args.eval_it == 0:
            with torch.no_grad():
                save_dir = os.path.join(".", "logged_files", args.dataset, wandb.run.name)
                print("Saving to {}".format(save_dir))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                ema_image_syn = ema.module[0]
                ema_text_syn = ema.module[1]
                if args.sim_type == 'lowrank':
                    ema_sim_params = ema.module[2:5]
                elif args.sim_type == 'full':
                    ema_sim_params = [ema.module[2]]

                image_save = ema_image_syn.detach().cpu()
                text_save = ema_text_syn.detach().cpu()
                sim_params = ema_sim_params
                sim_mat = sim_generator.generate_with_param(sim_params)

                torch.save({
                    "image": image_save,
                    "text": text_save,
                    "similarity_params": [x.detach().cpu() for x in sim_params],
                    "similarity_mat": sim_mat.detach().cpu(),
                    "syn_lr_img": syn_lr_img.detach().cpu(),
                    "syn_lr_txt": syn_lr_txt.detach().cpu(),
                }, os.path.join(save_dir, f"distilled_{args.current_subset_num}_{it}.pt") )

                if args.draw:
                    wandb.log({"Pixels": wandb.Histogram(torch.nan_to_num(image_syn.detach().cpu()))}, step=it)  # Move tensor to CPU before converting to NumPy

                    if args.ipc < 50 or args.force_save:
                        upsampled = image_save[:90]
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        sentence_list = nearest_neighbor(train_sentences, text_syn.cpu(), train_caption_embed)
                        sentence_list = sentence_list[:90]
                        torchvision.utils.save_image(grid, os.path.join(save_dir, "synthetic_images_{}.png".format(it)))
                        
                        with open(os.path.join(save_dir, "synthetic_sentences_{}.txt".format(it)), "w") as file:
                            file.write('\n'.join(sentence_list))
                        wandb.log({"Synthetic_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it) 
                        wandb.log({'Synthetic_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it) 
                        wandb.log({"Synthetic_Sentences": wandb.Html('<br>'.join(sentence_list))}, step=it)
                        print("finish saving images")

                        for clip_val in [2.5]:
                            std = torch.std(image_save)
                            mean = torch.mean(image_save)
                            upsampled = torch.clip(image_save, min=mean-clip_val*std, max=mean+clip_val*std).cpu()  # Move to CPU
                            if args.dataset != "ImageNet":
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                            grid = torchvision.utils.make_grid(upsampled[:90], nrow=10, normalize=True, scale_each=True)
                            wandb.log({"Clipped_Synthetic_Images/std_{}".format(clip_val): wandb.Image(torch.nan_to_num(grid))}, step=it)
                            torchvision.utils.save_image(grid, os.path.join(save_dir, "clipped_synthetic_images_{}_std_{}.png".format(it, clip_val)))
                    if args.zca:
                        raise AssertionError("we do not use ZCA transformation")

        wandb.log({"Synthetic_LR/Image": syn_lr_img.detach().cpu()}, step=it)
        wandb.log({"Synthetic_LR/Text": syn_lr_txt.detach().cpu()}, step=it)
        if args.sim_type == 'lowrank':
            wandb.log({"ema_Synthetic_LR/Image": ema.module[5].detach().cpu()}, step=it)
            wandb.log({"ema_Synthetic_LR/Text": ema.module[6].detach().cpu()}, step=it)
        elif args.sim_type == 'full':
            wandb.log({"ema_Synthetic_LR/Image": ema.module[3].detach().cpu()}, step=it)
            wandb.log({"ema_Synthetic_LR/Text": ema.module[4].detach().cpu()}, step=it)

        # 5.3 optimize distillation data
        # === record time and memory overhead
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _t0 = time.perf_counter()
        # ===

        # student net
        torch.cuda.empty_cache()
        student_net = CLIPModel_full(args, temperature=args.temperature).to(args.device)
        if args.image_trainable:
            img_student_net = ReparamModule(student_net.image_encoder.to('cpu')).to('cuda')
        else:
            img_student_net = ReparamModule(student_net.image_projection.to('cpu')).to('cuda')
        txt_student_net = ReparamModule(student_net.text_projection.to('cpu')).to('cuda')

        if args.distributed:
            img_student_net = torch.nn.DataParallel(img_student_net)
            txt_student_net = torch.nn.DataParallel(txt_student_net)

        img_student_net.train()
        txt_student_net.train()

        # trajectory
        img_expert_trajectory = img_buffer[expert_idx]
        txt_expert_trajectory = txt_buffer[expert_idx]
        expert_idx += 1
        if expert_idx == len(img_buffer):
            expert_idx = 0
            file_idx += 1
            if file_idx == len(img_expert_files): 
                file_idx = 0
                img_expert_files, txt_expert_files = shuffle_files(img_expert_files, txt_expert_files)
            if args.max_files != 1:
                del img_buffer
                del txt_buffer
                img_buffer = torch.load(img_expert_files[file_idx])
                txt_buffer = torch.load(txt_expert_files[file_idx])
        start_epoch = np.random.randint(args.min_start_epoch, args.max_start_epoch)
        img_starting_params = img_expert_trajectory[start_epoch]
        txt_starting_params = txt_expert_trajectory[start_epoch]
        img_target_params = img_expert_trajectory[start_epoch+args.expert_epochs]
        txt_target_params = txt_expert_trajectory[start_epoch+args.expert_epochs]
        img_target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in img_target_params], 0)
        txt_target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in txt_target_params], 0)
        img_student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in img_starting_params], 0).requires_grad_(True)]
        txt_student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in txt_starting_params], 0).requires_grad_(True)]
        img_starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in img_starting_params], 0)
        txt_starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in txt_starting_params], 0)
        syn_images = image_syn
        syn_texts = text_syn

        x_list = []
        y_list = []
        sim_list = []   # the parameters of parameterized similarity matrix
        sim_param_list = []   # the parameters of parameterized similarity matrix
        grad_sum_img = torch.zeros(img_student_params[-1].shape).to(args.device)
        grad_sum_txt = torch.zeros(txt_student_params[-1].shape).to(args.device)

        syn_image_gradients = torch.zeros(syn_images.shape).to(args.device)
        syn_txt_gradients = torch.zeros(syn_texts.shape).to(args.device)
        syn_sim_param_gradients = [torch.zeros(x.shape).to(args.device) for x in sim_generator.get_indexed_parameters()]

        indices_chunks_copy = []
        indices = torch.randperm(len(syn_images))
        index = 0
        
        # Use the distillation dataset to train student_net, select mini_batch_size image text pairs for training each time, and optimize syn_steps times
        these_indices_list = []
        for _ in range(args.syn_steps):
            if args.mini_batch_size + index > len(syn_images):
                indices = torch.randperm(len(syn_images))
                index = 0
            these_indices = indices[index : index + args.mini_batch_size]
            these_indices_list.append(these_indices)
            index += args.mini_batch_size
            indices_chunks_copy.append(these_indices)

            x = syn_images[these_indices]
            this_y = syn_texts[these_indices]
            x_list.append(x.clone())
            y_list.append(this_y.clone())

            this_sim_param = sim_generator.get_indexed_parameters(these_indices)
            this_sim = sim_generator.generate_with_param(this_sim_param)
            if args.distributed:
                img_forward_params = img_student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
                txt_forward_params = txt_student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                img_forward_params = img_student_params[-1]
                txt_forward_params = txt_student_params[-1]
            if not args.image_trainable:
                x = student_net.image_encoder(x)

            x = img_student_net(x, flat_param=img_forward_params)
            x = x / x.norm(dim=1, keepdim=True)
            this_y = txt_student_net(this_y, flat_param=txt_forward_params)
            this_y = this_y / this_y.norm(dim=1, keepdim=True)
            image_logits = logit_scale.exp() * x.float() @ this_y.float().t()
            
            sim_list.append(this_sim)
            sim_param_list.append(this_sim_param)

            contrastive_loss = contrastive_criterion(image_logits, this_sim)
            
            img_grad = torch.autograd.grad(contrastive_loss, img_student_params[-1], create_graph=True)[0]
            txt_grad = torch.autograd.grad(contrastive_loss, txt_student_params[-1], create_graph=True)[0]

            img_detached_grad = img_grad.detach().clone()
            img_student_params.append(img_student_params[-1] - syn_lr_img.item() * img_detached_grad)
            grad_sum_img += img_detached_grad
            
            txt_detached_grad = txt_grad.detach().clone()
            txt_student_params.append(txt_student_params[-1] - syn_lr_txt.item() * txt_detached_grad)
            grad_sum_txt += txt_detached_grad

            del img_grad
            del txt_grad
        
        # manual gradient
        img_param_dist = torch.tensor(0.0).to(args.device)
        txt_param_dist = torch.tensor(0.0).to(args.device)

        img_param_dist += torch.nn.functional.mse_loss(img_starting_params, img_target_params, reduction="sum")
        txt_param_dist += torch.nn.functional.mse_loss(txt_starting_params, txt_target_params, reduction="sum")

        for i in range(args.syn_steps):
            x, y = x_list[i], y_list[i]
            if not args.image_trainable:
                x = student_net.image_encoder(x)
            x = img_student_net(x, flat_param=img_student_params[i])
            x = x / x.norm(dim=1, keepdim=True)
            this_y = txt_student_net(y, flat_param=txt_student_params[i])
            this_y = this_y / this_y.norm(dim=1, keepdim=True)
            image_logits = logit_scale.exp() * x.float() @ this_y.float().t() 
            loss_i = contrastive_criterion(image_logits, sim_list[i])

            img_single_term = syn_lr_img.item() * (img_target_params - img_starting_params)
            img_square_term = (syn_lr_img.item() ** 2) * grad_sum_img
            txt_single_term = syn_lr_txt.item() * (txt_target_params - txt_starting_params) 
            txt_square_term = (syn_lr_txt.item() ** 2) * grad_sum_txt

            img_grad_i = torch.autograd.grad(loss_i, img_student_params[i], create_graph=True, retain_graph=True)[0]
            txt_grad_i = torch.autograd.grad(loss_i, txt_student_params[i], create_graph=True, retain_graph=True)[0]
            
            img_syn_real_dist = (img_single_term + img_square_term) @ img_grad_i
            txt_syn_real_dist = (txt_single_term + txt_square_term) @ txt_grad_i
            grand_loss_i = img_syn_real_dist / img_param_dist + txt_syn_real_dist / txt_param_dist

            multiple_gradients_in_one_time = torch.autograd.grad(
                2 * grand_loss_i, 
                [x_list[i], y_list[i]] + sim_param_list[i]
            )

            img_gradients = multiple_gradients_in_one_time[0]
            txt_gradients = multiple_gradients_in_one_time[1]
            sim_param_gradients = multiple_gradients_in_one_time[2:]
            
            with torch.no_grad():
                ids = indices_chunks_copy[i]
                syn_image_gradients[ids] += img_gradients
                syn_txt_gradients[ids] += txt_gradients

                assert len(sim_param_gradients) == len(syn_sim_param_gradients), f"{len(sim_param_gradients)}, {len(syn_sim_param_gradients)}"
                for g_idx in range(len(sim_param_gradients)):
                    if args.sim_type == 'full':
                        syn_sim_param_gradients[g_idx][ids[:,None], ids] += sim_param_gradients[g_idx]
                        # !!! gradient will be lost if using xxxx[ids, :][:, ids]
                    else:
                        syn_sim_param_gradients[g_idx][ids, ...] += sim_param_gradients[g_idx]

        # ---------end of computing input image gradients and learning rates--------------
        syn_images.grad = syn_image_gradients
        syn_texts.grad = syn_txt_gradients
        for g_idx, param in enumerate(sim_generator.get_indexed_parameters()):
            param.grad = syn_sim_param_gradients[g_idx]

        img_grand_loss = img_starting_params - syn_lr_img * grad_sum_img - img_target_params 
        txt_grand_loss = txt_starting_params - syn_lr_txt * grad_sum_txt - txt_target_params 
        img_grand_loss = img_grand_loss.dot(img_grand_loss) / img_param_dist
        txt_grand_loss = txt_grand_loss.dot(txt_grand_loss) / txt_param_dist
        grand_loss = img_grand_loss + txt_grand_loss
        img_lr_grad = torch.autograd.grad(img_grand_loss, syn_lr_img)[0]
        syn_lr_img.grad = img_lr_grad
        txt_lr_grad = torch.autograd.grad(txt_grand_loss, syn_lr_txt)[0]
        syn_lr_txt.grad = txt_lr_grad
        if math.isnan(img_grand_loss):
            break

        wandb.log({
            "Loss/grand": grand_loss.detach().cpu(),
            # "Start_Epoch": start_epoch,
            "Loss/img_grand": img_grand_loss.detach().cpu(),
            "Loss/txt_grand": txt_grand_loss.detach().cpu(),
        }, step=it)
        wandb.log({"Synthetic_LR/grad_syn_lr_img": syn_lr_img.grad.detach().cpu()}, step=it)
        wandb.log({"Synthetic_LR/grad_syn_lr_txt": syn_lr_txt.grad.detach().cpu()}, step=it)
        optimizer.step()
        ema.update([image_syn, text_syn] + sim_generator.get_indexed_parameters() + [syn_lr_img, syn_lr_txt])
        
        if args.clamp_lr:
            syn_lr_img.data = torch.clamp(syn_lr_img.data, min=args.clamp_lr)
            syn_lr_txt.data = torch.clamp(syn_lr_txt.data, min=args.clamp_lr)
        for _ in img_student_params:
            del _
        for _ in txt_student_params:
            del _
        if it % 10 == 0:
            print('%s subset_num = %d, iter = %04d, loss = %.4f' % (get_time(), args.current_subset_num, it, grand_loss.item()))

        # === record time and memory overhead
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            _alloc_samples_bytes.append(torch.cuda.memory_allocated())
        _distill_time_accum_s += (time.perf_counter() - _t0)
        _distill_iter_count += 1
        # ===

    # === record time and memory overhead
    def _bytes_to_mb(x_bytes: float) -> float:
        return x_bytes / (1024.0 ** 2) # MiB / MB
    _avg_s_per_iter = (_distill_time_accum_s / _distill_iter_count) if _distill_iter_count > 0 else float('nan')
    _avg_alloc_mb = _bytes_to_mb(np.mean(_alloc_samples_bytes)) if len(_alloc_samples_bytes) > 0 else float('nan')
    _peak_alloc_mb = _bytes_to_mb(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else float('nan')
    print(f"[Subset {args.current_subset_num}] Avg time per iter (distill only): {_avg_s_per_iter:.4f} s/iter")
    print(f"[Subset {args.current_subset_num}] Avg allocated GPU memory: {_avg_alloc_mb:.1f} MB")
    print(f"[Subset {args.current_subset_num}] Peak allocated GPU memory: {_peak_alloc_mb:.1f} MB")
    try:
        wandb.summary[f"Distill/S{args.current_subset_num}/avg_s_per_iter"] = float(_avg_s_per_iter)
        wandb.summary[f"Distill/S{args.current_subset_num}/avg_alloc_MB"]   = float(_avg_alloc_mb)
        wandb.summary[f"Distill/S{args.current_subset_num}/peak_alloc_MB"]  = float(_peak_alloc_mb)
    except Exception:
        pass
    # ===

    # 6. PTM part
    if args.sim_type == 'lowrank':
        nsg = LowRankSimilarityGenerator(args.num_queries, args.sim_rank, args.alpha)
        params = ema.module[2:5]
        a = params[0].detach().cpu()
        l = params[1].detach().cpu()
        r = params[2].detach().cpu()
        nsg.load_params([a, l, r])
        current_syn_data = syn_dataset(ema.module[0].detach().cpu(), ema.module[1].detach().cpu(),
            nsg, copy.deepcopy(ema.module[5].detach().item()), copy.deepcopy(ema.module[6].detach().item()))
    elif args.sim_type == 'full':
        nsg = FullSimilarityGenerator(args.num_queries)
        params = [ema.module[2].detach().cpu()]
        nsg.load_params(params)
        current_syn_data = syn_dataset(ema.module[0].detach().cpu(), ema.module[1].detach().cpu(),
            nsg, copy.deepcopy(ema.module[3].detach().item()), copy.deepcopy(ema.module[4].detach().item()))
    syn_data_list.append(current_syn_data)
    return syn_data_list



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='flickr', help='dataset')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode, check utils.py for more info')
    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')
    parser.add_argument('--eval_it', type=int, default=50, help='how often to evaluate')
    parser.add_argument('--epoch_eval_train', type=int, default=100, help='epochs to train a model with synthetic data')
    parser.add_argument('--lr_lr', type=float, default=0.01, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher_img', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--lr_teacher_txt', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--loss_type', type=str, default='WBCE')
    parser.add_argument('--batch_train', type=int, default=128, help='batch size for training networks')
    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"],
        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--txt_init', type=str, default='real', choices=["noise", "real"],
        help='noise/real: initialize synthetic texts from random noise or randomly sampled real images.')
    parser.add_argument('--dsa', type=str, default='False', choices=['True', 'False'], 
        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='./data/Flickr30k/', help='dataset path')
    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")
    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")
    parser.add_argument('--no_aug', action="store_true", default=True, help='this turns off diff aug during distillation')
    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')
    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')
    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    parser.add_argument('--name', type=str, default=current_time, help='name of wandb run')
    parser.add_argument('--mini_batch_size', type=int, default=20, help='number of queries')
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
    parser.add_argument('--text_encoder', type=str, default='bert', choices=['bert', 'clip', 'distilbert', 'bge'], help='text encoder')
    parser.add_argument('--text_pretrained', type=bool, default=True, help='text_pretrained')
    parser.add_argument('--image_pretrained', type=bool, default=True, help='image_pretrained')
    parser.add_argument('--text_trainable', type=bool, default=False, help='text_trainable')
    parser.add_argument('--image_trainable', default=False, action="store_true", help='image_trainable')
    parser.add_argument('--distill', type=bool, default=True, help='whether distill')
    parser.add_argument('--optimize', type=str, default='reparam', choices=['reparam', 'ift'], help='matching_train')
    parser.add_argument('--image_only', type=bool, default=False, help='None')
    parser.add_argument('--text_only', type=bool, default=False, help='None')
    parser.add_argument('--draw', type=bool, default=False, help='None')
    parser.add_argument('--std', type=bool, default=True, help='standard deviation')
    parser.add_argument('--disabled_wandb', type=bool, default=False, help='disable wandb')
    parser.add_argument('--test_with_norm', type=bool, default=False, help='')
    parser.add_argument('--clamp_lr', type=float, default=None, help='')

    # for LoRS    
    parser.add_argument('--sim_type', type=str, default="lowrank", choices=["full", "lowrank"], help='similarity matrix type')
    parser.add_argument('--alpha', type=float, default=1, help='alpha in LoRA')
    parser.add_argument('--lr_sim', type=float, default=100, help='learning rate for updating similarity mat learning rate')
    parser.add_argument('--temperature', type=float, default=0.07, help="temperature of CLIP model")
    parser.add_argument('--momentum_lr', type=float, default=0.5)
    parser.add_argument('--momentum_syn', type=float, default=0.5)
    parser.add_argument('--momentum_sim', type=float, default=0.5)
    
    # for PTM
    parser.add_argument('--subset_num', type=int, default=2)
    parser.add_argument('--current_subset_num', type=int, default=0)
    parser.add_argument('--expert_epochs', default=1, type=int)
    parser.add_argument('--num_queries', type=int, default=[200, 300], nargs='+')
    parser.add_argument('--Iteration', type=int, default=[2000, 3000], nargs='+', help='how many distillation steps to perform')
    parser.add_argument('--sim_rank', type=int, default=[5, 10], nargs='+', help='similarity matrix rank')
    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_txt', type=float, default=1000, help='learning rate for updating synthetic texts')
    parser.add_argument('--syn_steps', type=int, default=8, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=str, nargs='+')
    parser.add_argument('--min_start_epoch', type=str, nargs='+')
    parser.add_argument('--buffer_path', type=str, default=['./buffer/flickr/nfnet_bert/InfoNCE/normal'], nargs='+', help='buffer path')

    # ema
    parser.add_argument('--ema_decay', type=float, default=0.99)


    parser.add_argument('--out_embedding', type=int, default='2048')
    parser.add_argument('--debug', action="store_true", default=False)

    args = parser.parse_args()

    args.image_root = {
        'flickr': "./distill_utils/data/Flickr30k",
        'coco': "./distill_utils/data/COCO",
        'cc3m': "./distill_utils/data/CC3M"
    }[args.dataset]
    

    # subset list and wandb
    syn_data_list = []
    if args.disabled_wandb:
        wandb.init(mode = 'disabled')
    elif args.debug:
        wandb.init(project='debug', config=args, name=args.name+"_"+args.dataset+f"{args.subset_num}"+"_"+f"{args.num_queries}"+"_"+make_timestamp(), mode="online")
    else:
        run_name = args.name + "_" + args.dataset + str(args.subset_num) + "_" + str(args.num_queries)
        run_name +=  make_timestamp()
        wandb.init(project='distill', config=args, name=run_name, mode="online")

    for i in range(args.subset_num):
        args_copy = copy.deepcopy(args)
        args_copy.current_subset_num = i
        args_copy.num_queries = args_copy.num_queries[i]
        args_copy.min_start_epoch = args_copy.min_start_epoch[i]
        args_copy.max_start_epoch = args_copy.max_start_epoch[i]
        args_copy.buffer_path = args_copy.buffer_path[i]
        if args_copy.sim_type == 'lowrank':
            args_copy.sim_rank = args_copy.sim_rank[i]
        syn_data_list = main(args_copy, syn_data_list)
        
    wandb.finish()
