# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "2, 3, 4, 5, 6, 7"

import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import mxnet as mx

# from dit_models import DiT_models
# from models import GPT_Latent, UNet_Latent, GPTConfig, GPTLargeConfig
from diffusers.models import AutoencoderKL
# from sample import sample_func_autoregressive
from torchvision.utils import save_image
from bae.binaryae import BinaryGAN, BinaryAutoEncoder, load_pretrain
from hparams import get_vqgan_hparams
from sample import sample_func_autoregressive

from einops import rearrange, reduce, pack, unpack

import math
# from matplotlib import pyplot as plt

#################################################################################
#                             Training Helper Functions                         #
#################################################################################
#                               Set Weight Decay                                #
def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list or 'denoise_mlp' in name:
            no_decay.append(param)  # no weight decay on bias, norm and diffloss
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]
    
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

def pack_one(t, pattern):
    return pack([t], pattern)

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class mxImageNetDataset(Dataset):
    def __init__(self,
                 rec_file,
                 transform=None,
                 to_rgb=True):
        assert transform is not None
        self.transform = transform
        self.to_rgb = to_rgb

        self.record = mx.recordio.MXIndexedRecordIO(
            rec_file + '.idx', rec_file + '.rec', 'r')
        self.rec_index = list(sorted(self.record.keys))

        self.reckey2info = dict()
        index_file = rec_file + '.index'
        with open(index_file) as f:
            lines = f.readlines()
        for line in lines:
            split_parts = line.strip().split("\t")
            reckey, label, cls_name = split_parts[0], split_parts[2], split_parts[3]
            self.reckey2info[int(reckey)] = [label, cls_name]

        print("#images: ", len(self.rec_index), self.rec_index[:5])

    def __getitem__(self, idx):
        key = self.rec_index[idx]
        img = self.record.read_idx(int(key))
        head, im = mx.recordio.unpack_img(img)  # NOTE: BGR
        cls = head.label  # label in rec is numpy array.

        if self.to_rgb:
            im = im[:, :, ::-1]
        im = Image.fromarray(im)
        im = self.transform(im)

        return im, int(cls)

    def __len__(self):
        return len(self.rec_index)
    
def get_softlabels(binary_codes, dis=4):
    code_dim = binary_codes.shape[-1]
    device = binary_codes.device
    binary_codes = binary_codes.reshape(-1, code_dim)
    all_codes = torch.arange(int(2 ** code_dim))
    multipler = 2 ** (torch.arange(code_dim-1, -1, -1)) #tensor([32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1])
    codebook = ((all_codes[..., None] & multipler) != 0).float().to(device) 
    # breakpoint()
    dists = torch.where(binary_codes == 1, binary_codes, -1) @ (torch.where(codebook == 1, codebook, -1)).T
    # breakpoint()
    dists = (code_dim - dists) / 2
    # breakpoint()
    # dists = torch.where(dists>1, dists, 1)
    dists = torch.where(dists<=dis, dists, float('Inf'))
    # weights = torch.nn.functional.softmax(dists, dim=-1)
    # breakpoint()
    dists = torch.exp(dists * 3)
    weights = torch.nn.functional.normalize(1 / dists, p=1, dim=-1)
    # soft_labels_idx = torch.nonzero(dists <= dis)
    # breakpoint()
    # soft_labels = torch.index_select(codebook, dim=0, index=soft_labels_idx)
    return weights

def get_lr(it, learning_rate, min_lr, warmup_iters, lr_decay_iters):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def init_dist(launcher="pytorch", backend='nccl', port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == 'pytorch':
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, **kwargs)
    else:
        raise NotImplementedError(f'Not implemented launcher type: `{launcher}`!')
    
    return local_rank
#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args, args_ae):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP
    # dist.init_process_group("nccl")
    # Initialize distributed training
    local_rank      = init_dist()
    global_rank     = dist.get_rank()
    num_processes   = dist.get_world_size()
    is_main_process = global_rank == 0
    # dist.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=5000))
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    device = local_rank
    
    
    # seed = args.global_seed * dist.get_world_size() + rank
    seed = args.global_seed + global_rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={global_rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # ae.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        model_args = str(args.token_each)+'-'+str(args.code_dim)+'-g'+str(args.gen_iter_num) + args.extra_info
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-{model_args}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        save_dir = f"{experiment_dir}/sample_images"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        for k in args.__dict__:
            logger.info(k + ": " + str(args.__dict__[k]))
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 16
    
    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p
        
    #### GPT model ####
    if args.modeling == 'mlm':
        from llama.mlm_model import GPT_models
    elif args.modeling == 'ar':
        from llama.ar_model import GPT_models
        
    if args.modeling == 'mlm':
        model = GPT_models[args.model](
            use_adaLN=args.use_adaLN,
            mask_schstep=args.mask_schstep,
            mask_schtype=args.mask_schtype,
            norm_type=args.norm_type,
            maskpersample=args.maskpersample,
            gen_iter=args.gen_iter_num,
            pos_type=args.pos_type,
            smoothing=args.smoothing,
            token_each=args.token_each,
            code_dim=args.code_dim,
            vocab_size=args.vocab_size,
            block_size=latent_size ** 2,
            num_classes=args.num_classes,
            resid_dropout_p=dropout_p,
            ffn_dropout_p=dropout_p,
            drop_path_rate=args.drop_path_rate,
            token_dropout_p=args.token_dropout_p,
        ).to(device)
    else:
        model = GPT_models[args.model](
            attn_type=args.attn_type,
            use_adaLN=args.use_adaLN,
            token_each=args.token_each,
            code_dim=args.code_dim,
            vocab_size=args.vocab_size,
            block_size=latent_size ** 2,
            pos_type=args.pos_type,
            num_classes=args.num_classes,
            resid_dropout_p=dropout_p,
            ffn_dropout_p=dropout_p,
            drop_path_rate=args.drop_path_rate,
            token_dropout_p=args.token_dropout_p,
        ).to(device)
    # print(model)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    
    if args.resume is not None:
        resume_path = args.resume
        print("Resuming from checkpoint: {}".format(args.resume))
        resume_ckpt = torch.load(resume_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(resume_ckpt)
        ema.load_state_dict(resume_ckpt)
        
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    
    if args.mixed_precision is not None:
        print(f"Using mixed precision - {args.mixed_precision}")
    # compile the model
    if args.compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0
    model = DDP(model, device_ids=[device], find_unused_parameters=True)
    
    ############ Binary VAE ##############
    binaryae = BinaryAutoEncoder(args_ae).to(device)
    bae_code_dim = args.token_each * args.code_dim
    
    if args.deter_ae:
        bae_path = 'ckpts/bae/bae_' + str(bae_code_dim) + '_deter/binaryae_ema.th'
    else:
        bae_path = 'ckpts/bae/bae_' + str(bae_code_dim) + '/binaryae_ema.th'
    binaryae = load_pretrain(binaryae, bae_path)
    
    
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    # opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, args.beta2))
    param_groups = add_weight_decay(model.module, weight_decay=args.wd)
    opt = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(opt)
    # if args.resume is not None:
    #     opt.load_state_dict(resume_ckpt["opt"])
    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    dataset = ImageFolder(args.data_path, transform=transform)
    batch_size = int(args.global_batch_size // dist.get_world_size())
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=global_rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
        
    # Prepare models for training:
    model.train()  
    ema.eval()  # EMA model should always be in eval mode
    binaryae.eval()
    
    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    running_acc = 0
    # if args.resume is not None:
    #     resume_iter = int(args.resume.split('/')[-1].split('.')[0])
    #     train_steps = resume_iter
    #     print("We resume training from {} steps.".format(resume_iter))
        
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    train_losses = []
    m_losses = []
    m_num = []
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            if args.lr_decay:
                lr = get_lr(train_steps, args.lr, args.min_lr, args.warmup_iters, args.lr_decay_iters) if args.lr_decay else args.lr
                opt.param_groups[0]['lr'] = lr  
            model.train()
            x = x.to(device)
            y = y.to(device)
            
            with torch.no_grad():
                # Map input images to binary latent space:
                quant, binary_code = binaryae.encode(x) #binary_code: b, dim, h, w
                binary_code = rearrange(binary_code, 'b c h w -> b (h w) c') # b, 256, 64
                binary_code =  torch.stack(torch.chunk(binary_code, args.token_each, dim=-1), 2) # b, 256, 4, 16
                multipler = 2 ** (torch.arange(args.code_dim-1, -1, -1)).to(device) #tensor([32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1])
                indices = reduce(binary_code.long() * multipler,'b n m c -> b n m', 'sum') #b, 256, 4

            if args.hm_dist > 0:
                weights = get_softlabels(binary_code, args.hm_dist)
            else:
                weights = None
            with torch.cuda.amp.autocast(dtype=ptdtype):
                if args.modeling == 'mlm':
                    acc = torch.tensor([0.0])
                    if args.masksch:
                        _, loss, mask_num = model(idx=indices, cond=y, targets=indices.clone().detach().long(), train_iter_loc=train_steps, weights=weights)
                    else:
                        _, loss, mask_num = model(idx=indices, cond=y, targets=indices.clone().detach().long(), weights=weights)
                    # print(f'train step {train_steps}, loss:{loss.item()}, mask num{mask_num.item()}')
                else:
                    _, loss, acc = model(idx=indices[:, :-1, :], cond=y, targets=indices)

            opt.zero_grad()
            
            scaler.scale(loss).backward()
            if args.grad_clip != 0.0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            
            update_ema(ema, model.module._orig_mod if args.compile else model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_acc = torch.tensor(running_acc / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                dist.all_reduce(avg_acc, op=dist.ReduceOp.SUM)
                avg_acc = avg_acc.item() / dist.get_world_size()
                # train_losses.append(avg_loss)
                
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, lr: {opt.param_groups[0]['lr']}")
                # Reset monitoring variables:
                running_loss = 0
                running_acc = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps>=0:
                if global_rank == 0:
                    if args.compile:
                        model_weight = model.module._orig_mod.state_dict()
                    else:
                        model_weight = model.module.state_dict()  
                    checkpoint = {
                        "model": model_weight,
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
                    m_losses = []
                    m_num = []
                    # save_path = f"{save_dir}/{train_steps:07d}.png"
                    save_path = f"{save_dir}/{train_steps:07d}_cfg{args.cfg_scale}_t{args.temperature}.png"
                    
                    model_module = model.module._orig_mod if args.compile else model.module
                    model_without_ddp = deepcopy(model_module)
                    ### generate samples ###############
                    with torch.no_grad():
                        sample_func_autoregressive(
                            model_without_ddp, binaryae, save_path, args,
                            num_classes=args.num_classes)
                    
                    logger.info(f"Saved sampled images to {save_path}")
                    #################################
                    
                dist.barrier()
            train_steps += 1
    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    if global_rank == 0:
        save_path = f"{save_dir}/{train_steps:07d}.png"
        model_without_ddp = deepcopy(model.module)
        logger.info(f"Saved sampled images to {save_path}")
    dist.barrier()
                    
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--modeling", type=str, default="mlm", choices=['mlm', 'ar'])
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50000)
    parser.add_argument("--time-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_decay", default=False, action='store_true')
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--lr_decay_iters", type=int, default=800000)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--schedule", type=str, default='linear', choices=["squaredcos_cap_v2", "linear"])
    parser.add_argument("--dataset", type=str, required=True)
    
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--smoothing", type=float, default=0.0)
    parser.add_argument("--resume", type=str, default=None)
    
    
    ### GPT hparams
    parser.add_argument("--vocab-size", type=int, default=65536, help="vocabulary size of visual tokenizer")
    parser.add_argument("--token-each", type=int, default=4, help="number of tokens on each position")
    parser.add_argument("--code-dim", type=int, default=16, help="binary code dimension")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path-rate", type=float, default=0.0, help="using stochastic depth decay")
    parser.add_argument("--hm-dist", type=int, default=1, help="vocabulary size of visual tokenizer")

    
    parser.add_argument("--deter-ae", action="store_true")
    parser.add_argument("--pos_type", type=str, default="rope2d")
    parser.add_argument("--maskpersample", action="store_true")
    parser.add_argument("--postnorm", action="store_true")
    parser.add_argument("--norm_type", type=str, default="RMS")
    parser.add_argument("--mask_schstep", type=int, default=10000,help="")
    parser.add_argument("--mask_schtype", type=str, default='cos',help="")
    parser.add_argument("--masksch", action="store_true")
    parser.add_argument("--cfg_schedule", type=str, default='constant')
    parser.add_argument("--scale_pow", type=float, default=1.)
    parser.add_argument("--remask", action="store_true")
    
    parser.add_argument("--top-k", type=int, default=10,help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--extra_info", type=str, default="")
    parser.add_argument("--gen_iter_num", type=int, default=10, help="binary code dimension")
    
    parser.add_argument("--use_adaLN", action='store_true')
    parser.add_argument("--attn_type", type=str, default="sdp")
    
    args_ae = get_vqgan_hparams(parser)
    args = parser.parse_args()
    main(args, args_ae)
