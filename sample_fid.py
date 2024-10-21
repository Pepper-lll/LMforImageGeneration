

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
import argparse
import os
from glob import glob
from hparams import get_vqgan_hparams
from bae.binaryae import BinaryGAN, BinaryAutoEncoder, load_pretrain
# from llama.gpt_mlm_new import GPT_models
# from llama.gpt_mlm_cattoken import GPT_models

from einops import rearrange, reduce, pack, unpack

from time import time
import torch.distributed as dist
from tqdm import tqdm
from PIL import Image

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def sample_func_autoregressive(model, bae, args, class_labels, seed=0, image_size=256):
    # Setup PyTorch:
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    latent_size = image_size // 16
    
    model.eval()  # important!
    bae.eval()
    
    # Create sampling noise:
    n = len(class_labels)
    y = class_labels.to(device)
    
    bs = y.shape[0]
    # Setup classifier-free guidance:
    indices, logits = model.generate_cfg(idx=None, cond=y, num_iter=args.gen_iter_num, remask=args.remask, cfg_schedule=args.cfg_schedule,
                    temperature=args.temperature, top_k=args.top_k, cfg_scale=args.cfg_scale) # bs, 16*16, 4
    
    device = indices.device
    all_codes = torch.arange(int(2 ** args.code_dim))  #tensor([    0,     1,     2,  ..., 65533, 65534, 65535])
    multipler = 2 ** (torch.arange(args.code_dim-1, -1, -1)) #tensor([32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1])
    codebook = ((all_codes[..., None] & multipler) != 0).float().to(device) #8, 256, 4, 16
    
    codes = codebook[indices].reshape(bs, latent_size ** 2, -1) #8, 256, 64
    codes = rearrange(codes, 'b l c -> b c l')
    codes = codes.reshape(bs, -1, latent_size, latent_size)
    samples = bae.decode(codes, is_bin=True)
    
    samples = torch.clamp(255 * samples, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    
    # del model, bae
    return samples

def main(args, args_ae):
    """
    Run sampling.
    """
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    assert args.ckpt
    # print('******')
    
    #### GPT model ####
    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p
    
    latent_size = args.image_size // 16
    if args.modeling == 'mlm':
        from llama.mlm_model import GPT_models
    else:
        from llama.ar_model import GPT_models
    
    model = GPT_models[args.model](
        attn_type=args.attn_type,
        use_adaLN=args.use_adaLN,
        pos_type=args.pos_type,
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
    print("finish model builiding")
    
    binaryae = BinaryAutoEncoder(args_ae).to(device)
    bae_code_dim = args.token_each * args.code_dim
    print("bae code dimension:", bae_code_dim)
    
    if args.deter_ae:
        bae_path = 'ckpts/bae/bae_' + str(bae_code_dim) + '_deter/binaryae_ema.th'
    else:
        bae_path = 'ckpts/bae/bae_' + str(bae_code_dim) + '/binaryae_ema.th'
    
    binaryae = load_pretrain(binaryae, bae_path)
    ####################
    
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    
    model.load_state_dict(checkpoint)
        

    # Create folder to save samples:
    model_string_name = args.ckpt.split('/')[-1].split('.')[0]
    folder_name = f"{model_string_name}-{args.cfg_schedule}cfg-{args.cfg_scale}-t{args.temperature}-g{args.gen_iter_num}-top{args.top_k}"
    
    if not args.deterministic:
        folder_name += '-nd'
    if args.remask:
        folder_name += '-remask'
    sample_folder_dir = f"{args.sample_dir}/{folder_name}{args.extra_info}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Generate samples for model in {args.ckpt}, ***cfg={args.cfg_scale}, ***temperature={args.temperature}, ***gen iter num={args.gen_iter_num}")
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    # total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    total_samples = args.num_images
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    # assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    # assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"

    # Setup PyTorch:
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()  # important!
    binaryae.eval()
    
    # Labels to condition the model with (feel free to change):
    class_list_all = list(range(args.num_classes))
    
    # Create sampling noise:
    total = 0
    num_per_cls = int(args.num_images // args.num_classes)
    print('number of samples per class:', num_per_cls)
    cur_cls = 0
    for iter_loc in tqdm(range(int(args.num_images // global_batch_size))): ### int(args.num_images // global_batch_size) == classes_num // world_size
        
        # y = torch.randint(0, args.num_classes, (n,), device=device)
        y = (torch.ones((n,)) * cur_cls).long().to(device)
        bs = y.shape[0]
        # Setup classifier-free guidance:
        with torch.no_grad():
            samples = sample_func_autoregressive(model, binaryae, args, y, seed + iter_loc * dist.get_world_size())
            # print('y:', y, 'sample:', samples.shape)
        
        for i, sample in enumerate(samples):
            # label = label_i * dist.get_world_size() + rank
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/class{cur_cls}_{index:06d}_{seed:04d}.png")
        total += global_batch_size
        cur_cls = int(total/num_per_cls)
    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="GPT-L")
    parser.add_argument("--modeling", type=str, default="mlm", choices=['mlm', 'ar'])
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--save-path", type=str, default="samples")
    parser.add_argument("--schedule", type=str, default='linear', choices=["squaredcos_cap_v2", "linear"])
    parser.add_argument("--dataset", type=str, required=True)
    
    ### sample config
    parser.add_argument("--cfg-scale", type=float, default=10.0)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=2000,help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--gen_iter_num", type=int, default=10, help="binary code dimension")
    parser.add_argument("--pos_type", type=str, default="rope2d")
    
    ### GPT hparams
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--token-each", type=int, default=4, help="number of tokens on each position")
    parser.add_argument("--code-dim", type=int, default=16, help="binary code dimension")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.0, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path-rate", type=float, default=0.0, help="using stochastic depth decay")
    parser.add_argument("--per_proc_batch_size", type=int, default=32)
    parser.add_argument("--sample_dir", type=str, default='images_fid')
    parser.add_argument("--num_images", type=int, default=50000)
    
    parser.add_argument("--deter-ae", action="store_true")
    parser.add_argument("--remask", action="store_true")
    parser.add_argument("--cfg_schedule", type=str, default='constant')
    
    parser.add_argument("--use_adaLN", action='store_true')
    parser.add_argument("--postnorm", action="store_true")
    parser.add_argument("--norm_type", type=str, default="RMS")
    parser.add_argument("--attn_type", type=str, default="sdp")
    parser.add_argument("--extra_info", type=str, default='')
    
    args_ae = get_vqgan_hparams(parser)
    args = parser.parse_args()
    main(args, args_ae)