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
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from einops import rearrange, reduce, pack, unpack
# from llama.generate import generate
import cv2


def find_model(model_name):
    """
    Finds a pre-trained DiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """

    assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["model"]
    return checkpoint

def sample_func_autoregressive(model, bae, save, args, seed=0, image_size=256, num_classes=1000):
    # Setup PyTorch:
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device

    # if ckpt is None:
    #     assert model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
    #     assert image_size in [256, 512]
    #     assert num_classes == 1000

    # # Load model:
    latent_size = image_size // 16
    
    model.eval()  # important!
    bae.eval()
    
    # Labels to condition the model with (feel free to change):
    if num_classes == 1000:
        class_labels = [207, 360, 387, 974, 88, 979, 417, 4]
    elif num_classes == 1:
        class_labels = [0, 0, 0, 0, 0, 0, 0, 0]
    elif num_classes == 3:
        class_labels = [0, 1, 2, 0, 1, 2, 0, 2]
        

    # Create sampling noise:
    n = len(class_labels)
    y = torch.tensor(class_labels, device=device)
    
    bs = y.shape[0]
    # Setup classifier-free guidance:
    indices, logits = model.generate_cfg(idx=None, cond=y, num_iter=args.gen_iter_num,
                    temperature=args.temperature, top_k=args.top_k, cfg_scale=args.cfg_scale, remask=args.remask,
                    cfg_schedule=args.cfg_schedule, scale_pow=args.scale_pow) # bs, 16*16, 4
    # breakpoint()
    device = indices.device
    all_codes = torch.arange(int(2 ** args.code_dim))  #tensor([    0,     1,     2,  ..., 65533, 65534, 65535])
    multipler = 2 ** (torch.arange(args.code_dim-1, -1, -1)) #tensor([32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1])
    codebook = ((all_codes[..., None] & multipler) != 0).float().to(device) # 2**16, 16
    # breakpoint()
    codes = codebook[indices].reshape(bs, latent_size ** 2, -1) #8, 256, 64
    # breakpoint()
    codes = rearrange(codes, 'b l c -> b c l')
    # breakpoint()
    codes = codes.reshape(bs, -1, latent_size, latent_size)
    # breakpoint()
    samples = bae.decode(codes, is_bin=True)
    
    # samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    # samples = vae.decode(samples / 0.18215).sample
    
    # Save and display images:
    save_image(samples, save, nrow=4, normalize=True, value_range=(0, 1))
    
    del model, bae

def blending(x, sample, blend_width=128, v_expand=False):
    if v_expand:
        img1 = np.rot90(x[:, :, :])
        img2 = np.rot90(sample[:, :, :])
    else:
        img1 = x[:, :, :]
        img2 = sample[:, :, :]
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    blend_width = blend_width
    overlap_region1 = img1[:, w1-blend_width:w1]
    overlap_region2 = img2[:, :blend_width]
    alpha = np.linspace(0, 1, blend_width).reshape(1, -1, 1)
    alpha = np.tile(alpha, (h1, 1, 3))

    blended_region = overlap_region1 * (1-alpha) +overlap_region2 * alpha
    # breakpoint()
    result = np.hstack((img1[:, :w1-blend_width], blended_region, img2[:, blend_width:]))
    if v_expand:
        result = np.rot90(result, -1)
    return result

def main(args, args_ae):
    # Setup PyTorch:
    torch.set_grad_enabled(False)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device

    assert args.ckpt
    
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
    
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    print("successfully load checkpoint!!!")
    
    binaryae = BinaryAutoEncoder(args_ae).to(device)
    # breakpoint()
    bae_code_dim = args.token_each * args.code_dim
    # breakpoint()
    print("bae code dimension:", bae_code_dim)
    
    if args.deter_ae:
        bae_path = 'ckpts/bae/bae_' + str(bae_code_dim) + '_deter/binaryae_ema.th'
    else:
        bae_path = 'ckpts/bae/bae_' + str(bae_code_dim) + '/binaryae_ema.th'
    
    binaryae = load_pretrain(binaryae, bae_path)
    
    sample_index = args.ckpt.split('/')[-1].split('.')[0]
    paths = args.ckpt.split('/')
    model.eval()  # important!
    binaryae.eval()
    os.makedirs(args.save_dir, exist_ok=True) 
    if args.v_expand or args.h_expand:
        if args.image_path is None:
            for cur_num in range(args.gen_num):
                bs = 1
                y = torch.from_numpy(np.array([args.class_labels])).long().to(device) if args.class_labels is not None else None
                indices, logits = model.generate_cfg(idx=None, cond=y, num_iter=args.gen_iter_num,
                                                    temperature=args.temperature, top_k=args.top_k, cfg_scale=args.cfg_scale, remask=args.remask,
                                                    cfg_schedule=args.cfg_schedule, scale_pow=args.scale_pow)
                all_codes = torch.arange(int(2 ** args.code_dim))  #tensor([    0,     1,     2,  ..., 65533, 65534, 65535])
                multipler = 2 ** (torch.arange(args.code_dim-1, -1, -1)) #tensor([32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1])
                codebook = ((all_codes[..., None] & multipler) != 0).float().to(device) #8, 256, 4, 16
                
                codes = codebook[indices].reshape(bs, latent_size ** 2, -1) #8, 256, 64
                codes = rearrange(codes, 'b l c -> b c l')
                codes = codes.reshape(bs, -1, latent_size, latent_size)
                dec0 = binaryae.decode(codes, is_bin=True).squeeze(0)
                image0 = torch.clamp(255 * dec0, 0, 255).permute(1, 2, 0).to("cpu", dtype=torch.uint8).numpy() #h, w, 3
                images = []
                images.append(image0)
                for _ in range(args.expand_time):
                    if args.v_expand:
                        indices = indices[:, -args.overlap_width:, :]
                        bs = 1
                        # breakpoint()
                        with torch.no_grad():
                            gen_indices, logits = model.generate_cfg(idx=indices, cond=y, num_iter=args.gen_iter_num-args.overlap_width, remask=args.remask, cfg_schedule=args.cfg_schedule,
                                        temperature=args.temperature, top_k=args.top_k, cfg_scale=args.cfg_scale) # bs, 16*16, 4
                    else:
                        bs = 1
                        split_l = int(args.overlap_width / 16)
                        indices = indices.reshape(bs, latent_size, latent_size, -1)
                        indices = indices[:, :, -split_l:, :]
                        
                        for r in range(indices.shape[1]):
                            if r == 0:
                                cur_idx = indices[:, r, :, :]
                            else:
                                cur_idx = torch.cat((gen_indices, indices[:, r, :, :]), dim=1)
                            
                            with torch.no_grad():
                                gen_indices, logits = model.generate_cfg(idx=cur_idx, cond=y, num_iter=latent_size-split_l, remask=args.remask, cfg_schedule=args.cfg_schedule,
                                        temperature=args.temperature, top_k=args.top_k, cfg_scale=args.cfg_scale) # bs, 16*16, 4

                    all_codes = torch.arange(int(2 ** args.code_dim))  #tensor([    0,     1,     2,  ..., 65533, 65534, 65535])
                    multipler = 2 ** (torch.arange(args.code_dim-1, -1, -1)) #tensor([32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1])
                    codebook = ((all_codes[..., None] & multipler) != 0).float().to(device) #8, 256, 4, 16
                   
                    codes = codebook[gen_indices].reshape(bs, latent_size ** 2, -1) #8, 256, 64
                    indices = gen_indices
                    codes = rearrange(codes, 'b l c -> b c l')
                    
                    codes = codes.reshape(bs, -1, latent_size, latent_size)
                    dec = binaryae.decode(codes, is_bin=True).squeeze(0)
                    sample = dec
                    images.append(torch.clamp(255 * sample, 0, 255).permute(1, 2, 0).to("cpu", dtype=torch.uint8).numpy())
                
                blend_images = []
                
                for num in range(args.expand_time):
                    if num == 0:
                        result_image = blending(images[0], images[num+1], args.overlap_width, args.v_expand)
                        # ref_image = result_image[:-256, :-256, :]
                        ref = result_image
                    else:
                        result_image = blending(ref, images[num+1], args.overlap_width, args.v_expand)
                        ref = result_image
                    
                    blend_images.append(result_image)
                if args.v_expand:
                    # result = np.concatenate(result_images, axis=0)
                    image_name = str(args.class_labels)+'vert_'+str(args.overlap_width)
                else:
                    # result = np.concatenate(result_images, axis=1)
                    image_name = str(args.class_labels)+'hori_'+str(args.overlap_width)
                Image.fromarray(result_image.astype(np.uint8)).save(f"{args.save_dir}/{image_name}_{cur_num}_cfg{args.cfg_scale}_top{args.top_k}{args.extra_info}.png")
                print(f'finish generating {cur_num} images')
                print(f"final image size: {result_image.shape}")
     
            
        if args.image_path is not None:
            v_expand = False
            h_expand = False
            ori_img = Image.open(args.image_path)
            img = transforms.ToTensor()(ori_img).to(device)
            _, h, w = img.shape
            if h > w:
                v_expand = True
            else:
                h_expand = True
            if v_expand:
                ratio = args.image_size / w
                img = transforms.Resize([int(h*ratio), args.image_size])(img)
            elif h_expand:
                ratio = args.image_size / h
                img = transforms.Resize([args.image_size, int(w*ratio) ])(img)
            
            x = img[:, :args.image_size, :args.image_size].unsqueeze(0)
            
            quant, binary_code = binaryae.encode(x) #binary_code: b, dim, h, w
            binary_code = rearrange(binary_code, 'b c h w -> b (h w) c') # b, 256, 64
            binary_code =  torch.stack(torch.chunk(binary_code, args.token_each, dim=-1), 2) # b, 256, 4, 16 
            multipler = 2 ** (torch.arange(args.code_dim-1, -1, -1)).to(device) #tensor([32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1])
            indices = reduce(binary_code.long() * multipler,'b n m c -> b n m', 'sum') #b, 256, token_each
            
            y = torch.from_numpy(np.array([args.class_labels])).long().to(device) if args.class_labels is not None else None
            
            latent_size = args.image_size // 16
            
            if v_expand:
                indices = indices[:, args.overlap_width:, :]
                bs = 1
                with torch.no_grad():
                    gen_indices, logits = model.generate_cfg(idx=indices, cond=y, num_iter=args.gen_iter_num-128, remask=args.remask, cfg_schedule=args.cfg_schedule,
                                temperature=args.temperature, top_k=args.top_k, cfg_scale=args.cfg_scale) # bs, 16*16, 4
                
            elif h_expand:
                bs = 1
                split_l = int(args.overlap_width / 16)
                indices = indices.reshape(bs, latent_size, latent_size, -1)
                indices = indices[:, :, split_l:, :]
                
                for r in range(indices.shape[1]):
                    if r == 0:
                        cur_idx = indices[:, r, :, :]
                    else:
                        cur_idx = torch.cat((gen_indices, indices[:, r, :, :]), dim=1)
                    with torch.no_grad():
                        gen_indices, logits = model.generate_cfg(idx=cur_idx, cond=y, num_iter=split_l, remask=args.remask, cfg_schedule=args.cfg_schedule,
                                temperature=args.temperature, top_k=args.top_k, cfg_scale=args.cfg_scale) # bs, 16*16, 4
                    
            all_codes = torch.arange(int(2 ** args.code_dim))  #tensor([    0,     1,     2,  ..., 65533, 65534, 65535])
            multipler = 2 ** (torch.arange(args.code_dim-1, -1, -1)) #tensor([32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1])
            codebook = ((all_codes[..., None] & multipler) != 0).float().to(device) #8, 256, 4, 16
            codes = codebook[gen_indices].reshape(bs, latent_size ** 2, -1) #8, 256, 64
            codes = rearrange(codes, 'b l c -> b c l')
            codes = codes.reshape(bs, -1, latent_size, latent_size)
            sample = binaryae.decode(codes, is_bin=True)
            
            if v_expand:
                sample = torch.cat((img[:, :args.overlap_width, :], sample.squeeze(0)[:, :, :]), dim=1)
                args.extra_info += '_vert'
                
            elif h_expand:
                sample = torch.cat((img[:, :, :args.overlap_width], sample.squeeze(0)[:, :, :]), dim=2)
                args.extra_info += '_hori'
            
            image_name = args.image_path.split('/')[-1].split('.')[0]    
            sample = torch.clamp(255 * sample, 0, 255).permute(1, 2, 0).to("cpu", dtype=torch.uint8).numpy()
            Image.fromarray(sample).save(f"{args.save_dir}/{image_name}_expand{args.extra_info}.png")
            
            
            if args.class_labels is None:
                args.extra_info += '_ncls'
            else:
                args.extra_info += f'_{str(args.class_labels)}'
                args.extra_info += f'_{str(args.cfg_scale)}'
                args.extra_info += f'_{str(args.top_k)}'
                args.extra_info += f'_{str(args.token_each)}_{str(args.code_dim)}'
            
            x = torch.clamp(255 * x.squeeze(0), 0, 255).permute(1, 2, 0).to("cpu", dtype=torch.uint8).numpy()
            Image.fromarray(x).save(f"{args.save_dir}/{image_name}_oripart.png")
            
            ### blending 
            if v_expand:
                img1 = np.rot90(x[:148, :, :])
                img2 = np.rot90(sample[128:, :, :])
            else:
                img1 = x[:, :148, :]
                img2 = sample[:, 128:, :]
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            blend_width = 20
            overlap_region1 = img1[:, w1-blend_width:w1]
            overlap_region2 = img2[:, :blend_width]
            alpha = np.linspace(0, 1, blend_width).reshape(1, -1, 1)
            alpha = np.tile(alpha, (h1, 1, 3))

            blended_region = overlap_region1 * (1-alpha) +overlap_region2 * alpha
            
            result = np.hstack((img1[:, :w1-blend_width], blended_region, img2[:, blend_width:]))
            if v_expand:
                result = np.rot90(result, -1)
            
            Image.fromarray(result.astype(np.uint8)).save(f"{args.save_dir}/{image_name}_blend{args.extra_info}.png")

    else:
        for cur_num in range(args.gen_num):
            save_path = args.save_dir + '/' + f"cfg{args.cfg_scale}{args.cfg_schedule}{args.scale_pow}_top{args.top_k}_t{args.temperature}_g{args.gen_iter_num}_{sample_index}_{cur_num}.png"
            print("Saving to {}".format(save_path))
            sample_func_autoregressive(
                model, binaryae, save_path, args,
                seed=args.seed+cur_num, image_size=args.image_size, num_classes=args.num_classes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeling", type=str, default="mlm", choices=['mlm', 'ar'])
    parser.add_argument("--model", type=str, default="GPT-L")
    parser.add_argument("--device", type=str, default="cuda")
    
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--time-steps", type=int, default=500)
    parser.add_argument("--save-path", type=str, default="samples")
    parser.add_argument("--schedule", type=str, default='linear', choices=["squaredcos_cap_v2", "linear"])
    parser.add_argument("--dataset", type=str, required=True)
    
    ### sample config
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--top-k", type=int, default=10,help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    
    ### GPT hparams
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--token-each", type=int, default=4, help="number of tokens on each position")
    parser.add_argument("--code-dim", type=int, default=16, help="binary code dimension")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path-rate", type=float, default=0.0, help="using stochastic depth decay")
    parser.add_argument("--pos_type", type=str, default="rope")

    parser.add_argument("--deter-ae", action="store_true")
        
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--gen_iter_num", type=int, default=10, help="binary code dimension")
    parser.add_argument("--remask", action="store_true")
    parser.add_argument("--cfg_schedule", type=str, default='constant')
    parser.add_argument("--scale_pow", type=float, default=1.)
    
    parser.add_argument("--save_dir", type=str, default="pics")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--v_expand", action="store_true")
    parser.add_argument("--h_expand", action="store_true")
    parser.add_argument("--class_labels", type=int, default=None)
    parser.add_argument("--blend_width", type=int, default=128)
    parser.add_argument("--overlap_width", type=int, default=128)
    parser.add_argument("--expand_time", type=int, default=2)
    parser.add_argument("--gen_num", type=int, default=2)
    
    parser.add_argument("--extra_info", type=str, default="")
    
    
    
    args_ae = get_vqgan_hparams(parser)
    args = parser.parse_args()
    main(args, args_ae)
