from importlib.metadata import distribution
import os
from pyrsistent import l
import torch
from tqdm import tqdm

from .log_utils import save_latents, log
from models import TransformerBD, BinaryDiffusion
import pdb
import numpy as np 
import misc
import time

def get_sampler(H, embedding_weight):

    if H.sampler == 'bld':
        denoise_fn = TransformerBD(H).cuda()
        print(denoise_fn)
        sampler = BinaryDiffusion(
            H, denoise_fn, H.codebook_size, embedding_weight)
    else:
        raise NotImplementedError

    return sampler


@torch.no_grad()
def get_samples_temp(H, generator, sampler, x=None, ee=False):

    if x is None:
        latents_all = []

        sampler.eval()
        print('Sampling')
        t0 = time.time()
        if ee:
            for t in np.linspace(0.2, 1.0, num=5):
                for f in [False, True]:
                    latents = sampler.sample(sample_steps=H.sample_steps, temp=t, full=f)
                    latents = latents[:10]
                    latents_all.append(latents)
        else:
            for t in np.linspace(0.2, 1.0, num=10):
                latents = sampler.sample(sample_steps=H.sample_steps, temp=t, full=f)
                latents = latents[:10]
                latents_all.append(latents)
        latents = torch.cat(latents_all, dim=0)
        sampler.train()

        print('Sampling done at %.1fs' %((time.time() - t0)))
    else:
        latents = x

    with torch.cuda.amp.autocast():
        size = min(5, latents.shape[0])
        images = []
        for i in range(len(latents)//size):
            latent = latents[i*size : (i+1)*size]

            latent = (latent * 1.0) 

            if H.use_tanh:
                latent = (latent - 0.5) * 2.0
            if not H.norm_first:
                latent = latent / float(H.codebook_size)
            latent = latent @ sampler.embedding_weight

            latent = latent.permute(0,2,1)
            latent = latent.reshape(*latent.shape[:-1], H.latent_shape[1], H.latent_shape[2])
            img = generator(latent.float())
            images.append(img)
        images = torch.cat(images, 0)

    return images

@torch.no_grad()
def get_samples_test(H, generator, sampler, x=None, t=1.0, n_samples=20, return_all=False, label=None, mask=None, guidance=None):
    generator.eval()
    sampler.eval()
    latents = sampler.sample(sample_steps=H.sample_steps, temp=t, b=n_samples, return_all=return_all, label=label, mask=mask, guidance=guidance)
    

    if mask is not None:
        latents = torch.cat([mask['latent'].unsqueeze(0), latents], 0)

    with torch.cuda.amp.autocast():
        size = min(25, latents.shape[0])
        if H.latent_shape[-1] == 32:
            size = 5
        images = []
        for i in range(len(latents)//size):
            latent = latents[i*size : (i+1)*size]

            latent = (latent * 1.0) 

            if H.use_tanh:
                latent = (latent - 0.5) * 2.0
            # latents = latents / (latents.sum(dim=-1, keepdim=True)+1e-6)
            if not H.norm_first:
                latent = latent / float(H.codebook_size)
            latent = latent @ sampler.embedding_weight

            latent = latent.permute(0,2,1)
            latent = latent.reshape(*latent.shape[:-1], H.latent_shape[1], H.latent_shape[2])
            img = generator(latent.float())
            images.append(img)
        images = torch.cat(images, 0)

    return images

@torch.no_grad()
def get_samples_guidance(H, generator, sampler, x=None):

    if x is None:
        latents_all = []

        sampler.eval()
        print('Sampling')
        for g in [None, 0.1, 0.5, 1.0, 2.0, 5.0]:
            for t in [0.5, 0.9]:
                latents = sampler.sample(sample_steps=H.sample_steps, temp=t, guidance=g)
                
                latents = latents[:10]
                latents_all.append(latents)
        latents = torch.cat(latents_all, dim=0)
        sampler.train()

    else:
        latents = x

    print('Sampling done')

    with torch.cuda.amp.autocast():
        size = min(5, latents.shape[0])
        images = []
        for i in range(len(latents)//size):
            latent = latents[i*size : (i+1)*size]

            latent = (latent * 1.0) 

            if H.use_tanh:
                latent = (latent - 0.5) * 2.0
            # latents = latents / (latents.sum(dim=-1, keepdim=True)+1e-6)
            if not H.norm_first:
                latent = latent / float(H.codebook_size)
            latent = latent @ sampler.embedding_weight

            latent = latent.permute(0,2,1)
            latent = latent.reshape(*latent.shape[:-1], H.latent_shape[1], H.latent_shape[2])
            img = generator(latent.float())
            images.append(img)
        images = torch.cat(images, 0)

    return images

@torch.no_grad()
def get_t2i_samples_guidance(H, generator, sampler, label, x=None,):
    
    if isinstance(label, list):
        batch_size = label[0].shape[0]
    else:
        batch_size = label.shape[0]

    if x is None:
        latents_all = []

        sampler.eval()
        print('Sampling')
        for g in [None, 0.1, 0.5, 1.0, 3.0, 10.0]:
            for t in [0.6, 1.0]:
                latents = sampler.sample(sample_steps=H.sample_steps, b=batch_size, temp=t, label=label, guidance=g)
                
                latents_all.append(latents)
        latents = torch.cat(latents_all, dim=0)
        sampler.train()
    else:
        latents = x

    print('Sampling done')

    with torch.cuda.amp.autocast():
        size = min(5, latents.shape[0])
        images = []
        for i in range(len(latents)//size):
            latent = latents[i*size : (i+1)*size]

            latent = (latent * 1.0) 

            if H.use_tanh:
                latent = (latent - 0.5) * 2.0
            if not H.norm_first:
                latent = latent / float(H.codebook_size)
            latent = latent.permute(0,2,1)
            latent = latent.reshape(*latent.shape[:-1], H.latent_shape[1], H.latent_shape[2])
            img, _, _ = generator(None, code=latent.float())
            images.append(img)
        images = torch.cat(images, 0)

    return images

@torch.no_grad()
def get_online_samples(H, generator, sampler, x=None,):

    if x is None:
        latents_all = []

        sampler.eval()

        print('Sampling')
        for t in np.linspace(0.55, 1.0, num=10):
            latents = sampler.sample(sample_steps=H.sample_steps, temp=t)
            latents_all.append(latents)
        latents = torch.cat(latents_all, dim=0)
        sampler.train()

    else:
        latents = x

    print('Sampling done')

    with torch.cuda.amp.autocast():
        # latents_one_hot = latent_ids_to_onehot(latents, H.latent_shape, H.codebook_size)
        size = min(5, latents.shape[0])
        images = []
        for i in range(len(latents)//size):
            latent = latents[i*size : (i+1)*size]

            latent = (latent * 1.0) 

            if H.use_tanh:
                latent = (latent - 0.5) * 2.0

            if not H.norm_first:
                latent = latent / float(H.codebook_size)

            latent = latent.permute(0,2,1)
            latent = latent.reshape(*latent.shape[:-1], H.latent_shape[1], H.latent_shape[2])
            img, _, _ = generator(None, code=latent.float())
            images.append(img)
        images = torch.cat(images, 0)

    return images

@torch.no_grad()
def get_t2i_samples_guidance_test(H, generator, sampler, label, x=None, g=None, t=1.0, return_latent=False):

    if isinstance(label, list):
        batch_size = label[0].shape[0]
    else:
        batch_size = label.shape[0]

    if x is None:
        latents_all = []
        sampler.eval()
        print('Sampling')
        t0 = time.time()
        latents = sampler.sample(sample_steps=H.sample_steps, b=batch_size, temp=t, label=label, guidance=g)
        
        latents_all.append(latents)
        latents = torch.cat(latents_all, dim=0)

        print('Sampling done at %.1fs' %((time.time() - t0)))
        sampler.train()
    else:
        latents = x


    with torch.cuda.amp.autocast():
        # latents_one_hot = latent_ids_to_onehot(latents, H.latent_shape, H.codebook_size)
        size = min(5, latents.shape[0])
        images = []
        for i in range(len(latents)//size):
            latent = latents[i*size : (i+1)*size]

            latent = (latent * 1.0) 

            if H.use_tanh:
                latent = (latent - 0.5) * 2.0
            if not H.norm_first:
                latent = latent / float(H.codebook_size)

            latent = latent.permute(0,2,1)
            latent = latent.reshape(*latent.shape[:-1], H.latent_shape[1], H.latent_shape[2])
            img, _, _ = generator(None, code=latent.float())
            images.append(img)
        images = torch.cat(images, 0)

    if return_latent:
        return images, latents
    else:
        return images

@torch.no_grad()
def get_online_samples_guidance(H, generator, sampler, x=None,):

    if x is None:
        latents_all = []

        sampler.eval()

        print('Sampling')
        for g in [None, 1.0, 2.0, 5.0, 10.0]:
            for t in [0.5, 0.9]:
                latents = sampler.sample(sample_steps=H.sample_steps, temp=t, guidance=g)
                latents_all.append(latents)
                # print('done')
        latents = torch.cat(latents_all, dim=0)
        sampler.train()
    else:
        latents = x

    print('Sampling done')

    with torch.cuda.amp.autocast():
        size = min(5, latents.shape[0])
        images = []
        for i in range(len(latents)//size):
            latent = latents[i*size : (i+1)*size]

            latent = (latent * 1.0) 

            if H.use_tanh:
                latent = (latent - 0.5) * 2.0
            if not H.norm_first:
                latent = latent / float(H.codebook_size)
            latent = latent.permute(0,2,1)
            latent = latent.reshape(*latent.shape[:-1], H.latent_shape[1], H.latent_shape[2])
            img, _, _ = generator(None, code=latent.float())
            images.append(img)
        images = torch.cat(images, 0)

    return images

def get_samples_idx(H, generator, sampler, idx):

    latents_one_hot = latent_ids_to_onehot(idx, H.latent_shape, H.codebook_size)
    q = sampler.embed(latents_one_hot)
    images = generator(q.float())

    return images


def latent_ids_to_onehot(latent_ids, latent_shape, codebook_size):
    min_encoding_indices = latent_ids.view(-1).unsqueeze(1)
    encodings = torch.zeros(
        min_encoding_indices.shape[0],
        codebook_size
    ).to(latent_ids.device)
    encodings.scatter_(1, min_encoding_indices, 1)
    one_hot = encodings.view(
        latent_ids.shape[0],
        latent_shape[1],
        latent_shape[2],
        codebook_size
    )
    return one_hot.reshape(one_hot.shape[0], -1, codebook_size)



# TODO: rethink this whole thing - completely unnecessarily complicated
def retrieve_autoencoder_components_state_dicts(H, components_list, remove_component_from_key=False):
    state_dict = {}
    # default to loading ema models first
    ae_load_path = f"{H.ae_load_dir}/saved_models/binaryae_ema_{H.ae_load_step}.th"
    if not os.path.exists(ae_load_path):
        ae_load_path = f"{H.ae_load_dir}/saved_models/binaryae_{H.ae_load_step}.th"
    log(f"Loading Binary Autoencoder from {ae_load_path}")
    full_vqgan_state_dict = torch.load(ae_load_path, map_location="cpu")

    for key in full_vqgan_state_dict:
        for component in components_list:
            if component in key:
                new_key = key[3:]  # remove "ae."
                if remove_component_from_key:
                    new_key = new_key[len(component)+1:]  # e.g. remove "quantize."

                state_dict[new_key] = full_vqgan_state_dict[key]
    del full_vqgan_state_dict
    return state_dict
