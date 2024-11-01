This is the codebase for the paper

# Elucidating the design space of language models for image generation

[![Project Page](https://img.shields.io/badge/Webpage-0054a6?logo=Google%20chrome&logoColor=white)](https://Pepper-lll.github.io/LMforImageGeneration/)
[![arXiv](https://img.shields.io/badge/arXiv-2410.16257%20-b31b1b)](https://arxiv.org/abs/2410.16257)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ELM-blue)](https://huggingface.co/xuantonglll/ELM)
[![Colab Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1l7FRzS8HSlmSjpVJW7mOfsV2GJWtmGsR?usp=sharing)
![License](https://img.shields.io/github/license/haoosz/ConceptExpress?color=lightgray)

[Xuantong Liu](https://github.com/Pepper-lll), [Shaozhe Hao](https://haoosz.github.io/), [Xianbiao Qi*](https://scholar.google.com/citations?user=odjSydQAAAAJ&hl=en), [Tianyang Hu#](https://hu-tianyang.github.io/), [Jun Wang](https://scholar.google.com/citations?user=mX8s9ZgAAAAJ), [Rong Xiao](https://scholar.google.com/citations?user=Zb5wT08AAAAJ&hl=en), [Yuan Yao#](https://yao-lab.github.io/)\
The Hong Kong University of Science and Technology, The University of Hong Kong, Intellifusion, Huawei Noah's Ark Lab\
(*: Project leader; #: Corresponding authors)

[**[Project Page]**](https://Pepper-lll.github.io/LMforImageGeneration/) [**[arXiv]**](https://arxiv.org/abs/2410.16257) [**[Colab]**](https://colab.research.google.com/drive/1l7FRzS8HSlmSjpVJW7mOfsV2GJWtmGsR?usp=sharing)

![image](https://github.com/Pepper-lll/LMforImageGeneration/blob/main/first_pic.png)

# Introduction 💡

We explore the design space of using language models on the image generation task, including the **image tokenizer choice** (Binary Autoencoder or Vector-Quantization Autoencoder), **language modeling method** (AutoRegressive or Masked Language Model), **vocabulary design** based on BAE and sampling strategies and **sampling strategies**. We achieve a strong baseline (1.54 FID on ImageNet 256*256) compared to language-model-based and diffusion-model-based image generation models. We also analyze the **fundamental difference between image and languege sequence generation** and **the learning behavior of language models on image generation**, demonstrating the scaling law and the great potential of AR models across different domains.

We provide 4 BAE tokenizers with code dimension 16, 10, 24 and 32, each trained for 1,000,000 iterations with batch size 256. We also provide the checkpoints for all the generation models we discussed in the paper. All the download links are provided.

# Set up 🔩
You can simply install the environment with the file ```environment.yml``` by:

```
conda env create -f environment.yml
conda activate ELM
```
# Download 💡
You can download the checkpoints for the image tokenizers (BAE) and generation models from [link](https://huggingface.co/xuantonglll/ELM).

### Image Tokenizers (BAEs) 🧩

| Code Dim  | Bernoulli Sampling | Link | Size |
| ------------- | ------------- |-------------|-------------|
| 16  | ✅  | [link](https://huggingface.co/xuantonglll/ELM/resolve/main/bae/bae_16/binaryae_ema.th?download=true) | 332MB |
| 16  | ❌ | [link](https://huggingface.co/xuantonglll/ELM/resolve/main/bae/bae_16_deter/binaryae_ema.th?download=true) | 332MB|
| 20  | ✅  | [link](https://huggingface.co/xuantonglll/ELM/resolve/main/bae/bae_20/binaryae_ema.th?download=true) | 332MB |
| 24  | ✅  | [link](https://huggingface.co/xuantonglll/ELM/resolve/main/bae/bae_24/binaryae_ema.th?download=true)| 332MB |

### Generation Models (GPTs) ⚙️
| Model  | Link | Size |
| ------------- | -------------| -------------|
|AR-L |[[1-16]](https://huggingface.co/xuantonglll/ELM/resolve/main/gpt/L-1-16.pth?download=true)  [[2-8]](https://huggingface.co/xuantonglll/ELM/resolve/main/gpt/L-2-8.pth?download=true) [[2-10]](https://huggingface.co/xuantonglll/ELM/resolve/main/gpt/L-2-10.pth?download=true) [[2-12]](https://huggingface.co/xuantonglll/ELM/resolve/main/gpt/L-2-12.pth?download=true)| 1.25GB~1.77GB|
|AR-XL | [[1-16]](https://huggingface.co/xuantonglll/ELM/resolve/main/gpt/XL-1-16.pth?download=true) [[2-8]](https://huggingface.co/xuantonglll/ELM/resolve/main/gpt/XL-2-8.pth?download=true) [[2-10]](https://huggingface.co/xuantonglll/ELM/resolve/main/gpt/XL-2-10.pth?download=true)  [[2-12]](https://huggingface.co/xuantonglll/ELM/resolve/main/gpt/XL-2-12.pth?download=true) | 2.95GB~3.6GB|
|AR-XXL | [[1-16]](https://huggingface.co/xuantonglll/ELM/resolve/main/gpt/XXL-1-16.pth?download=true) [[2-10]](https://huggingface.co/xuantonglll/ELM/resolve/main/gpt/XXL-2-10.pth?download=true)  [[2-12]](https://huggingface.co/xuantonglll/ELM/resolve/main/gpt/XXL-2-12.pth?download=true) | 5.49GB~6.25GB|
|AR-2B | [[2-12]](https://huggingface.co/xuantonglll/ELM/resolve/main/gpt/2B-2-12.pth?download=true) | 7.64GB|
|MLM-L | [[1-16]](https://huggingface.co/xuantonglll/ELM/resolve/main/gpt/mlmL-1-16.pth?download=true) | 1.51GB|
|MLM-XL | [[1-16]](https://huggingface.co/xuantonglll/ELM/resolve/main/gpt/mlmXL-1-16.pth?download=true) | 3.27GB|
|MLM-XXL | [[1-16]](https://huggingface.co/xuantonglll/ELM/resolve/main/gpt/mlmXXL-1-16.pth?download=true) | 5.86GB|

# Image Generation 🌟
If you want to generate samples with our pretrained models, run
```
bash inference.sh
```
You need to specify the checkpoint path in ```--ckpt```. The default setting is generated samples from 8 classes [207, 360, 387, 974, 88, 979, 417, 279].
If you want to generated images larger than 256 $\times$ 256
, activate ```--v_expand``` (for vertical expanding) or ```--h_expand``` (for horizontal expanding) in ```inference.sh```, ```--overlap_width``` sets the length of the preceding sequence each time, ```--expand_time``` sets how many times to expand, ```--gen_num``` specify the number of generated samples.


# Train 🌟
If you want to train EML-L with vocabulary 2-10 on 1 GPU node with 8 GPUs, just run
```
bash train.sh
```
You need to specify the ImageNet dataset path at ```--data-path```. You can change the model size through ```--model``` (L, XL, XXL and 2B), modeling method through ```--modeling``` (ar or mlm), number of sub-codes through ```--token-each``` (1, 2, 3, ...), dimension of each code through ```--code-dim```. Remember the **```codebook_size``` should be equal to 
```token-each``` *  ```code-dim```**. ```--hm-dist``` larger than 1 means the soft label according to Hamming Distance is used, however, we found it is kind of useless, and we have not utilized it or discussed it in our paper. You are free to have a try!

We train L/XL-sized models using 8 A800 GPUs, XXL/2B-sized models using 32 A800 GPUs on 4 nodes.

# Additional Results 🌟
### FID without cfg
For each model size, we test the 50k-FID without cfg with the most suitable tokenizer using ```pytorch_fid```.
|Model|FID|
|---|---|
|XL, 2-10|17.95|
|XL, 2-10|13.70|
|XXL, 2-12| 11.41|

### Training loss curve
The training loss for token-prediction-based image generation can not converge  well but still ensures high image generation capability. The rationale behind this is discussed in our paper.
We show the training loss curve of the model of different sizes with the same tokenizer, where the scaling law is also presented.

![image](https://github.com/Pepper-lll/LMforImageGeneration/blob/main/losses2-12.png)

However, the training loss trend of models with different tokenizers (such as L with 1-16, 2-8, 2-10, ...) is not compared. Because different tokenizers have different vocabulary sizes, the losses are not of the same magnitude and cannot be compared.
