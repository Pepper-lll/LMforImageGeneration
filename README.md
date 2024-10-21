This is the codebase for the paper

### Elucidating the design space of language models for image generation
[Xuantong Liu](https://github.com/Pepper-lll), [Shaozhe Hao](https://haoosz.github.io/), [Xianbiao Qi*](https://scholar.google.com/citations?user=odjSydQAAAAJ&hl=en), [Tianyang Hu#](https://hu-tianyang.github.io/), [Jun Wang](https://scholar.google.com/citations?user=mX8s9ZgAAAAJ), [Rong Xiao](https://scholar.google.com/citations?user=Zb5wT08AAAAJ&hl=en), [Yuan Yao#](https://yao-lab.github.io/)\
The Hong Kong University of Science and Technology, The University of Hong Kong, Intellifusion, Huawei Noah's Ark Lab\
(*: Project leader; #: Corresponding authors)

[[Project Page]](https://Pepper-lll.github.io/LMforImageGeneration/)
![image](https://github.com/Pepper-lll/LMforImageGeneration/blob/main/first_pic.png)

### Introduction 💡

We explore the design space of using language models on the image generation task, including the **image tokenizer choice** (Binary Autoencoder or Vector-Quantization Autoencoder), **language modeling method** (AutoRegressive or Masked Language Model), **vocabulary design** based on BAE and sampling strategies and **sampling strategies**. We achieve a strong baseline (1.54 FID on ImageNet 256*256) compared to language-model-based and diffusion-model-based image generation models. We also analyze the **fundamental difference between image and languege sequence generation** and **the learning behavior of language models on image generation**, demonstrating the scaling law and the great potential of AR models across different domains.

We provide 4 BAE tokenizers with code dimension 16, 10, 24 and 32, each trained for 1,000,000 iterations with batch size 256. We also provide the checkpoints for all the generation models we discussed in the paper. All the download links are provided.

### Set up 🔩
You can simply install the environment with the file ```environment.yml``` by:

```
conda env create -f environment.yml
conda activate ELM
```

### Image Generation 🌟
If you want to generated samples with our pretrained models, run
```
bash inference.sh
```
The default setting is generated samples from 8 classes.
If you want to generated images larger than 256 $\times$ 256
, activate ```--v_expand``` (for vertical expanding) or ```--h_expand``` (for horizontal expanding) in ```inference.sh```.


### Train 🌟
If you want to train EML-L with vocabulary 2-10 on 1 GPU node with 8 GPUs, just run
```
bash train.sh
```
You need to specify the ImageNet dataset path at ```--data-path```.

We train L/XL-sized models using 8 A800 GPUs, XXL/2B-sized models using 32 A800 GPUs on 4 nodes.
