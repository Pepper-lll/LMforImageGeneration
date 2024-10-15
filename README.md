# LMforImageGeneration
This is the codebase for the paper

**Elucidating the design space of language models for image generation**

![image](https://github.com/Pepper-lll/LMforImageGeneration/blob/main/first_pic.png)

We explore the design space of using language models on the image generation task, including the image tokenizer choice (*Binary Autoencoder* or *Vector-Quantization Autoencoder*), language modeling method (*AutoRegressive* or *Masked Language Model*), vocabulary design based on BAE and sampling strategies. We achieve a strong baseline (1.54 FID on ImageNet 256$\times$256)  compared to language-model-based and diffusion-model-based image generation models. We also analyze the learning behavior of language models on image generation, demonstrating the great potential of AR models across different domains.
