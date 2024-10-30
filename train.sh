###gpt-L 2-10
TORCH_DISTRIBUTED_DEBUG=INFO TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 NCCL_DEBUG=INFO torchrun --nnodes=1 --nproc_per_node=8 train_ddp.py \
 --model GPT-L --modeling ar --results-dir results-ar --token-each 2 --code-dim 10 --hm-dist 0 \
 --mixed-precision bf16 --compile --smoothing 0.0 \
 --img_size 256 --latent_shape 1 16 16 --codebook_size 20 --norm_first --dataset custom \
 --data-path /your/path/to/ImageNet1k --num-classes 1000 \
 --global-batch-size 256 --ckpt-every 10000 --log-every 100 --epochs 400 \
 --lr 1e-4 \
 --cfg-scale 2.0 --temperature 1.0 --gen_iter_num 256 \
 --wd 0.05 --beta2 0.95 --pos_type 'rope2d' \
