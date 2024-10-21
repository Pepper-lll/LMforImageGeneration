

#################################################### cat token L 2-10 #####################################################
TORCH_DISTRIBUTED_DEBUG=INFO torchrun --nnodes=1 --nproc_per_node=8 sample_fid.py \
 --model GPT-L --modeling ar --sample_dir images_fid_ar \
 --ckpt ckpts/gpt/L-2-10.pth \
 --dataset custom --codebook_size 20 --norm_first --num-classes 1000 \
 --code-dim 10 --token-each 2 --pos_type rope2d --cfg-scale 4.0 --cfg_schedule linear \
 --latent_shape 1 16 16 --deterministic --top-k 1000 --gen_iter_num 256 --temperature 1.0 \
 --num_images 32000 --per_proc_batch_size 4 --global_seed 64

python -m pytorch_fid /app/ms/AIGC/haoshaozhe/fid_test/pytorch_fid/imagenet/train.npz images_fid_ar/L-2-10-g256-linearcfg-4.0-t1.0-g256-top1000 --num-workers 4
