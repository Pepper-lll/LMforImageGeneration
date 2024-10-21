

CUDA_VISIBLE_DEVICES="7" python sample.py --modeling ar --model GPT-XXL \
 --ckpt ckpts/gpt/XXL-2-12.pth \
 --save_dir expanding \
 --dataset custom --codebook_size 24 --norm_first \
 --code-dim 12 --token-each 2 --pos_type rope2d --cfg-scale 2.0 --cfg_schedule constant \
 --latent_shape 1 16 16 --top-k 400 --gen_iter_num 256 --temperature 1.0 --gen_num 4 \
 --class_labels 985 --h_expand --overlap_width 112 --gen_num 20 --expand_time 4 \