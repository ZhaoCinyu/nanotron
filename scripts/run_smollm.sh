export CUDA_DEVICE_MAX_CONNECTIONS=1 
export CUDA_VISIBLE_DEVICES=2,3

# torchrun --master_port 37254 --nproc_per_node=4 run_train.py \
#     --config-file examples/config_smollm1_135M.yaml

torchrun --nproc_per_node=1 --master_port 35645 examples/llama/convert_nanotron_to_hf.py \
 --checkpoint_path='/playpen/xinyu/checkpoints/pt_smollm/30000'\
 --save_path='/playpen/xinyu/checkpoints/pt_smollm_30000_hf'

# torchrun --nproc_per_node=1 --master_port 37223 run_generate.py \
#     --ckpt-path /playpen/xinyu/checkpoints/pt_smollm/30000 --tp 1 --pp 1
