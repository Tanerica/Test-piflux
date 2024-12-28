# Enable debugging: prints each command before executing it
set -x

# Add the current directory to PYTHONPATH
export PYTHONPATH="$PWD:$PYTHONPATH"

mkdir -p ./results

# task args
TASK_ARGS="--height 1024 --width 1024 --no_use_resolution_binning"

# On 8 gpus, pp=2, ulysses=2, ring=1, cfg_parallel=2 (split batch)
N_GPUS=2
#PARALLEL_ARGS="--pipefusion_parallel_degree 2 --ulysses_degree 2 --ring_degree 2"
PARALLEL_ARGS="--pipefusion_parallel_degree 1 --ulysses_degree 2 --ring_degree 1"
# CFG_ARGS="--use_cfg_parallel"

# By default, num_pipeline_patch = pipefusion_degree, and you can tune this parameter to achieve optimal performance.
# PIPEFUSION_ARGS="--num_pipeline_patch 8 "

# For high-resolution images, we use the latent output type to avoid runing the vae module. Used for measuring speed.
# OUTPUT_ARGS="--output_type latent"

# PARALLLEL_VAE="--use_parallel_vae"

# Another compile option is `--use_onediff` which will use onediff's compiler.
COMPILE_FLAG="--use_torch_compile"


# Use this flag to quantize the T5 text encoder, which could reduce the memory usage and have no effect on the result quality.
# QUANTIZE_FLAG="--use_fp8_t5_encoder"

# export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --nproc_per_node=$N_GPUS ./examples/flux_usp_example.py \
--model "black-forest-labs/FLUX.1-schnell" \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps 8 \
--warmup_steps 4 \
--prompt "photograph of a woman with mermaid waves hair, dark-blonde hair, light blue eyes, dressed in unzipped dark blue hoodie, showing off a bit of cleavage, pink shorts, textured fabric, resorts, palm trees, clear sky, standing on a balcony railing of a high rise resort hotel, facing the viewer" \
--seed 62 \
$CFG_ARGS \
$PARALLLEL_VAE \
$COMPILE_FLAG \
$QUANTIZE_FLAG \
