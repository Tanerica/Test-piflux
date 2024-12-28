N_GPUS=2
torchrun --nproc_per_node=$N_GPUS run_flux_tan.py --output 'results'