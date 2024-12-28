N_GPUS=2
torchrun --nproc_per_node=$N_GPUS run_flux.py --output './image_out.png'