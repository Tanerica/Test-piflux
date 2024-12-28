import argparse
import torch
from diffusers import FluxPipeline
import time
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Choose option for torch compile')
    parser.add_argument('--model_name_or_path', type=str, default='black-forest-labs/FLUX.1-schnell')
    parser.add_argument('--steps', default=8, type=int, help='Number of inference steps')
    parser.add_argument('--height', default=1024, type=int, help='Height size of sample images')
    parser.add_argument('--width', default=1024, type=int, help='Width of sample images')
    parser.add_argument('--seed', default=62, type=int, help='Randome seed for reproduct result')
    parser.add_argument('--output_dir', default='./normal_output', type=str, help='Output of generated images')
    args = parser.parse_args()

    pipe = FluxPipeline.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16).to('cuda')
    pipe.transformer = torch.compile(pipe.transformer, options={"triton.cudagraphs": True}, fullgraph=True)
    pipe.text_encoder = torch.compile(pipe.text_encoder, options={"triton.cudagraphs": True}, fullgraph=True)
    pipe.text_encoder_2 = torch.compile(pipe.text_encoder_2, options={"triton.cudagraphs": True}, fullgraph=True)
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, options={"triton.cudagraphs": True}, fullgraph=True)
    memory = (torch.cuda.memory_allocated() / (1024 ** 3)) 
    print(f"Model size in GPU: {memory=:.3f} GB")
    torch.cuda.reset_peak_memory_stats()
    # warm-up
    dum_prompt = "A cat holding a sign that says hello world"
    num_warmup = 4
    for i in range(num_warmup):
        torch.cuda.synchronize()
        start_time = time.time()
        dum_out = pipe(
        prompt=dum_prompt,
        guidance_scale=0.,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        max_sequence_length=256,
    ).images[0]
        torch.cuda.synchronize()
        end_time = time.time()
        print(f'Time warm up {i}: ', {round(end_time - start_time, 3)})

    
    from create_prompts import list_prompts
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for i in range(len(list_prompts)):
        torch.cuda.synchronize()
        start_time = time.time()
        out = pipe(
            prompt=list_prompts[i],
            guidance_scale=0.,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            max_sequence_length=256,
            generator = torch.Generator('cuda').manual_seed(args.seed)
        ).images[0]
        torch.cuda.synchronize()
        end_time = time.time()
        #print(f"Length of prompt {i}: ", len(prompts[i].split(" ")))
        out.save(os.path.join(args.output_dir, f'{i}.png'))
        print(f'Time compute prompt {i}: ', {round(end_time - start_time, 3)})
    print('Max mem allocated: ', torch.cuda.max_memory_allocated() / (1024 ** 3))