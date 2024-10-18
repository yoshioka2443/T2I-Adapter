from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
from controlnet_aux.midas import MidasDetector
import torch

# load adapter
adapter = T2IAdapter.from_pretrained(
  "TencentARC/t2i-adapter-depth-midas-sdxl-1.0", torch_dtype=torch.float16, varient="fp16"
).to("cuda")

# load euler_a scheduler
model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
    model_id, vae=vae, adapter=adapter, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16", 
).to("cuda")
pipe.enable_xformers_memory_efficient_attention()

midas_depth = MidasDetector.from_pretrained(
  "valhalla/t2iadapter-aux-models", filename="dpt_large_384.pt", model_type="dpt_large"
).to("cuda")

url = "https://huggingface.co/Adapter/t2iadapter/resolve/main/figs_SDXLV1.0/org_mid.jpg"
image = load_image(url)

# output
path = "/workspace/HOIDiffusion/output_test5/depth/002_master_chef_can_0_0_4.png"
image = load_image(path)
image = midas_depth(
  image, detect_resolution=512, image_resolution=1024
)

# prompt = "A photo of a room, 4k photo, highly detailed"
prompt = "a hand grasping a master chef can, A photo of a room, 4k photo, highly detailed"
negative_prompt = "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"

gen_images = pipe(
  prompt=prompt,
  negative_prompt=negative_prompt,
  image=image,
  num_inference_steps=30,
  adapter_conditioning_scale=1,
  guidance_scale=7.5,  
).images[0]
# gen_images.save('output_image/out_mid.png')
gen_images.save('output_image/can_depth.png')