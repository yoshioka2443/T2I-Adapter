from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
from controlnet_aux import OpenposeDetector
import torch
import numpy as np
from PIL import Image

# load adapter
adapter = T2IAdapter.from_pretrained(
  "TencentARC/t2i-adapter-openpose-sdxl-1.0", torch_dtype=torch.float16
).to("cuda")

# load euler_a scheduler
model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
    model_id, vae=vae, adapter=adapter, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16", 
).to("cuda")
pipe.enable_xformers_memory_efficient_attention()

open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")

# url = "https://huggingface.co/Adapter/t2iadapter/resolve/main/people.jpg"
# image = load_image(url)
path = "/workspace/HOIDiffusion/output_test5/skeleton/002_master_chef_can_0_0_4.jpg"
image = load_image(path)
image = open_pose(image, detect_resolution=512, image_resolution=1024)
image = np.array(image)[:, :, ::-1]           
image = Image.fromarray(np.uint8(image))

# prompt = "A couple, 4k photo, highly detailed, in the forest"
# prompt = "A couple, 4k photo, highly detailed"
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
# gen_images.save('output_image/forest_couple.png')
gen_images.save('output_image/can_skele1.png')


