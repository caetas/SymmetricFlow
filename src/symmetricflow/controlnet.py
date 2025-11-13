import cv2
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import numpy as np
from diffusers.utils import load_image
import time

image = load_image("https://huggingface.co/lllyasviel/sd-controlnet-hed/blob/main/images/man.png")
image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = Image.fromarray(image)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()

# Reset CUDA peak counters so the reported peak is for this run only
if torch.cuda.is_available():
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

start = time.time()

image = pipe("", image, num_inference_steps=20).images[0]

end = time.time()
print(f"Time taken: {1000*(end - start)} milliseconds")

# Print peak VRAM allocated (MB) for this run
if torch.cuda.is_available():
    try:
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"[controlnet] peak VRAM allocated: {peak_mb:.1f} MB")
    except Exception:
        pass

image.save('images/bird_canny_out.png')
