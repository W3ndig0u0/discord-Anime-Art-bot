import discord
from PIL import Image
import os
import time
import math
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPTextConfig
from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL


class AnimeArtist:
    def __init__(self):
        self.progress = 0
        self.total_steps = 0
        self.generation_complete = False
        self.estimated_time = None
        self.generator = None
        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    def load_generator(self, model_name, vae_name, cache_dir):
        self.generator = load_modelDiff(model_name, vae_name, cache_dir, self.device)

    def image_grid(self, imgs, rows, cols):
        w, h = imgs[0].size
        max_size = 512

        if w > max_size:
            w = max_size
            h = int((max_size / imgs[0].width) * imgs[0].height)

        if h > max_size:
            h = max_size
            w = int((max_size / imgs[0].height) * imgs[0].width)

        grid = Image.new("RGB", size=(cols * w, rows * h))
        for i, img in enumerate(imgs):
            img = img.resize((w, h), Image.ANTIALIAS)
            grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid

    def generate_art(
        self,
        input_prompt,
        height,
        width,
        num_inference_steps,
        eta,
        negative_prompt,
        guidance_scale,
        save_folder,
        seed,
        batch_size,
        model_name,
        vae_name,
        initial_generation=False,
    ):
        cache_dir = "./"
        if initial_generation or self.generator is None:
            self.progress = 0

        self.load_generator(model_name, vae_name, cache_dir)

        self.total_steps = batch_size
        self.generation_complete = False
        self.estimated_time = None

        print(model_name)
        print(self.device)

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.generator.enable_model_cpu_offload()

        self.generator.enable_attention_slicing()

        with torch.no_grad():
            current_images = [
                Image.new("RGB", (width, height)) for _ in range(batch_size)
            ]

            os.makedirs(save_folder, exist_ok=True)
            existing_files = os.listdir(save_folder)
            existing_numbers = [
                int(file_name.split("_")[-1].split(".")[0])
                for file_name in existing_files
                if file_name.endswith(".png") and file_name.split(".")[0].isdigit()
            ]
            last_file_number = max(existing_numbers) if existing_numbers else 0

            randomSeeds = [
                torch.Generator(self.device).manual_seed(seed)
                if seed != -1 and step == 0
                else torch.Generator(self.device).manual_seed(
                    torch.randint(0, 2**32, (1,)).item()
                )
                for step in range(batch_size)
            ]

            start_time = time.time()
            file_number = last_file_number
            for step in range(batch_size):
                generated = self.generator(
                    prompt=input_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    eta=eta,
                    negative_prompt=negative_prompt,
                    generator=randomSeeds[step],
                )

                current_images[step] = generated.images[0]

                file_number += 1

                file_path = os.path.join(save_folder, f"{file_number}.png")
                current_images[step].save(file_path)

                self.progress = step + 1

                elapsed_time = time.time() - start_time
                if step > 0:
                    average_time_per_step = elapsed_time / step
                    remaining_steps = num_inference_steps - step
                    estimated_time = average_time_per_step * remaining_steps
                    self.estimated_time = round(estimated_time, 2)

            self.generation_complete = True

        if batch_size > 1:
            file_number += 1
            grid_size = math.ceil(math.sqrt(batch_size))
            generated_images = self.image_grid(current_images, grid_size, grid_size)
            save_path = os.path.join(save_folder, f"{file_number}.png")
            generated_images.save(save_path)

        final_file_number = file_number

        return save_folder, final_file_number


def load_modelDiff(model_name, vae_name, cache_dir, device):
    var_cache_dir = os.path.join("./")
    safe_cache_dir = os.path.join("./")
    vae = AutoencoderKL.from_pretrained(
        vae_name, torch_dtype=torch.float16, cache_dir=var_cache_dir
    )

    print("Using " + model_name + " NOW!")
    if model_name.endswith(".safetensors") or model_name.endswith(".ckpt"):
        model = StableDiffusionPipeline.from_ckpt(
            model_name,
            cache_dir=safe_cache_dir,
            torch_dtype=torch.float16,
            vae=vae,
            local_files_only=True,
        )
    else:
        model = StableDiffusionPipeline.from_pretrained(
            model_name,
            cache_dir=safe_cache_dir,
            torch_dtype=torch.float16,
            vae=vae,
            local_files_only=True,
        )

    print("Using " + model_name + " NOW!")
    model = StableDiffusionPipeline.from_pretrained(
        model_name,
        cache_dir=safe_cache_dir,
        torch_dtype=torch.float16,
        vae=vae,
        local_files_only=True,
        use_safetensors=True,
    )

    model = model.to(device)
    if device == "cuda":
        model.enable_xformers_memory_efficient_attention()
    elif device == "mps":
        torch.backends.mps.enable_xformers_memory_efficient_attention()

    return model
