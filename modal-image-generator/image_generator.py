import modal

# Create a Modal app
app = modal.App("stable-diffusion-api")

# Specify Python version and required libraries
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch==2.0.1",
    "torchvision==0.15.2",
    "diffusers==0.31.0",
    "transformers",
    "accelerate"
)

# Function to generate images
@app.function(image=image, gpu="a10g", timeout=600)
def generate_image(prompt: str) -> str:
    import torch
    from diffusers import StableDiffusionPipeline

    # Load the Stable Diffusion model
    model_id = "CompVis/stable-diffusion-v1-4"
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipeline = pipeline.to("cuda")

    # Generate an image
    image = pipeline(prompt).images[0]

    # Save the image locally
    local_path = "/tmp/generated_image.png"
    image.save(local_path)

    # Upload to Modal's cloud storage
    modal_path = modal.upload_file(local_path)
    return modal_path
