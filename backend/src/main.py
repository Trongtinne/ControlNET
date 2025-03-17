import os
import sys
import numpy as np
import torch
import einops
import cv2
import random
from typing import Optional
from PIL import Image
import io
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# Add the current directory to path to ensure imports work
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import model-related utilities
from pytorch_lightning import seed_everything
from utils.data import HWC3, apply_color, resize_image
from utils.ddim import DDIMSampler
from utils.model import create_model, load_state_dict

# Create FastAPI app
app = FastAPI(title="Image Colorization API",
              description="API for colorizing grayscale images using Stable Diffusion")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a class for colorization parameters
class ColorizationParams(BaseModel):
    prompt: str = "Colorize this image"
    a_prompt: str = "best quality, natural colors"
    n_prompt: str = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    num_samples: int = 1
    image_resolution: int = 512
    ddim_steps: int = 20
    guess_mode: bool = False
    strength: float = 1.0
    scale: float = 9.0
    seed: int = -1
    eta: float = 0.0

# Function to load the model
def load_model():
    model = create_model('backend/src/models/cldm_v21.yaml').cpu()
    model.load_state_dict(load_state_dict(
        r'backend/src/checkpoints/epoch=2-step=44357.ckpt', location='cuda'))
    model = model.cuda()
    return model, DDIMSampler(model)

# Function to process and colorize images
def process(input_image, params):
    with torch.no_grad():
        # Convert to numpy array if it's a PIL image
        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image.convert("RGB"))
            
        input_image = HWC3(input_image)
        img = resize_image(input_image, params.image_resolution)
        H, W, C = img.shape

        control = torch.from_numpy(img.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(params.num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if params.seed == -1:
            params.seed = random.randint(0, 65535)
        seed_everything(params.seed)

        model, ddim_sampler = load_model()

        cond = {"c_concat": [control], "c_crossattn": [
            model.get_learned_conditioning([params.prompt + ', ' + params.a_prompt] * params.num_samples)]}
        un_cond = {"c_concat": None if params.guess_mode else [control], "c_crossattn": [
            model.get_learned_conditioning([params.n_prompt] * params.num_samples)]}
        shape = (4, H // 8, W // 8)

        model.control_scales = [params.strength * (0.825 ** float(12 - i)) for i in range(13)] if params.guess_mode else (
            [params.strength] * 13)
        samples, _ = ddim_sampler.sample(params.ddim_steps, params.num_samples,
                                      shape, cond, verbose=False, eta=params.eta,
                                      unconditional_guidance_scale=params.scale,
                                      unconditional_conditioning=un_cond)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c')
                     * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(params.num_samples)]
        colored_results = [apply_color(img, result) for result in results]
        
        return [img] + colored_results

# Define FastAPI routes
@app.post("/api/colorize")
async def colorize_api(file: UploadFile = File(...), 
                       prompt: str = Form("Colorize this image"),
                       a_prompt: str = Form("best quality, natural colors"),
                       n_prompt: str = Form("longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"),
                       num_samples: int = Form(1),
                       image_resolution: int = Form(512),
                       ddim_steps: int = Form(20),
                       guess_mode: bool = Form(False),
                       strength: float = Form(1.0),
                       scale: float = Form(9.0),
                       seed: int = Form(-1),
                       eta: float = Form(0.0)):
    
    # Read image from request
    contents = await file.read()
    try:
        input_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    
    # Create parameters object
    params = ColorizationParams(
        prompt=prompt,
        a_prompt=a_prompt,
        n_prompt=n_prompt,
        num_samples=num_samples,
        image_resolution=image_resolution,
        ddim_steps=ddim_steps,
        guess_mode=guess_mode,
        strength=strength,
        scale=scale,
        seed=seed,
        eta=eta
    )
    
    # Process the image
    try:
        results = process(input_image, params)
        # Return the colored image (second item in results)
        if len(results) > 1:
            colored_image = results[1]
            # Ensure the image is in RGB format
            colored_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB)
            img_bytes = cv2.imencode('.png', colored_image)[1].tobytes()
            return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")
        else:
            raise HTTPException(status_code=500, detail="Failed to colorize image")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during colorization: {str(e)}")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Image Colorization API. Use /api/colorize to colorize images."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)