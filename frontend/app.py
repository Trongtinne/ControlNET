import os
import sys
import numpy as np
import torch
import einops
import cv2
import random
import base64
from typing import Optional
from PIL import Image
import io
import streamlit as st
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn
from starlette.staticfiles import StaticFiles
from starlette.responses import StreamingResponse

# Add the current directory to path to ensure imports work
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import model-related utilities
from pytorch_lightning import seed_everything
from backend.src.models import HWC3, apply_color, resize_image
from backend.src.models import DDIMSampler
from backend.src.models import create_model, load_state_dict

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

# Function to load the model (cached for streamlit)
@st.cache_resource
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
        input_image = Image.open(io.BytesIO(contents))
        input_image = input_image.convert("RGB")
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
            img_bytes = cv2.imencode('.png', colored_image)[1].tobytes()
            return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")
        else:
            raise HTTPException(status_code=500, detail="Failed to colorize image")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during colorization: {str(e)}")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Image Colorization API. Use /api/colorize to colorize images."}

# Streamlit UI
def streamlit_ui():
    # Get base64 of background image
    def get_base64_of_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    # Try to get background image, use fallback if not found
    try:
        bg_image = get_base64_of_image(r"D:\Diffusion\Colorization-for-grayscale-images\image.jpg")
    except:
        # Fallback to no background if image not found
        bg_image = None
        
    # CSS styling with conditional background
    bg_style = f"""
        .stApp {{
            background: url("data:image/jpg;base64,{bg_image}") no-repeat center center fixed;
            background-size: cover;
        }}
    """ if bg_image else ""
    
    st.markdown(f"""
        <style>
            {bg_style}

            .block-container {{
                padding-left: 30px !important;
                padding-right: 30px !important;
                max-width: 100% !important;
            }}

            .title {{
                font-size: 50px;
                font-weight: 550;
                color: #4CAF50;
                margin-bottom: 5px;
            }}

            .image-box {{
                border: 2px solid #ddd;
                padding: 15px;
                border-radius: 10px;
                background-color: rgba(255, 255, 255, 0.8);
                box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
                text-align: center;
                width: 100%;
            }}

            .stAlert {{
                color: #66BB6A !important;
                font-weight: bold !important;
            }}
            
            .sidebar .image-selector {{
                padding: 10px;
                border-radius: 5px;
                background-color: rgba(255, 255, 255, 0.8);
                margin-bottom: 15px;
            }}
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state for storing images
    if 'input_image' not in st.session_state:
        st.session_state.input_image = None
    if 'output_image' not in st.session_state:
        st.session_state.output_image = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False

    # Title
    st.markdown('<p class="title">Image Colorization with Stable Diffusion</p>', unsafe_allow_html=True)
    
    # Create a sidebar for parameters
    with st.sidebar:
        st.markdown("## Colorization Parameters")
        prompt = st.text_input("Prompt", "Colorize this image")
        a_prompt = st.text_input("Added Prompt", "best quality, natural colors")
        
        with st.expander("Advanced Options", expanded=False):
            n_prompt = st.text_input("Negative Prompt", 
                                     "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality")
            num_samples = st.slider("Number of Samples", 1, 4, 1)
            image_resolution = st.slider("Image Resolution", 256, 768, 512, 64)
            ddim_steps = st.slider("Steps", 1, 100, 20)
            strength = st.slider("Control Strength", 0.0, 2.0, 1.0, 0.01)
            scale = st.slider("Guidance Scale", 0.1, 30.0, 9.0, 0.1)
            seed = st.slider("Seed", -1, 2147483647, -1)
            eta = st.number_input("eta (DDIM)", 0.0)
            guess_mode = st.checkbox("Guess Mode", False)
    
    # Main content area
    st.markdown("### Upload a grayscale image")
    uploaded_file = st.file_uploader("Choose a grayscale image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Store input image
        image = Image.open(uploaded_file)
        st.session_state.input_image = image
        
        # Process button
        if st.button("Colorize Image"):
            st.session_state.processing = True
            
            # Create params for processing
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
            
            with st.spinner("🔄 Colorizing..."):
                results = process(image, params)
                if len(results) > 1:
                    # Store output image
                    st.session_state.output_image = Image.fromarray(results[1])
                    st.session_state.processing = False
        
        # Display both images side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📷 Original Image (B&W)")
            if st.session_state.input_image is not None:
                st.image(st.session_state.input_image, use_column_width=True)
        
        with col2:
            st.markdown("### 🎨 Colorized Image")
            if st.session_state.output_image is not None:
                st.image(st.session_state.output_image, use_column_width=True)
                
                # Add download button for colorized image
                buf = io.BytesIO()
                st.session_state.output_image.save(buf, format="PNG")
                buf.seek(0)
                st.download_button(
                    label="Download Colorized Image",
                    data=buf,
                    file_name="colorized_image.png",
                    mime="image/png"
                )
            elif st.session_state.processing:
                st.info("Processing... please wait")
            else:
                st.info("Click 'Colorize Image' to generate a colorized version")

# Main function to run the application
def main():
    # Check if running as streamlit app or as FastAPI
    if os.environ.get("STREAMLIT_RUN"):
        streamlit_ui()
    else:
        # Mount the Streamlit app for the root path
        @app.get("/streamlit")
        async def redirect_to_streamlit():
            streamlit_command = f"streamlit run {__file__} --server.port 8501"
            return {"message": f"Streamlit UI is available at http://localhost:8501. Run: {streamlit_command}"}
        
        # Start the FastAPI server
        uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    # Set environment variable to indicate Streamlit is running
    if "streamlit" in sys.argv[0].lower():
        os.environ["STREAMLIT_RUN"] = "1"
    main()