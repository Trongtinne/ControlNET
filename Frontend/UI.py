import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import torch
import einops
import base64

from utils.data import HWC3, apply_color, resize_image
from utils.ddim import DDIMSampler
from utils.model import create_model, load_state_dict
from pytorch_lightning import seed_everything

# Hàm chuyển ảnh sang base64
def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Lấy dữ liệu base64 của ảnh nền
bg_image = get_base64_of_image(r"D:\Diffusion\Colorization-for-grayscale-images\image.jpg")

# CSS tùy chỉnh
st.markdown(f"""
    <style>
        .stApp {{
            background: url("data:image/jpg;base64,{bg_image}") no-repeat center center fixed;
            background-size: cover;
        }}

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

        /* Đổi màu chữ của st.info() thành xanh lá */
        .stAlert {{
            color: #66BB6A !important;  /* Xanh lá đậm */
            font-weight: bold !important;
        }}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = create_model('./Colorization-for-grayscale-images/models/cldm_v21.yaml').cpu()
    model.load_state_dict(load_state_dict(
        'lightning_logs/version_6/checkpoints/colorizenet-sd21.ckpt', location='cuda'))
    model = model.cuda()
    return model, DDIMSampler(model)

model, ddim_sampler = load_model()

# Hàm inferencing ảnh
def colorize_image(input_image):
    input_image = np.array(input_image.convert("RGB"))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    input_image = HWC3(input_image)
    img = resize_image(input_image, resolution=512)
    H, W, C = img.shape

    num_samples = 1
    control = torch.from_numpy(img.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    seed = 1294574436
    seed_everything(seed)
    prompt = "Colorize this image"
    n_prompt = ""
    guess_mode = False
    strength = 1.0
    eta = 0.0
    ddim_steps = 20
    scale = 9.0

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt] * num_samples)]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
    shape = (4, H // 8, W // 8)

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
    samples, _ = ddim_sampler.sample(ddim_steps, num_samples, shape, cond, verbose=False, eta=eta,
                                     unconditional_guidance_scale=scale, unconditional_conditioning=un_cond)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c')
                 * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

    result = apply_color(img, x_samples[0])
    return result

# Giao diện
cols = st.columns([1, 1], gap="large")

# Cột 1: Upload ảnh
with cols[0]:
    st.info("📷 Ảnh Gốc (B&W)", icon="ℹ️")
    uploaded_file = st.file_uploader("Chọn ảnh đen trắng", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Ảnh tải lên", use_column_width=True)

# Cột 2: Hiển thị ảnh đã tô màu
with cols[1]:
    st.info("🎨 Ảnh Sau Khi Tô Màu", icon="ℹ️")

    if uploaded_file is not None:
        with st.spinner("🔄 Đang tô màu..."):
            colored_image = colorize_image(image)
            st.image(colored_image, caption="🌈 Ảnh sau khi tô màu", use_column_width=True)


# import sys
# import os

# sys.path.append(os.path.abspath(os.path.dirname(__file__)))
