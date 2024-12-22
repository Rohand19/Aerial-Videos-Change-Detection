import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import os

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# Import custom modules
from models import GeneratorUNet_CBAM, Discriminator
from video_processing import VideoDataset
from test import main as test_main
from create_video import create_comparison_video

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_model(model_path, model_type='generator'):
    """
    Load the generator or discriminator model
    """
    device = get_device()
    
    if model_type == 'generator':
        model = GeneratorUNet_CBAM(in_channels=3).to(device)
    else:
        model = Discriminator().to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def analyze_change_detection_metrics(val_dataloader, generator, discriminator):
    """
    Compute detailed metrics for change detection
    """
    device = get_device()
    criterion_GAN = torch.nn.MSELoss().to(device)
    criterion_pixelwise = torch.nn.L1Loss().to(device)
    lambda_pixel = 100
    
    metrics = {
        'frame_names': [],
        'loss_G': [],
        'loss_pixel': [],
        'loss_GAN': [],
        'pixel_difference': []
    }
    
    with torch.no_grad():
        for batch in val_dataloader:
            img_A = batch["A"].to(device)
            img_B = batch["B"].to(device)
            name = batch["NAME"][0]
            
            valid = torch.ones((img_A.size(0), 1, 16, 16), device=device)
            
            gener_output = generator(img_A, img_B)
            gener_output_pred = discriminator(gener_output, img_A)
            
            loss_GAN = criterion_GAN(gener_output_pred, valid)
            loss_pixel = criterion_pixelwise(gener_output, img_A)
            loss_G = loss_GAN + lambda_pixel * loss_pixel
            
            pixel_diff = torch.mean(torch.abs(gener_output - img_A)).item()
            
            metrics['frame_names'].append(name)
            metrics['loss_G'].append(loss_G.item())
            metrics['loss_pixel'].append(loss_pixel.item())
            metrics['loss_GAN'].append(loss_GAN.item())
            metrics['pixel_difference'].append(pixel_diff)
    
    return pd.DataFrame(metrics)

def main():
    st.set_page_config(page_title="Change Detection Analysis", layout="wide")
    st.title("Change Detection Model Metrics & Visualization")
    
    # Sidebar Configuration
    st.sidebar.header("Model & Data Configuration")
    
    # Video Input
    past_video = st.sidebar.file_uploader("Upload Past Video", type=['mp4', 'avi'])
    present_video = st.sidebar.file_uploader("Upload Present Video", type=['mp4', 'avi'])
    
    # Model Configuration
    save_name = st.sidebar.selectbox("Dataset", ["levir", "other_datasets"])
    frame_interval = st.sidebar.slider("Frame Extraction Interval", 1, 10, 1)
    
    # Analysis Options
    st.sidebar.header("Analysis Options")
    show_metrics = st.sidebar.checkbox("Show Metrics Table")
    show_plots = st.sidebar.checkbox("Show Metrics Plots")
    # generate_video = st.sidebar.checkbox("Generate Comparison Video")
    
    if st.sidebar.button("Run Analysis"):
        if past_video and present_video:
            # Temporary file handling
            with open(os.path.join("/tmp", "past_video.mp4"), "wb") as f:
                f.write(past_video.getbuffer())
            with open(os.path.join("/tmp", "present_video.mp4"), "wb") as f:
                f.write(present_video.getbuffer())
            
            # Configure paths
            root_path = "/tmp/change_detection"
            os.makedirs(root_path, exist_ok=True)
            
            # Extract Frames
            st.write("Extracting Video Frames...")
            from video_processing import extract_frames_from_two_videos
            extract_frames_from_two_videos(
                "/tmp/past_video.mp4", 
                "/tmp/present_video.mp4", 
                root_path, 
                frame_interval
            )
            
            # Prepare Transformations
            transforms_ = A.Compose([
                A.Resize(256, 256),
                A.Normalize(), 
                ToTensorV2()
            ])
            
            # Create Dataset
            val_dataset = VideoDataset(root_path, transforms=transforms_)
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset, batch_size=1, shuffle=False
            )
            
            # Load Models
            generator = load_model(f"saved_models/{save_name}/generator_9.pth")
            discriminator = load_model(f"saved_models/{save_name}/discriminator_9.pth", 'discriminator')
            
            # Analyze Metrics
            metrics_df = analyze_change_detection_metrics(val_dataloader, generator, discriminator)
            
            # Display Results
            if show_metrics:
                st.subheader("Metrics Table")
                st.dataframe(metrics_df)
            
            if show_plots:
                st.subheader("Metrics Visualization")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots()
                    metrics_df.boxplot(column=['loss_G', 'loss_pixel', 'loss_GAN'])
                    st.pyplot(fig)
                
                with col2:
                    fig, ax = plt.subplots()
                    sns.histplot(metrics_df['pixel_difference'], kde=True)
                    st.pyplot(fig)
            
            # if generate_video:
            #     st.write("Generating Comparison Video...")
            #     video_output_path = os.path.join(root_path, "change_detection.mp4")
            #     create_comparison_video(root_path, video_output_path)
                
            #     with open(video_output_path, "rb") as video_file:
            #         st.video(video_file.read())
        
        else:
            st.error("Please upload both past and present videos")

if __name__ == "__main__":
    main()