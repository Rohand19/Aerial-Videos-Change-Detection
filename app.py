import streamlit as st
import os
import torch
from models import GeneratorUNet_CBAM, Discriminator
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from video_processing import extract_frames_from_two_videos, VideoDataset
from torch.utils.data import DataLoader
import tempfile
from create_video import create_comparison_video  # Assuming this can be adapted or we'll create a new one
from torchvision.utils import save_image
import shutil
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="AI Video Change Detection with Pixel Visual",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (same as before)
st.markdown("""
    <style>
        .stApp {
            background-color: #f5f7f9;
        }
        .main > div {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3rem;
            background-color: #FF4B4B;
            color: white;
        }
        .stButton>button:hover {
            background-color: #FF3333;
        }
        .upload-section {
            background-color: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .results-section {
            margin-top: 2rem;
            background-color: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .info-box {
            background-color: #e1f5fe;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def pixel_visual_streamlit(gener_output_, img_B_):  # Modified for Streamlit context
    gener_output = gener_output_.cpu().clone().detach().squeeze()
    img_B = img_B_.cpu().clone().detach().squeeze()

    pixel_loss_tensor = torch.abs(gener_output - img_B)
    pixel_loss_pil = to_pil_image(pixel_loss_tensor)
    trans = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()])
    pixel_loss = trans(pixel_loss_pil)

    thre_num = 0.7
    threshold = torch.nn.Threshold(thre_num, 0.)
    pixel_loss_thresholded = threshold(pixel_loss)
    return pixel_loss_thresholded.cpu() # Return CPU tensor for saving/display


def create_comparison_video_with_pixel_loss(results_base_dir, output_video_path, fps=0.5):
    frame_dirs = sorted([d for d in os.listdir(results_base_dir) if os.path.isdir(os.path.join(results_base_dir, d)) and d.startswith('frame_')])
    if not frame_dirs:
        raise Exception(f"No frame directories found in {results_base_dir}")

    frame_size = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (0, 0)) # Initialize without size

    for frame_dir_name in frame_dirs:
        frame_dir = os.path.join(results_base_dir, frame_dir_name)
        input_t1_path = os.path.join(frame_dir, 'input_t1.png')
        input_t2_path = os.path.join(frame_dir, 'input_t2.png')
        change_mask_path = os.path.join(frame_dir, 'change_mask.png')
        pixel_loss_path = os.path.join(frame_dir, 'pixel_loss.png') # Pixel loss image path

        if not all(os.path.exists(path) for path in [input_t1_path, input_t2_path, change_mask_path, pixel_loss_path]): # Include pixel_loss
            st.error(f"Missing images in {frame_dir}. Skipping frame.") # Streamlit error for debugging
            continue

        img_t1 = cv2.imread(input_t1_path)
        img_t2 = cv2.imread(input_t2_path)
        change_mask = cv2.imread(change_mask_path)
        pixel_loss_img = cv2.imread(pixel_loss_path) # Load pixel loss image

        if img_t1 is None or img_t2 is None or change_mask is None or pixel_loss_img is None: # Check pixel_loss
             st.error(f"Error loading images in {frame_dir}. Skipping frame.") # Streamlit error for debugging
             continue

        h1, w1, _ = img_t1.shape
        h2, w2, _ = img_t2.shape
        hc, wc, _ = change_mask.shape
        hp, wp, _ = pixel_loss_img.shape # Pixel loss shape

        if frame_size is None:
            frame_size = (w1 + w2 + wc + wp, max(h1, h2, hc, hp)) # Adjust width for pixel loss
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size) # Initialize with correct size


        combined_frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        combined_frame[:h1, :w1, :] = img_t1
        combined_frame[:h2, w1:w1+w2, :] = img_t2
        combined_frame[:hc, w1+w2:w1+w2+wc, :] = change_mask
        combined_frame[:hp, w1+w2+wc:w1+w2+wc+wp, :] = pixel_loss_img # Add pixel loss

        video_writer.write(combined_frame)

    video_writer.release()
    if not os.path.exists(output_video_path):
        raise Exception(f"Video file was not created at {output_video_path}")


def process_videos(video_path_past, video_path_present, progress_bar, status_area):
    with tempfile.TemporaryDirectory() as temp_dir:
        progress_bar.progress(0.1)
        with status_area:
            st.info("📥 Extracting frames from videos...")

        try:
            num_frames = extract_frames_from_two_videos(
                video_path_past,
                video_path_present,
                temp_dir,
                frame_interval=1
            )
            with status_area:
                st.success(f"✅ Successfully extracted {num_frames} frame pairs")
        except Exception as e:
            with status_area:
                st.error(f"❌ Error during frame extraction: {str(e)}")
            return None

        progress_bar.progress(0.3)
        with status_area:
            st.info("🔧 Initializing AI model...")

        device = get_device()
        with status_area:
            st.success(f"💻 Using device: {device}")

        generator = GeneratorUNet_CBAM(in_channels=3).to(device)

        try:
            generator.load_state_dict(torch.load(
                "saved_models/levir/generator_9.pth",
                map_location=device
            ))
            generator.eval()
        except Exception as e:
            with status_area:
                st.error(f"❌ Error loading model weights: {str(e)}")
            return None

        progress_bar.progress(0.5)
        with status_area:
            st.info("🔄 Processing frames...")

        transforms_ = A.Compose([
            A.Resize(256, 256),
            A.Normalize(),
            ToTensorV2()
        ])

        val_dataset = VideoDataset(temp_dir, transforms=transforms_)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )

        results_base_dir = os.path.join(temp_dir, 'results')
        os.makedirs(results_base_dir, exist_ok=True)

        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                img_A = batch["A"].to(device)
                img_B = batch["B"].to(device)
                name = batch["NAME"]

                gener_output = generator(img_A, img_B)
                pixel_loss_output = pixel_visual_streamlit(gener_output, img_B) # Get pixel loss tensor

                frame_dir = os.path.join(results_base_dir, f"frame_{i:06d}")
                os.makedirs(frame_dir, exist_ok=True)

                save_image(img_A.cpu(), os.path.join(frame_dir, 'input_t1.png'), normalize=True)
                save_image(img_B.cpu(), os.path.join(frame_dir, 'input_t2.png'), normalize=True)
                save_image(gener_output.cpu(), os.path.join(frame_dir, 'change_mask.png'), normalize=True)
                save_image(pixel_loss_output, os.path.join(frame_dir, 'pixel_loss.png')) # Save pixel loss

                progress_bar.progress(0.5 + (0.4 * (i + 1) / len(val_dataloader)))

        with status_area:
            st.info("🎬 Creating final visualization with pixel loss...")
        output_video_path = os.path.join(temp_dir, 'output.mp4')

        try:
            create_comparison_video_with_pixel_loss(results_base_dir, output_video_path, fps=0.5) # Use new video creation

            with open(output_video_path, 'rb') as f:
                video_bytes = f.read()

            progress_bar.progress(1.0)
            return video_bytes

        except Exception as e:
            with status_area:
                st.error(f"❌ Error creating video with pixel loss: {str(e)}")
                st.write("📁 Debug: Contents of results directory:")
                for root, dirs, files in os.walk(results_base_dir):
                    st.write(f"Directory: {root}")
                    st.write(f"Files: {files}")
            return None


import cv2 # Import OpenCV here for video writing

def main():
    # Sidebar (same as before)
    with st.sidebar:
        st.image("/Users/rohandivakar/Desktop/CDRL/datasets/compressed_b7b2f052a98f687d5cfbe0108b35734a.webp", width=150)
        st.title("About")
        st.markdown("""
        This AI-powered tool detects and visualizes changes between two videos
        of the same location taken at different times.

        ### Applications
        - 🏗️ Construction monitoring
        - 🌳 Environmental change detection
        - 🏙️ Urban development tracking
        - 🔍 Security surveillance
        """)

        st.markdown("---")
        st.markdown("### Technical Details")
        with st.expander("Model Architecture"):
            st.markdown("""
            - Based on U-Net architecture
            - Enhanced with CBAM attention
            - Trained on LEVIR-CD dataset
            """)

    # Main content (same as before)
    st.title("🎥 Video Change Detection AI with Pixel Visual")
    st.markdown("""
    Upload two videos of the same location taken at different times to detect and visualize changes.
    Our AI model will analyze the differences and generate a detailed change detection visualization including Pixel Loss.
    """)

    # Upload section (same as before)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("📤 Past Video")
        video_past = st.file_uploader("Upload video from earlier time", type=['mp4', 'avi', 'mov'])
        if video_past:
            st.success("✅ Past video uploaded")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("📤 Present Video")
        video_present = st.file_uploader("Upload recent video", type=['mp4', 'avi', 'mov'])
        if video_present:
            st.success("✅ Present video uploaded")
        st.markdown('</div>', unsafe_allow_html=True)

    # Guidelines (same as before)
    with st.expander("📋 Guidelines for Best Results"):
        st.markdown("""
        1. **Video Requirements**
           - Same scene/location
           - Similar duration
           - Stable camera position
           - Good lighting conditions

        2. **Supported Formats**
           - MP4
           - AVI
           - MOV

        3. **Processing Time**
           - Depends on video length
           - Typically 2-5 minutes
           - Please be patient during processing
        """)

    # Process button (same as before)
    if st.button('🚀 Generate Change Detection with Pixel Visual', disabled=not (video_past and video_present)):
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        st.subheader("🔄 Processing Status")

        progress_bar = st.progress(0)
        status_area = st.empty()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f1:
            f1.write(video_past.read())
            video_path_past = f1.name

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f2:
            f2.write(video_present.read())
            video_path_present = f2.name

        try:
            video_bytes = process_videos(video_path_past, video_path_present, progress_bar, status_area)

            if video_bytes:
                st.success("✨ Change detection with pixel visual completed successfully!")
                st.subheader("🎦 Results Visualization")
                st.video(video_bytes)

                st.markdown("""
            ### 📊 Visualization Guide
            - **Bright regions**: No Significant changes detected
            - **Dark areas**: Significant changes detected
            - **Brightness**: Indicates change intensity
            """)

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.warning("Please check that your videos are valid and try again.")

        finally:
            os.unlink(video_path_past)
            os.unlink(video_path_present)

        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
