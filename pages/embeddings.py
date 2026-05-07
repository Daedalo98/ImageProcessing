import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import os
import urllib.request
from hilbertcurve.hilbertcurve import HilbertCurve
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

st.set_page_config(page_title="Embeddings & Decoder", page_icon="🧬", layout="wide")

# ==========================================
# 1. SETUP & MEDIAPIPE EMBEDDER
# ==========================================
MODEL_PATH = 'mobilenet_v3_large.tflite'

@st.cache_resource
def load_embedder():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading MediaPipe Embedder Model...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_large/float32/1/mobilenet_v3_large.tflite", 
            MODEL_PATH
        )
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.ImageEmbedderOptions(base_options=base_options)
    return vision.ImageEmbedder.create_from_options(options)

embedder = load_embedder()

# ==========================================
# 2. HILBERT CURVE MAPPING
# ==========================================
def vector_to_hilbert_image(embedding_vector):
    """
    Maps an embedding vector (up to 1024 elements) into a 32x32 Hilbert curve grid.
    """
    target_size = 1024  # 32x32
    
    # Pad or truncate to exact 1024
    if len(embedding_vector) < target_size:
        embedding_vector = np.pad(embedding_vector, (0, target_size - len(embedding_vector)))
    elif len(embedding_vector) > target_size:
        embedding_vector = embedding_vector[:target_size]
        
    # Normalize to 0-255 (Grayscale)
    min_val, max_val = np.min(embedding_vector), np.max(embedding_vector)
    if max_val > min_val:
        normalized = (embedding_vector - min_val) / (max_val - min_val) * 255.0
    else:
        normalized = embedding_vector * 0
    normalized = normalized.astype(np.uint8)
    
    # Map to 32x32 using 2D Hilbert Curve (p=5 means grid size 2^5 = 32)
    img = np.zeros((32, 32), dtype=np.uint8)
    hc = HilbertCurve(p=5, n=2)
    
    for i in range(target_size):
        coords = hc.point_from_distance(i)
        img[coords[0], coords[1]] = normalized[i]
        
    return img

# ==========================================
# 3. DECODER MODEL ARCHITECTURE
# ==========================================
class EmbeddingDecoder(nn.Module):
    def __init__(self, embedding_size=1024, output_channels=3):
        super(EmbeddingDecoder, self).__init__()
        # Ingests 1024 vector, outputs 128x128 image
        self.fc = nn.Linear(embedding_size, 256 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1), # 128x128
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 8, 8)
        x = self.decoder(x)
        return x

# ==========================================
# 4. STREAMLIT UI PIPELINE
# ==========================================
st.title("🧬 Embeddings & Decoder Pipeline")

tab1, tab2 = st.tabs(["Extraction & Hilbert Mapping", "Train Decoder Model"])

with tab1:
    st.header("1. Extract & Visualize Embeddings")
    uploaded_file = st.file_uploader("Upload Image or Video", type=['png', 'jpg', 'jpeg', 'mp4', 'mov'])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        
        # Check if it's an image
        if uploaded_file.type.startswith('image'):
            img = cv2.imdecode(file_bytes, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            st.image(img_rgb, caption="Original Image", width=300)
            
            if st.button("Extract Embedding"):
                # Extract
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                embed_result = embedder.embed(mp_image)
                embedding_array = embed_result.embeddings[0].embedding
                
                # Render Hilbert Curve
                hilbert_img = vector_to_hilbert_image(embedding_array)
                hilbert_img_resized = cv2.resize(hilbert_img, (256, 256), interpolation=cv2.INTER_NEAREST)
                
                col1, col2 = st.columns(2)
                col1.image(hilbert_img_resized, caption="32x32 Hilbert Curve Embedding (Scaled up)", clamp=True)
                
                # Save options
                st.write("### Save Embedding")
                
                json_data = json.dumps({"embedding": embedding_array.tolist()})
                st.download_button("Download as JSON", json_data, file_name="embedding.json", mime="application/json")
                
                success, encoded_image = cv2.imencode('.jpeg', hilbert_img)
                st.download_button("Download as JPEG (Grayscale)", encoded_image.tobytes(), file_name="embedding_hilbert.jpeg", mime="image/jpeg")

        elif uploaded_file.type.startswith('video'):
            st.write("### Processing Video...")
            st.info("Extracting embeddings frame-by-frame. You will see a live preview below.")
            
            input_path = "temp_in.mp4"
            output_emb_path = "temp_out_emb.webm"
            output_orig_path = "temp_out_orig.webm"
            
            # Save uploaded bytes
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fps == 0 or np.isnan(fps): fps = 30.0
            
            # 1. Setup VideoWriters (VP8 web codec)
            fourcc = cv2.VideoWriter_fourcc(*'vp80')
            
            # Writer for Embedding (Fixed 512x512)
            out_emb_size = (512, 512)
            out_emb = cv2.VideoWriter(output_emb_path, fourcc, fps, out_emb_size, isColor=True)
            
            # Writer for Original Video (Original dimensions)
            out_orig = cv2.VideoWriter(output_orig_path, fourcc, fps, (orig_width, orig_height), isColor=True)
            
            progress_bar = st.progress(0)
            
            # UI Placeholders for live preview
            preview_col1, preview_col2 = st.columns(2)
            frame_window_orig = preview_col1.empty()
            frame_window_emb = preview_col2.empty()
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Write the original frame to our browser-safe webm file (OpenCV expects BGR)
                out_orig.write(frame)
                
                # Convert for MediaPipe and UI (RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Extract Embedding
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                embed_result = embedder.embed(mp_image)
                embedding_array = embed_result.embeddings[0].embedding
                
                # Render Hilbert Curve
                hilbert_img_gray = vector_to_hilbert_image(embedding_array)
                hilbert_img_resized = cv2.resize(hilbert_img_gray, out_emb_size, interpolation=cv2.INTER_NEAREST)
                
                # Write to Video (Convert to BGR for the VideoWriter)
                hilbert_bgr = cv2.cvtColor(hilbert_img_resized, cv2.COLOR_GRAY2BGR)
                out_emb.write(hilbert_bgr)
                
                # Live UI Preview (Update every 3rd frame to prevent Streamlit lag)
                if frame_idx % 3 == 0:
                    frame_window_orig.image(frame_rgb, channels="RGB", caption=f"Original Frame {frame_idx}")
                    frame_window_emb.image(hilbert_img_resized, caption=f"Embedding Frame {frame_idx}")
                    
                frame_idx += 1
                if total_frames > 0:
                    progress_bar.progress(min(frame_idx / total_frames, 1.0))
                    
            cap.release()
            out_emb.release()
            out_orig.release()
            
            st.success("Video processing complete!")
            
            st.write("---")
            st.write("### Final Results")
            
            # Clear the live preview windows
            frame_window_orig.empty()
            frame_window_emb.empty()
            
            # Display Parallel Videos
            vid_col1, vid_col2 = st.columns(2)
            with vid_col1:
                st.write("**Original Video (Browser Safe)**")
                st.video(output_orig_path, format="video/webm")
            with vid_col2:
                st.write("**Hilbert Embedding Video**")
                st.video(output_emb_path, format="video/webm") 
                
            # Download Button
            with open(output_emb_path, "rb") as f:
                st.download_button(
                    label="Download Embedding Video",
                    data=f,
                    file_name="embedding_hilbert.webm",
                    mime="video/webm"
                )

with tab2:
    st.header("2. Train Decoder Local Model")
    st.write("Train a model to reconstruct images from their embeddings.")
    
    data_folder = st.text_input("Dataset Folder Path (containing images in subfolders)", "dataset/")
    
    col1, col2, col3 = st.columns(3)
    epochs = col1.number_input("Epochs", min_value=1, value=10)
    batch_size = col2.number_input("Batch Size", min_value=1, value=8)
    lr = col3.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
    
    if st.button("Start Training Decoder"):
        if not os.path.exists(data_folder):
            st.error("Folder does not exist!")
        else:
            st.info("Preparing Dataset and Extracting Embeddings... This may take a while.")
            
            # Simple Dataset Builder
            images = []
            targets = []
            
            target_transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor()
            ])
            
            for root, _, files in os.walk(data_folder):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(root, file)
                        try:
                            # Load for MediaPipe
                            img_cv = cv2.imread(img_path)
                            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                            emb = embedder.embed(mp_image).embeddings[0].embedding
                            
                            # Padding/truncating
                            if len(emb) < 1024: emb = np.pad(emb, (0, 1024 - len(emb)))
                            elif len(emb) > 1024: emb = emb[:1024]
                            
                            # Load target image for PyTorch
                            pil_img = Image.open(img_path).convert('RGB')
                            tensor_img = target_transform(pil_img)
                            
                            images.append(torch.tensor(emb, dtype=torch.float32))
                            targets.append(tensor_img)
                        except Exception as e:
                            pass
            
            if len(images) == 0:
                st.error("No valid images found or embedding extraction failed.")
            else:
                dataset = list(zip(images, targets))
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = EmbeddingDecoder().to(device)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Training Loop
                for epoch in range(epochs):
                    model.train()
                    epoch_loss = 0
                    for batch_idx, (emb, img_tgt) in enumerate(dataloader):
                        emb, img_tgt = emb.to(device), img_tgt.to(device)
                        
                        optimizer.zero_grad()
                        outputs = model(emb)
                        loss = criterion(outputs, img_tgt)
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        
                    avg_loss = epoch_loss / len(dataloader)
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
                    
                st.success("Training Complete!")
                torch.save(model.state_dict(), "decoder_model.pth")
                st.write("Model saved to `decoder_model.pth`")